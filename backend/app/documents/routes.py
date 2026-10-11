import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlmodel import Session, func, select

from app.config import get_settings
from app.db import engine, get_session
from app.ingest import pipeline
from app.ingest.pdf import PdfError, is_pdf
from app.memory import trim
from app.models import LIBRARY, Chunk, Document
from app.ratelimit import enforce
from app.sse import Channel, channel_response
from app.workspaces import Workspace, delete_document, get_workspace, require_workspace

logger = logging.getLogger("documind.documents")
router = APIRouter(tags=["documents"])


def _out(doc: Document) -> dict:
    return {**doc.model_dump(), "scope": "library" if doc.collection == LIBRARY else "workspace"}


def _visible(session: Session, doc_id: str, ws: Workspace) -> Document:
    doc = session.get(Document, doc_id)
    if not doc or not ws.can_see(doc.collection):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@router.get("/documents")
def list_documents(ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)):
    docs = session.exec(
        select(Document).where(Document.collection.in_(ws.collections())).order_by(Document.created_at)  # type: ignore[attr-defined]
    ).all()
    return [_out(d) for d in docs]


@router.get("/documents/{doc_id}")
def get_document(
    doc_id: str, ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)
):
    return _out(_visible(session, doc_id, ws))


@router.get("/documents/{doc_id}/chunks")
def get_chunks(doc_id: str, ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)):
    _visible(session, doc_id, ws)
    return session.exec(select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.ordinal)).all()


@router.get("/documents/{doc_id}/file")
def get_file(doc_id: str, ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)):
    doc = _visible(session, doc_id, ws)
    s = get_settings()
    path = (
        (s.samples_dir / doc.filename)
        if doc.collection == LIBRARY
        else (s.upload_dir / doc.collection / f"{doc.id}.pdf")
    )
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not available")
    return FileResponse(
        path, media_type="application/pdf", filename=doc.filename, content_disposition_type="inline"
    )


@router.delete("/documents/{doc_id}", status_code=204)
async def remove_document(
    doc_id: str, ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)
):
    doc = _visible(session, doc_id, ws)
    if doc.collection == LIBRARY:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "The sample library is read-only")
    if doc.status == "processing":
        raise HTTPException(status.HTTP_409_CONFLICT, "Still processing, try again in a moment")
    await asyncio.to_thread(delete_document, session, doc)


@router.post("/documents", status_code=202)
async def upload(
    file: UploadFile,
    request: Request,
    ws: Workspace = Depends(get_workspace),
    session: Session = Depends(get_session),
):
    """Upload a PDF to your private workspace. Returns at once; follow `/ingest/{job}/events`."""
    s = get_settings()
    collection = require_workspace(ws)
    count = session.exec(
        select(func.count()).select_from(Document).where(Document.collection == collection)
    ).one()
    if count >= s.max_docs_per_workspace:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Workspace limit is {s.max_docs_per_workspace} documents; delete one first",
        )
    enforce(request, "upload", s.uploads_per_hour, 3600, "uploads")

    max_bytes = int(s.max_upload_mb * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File is larger than {s.max_upload_mb:g} MB")
    if not is_pdf(data):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are supported")
    try:  # cheap structural check up front so obvious problems fail the request, not the job
        await asyncio.to_thread(_check_pages, data, s.max_pages)
    except PdfError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e

    name = (file.filename or "document.pdf").replace("/", "_").replace("\\", "_")[-120:]
    doc = Document(collection=collection, filename=name, title=name.rsplit(".", 1)[0], size_bytes=len(data))
    session.add(doc)
    session.commit()
    session.refresh(doc)
    path = s.upload_dir / collection / f"{doc.id}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

    channel = Channel()
    request.app.state.jobs[doc.id] = channel
    task = asyncio.create_task(_run_ingest(doc.id, data, channel))
    request.app.state.tasks.add(task)
    task.add_done_callback(request.app.state.tasks.discard)
    return {"job": doc.id, "document": _out(doc)}


def _check_pages(data: bytes, max_pages: int) -> None:
    from io import BytesIO

    from pypdf import PdfReader

    try:
        n = len(PdfReader(BytesIO(data)).pages)
    except Exception as e:
        raise PdfError("Could not read this PDF") from e
    if n > max_pages:
        raise PdfError(f"PDF has {n} pages; the limit is {max_pages}")


async def _run_ingest(doc_id: str, data: bytes, channel: Channel) -> None:
    def work():
        with Session(engine) as session:
            doc = session.get(Document, doc_id)
            if doc is None:
                return None
            doc = pipeline.ingest(
                session,
                doc,
                data,
                lambda stage, p: channel.publish_threadsafe(
                    {"type": "progress", "stage": stage, "progress": round(p, 2)}
                ),
            )
            return _out(doc)

    try:
        result = await asyncio.to_thread(work)
        trim()
        await channel.publish({"type": "done", "document": result}, final=True)
    except Exception:
        logger.exception("Ingestion job %s crashed", doc_id)
        await channel.publish({"type": "done", "document": None, "error": "Ingestion failed"}, final=True)


@router.get("/ingest/{job}/events")
async def ingest_events(job: str, request: Request):
    channel = request.app.state.jobs.get(job)
    if channel is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such ingestion job")
    return channel_response(channel, request)
