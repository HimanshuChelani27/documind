import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlmodel import Session, col, delete, func, or_, select

from app.config import get_settings
from app.db import get_session
from app.eval.runner import ANSWER_ITEMS_LIMIT, run_eval
from app.models import Document, EvalDataset, EvalItem, EvalResult, EvalRun
from app.ratelimit import enforce
from app.retrieval.strategies import STRATEGIES, Strategy
from app.sse import Channel, channel_response
from app.workspaces import Scope, Workspace, get_workspace, require_workspace

router = APIRouter(prefix="/eval", tags=["evaluation"])
MAX_CUSTOM_ITEMS = 30


def _dataset(session: Session, dataset_id: str, ws: Workspace, write: bool = False) -> EvalDataset:
    ds = session.get(EvalDataset, dataset_id)
    if not ds or not (ds.builtin or ds.owner == ws.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    if write and ds.builtin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Built-in datasets are read-only")
    return ds


def _run(session: Session, run_id: str, ws: Workspace) -> EvalRun:
    run = session.get(EvalRun, run_id)
    if not run or not (run.owner is None or run.owner == ws.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return run


@router.get("/datasets")
def list_datasets(ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)):
    rows = session.exec(
        select(EvalDataset)
        .where(or_(col(EvalDataset.builtin), EvalDataset.owner == ws.id))
        .order_by(col(EvalDataset.builtin).desc(), EvalDataset.created_at)
    ).all()
    counts = dict(session.exec(select(EvalItem.dataset_id, func.count()).group_by(EvalItem.dataset_id)).all())
    return [
        {
            **d.model_dump(),
            "items": counts.get(d.id, 0),
            "scope": "library" if d.collection == "library" else "workspace",
        }
        for d in rows
    ]


class DatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    scope: Scope = "library"


@router.post("/datasets", status_code=201)
def create_dataset(
    body: DatasetIn, ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)
):
    require_workspace(ws)
    if (
        session.exec(select(func.count()).select_from(EvalDataset).where(EvalDataset.owner == ws.id)).one()
        >= 5
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, "You can keep up to 5 datasets")
    ds = EvalDataset(name=body.name.strip(), collection=ws.resolve(body.scope), owner=ws.id)
    session.add(ds)
    session.commit()
    session.refresh(ds)
    return {**ds.model_dump(), "items": 0, "scope": body.scope}


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(
    dataset_id: str, ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)
):
    ds = _dataset(session, dataset_id, ws, write=True)
    session.exec(delete(EvalItem).where(col(EvalItem.dataset_id) == ds.id))
    session.delete(ds)
    session.commit()


@router.get("/datasets/{dataset_id}/items")
def list_items(
    dataset_id: str, ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)
):
    _dataset(session, dataset_id, ws)
    return session.exec(select(EvalItem).where(EvalItem.dataset_id == dataset_id)).all()


class ItemIn(BaseModel):
    question: str = Field(min_length=3, max_length=300)
    doc_id: str
    gold_pages: list[int] = Field(min_length=1, max_length=5)
    answer: str | None = Field(default=None, max_length=500)
    type: str | None = Field(default=None, pattern="^(keyword|paraphrase)$")


@router.post("/datasets/{dataset_id}/items", status_code=201)
def add_item(
    dataset_id: str,
    body: ItemIn,
    ws: Workspace = Depends(get_workspace),
    session: Session = Depends(get_session),
):
    ds = _dataset(session, dataset_id, ws, write=True)
    doc = session.get(Document, body.doc_id)
    if not doc or doc.collection != ds.collection:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Pick a document from this dataset's collection"
        )
    if any(p < 1 or p > doc.pages for p in body.gold_pages):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Pages must be between 1 and {doc.pages}")
    n = session.exec(select(func.count()).select_from(EvalItem).where(EvalItem.dataset_id == ds.id)).one()
    if n >= MAX_CUSTOM_ITEMS:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Datasets hold up to {MAX_CUSTOM_ITEMS} questions")
    item = EvalItem(
        dataset_id=ds.id,
        question=body.question.strip(),
        answer=body.answer,
        doc=doc.filename,
        gold_pages=sorted(set(body.gold_pages)),
        type=body.type,
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


@router.delete("/datasets/{dataset_id}/items/{item_id}", status_code=204)
def delete_item(
    dataset_id: str,
    item_id: str,
    ws: Workspace = Depends(get_workspace),
    session: Session = Depends(get_session),
):
    _dataset(session, dataset_id, ws, write=True)
    item = session.get(EvalItem, item_id)
    if not item or item.dataset_id != dataset_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Question not found")
    session.delete(item)
    session.commit()


class RunIn(BaseModel):
    dataset_id: str
    strategies: list[Strategy] = Field(default_factory=lambda: list(STRATEGIES), min_length=1)
    k: int = Field(default=5, ge=1, le=10)
    with_answers: bool = False


@router.post("/runs", status_code=202)
async def start_run(
    body: RunIn,
    request: Request,
    ws: Workspace = Depends(get_workspace),
    session: Session = Depends(get_session),
):
    ds = _dataset(session, body.dataset_id, ws)
    n = session.exec(select(func.count()).select_from(EvalItem).where(EvalItem.dataset_id == ds.id)).one()
    if not n:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "This dataset has no questions yet")
    enforce(request, "eval", get_settings().eval_runs_per_hour, 3600, "eval runs")
    run = EvalRun(
        dataset_id=ds.id,
        collection=ds.collection,
        owner=ws.id,
        items=n,
        strategies=list(dict.fromkeys(body.strategies)),
        k=body.k,
        with_answers=body.with_answers,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    channel = Channel()
    request.app.state.jobs[f"eval:{run.id}"] = channel
    task = asyncio.create_task(run_eval(run.id, channel))
    request.app.state.tasks.add(task)
    task.add_done_callback(request.app.state.tasks.discard)
    return {**run.model_dump(), "answer_items_limit": ANSWER_ITEMS_LIMIT}


@router.get("/runs")
def list_runs(ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)):
    runs = session.exec(
        select(EvalRun)
        .where(or_(col(EvalRun.owner).is_(None), EvalRun.owner == ws.id))
        .order_by(col(EvalRun.created_at).desc())
        .limit(30)
    ).all()
    names = {d.id: d.name for d in session.exec(select(EvalDataset)).all()}
    return [
        {
            **r.model_dump(),
            "dataset_name": names.get(r.dataset_id, "deleted dataset"),
            "shared": r.owner is None,
        }
        for r in runs
    ]


@router.get("/runs/{run_id}")
def get_run(run_id: str, ws: Workspace = Depends(get_workspace), session: Session = Depends(get_session)):
    run = _run(session, run_id, ws)
    items = {
        i.id: i for i in session.exec(select(EvalItem).where(EvalItem.dataset_id == run.dataset_id)).all()
    }
    results = session.exec(select(EvalResult).where(EvalResult.run_id == run.id)).all()
    per_item: dict[str, dict] = {}
    for r in results:
        item = items.get(r.item_id)
        row = per_item.setdefault(
            r.item_id,
            {
                "item_id": r.item_id,
                "question": item.question if item else "(deleted)",
                "reference": item.answer if item else None,
                "doc": item.doc if item else None,
                "gold_pages": item.gold_pages if item else [],
                "type": item.type if item else None,
                "strategies": {},
            },
        )
        row["strategies"][r.strategy] = {"metrics": r.metrics, "retrieved": r.retrieved, "answer": r.answer}
    return {**run.model_dump(), "results": list(per_item.values())}


@router.get("/runs/{run_id}/events")
async def run_events(
    run_id: str,
    request: Request,
    ws: Workspace = Depends(get_workspace),
    session: Session = Depends(get_session),
):
    run = _run(session, run_id, ws)
    channel = request.app.state.jobs.get(f"eval:{run.id}")
    if channel is None:
        if run.status == "running":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No live events for this run")
        channel = Channel()
        await channel.publish(
            {"type": "done", "status": run.status, "summary": run.summary, "error": run.error}, final=True
        )
    return channel_response(channel, request)
