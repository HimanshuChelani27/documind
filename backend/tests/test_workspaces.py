from datetime import UTC, datetime, timedelta

from sqlmodel import Session

from app.db import engine
from app.workspaces import cleanup_expired
from tests.conftest import make_pdf, read_sse

WS_A = {"X-Workspace": "workspace-aaaa-1111"}
WS_B = {"X-Workspace": "workspace-bbbb-2222"}

PAGES = [
    "Zephyr Labs onboarding guide. New hires receive a Zephyr badge on day one.",
    "The Zephyr cafeteria serves free lunch on Wednesdays. The quokka room is on floor 3.",
]


def _upload(client, headers, data=None, name="guide.pdf"):
    return client.post(
        "/documents", headers=headers, files={"file": (name, data or make_pdf(PAGES), "application/pdf")}
    )


def test_rejects_non_pdf_and_oversize(client):
    r = _upload(client, WS_A, b"hello, I am not a pdf", "evil.pdf")
    assert r.status_code == 415
    big = b"%PDF-1.4\n" + b"0" * (6 * 1024 * 1024)
    assert _upload(client, WS_A, big).status_code == 413
    assert (
        client.post("/documents", files={"file": ("x.pdf", make_pdf(PAGES), "application/pdf")}).status_code
        == 400
    )


def test_rejects_too_many_pages(client):
    assert _upload(client, WS_A, make_pdf(["page"] * 41)).status_code == 422


def test_upload_ingest_and_isolation(client):
    r = _upload(client, WS_A)
    assert r.status_code == 202
    job = r.json()["job"]
    events = read_sse(client.get(f"/ingest/{job}/events"))
    stages = [e.get("stage") for e in events if e["type"] == "progress"]
    assert stages[0] == "parse" and "embed" in stages and stages[-1] == "done"
    doc = events[-1]["document"]
    assert doc["status"] == "ready" and doc["pages"] == 2 and doc["scope"] == "workspace"

    # A sees it, B and anonymous visitors don't
    assert any(d["id"] == doc["id"] for d in client.get("/documents", headers=WS_A).json())
    assert all(d["id"] != doc["id"] for d in client.get("/documents", headers=WS_B).json())
    assert client.get(f"/documents/{doc['id']}/chunks", headers=WS_B).status_code == 404
    assert client.delete(f"/documents/{doc['id']}", headers=WS_B).status_code == 404

    # Chat over A's upload finds page 2; B's workspace is empty
    events = read_sse(
        client.post("/chat", headers=WS_A, json={"question": "When is lunch free?", "scope": "workspace"})
    )
    assert events[0]["hits"][0]["page"] == 2
    assert "Wednesdays" in events[-2]["answer"]
    empty = read_sse(client.post("/chat", headers=WS_B, json={"question": "lunch?", "scope": "workspace"}))
    assert empty[0]["hits"] == []

    assert client.delete(f"/documents/{doc['id']}", headers=WS_A).status_code == 204
    assert client.get(f"/documents/{doc['id']}", headers=WS_A).status_code == 404


def test_workspace_document_limit(client):
    headers = {"X-Workspace": "workspace-limit-3333"}
    for _ in range(5):
        job = _upload(client, headers).json()["job"]
        read_sse(client.get(f"/ingest/{job}/events"))
        client.app.state.limiter.reset()
    assert _upload(client, headers).status_code == 409


def test_ttl_cleanup_removes_expired_uploads(client):
    headers = {"X-Workspace": "workspace-ttl-4444"}
    job = _upload(client, headers).json()["job"]
    read_sse(client.get(f"/ingest/{job}/events"))
    assert client.get("/documents", headers=headers).json()[-1]["scope"] == "workspace"
    with Session(engine) as session:
        removed = cleanup_expired(session, now=datetime.now(UTC) + timedelta(hours=25))
    assert removed >= 1
    assert all(d["scope"] == "library" for d in client.get("/documents", headers=headers).json())


def test_invalid_workspace_header(client):
    assert client.get("/documents", headers={"X-Workspace": "../../etc"}).status_code == 400
