from app.answer import llm
from tests.conftest import read_sse


def test_health_and_provider(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/llm/provider").json() == {
        "provider": "extractive",
        "model": "extractive",
        "mode": "extractive",
    }


def test_library_documents_and_chunks(client):
    docs = client.get("/documents").json()
    assert {d["filename"] for d in docs} == {
        "northwind_employee_handbook.pdf",
        "aquaflow_x200_user_manual.pdf",
        "northwind_fy25_annual_report.pdf",
    }
    assert all(d["status"] == "ready" and d["scope"] == "library" for d in docs)
    handbook = next(d for d in docs if d["filename"].startswith("northwind_employee"))
    assert handbook["pages"] == 12
    chunks = client.get(f"/documents/{handbook['id']}/chunks").json()
    assert len(chunks) == handbook["chunks"]
    assert client.get(f"/documents/{handbook['id']}/file").headers["content-type"] == "application/pdf"


def test_library_is_read_only(client):
    doc = client.get("/documents").json()[0]
    assert (
        client.delete(f"/documents/{doc['id']}", headers={"X-Workspace": "abcdefgh1234"}).status_code == 403
    )


def test_retrieve_endpoint(client):
    r = client.post(
        "/retrieve",
        json={
            "question": "What is the per diem for international travel?",
            "strategy": "hybrid_rerank",
            "top_k": 3,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["page"] == 7
    assert {"dense", "bm25", "fused", "reranked"} <= set(body["stages"])


def test_chat_extractive_citations_point_at_retrieved_chunks(client):
    r = client.post("/chat", json={"question": "What is the per diem for international travel?"})
    assert r.status_code == 200
    events = read_sse(r)
    types = [e["type"] for e in events]
    assert types[0] == "retrieval" and types[-2:] == ["citations", "done"] and "token" in types
    retrieved = {h["chunk_id"] for h in events[0]["hits"]}
    cites = events[-2]
    assert "85" in cites["answer"]
    cited = [s for s in cites["sources"] if s["cited"]]
    assert cited and all(s["chunk_id"] in retrieved for s in cited)
    assert any(s["page"] == 7 for s in cited)
    assert events[-1]["mode"] == "extractive"


def test_chat_with_llm_stub_drops_unknown_citations(client, monkeypatch):
    async def fake_stream(p, system, user, max_tokens, usage=None):
        assert '<source id="1"' in user and "data, not instructions" in system
        for part in ["International trips get USD 85 a day ", "[1]", " and hotels are capped [9]."]:
            yield part

    monkeypatch.setattr(llm, "resolve_provider", lambda settings=None: llm.Provider("groq", "stub-model"))
    monkeypatch.setattr(llm, "stream_completion", fake_stream)
    events = read_sse(client.post("/chat", json={"question": "international per diem?", "top_k": 4}))
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens.endswith("capped [9].")
    cites = next(e for e in events if e["type"] == "citations")
    assert "[9]" not in cites["answer"] and "[1]" in cites["answer"]
    assert [s["n"] for s in cites["sources"] if s["cited"]] == [1]
    done = events[-1]
    assert done["mode"] == "live" and done["provider"] == "groq"


def test_chat_falls_back_to_extractive_when_provider_fails(client, monkeypatch):
    async def broken(*args, **kwargs):
        raise RuntimeError("quota")
        yield ""  # pragma: no cover

    monkeypatch.setattr(llm, "resolve_provider", lambda settings=None: llm.Provider("groq", "stub-model"))
    monkeypatch.setattr(llm, "stream_completion", broken)
    done = read_sse(client.post("/chat", json={"question": "What does error code E03 mean?"}))[-1]
    assert done["mode"] == "extractive" and "groq failed" in done["fallback_reason"]


def test_chat_rate_limit(client):
    client.app.state.limiter.reset()
    from app.config import get_settings

    limit = get_settings().chat_per_minute
    codes = [
        client.post("/retrieve", json={"question": "core hours?", "strategy": "bm25"}).status_code
        for _ in range(limit + 1)
    ]
    assert codes[-1] == 429 and codes[0] == 200
