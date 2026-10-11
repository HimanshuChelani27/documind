import time

from tests.conftest import read_sse

WS = {"X-Workspace": "workspace-eval-5555"}


def test_baseline_run_on_gold_set(client):
    for _ in range(100):
        runs = [r for r in client.get("/eval/runs").json() if r["shared"]]
        if runs and runs[0]["status"] != "running":
            break
        time.sleep(0.2)
    run = runs[0]
    assert run["status"] == "done" and run["items"] == 40
    summary = run["summary"]["strategies"]
    assert set(summary) == {"dense", "bm25", "hybrid", "hybrid_rerank"}
    assert summary["hybrid_rerank"]["recall"] >= summary["bm25"]["recall"]
    detail = client.get(f"/eval/runs/{run['id']}").json()
    assert len(detail["results"]) == 40
    assert set(run["summary"]["by_type"]["dense"]) == {"keyword", "paraphrase"}


def test_custom_dataset_run_with_extractive_answers(client):
    docs = client.get("/documents").json()
    manual = next(d for d in docs if d["filename"].startswith("aquaflow"))
    ds = client.post("/eval/datasets", headers=WS, json={"name": "Pump", "scope": "library"}).json()
    for q, pages in [("What does E07 mean?", [8]), ("What is the impeller part number?", [9, 6])]:
        r = client.post(
            f"/eval/datasets/{ds['id']}/items",
            headers=WS,
            json={"question": q, "doc_id": manual["id"], "gold_pages": pages},
        )
        assert r.status_code == 201
    bad = client.post(
        f"/eval/datasets/{ds['id']}/items",
        headers=WS,
        json={"question": "x?x", "doc_id": manual["id"], "gold_pages": [99]},
    )
    assert bad.status_code == 422
    # other workspaces can't see or edit it
    assert (
        client.get(f"/eval/datasets/{ds['id']}/items", headers={"X-Workspace": "someone-else-1"}).status_code
        == 404
    )
    assert (
        client.post(
            "/eval/datasets/gold/items",
            headers=WS,
            json={"question": "q??", "doc_id": manual["id"], "gold_pages": [1]},
        ).status_code
        == 403
    )

    run = client.post(
        "/eval/runs",
        headers=WS,
        json={"dataset_id": ds["id"], "strategies": ["bm25", "dense"], "k": 3, "with_answers": True},
    )
    assert run.status_code == 202
    events = read_sse(client.get(f"/eval/runs/{run.json()['id']}/events", headers=WS))
    assert [e["type"] for e in events].count("progress") == 4
    assert events[-1]["status"] == "done"
    detail = client.get(f"/eval/runs/{run.json()['id']}", headers=WS).json()
    assert detail["mode"] == "extractive"
    first = detail["results"][0]["strategies"]["bm25"]
    assert first["answer"] and {"hit", "recall", "mrr", "ndcg", "supported_share", "qa_similarity"} <= set(
        first["metrics"]
    )
    assert (
        client.get(f"/eval/runs/{run.json()['id']}", headers={"X-Workspace": "someone-else-1"}).status_code
        == 404
    )
