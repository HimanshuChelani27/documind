import json
import os
import tempfile
from pathlib import Path

# Must be set before the app (and its engine) is imported.
_DATA = tempfile.mkdtemp(prefix="documind-test-")
os.environ["DOCUMIND_DATA_DIR"] = _DATA
os.environ["DOCUMIND_DATABASE_URL"] = "sqlite://"
os.environ["DOCUMIND_LLM_PROVIDER"] = "extractive"

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

from app.main import app

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # runs lifespan: models, sample library, gold set
        yield c


@pytest.fixture(autouse=True)
def _reset_rate_limits(client):
    client.app.state.limiter.reset()


@pytest.fixture(scope="session")
def gold() -> list[dict]:
    return [json.loads(line) for line in (SAMPLES / "eval" / "gold.jsonl").read_text().splitlines() if line]


def make_pdf(pages: list[str]) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


def read_sse(response) -> list[dict]:
    events = []
    for block in response.text.split("\n\n"):
        data = [line[6:] for line in block.split("\n") if line.startswith("data: ")]
        if data:
            events.append(json.loads("\n".join(data)))
    return events
