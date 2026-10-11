import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

LIBRARY = "library"


def _now() -> datetime:
    return datetime.now(UTC)


def _id() -> str:
    return uuid.uuid4().hex[:12]


class Document(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    collection: str = Field(index=True)  # "library" or "ws_<workspace id>"
    filename: str
    title: str
    pages: int = 0
    chunks: int = 0
    size_bytes: int = 0
    status: str = "processing"  # processing | ready | failed
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)


class Chunk(SQLModel, table=True):
    id: str = Field(primary_key=True)  # "<doc id>:<ordinal>"
    doc_id: str = Field(index=True, foreign_key="document.id")
    collection: str = Field(index=True)
    ordinal: int
    page: int
    heading: str | None = None
    char_start: int
    text: str


class EvalDataset(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    name: str
    collection: str = Field(index=True)  # which collection the questions are about
    owner: str | None = Field(default=None, index=True)  # workspace id; None = built-in
    builtin: bool = False
    created_at: datetime = Field(default_factory=_now)


class EvalItem(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    dataset_id: str = Field(index=True, foreign_key="evaldataset.id")
    question: str
    answer: str | None = None  # reference answer (optional)
    doc: str  # filename the gold pages belong to
    gold_pages: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    type: str | None = None  # keyword | paraphrase


class EvalRun(SQLModel, table=True):
    id: str = Field(default_factory=_id, primary_key=True)
    dataset_id: str = Field(index=True)
    collection: str = Field(index=True)
    owner: str | None = Field(default=None, index=True)  # workspace id; None = shared baseline
    strategies: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    k: int = 5
    with_answers: bool = False
    items: int = 0
    status: str = "running"  # running | done | failed
    mode: str = "extractive"  # live (LLM judge) | extractive (keyless proxies)
    judge: str | None = None  # provider/model used as judge
    summary: dict = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None


class EvalResult(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, foreign_key="evalrun.id")
    item_id: str
    strategy: str
    metrics: dict = Field(default_factory=dict, sa_column=Column(JSON))
    retrieved: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    answer: str | None = None
