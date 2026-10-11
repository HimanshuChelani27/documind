"""Prepares data at build time: ingests the sample library, loads the gold set and runs the
baseline evaluation, so a production start only loads what is already on disk.

    uv run python -m app.prebuild
"""

import asyncio
import logging

from sqlmodel import Session

from app.config import get_settings
from app.db import engine, init_db
from app.eval.runner import ensure_baseline, seed_gold
from app.ingest.embed import warm_up
from app.ingest.pipeline import seed_library


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    warm_up()
    init_db()
    with Session(engine) as session:
        seed_library(session)
        seed_gold(session, settings.samples_dir / "eval" / "gold.jsonl")
    asyncio.run(ensure_baseline())
    logging.getLogger("documind").info("Prebuild complete: %s", settings.data_dir)


if __name__ == "__main__":
    main()
