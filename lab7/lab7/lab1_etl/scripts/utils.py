from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "etl.log"

load_dotenv(PROJECT_ROOT / ".env")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _db_url() -> str:
    return (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


def get_engine():
    return create_engine(_db_url())


@contextmanager
def db_connection() -> Iterator:
    engine = get_engine()
    with engine.begin() as connection:
        yield connection


def log_to_db(
    run_id: str,
    step: str,
    status: str,
    rows_count: int = 0,
    message: str = "",
    started_at: Optional[datetime] = None,
) -> None:
    started = started_at or datetime.now()
    try:
        with db_connection() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO etl_log (run_id, step, status, rows_count, message, started_at, ended_at)
                    VALUES (:run_id, :step, :status, :rows_count, :message, :started_at, :ended_at)
                    """
                ),
                {
                    "run_id": run_id,
                    "step": step,
                    "status": status,
                    "rows_count": rows_count,
                    "message": message,
                    "started_at": started,
                    "ended_at": datetime.now(),
                },
            )
    except Exception as exc:
        get_logger("etl.db_log").warning("Khong the ghi log vao DB: %s", exc)
