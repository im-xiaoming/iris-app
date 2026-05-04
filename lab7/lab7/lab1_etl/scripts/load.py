from __future__ import annotations

from datetime import datetime
from typing import Dict

import pandas as pd
from sqlalchemy import text

from utils import db_connection, get_logger, log_to_db

logger = get_logger("etl.load")


def _truncate_tables() -> None:
    with db_connection() as connection:
        connection.execute(text("TRUNCATE TABLE fact_sales RESTART IDENTITY CASCADE"))
        connection.execute(text("TRUNCATE TABLE dim_date CASCADE"))
        connection.execute(text("TRUNCATE TABLE dim_store CASCADE"))


def _load_table(df: pd.DataFrame, table_name: str) -> int:
    with db_connection() as connection:
        df.to_sql(table_name, con=connection.connection, if_exists="append", index=False, method="multi")
    return len(df)


def load_all(data: Dict[str, pd.DataFrame], run_id: str, skip_db: bool = False) -> None:
    start = datetime.now()
    logger.info("Bat dau load vao Data Warehouse")

    if skip_db:
        logger.info("Bo qua DB load vi skip_db=True")
        log_to_db(run_id, "load", "SUCCESS", len(data["fact_sales"]), "Skip DB load", start)
        return

    try:
        _truncate_tables()
        _load_table(data["dim_store"], "dim_store")
        _load_table(data["dim_date"], "dim_date")
        _load_table(data["fact_sales"], "fact_sales")
        log_to_db(run_id, "load", "SUCCESS", len(data["fact_sales"]), "Load thanh cong", start)
    except Exception as exc:
        logger.exception("Load that bai: %s", exc)
        log_to_db(run_id, "load", "FAILED", 0, str(exc), start)
        raise
