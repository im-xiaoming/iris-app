from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from extract import extract_from_api, extract_from_csv
from load import load_all
from transform import transform
from utils import get_logger

logger = get_logger("etl.main")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ETL pipeline for retail sales")
    parser.add_argument("--skip-db", action="store_true", help="Khong load vao PostgreSQL")
    args = parser.parse_args()

    run_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("Khoi dong pipeline | run_id=%s", run_id)

    csv_data = extract_from_csv(run_id)
    api_data = extract_from_api(run_id)
    transformed = transform(csv_data, api_data, run_id)
    load_all(transformed, run_id, skip_db=args.skip_db)
    logger.info("Pipeline hoan thanh")


if __name__ == "__main__":
    main()
