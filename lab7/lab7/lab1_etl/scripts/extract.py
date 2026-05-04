from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

from utils import get_logger, log_to_db

logger = get_logger("etl.extract")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_FILE = RAW_DIR / "api_weather_cache.csv"


def extract_from_csv(run_id: str) -> Dict[str, pd.DataFrame]:
    start = datetime.now()
    logger.info("Bat dau extract tu CSV")

    sales = pd.read_csv(RAW_DIR / "sales.csv")
    stores = pd.read_csv(RAW_DIR / "stores.csv")
    features = pd.read_csv(RAW_DIR / "features.csv")

    total_rows = len(sales) + len(stores) + len(features)
    log_to_db(run_id, "extract_csv", "SUCCESS", total_rows, "Da nap 3 file CSV", start)
    return {"sales": sales, "stores": stores, "features": features}


def extract_from_api(run_id: str) -> pd.DataFrame:
    start = datetime.now()
    logger.info("Bat dau extract du lieu bo sung tu API")

    api_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 10.8231,
        "longitude": 106.6297,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "Asia/Bangkok",
        "start_date": "2024-01-05",
        "end_date": "2024-01-19",
    }

    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json().get("daily", {})
        api_df = pd.DataFrame(
            {
                "date": payload.get("time", []),
                "api_temp_max": payload.get("temperature_2m_max", []),
                "api_temp_min": payload.get("temperature_2m_min", []),
            }
        )
        if not api_df.empty:
            api_df.to_csv(CACHE_FILE, index=False)
        log_to_db(run_id, "extract_api", "SUCCESS", len(api_df), "Lay du lieu weather API", start)
        return api_df
    except Exception as exc:
        logger.warning("Khong goi duoc API, thu cache: %s", exc)
        if CACHE_FILE.exists():
            cached = pd.read_csv(CACHE_FILE)
            log_to_db(run_id, "extract_api", "SUCCESS", len(cached), "Dung cache weather API", start)
            return cached

        log_to_db(run_id, "extract_api", "FAILED", 0, str(exc), start)
        return pd.DataFrame(columns=["date", "api_temp_max", "api_temp_min"])
