from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from utils import get_logger, log_to_db

logger = get_logger("etl.transform")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    )
    return df


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["weekly_sales"] = pd.to_numeric(df["weekly_sales"], errors="coerce")
    df["weekly_sales"] = df["weekly_sales"].fillna(df["weekly_sales"].median())
    df["isholiday"] = (
        df["isholiday"].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).fillna(False)
    )

    q1 = df["weekly_sales"].quantile(0.25)
    q3 = df["weekly_sales"].quantile(0.75)
    iqr = q3 - q1
    lower = max(0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    df["weekly_sales"] = df["weekly_sales"].clip(lower=lower, upper=upper)

    df = df.drop_duplicates(subset=["store", "dept", "date"]).dropna(subset=["date"])
    return df.rename(columns={"isholiday": "is_holiday"})


def clean_stores(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df)
    df["opened_date"] = pd.to_datetime(df["opened_date"], errors="coerce")
    df["type"] = df["type"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["region"] = df["region"].fillna("UNKNOWN").astype(str).str.strip().str.title()
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0).astype(int)
    return df.rename(
        columns={"store": "store_id", "type": "store_type", "size": "store_size"}
    )[["store_id", "store_type", "region", "store_size", "opened_date"]]


def clean_features(df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    markdown_cols = ["markdown1", "markdown2", "markdown3", "markdown4", "markdown5"]
    for col in markdown_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["temperature", "fuel_price", "cpi", "unemployment"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[["temperature", "fuel_price", "cpi", "unemployment"]] = df[
        ["temperature", "fuel_price", "cpi", "unemployment"]
    ].ffill().bfill()
    df["isholiday"] = (
        df["isholiday"].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).fillna(False)
    )
    df = df.rename(columns={"isholiday": "is_holiday"})

    api_df = api_df.copy()
    if not api_df.empty:
        api_df["date"] = pd.to_datetime(api_df["date"], errors="coerce")
        api_df["api_avg_temp"] = (
            pd.to_numeric(api_df["api_temp_max"], errors="coerce")
            + pd.to_numeric(api_df["api_temp_min"], errors="coerce")
        ) / 2
        df = df.merge(api_df[["date", "api_avg_temp"]], on="date", how="left")
        df["temperature"] = df["temperature"].fillna(df["api_avg_temp"])
        df = df.drop(columns=["api_avg_temp"])

    df["promo_markdown"] = df[markdown_cols].sum(axis=1)
    return df


def build_dim_date(fact_source: pd.DataFrame) -> pd.DataFrame:
    dim_date = pd.DataFrame({"date_id": pd.to_datetime(fact_source["date"]).drop_duplicates().sort_values()})
    dim_date["year"] = dim_date["date_id"].dt.year
    dim_date["month"] = dim_date["date_id"].dt.month
    dim_date["month_name"] = dim_date["date_id"].dt.month_name()
    dim_date["quarter"] = dim_date["date_id"].dt.quarter
    dim_date["week_of_year"] = dim_date["date_id"].dt.isocalendar().week.astype(int)
    dim_date["is_weekend"] = dim_date["date_id"].dt.dayofweek >= 5
    return dim_date


def build_fact_sales(sales_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    fact = sales_df.merge(
        features_df[["store", "date", "temperature", "fuel_price", "cpi", "unemployment", "promo_markdown"]],
        on=["store", "date"],
        how="left",
    )
    return fact.rename(columns={"store": "store_id", "date": "date_id"})[
        [
            "store_id",
            "dept",
            "date_id",
            "weekly_sales",
            "is_holiday",
            "temperature",
            "fuel_price",
            "cpi",
            "unemployment",
            "promo_markdown",
        ]
    ].assign(source_system="csv_api")


def transform(data: Dict[str, pd.DataFrame], api_df: pd.DataFrame, run_id: str) -> Dict[str, pd.DataFrame]:
    start = datetime.now()
    logger.info("Bat dau transform du lieu")

    sales_clean = clean_sales(data["sales"])
    stores_clean = clean_stores(data["stores"])
    features_clean = clean_features(data["features"], api_df)
    fact_sales = build_fact_sales(sales_clean, features_clean)
    dim_date = build_dim_date(fact_sales.rename(columns={"date_id": "date"}))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fact_sales.to_csv(PROCESSED_DIR / "fact_sales.csv", index=False)
    stores_clean.to_csv(PROCESSED_DIR / "dim_store.csv", index=False)
    dim_date.to_csv(PROCESSED_DIR / "dim_date.csv", index=False)
    features_clean.to_csv(PROCESSED_DIR / "features_enriched.csv", index=False)

    log_to_db(run_id, "transform", "SUCCESS", len(fact_sales), "Transform thanh cong", start)
    return {
        "fact_sales": fact_sales,
        "dim_store": stores_clean,
        "dim_date": dim_date,
        "features_enriched": features_clean,
    }
