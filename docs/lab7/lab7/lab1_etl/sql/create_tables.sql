CREATE TABLE IF NOT EXISTS dim_store (
    store_id INTEGER PRIMARY KEY,
    store_type VARCHAR(10) NOT NULL,
    region VARCHAR(50) NOT NULL,
    store_size INTEGER NOT NULL,
    opened_date DATE
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id DATE PRIMARY KEY,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name VARCHAR(20) NOT NULL,
    quarter INTEGER NOT NULL,
    week_of_year INTEGER NOT NULL,
    is_weekend BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_sales (
    sales_id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES dim_store(store_id),
    dept INTEGER NOT NULL,
    date_id DATE NOT NULL REFERENCES dim_date(date_id),
    weekly_sales NUMERIC(14, 2) NOT NULL,
    is_holiday BOOLEAN NOT NULL,
    temperature NUMERIC(8, 2),
    fuel_price NUMERIC(8, 3),
    cpi NUMERIC(10, 3),
    unemployment NUMERIC(8, 3),
    promo_markdown NUMERIC(12, 2),
    source_system VARCHAR(30) NOT NULL DEFAULT 'csv_api',
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS etl_log (
    log_id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) NOT NULL,
    step VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    rows_count INTEGER DEFAULT 0,
    message TEXT,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP NOT NULL DEFAULT NOW()
);
