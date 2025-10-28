# Starter Backlog

## P0 — MVP Foundations
- Create repo scaffolding (`src/`, `app/`, `tests/`, `configs/`).
- `get_current_gridpoint_forecast(lat, lon)` → pandas DataFrame; persist JSON cache.
- `get_ncei_daily_obs(station_id, start, end, vars)` → pandas DataFrame.
- `select_nearest_station(lat, lon, network='GHCND', radius_km=50)`.
- `align_daily_pairs(forecasts_df, obs_df)` → tidy pairs with `lead_hours`.
- `score_pairs(pairs_df, metrics=['mae', 'rmse', 'bias'])`.
- Streamlit MVP: sidebar inputs, forecast vs observed line chart, MAE vs lead curve.
- Parquet writers/readers; `catalog.yaml` or DuckDB metadata tables.

## P1 — Data Pipeline
- [x] NDFD archive reader (xarray/GRIB) for backfill (`src/ingest_forecast_archive.py`).
- Daily cron (e.g., GitHub Action) to append the latest cycle snapshots.
- Station coverage map (pydeck / leafmap).
- Basic data-quality report per station.

## P2 — Verification Enhancements
- Reliability diagram for PoP (Brier score).
- Bilinear interpolation option for grid-to-point mapping.
- Monthly aggregation job & seasonal facets.

## P3 — Hardening
- Add retries, caching, and logging instrumentation.
- Data coverage dashboards, last-update timestamps.
- Docker image & CI quick tests.

## Side Quest — Historical Coverage Census
- [ ] Fetch NCEI station metadata for target networks (e.g., GHCND) including available datatypes and coverage.
- [ ] Identify stations supporting the full variable set (TMAX, TMIN, PRCP) and flag partial coverage.
- [ ] Produce a coverage map comparing full vs. raw station sets to understand spatial gaps.
- [ ] Decide ETL strategy (station-only extraction vs. coarsened grid) based on coverage results.
- [ ] Prototype slow-ingest pipeline: download, subset, and persist station-focused forecast/obs history.
