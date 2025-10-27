# Forecast Accuracy App

Build a Streamlit application that fetches current National Weather Service (NWS) forecasts, pairs them with historical forecasts and NCEI observations, and visualizes daily forecast accuracy by variable and lead time for a chosen location. The MVP targets one WFO/state, three variables (TMAX, TMIN, 24‑h PRCP), and provides time-series plus lead-time skill plots backed by reproducible pipelines and relational storage.

## Getting Started

### 1. Bootstrap the Conda environment

```bash
conda env create -f environment.yml
conda activate forecast_accuracy_app
```

> Tip: For reproducibility, keep the environment name as provided. Run `conda env update -f environment.yml --prune` after dependency changes.

### 2. Configure secrets and runtime settings

Create an `.env` file in the project root (ignored by Git) and populate the following keys:

```env
NWS_USER_AGENT="ForecastAccuracyApp/0.1 (contact@email)"
NCEI_TOKEN="your_cdo_api_token"
DATABASE_URL="duckdb:///data/forecast_accuracy.duckdb"
```

* `NWS_USER_AGENT` — required by NWS; include an email or URL.
* `NCEI_TOKEN` — request from https://www.ncdc.noaa.gov/cdo-web/token.
* `DATABASE_URL` — SQLAlchemy URL for the relational store (DuckDB in local dev).

Optional overrides can live in `configs/app.toml`.

### 3. Initialize the project database

The app uses DuckDB (SQL standard) to catalog stations, forecast pairs, and scores. On first run the schema is created automatically, but you can pre-create it via:

```bash
python -m src.storage --init
```

### 4. Run the developer tests

```bash
pytest
```

### 5. Launch the Streamlit MVP

```bash
streamlit run app/StreamlitApp.py
```

In the app, you can adjust latitude/longitude manually or enable the map picker (via `streamlit-folium`) to click and move the verification target interactively.
The Streamlit tab uses `app/forecast_accuracy_app.ico` automatically when present; swap the icon to rebrand.
The **History** tab (enabled once archive data exists) plots observed vs. forecast values by lead hour, pulling from the Parquet backfill.

### 6. Backfill historical forecasts (optional)

Use the NDFD archive ingestor to download and persist historical grids by issue time:

```bash
python -m src.ingest_forecast_archive --issue 2024-05-01T00 --vars TMAX TMIN PRCP
```

Options:

- `--bbox MIN_LAT MIN_LON MAX_LAT MAX_LON` subsets the grid to a bounding box.
- `--overwrite-download` forces a fresh download even if the GRIB is cached under `data/ndfd/`.
- `--no-parquet` skips writing the tidy Parquet output (for dry runs).

Parquet outputs land in `data/forecasts/{var}/{issue}.parquet`, ready for matching with observations.

## Repository Layout

```
app/                 # Streamlit UI
configs/             # TOML/YAML configs and defaults
data/                # Local cache (parquet & duckdb) – ignored in Git
src/                 # Python packages (ingest, matcher, metrics, storage, plotting)
tests/               # Pytest test suite (alignment, metrics, storage)
environment.yml      # Conda environment definition
```

## MVP Deliverables (Phase 1)

- `get_current_gridpoint_forecast(lat, lon)` → current NWS forecast DataFrame (cached).
- `get_ncei_daily_obs(station_id, start, end)` → GHCND observations DataFrame.
- `select_nearest_station(lat, lon, network='GHCND', radius_km=50)` → best station metadata.
- `align_daily_pairs(forecasts_df, obs_df)` → joined daily forecast/obs pairs with lead hours.
- `score_pairs(pairs_df, metrics=['mae', 'rmse', 'bias'])`.
- Streamlit UI with sidebar controls, forecast vs obs chart, and MAE vs lead chart.
- Parquet + DuckDB storage with simple catalog (`src/storage.py`).

Subsequent phases add archival ingest, cron-style refresh, verification extras, and UI polish (see `docs/backlog.md`).

## Development Notes

- Use UTC for computations; convert to local time zones for display only.
- Respectful API usage: add retry & backoff, and persist fetched responses under `data/`.
- Station selection prefers GHCND sites with low missingness (COOP/ASOS). Extend via `src/station_lookup.py`.
- Parquet partitioning strategy: `/data/forecasts/{var}/yyyymmdd.parquet`, `/data/obs/{network}/yyyy.parquet`, `/data/scores/{var}/yyyymmdd.parquet`.
- The relational layer (DuckDB via SQLAlchemy) tracks metadata (`stations`, `pairs`, `scores_daily`, `scores_monthly`) and enables aggregations without re-hydrating all Parquet files.

## Contributing

1. Create a feature branch.
2. Update or add tests alongside code changes.
3. Run `pytest` and `streamlit run app/StreamlitApp.py` to verify.
4. Submit a PR with a short summary and testing notes.
