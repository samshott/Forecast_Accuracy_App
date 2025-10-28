from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.config import (
    APIConfig,
    AppConfig,
    DataConfig,
    ForecastArchiveConfig,
    Settings,
)
from src import ingest_forecast_archive as archive


def _make_settings(tmp_path: Path) -> Settings:
    cache_dir = tmp_path / "cache"
    forecast_dir = tmp_path / "forecasts"
    obs_dir = tmp_path / "obs"
    scores_dir = tmp_path / "scores"
    archive_dir = tmp_path / "archive"
    for directory in (cache_dir, forecast_dir, obs_dir, scores_dir, archive_dir):
        directory.mkdir(parents=True, exist_ok=True)
    app_cfg = AppConfig(
        title="Test",
        default_lat=44.9,
        default_lon=-93.1,
        default_state="MN",
        default_variables=("TMAX",),
        default_lead_hours=(0, 24),
    )
    data_cfg = DataConfig(
        cache_dir=cache_dir,
        forecast_dir=forecast_dir,
        obs_dir=obs_dir,
        scores_dir=scores_dir,
        archive_dir=archive_dir,
    )
    api_cfg = APIConfig(
        nws_base_url="https://example",
        nws_timeout=10,
        nws_retries=1,
        ncei_base_url="https://example",
        ncei_timeout=10,
        ncei_retries=1,
        nws_user_agent="TestAgent/1.0 (test@example.com)",
        ncei_token="TEST",
    )
    archive_cfg = ForecastArchiveConfig(
        base_url="https://example.com/conus",
        sector="",
        product_files={"TMAX": ("VP.001-003/ds.maxt.bin",)},
        grid="",
        archive_dir=archive_dir,
    )
    return Settings(
        app=app_cfg,
        data=data_cfg,
        api=api_cfg,
        forecast_archive=archive_cfg,
        database_url="duckdb:///test.duckdb",
    )


def test_ingest_ndfd_archive_writes_parquet(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    issue_time = datetime(2024, 5, 1, 0, tzinfo=timezone.utc)
    fake_grib = settings.data.archive_dir / "TMAX" / "20240501" / "tmax_2024050100.grib2"
    fake_grib.parent.mkdir(parents=True, exist_ok=True)
    fake_grib.write_bytes(b"fake")

    def fake_download(variable, rel_path, session=None, headers=None, config=None, timeout=None):
        return fake_grib

    monkeypatch.setattr(archive, "_download_tgftp_file", fake_download)

    times = pd.date_range(issue_time, periods=2, freq="24h")
    lat_vals = np.array([[44.8, 44.9], [45.0, 45.1]])
    lon_vals = np.array([[-93.2, -93.0], [-92.8, -92.6]])
    data_vals = np.array(
        [
            [[20.0, 21.0], [22.0, 23.0]],
            [[24.0, 25.0], [26.0, 27.0]],
        ]
    )
    data = xr.DataArray(
        data_vals,
        dims=("time", "y", "x"),
        coords={"time": times, "y": [0, 1], "x": [0, 1]},
        name="tmax",
    )
    lat = xr.DataArray(lat_vals, dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]}, name="latitude")
    lon = xr.DataArray(lon_vals, dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]}, name="longitude")
    dataset = xr.Dataset({"tmax": data, "latitude": lat, "longitude": lon})

    monkeypatch.setattr(archive, "_open_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(archive, "load_settings", lambda: settings)
    results = archive.ingest_ndfd_archive(
        issue_time,
        variables=["TMAX"],
        bbox=(44.7, -93.3, 45.2, -92.5),
        write_parquet=True,
    )
    assert "TMAX" in results
    result = results["TMAX"]
    assert result.row_count == 8
    assert result.parquet_path is not None
    parquet_data = pd.read_parquet(result.parquet_path)
    assert not parquet_data.empty
    assert parquet_data["var"].unique().tolist() == ["TMAX"]
    assert parquet_data["lead_hours"].max() == pytest.approx(24.0)


def test_build_remote_url_simple():
    issue_time = datetime(2024, 5, 1, 12, tzinfo=timezone.utc)
    cfg = ForecastArchiveConfig(
        base_url="https://tgftp.example/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus",
        sector="",
        product_files={"TMAX": ("VP.001-003/ds.maxt.bin",)},
        grid="",
        archive_dir=Path("/tmp"),
    )
    url = archive._build_remote_url(issue_time, "TMAX", cfg)
    assert url == "https://tgftp.example/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.maxt.bin"
