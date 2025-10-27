from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import tomllib
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "app.toml"


@dataclass(frozen=True)
class AppConfig:
    title: str
    default_lat: float
    default_lon: float
    default_state: str
    default_variables: tuple[str, ...]
    default_lead_hours: tuple[int, ...]


@dataclass(frozen=True)
class DataConfig:
    cache_dir: Path
    forecast_dir: Path
    obs_dir: Path
    scores_dir: Path
    archive_dir: Path


@dataclass(frozen=True)
class APIConfig:
    nws_base_url: str
    nws_timeout: int
    nws_retries: int
    ncei_base_url: str
    ncei_timeout: int
    ncei_retries: int
    nws_user_agent: str
    ncei_token: str


@dataclass(frozen=True)
class ForecastArchiveConfig:
    base_url: str
    sector: str
    product_files: dict[str, str]
    grid: str
    archive_dir: Path


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    data: DataConfig
    api: APIConfig
    forecast_archive: ForecastArchiveConfig
    database_url: str


def _load_toml_config() -> Dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")
    with _CONFIG_PATH.open("rb") as fh:
        return tomllib.load(fh)


def _coerce_tuple(values: Any) -> tuple:
    if isinstance(values, (list, tuple)):
        return tuple(values)
    return (values,)


def load_settings() -> Settings:
    load_dotenv()
    raw = _load_toml_config()

    data_section = raw.get("data", {})
    cache_dir = (_PROJECT_ROOT / data_section.get("cache_dir", "data")).resolve()
    forecast_dir = (_PROJECT_ROOT / data_section.get("forecast_dir", "data/forecasts")).resolve()
    obs_dir = (_PROJECT_ROOT / data_section.get("obs_dir", "data/obs")).resolve()
    scores_dir = (_PROJECT_ROOT / data_section.get("scores_dir", "data/scores")).resolve()
    archive_dir = (_PROJECT_ROOT / data_section.get("archive_dir", "data/ndfd")).resolve()
    for path in (cache_dir, forecast_dir, obs_dir, scores_dir, archive_dir):
        path.mkdir(parents=True, exist_ok=True)

    api_section = raw.get("api", {})
    nws_user_agent = os.getenv("NWS_USER_AGENT")
    if not nws_user_agent:
        raise EnvironmentError("NWS_USER_AGENT must be set in the environment")
    ncei_token = os.getenv("NCEI_TOKEN", "")

    database_url = os.getenv("DATABASE_URL", f"duckdb:///{cache_dir}/forecast_accuracy.duckdb")

    app_cfg = AppConfig(
        title=raw["app"]["title"],
        default_lat=float(raw["app"]["default_lat"]),
        default_lon=float(raw["app"]["default_lon"]),
        default_state=raw["app"]["default_state"],
        default_variables=_coerce_tuple(raw["app"]["default_variables"]),
        default_lead_hours=_coerce_tuple(raw["app"]["default_lead_hours"]),
    )
    data_cfg = DataConfig(
        cache_dir=cache_dir,
        forecast_dir=forecast_dir,
        obs_dir=obs_dir,
        scores_dir=scores_dir,
        archive_dir=archive_dir,
    )
    api_cfg = APIConfig(
        nws_base_url=api_section.get("nws_base_url", "https://api.weather.gov/gridpoints"),
        nws_timeout=int(api_section.get("nws_timeout", 30)),
        nws_retries=int(api_section.get("nws_retries", 3)),
        ncei_base_url=api_section.get("ncei_base_url", "https://www.ncdc.noaa.gov/cdo-web/api/v2"),
        ncei_timeout=int(api_section.get("ncei_timeout", 30)),
        ncei_retries=int(api_section.get("ncei_retries", 3)),
        nws_user_agent=nws_user_agent,
        ncei_token=ncei_token,
    )
    archive_section = raw.get("forecast_archive", {})
    forecast_archive_cfg = ForecastArchiveConfig(
        base_url=archive_section.get("base_url", ""),
        sector=archive_section.get("sector", "conus"),
        product_files=dict(archive_section.get("product_files", {})),
        grid=archive_section.get("grid", "ndfd"),
        archive_dir=archive_dir,
    )
    return Settings(
        app=app_cfg,
        data=data_cfg,
        api=api_cfg,
        forecast_archive=forecast_archive_cfg,
        database_url=database_url,
    )


__all__ = [
    "load_settings",
    "Settings",
    "AppConfig",
    "DataConfig",
    "APIConfig",
    "ForecastArchiveConfig",
]
