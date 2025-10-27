import os

import pytest


@pytest.fixture(autouse=True)
def default_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NWS_USER_AGENT", "ForecastAccuracyApp-Test/0.1 (test@example.com)")
    monkeypatch.setenv("NCEI_TOKEN", "DUMMY_TOKEN")
    db_path = tmp_path / "test.duckdb"
    monkeypatch.setenv("DATABASE_URL", f"duckdb:///{db_path}")
    yield
