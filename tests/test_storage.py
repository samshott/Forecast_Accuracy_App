import pandas as pd

from src.storage import append_pairs, get_engine, init_db, upsert_station


def test_storage_append_pairs(tmp_path, monkeypatch):
    db_path = tmp_path / "storage.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    init_db()
    station_meta = {
        "station_id": "STA001",
        "name": "Test Station",
        "network": "GHCND",
        "latitude": 44.9,
        "longitude": -93.1,
        "elevation": 250.0,
        "datacoverage": 0.95,
        "mindate": "2000-01-01",
        "maxdate": "2024-05-01",
    }
    upsert_station(station_meta)

    pairs_df = pd.DataFrame(
        {
            "station_id": ["STA001"],
            "var": ["TMAX"],
            "valid_date": pd.to_datetime(["2024-05-01"]),
            "lead_hours": [24],
            "issue_time": pd.to_datetime(["2024-04-30T12:00:00Z"]),
            "f_value": [20.0],
            "o_value": [18.0],
        }
    )

    append_pairs(pairs_df)

    engine = get_engine()
    stored = pd.read_sql_table("pairs", engine)
    assert len(stored) == 1
    assert stored.loc[0, "station_id"] == "STA001"
    assert stored.loc[0, "var"] == "TMAX"
