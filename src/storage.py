from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Identity,
    MetaData,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import Settings, load_settings

metadata_obj = MetaData()


class Base(DeclarativeBase):
    metadata = metadata_obj


class Station(Base):
    __tablename__ = "stations"

    station_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    network: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    elevation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    datacoverage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mindate: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    maxdate: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Pair(Base):
    __tablename__ = "pairs"
    __table_args__ = (
        UniqueConstraint("station_id", "var", "valid_date", "lead_hours", "issue_time", name="uq_pair"),
    )

    id: Mapped[int] = mapped_column(Integer, Identity(start=1, cycle=False), primary_key=True)
    station_id: Mapped[str] = mapped_column(String, ForeignKey("stations.station_id"))
    var: Mapped[str] = mapped_column(String, nullable=False)
    valid_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    lead_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    issue_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    f_value: Mapped[float] = mapped_column(Float, nullable=False)
    o_value: Mapped[float] = mapped_column(Float, nullable=False)

    station: Mapped[Station] = relationship("Station", backref="pairs")


class DailyScore(Base):
    __tablename__ = "scores_daily"
    __table_args__ = (
        UniqueConstraint("station_id", "var", "valid_date", "lead_hours", name="uq_score_daily"),
    )

    id: Mapped[int] = mapped_column(Integer, Identity(start=1, cycle=False), primary_key=True)
    station_id: Mapped[str] = mapped_column(String, ForeignKey("stations.station_id"))
    var: Mapped[str] = mapped_column(String, nullable=False)
    valid_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    lead_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    mae: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rmse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bias: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n: Mapped[int] = mapped_column(Integer, nullable=False)

    station: Mapped[Station] = relationship("Station", backref="scores")


def get_engine(settings: Optional[Settings] = None):
    settings = settings or load_settings()
    db_path = settings.database_url
    if db_path.startswith("duckdb:///"):
        Path(db_path.replace("duckdb:///", "")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_path)
    return engine


def init_db(settings: Optional[Settings] = None) -> None:
    engine = get_engine(settings)
    Base.metadata.create_all(engine)


def get_session(settings: Optional[Settings] = None) -> Session:
    engine = get_engine(settings)
    return sessionmaker(bind=engine)()


def _to_date(value):
    if value is None or value == "":
        return None
    return pd.to_datetime(value).date()


def upsert_station(station: pd.Series | dict, settings: Optional[Settings] = None) -> None:
    session = get_session(settings)
    data = station if isinstance(station, dict) else station.to_dict()
    station_id = data.get("station_id") or data.get("id")
    if not station_id:
        raise ValueError("station metadata must include station_id or id")
    record = session.get(Station, station_id)
    if record is None:
        record = Station(station_id=station_id)
    record.name = data.get("name") or getattr(record, "name", None)
    record.network = data.get("network") or data.get("id", "").split(":")[0]
    record.latitude = data.get("latitude")
    record.longitude = data.get("longitude")
    record.elevation = data.get("elevation")
    record.datacoverage = data.get("datacoverage")
    record.mindate = _to_date(data.get("mindate"))
    record.maxdate = _to_date(data.get("maxdate"))
    session.add(record)
    session.commit()
    session.close()


def append_pairs(pairs_df: pd.DataFrame, settings: Optional[Settings] = None) -> None:
    if pairs_df.empty:
        return
    session = get_session(settings)
    records = []
    for _, row in pairs_df.iterrows():
        records.append(
            Pair(
                station_id=row["station_id"],
                var=row["var"],
                valid_date=pd.to_datetime(row["valid_date"]).date(),
                lead_hours=int(row["lead_hours"]),
                issue_time=pd.to_datetime(row["issue_time"]),
                f_value=float(row["f_value"]),
                o_value=float(row["o_value"]),
            )
        )
    session.add_all(records)
    session.commit()
    session.close()


def append_daily_scores(scores_df: pd.DataFrame, settings: Optional[Settings] = None) -> None:
    if scores_df.empty:
        return
    session = get_session(settings)
    records = []
    for _, row in scores_df.iterrows():
        records.append(
            DailyScore(
                station_id=row["station_id"],
                var=row["var"],
                valid_date=pd.to_datetime(row["valid_date"]).date()
                if "valid_date" in row
                else pd.to_datetime(row["date"]).date(),
                lead_hours=int(row["lead_hours"]),
                mae=float(row.get("mae")) if row.get("mae") is not None else None,
                rmse=float(row.get("rmse")) if row.get("rmse") is not None else None,
                bias=float(row.get("bias")) if row.get("bias") is not None else None,
                n=int(row.get("n", 0)),
            )
        )
    session.add_all(records)
    session.commit()
    session.close()


def write_parquet(df: pd.DataFrame, path: Path, *, partition_cols: Optional[Iterable[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if partition_cols:
        df.to_parquet(path, partition_cols=list(partition_cols), index=False)
    else:
        df.to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Storage utilities for Forecast Accuracy App.")
    parser.add_argument("--init", action="store_true", help="Initialise the DuckDB database schema.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.init:
        init_db()


if __name__ == "__main__":
    main()


__all__ = [
    "init_db",
    "append_pairs",
    "append_daily_scores",
    "write_parquet",
    "upsert_station",
    "get_engine",
]
