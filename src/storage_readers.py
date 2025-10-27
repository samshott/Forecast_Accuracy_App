from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def load_forecast_parquet(
    directory: Path,
    *,
    issue_times: Optional[Iterable[str]] = None,
    variables: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    directory = Path(directory)
    if not directory.exists():
        return pd.DataFrame()

    vars_to_read = list(variables) if variables else [p.name for p in directory.iterdir() if p.is_dir()]
    frames = []
    for var in vars_to_read:
        var_dir = directory / var
        if not var_dir.exists():
            continue
        if issue_times:
            for issue in issue_times:
                parquet_path = var_dir / f"{issue}.parquet"
                if parquet_path.exists():
                    frames.append(pd.read_parquet(parquet_path))
        else:
            for parquet_path in var_dir.glob("*.parquet"):
                frames.append(pd.read_parquet(parquet_path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["issue_key"] = df["issue_time"].astype("datetime64[ns]").dt.strftime("%Y%m%d%H")
    return df


def load_observation_parquet(directory: Path, *, station_id: Optional[str] = None) -> pd.DataFrame:
    directory = Path(directory)
    if not directory.exists():
        return pd.DataFrame()
    frames = []
    for parquet_path in directory.glob("*.parquet"):
        df = pd.read_parquet(parquet_path)
        if station_id:
            df = df[df["station_id"] == station_id]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


__all__ = ["load_forecast_parquet", "load_observation_parquet"]
