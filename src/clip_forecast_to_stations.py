from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_OUTPUT_ROOT = Path("data/forecasts_station")
TARGET_NETWORKS = {"USW", "USC"}
TARGET_VARS = ("TMAX", "TMIN", "PRCP")


def load_inventory(path: Path, *, min_datacoverage: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ("latitude", "longitude", "datacoverage"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["network"] = df["station_id"].str.split(":", n=1).str[-1].str[:3]
    has_all = df[["has_tmax", "has_tmin", "has_prcp"]].all(axis=1)
    meets = has_all & df["network"].isin(TARGET_NETWORKS)
    meets &= df["datacoverage"].fillna(0) >= min_datacoverage
    return df[meets].reset_index(drop=True)


def clip_parquet_to_stations(
    parquet_path: Path,
    stations: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError(f"Parquet file {parquet_path} is empty.")

    coords = df[["latitude", "longitude"]].drop_duplicates()
    tree = cKDTree(coords[["latitude", "longitude"]].to_numpy())
    station_coords = stations[["latitude", "longitude"]].to_numpy()
    _, idx = tree.query(station_coords)
    matched_coords = coords.iloc[idx].copy().reset_index(drop=True)
    matched_coords["station_id"] = stations["station_id"].values

    df["lat_key"] = df["latitude"].round(5)
    df["lon_key"] = df["longitude"].round(5)
    matched_coords["lat_key"] = matched_coords["latitude"].round(5)
    matched_coords["lon_key"] = matched_coords["longitude"].round(5)

    clipped = df.merge(
        matched_coords[["station_id", "lat_key", "lon_key"]],
        on=["lat_key", "lon_key"],
        how="inner",
    )
    clipped.drop(columns=["lat_key", "lon_key"], inplace=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / parquet_path.name
    clipped.to_parquet(output_path, index=False)
    return output_path


def infer_variable_from_path(path: Path) -> str:
    parts = path.parts
    for var in TARGET_VARS:
        if var in parts:
            return var
    raise ValueError(f"Unable to infer variable from path {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clip forecast parquet files to station locations.")
    parser.add_argument(
        "parquet_paths",
        nargs="+",
        type=Path,
        help="Input forecast parquet files to clip.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/stations/station_inventory.parquet"),
        help="Station inventory parquet file.",
    )
    parser.add_argument(
        "--min-datacoverage",
        type=float,
        default=0.9,
        help="Minimum datacoverage for stations (default: 0.9).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to save clipped parquet files (organized by variable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.inventory.exists():
        raise SystemExit(f"Inventory file not found: {args.inventory}")

    stations = load_inventory(args.inventory, min_datacoverage=args.min_datacoverage)
    if stations.empty:
        raise SystemExit("No stations matched the filter; adjust datacoverage or inventory.")

    total_input = 0
    total_output = 0

    for path in args.parquet_paths:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        var = infer_variable_from_path(path)
        var_output_dir = args.output_root / var
        print(f"Clipping {path} ({var}) for {len(stations)} stations…", flush=True)
        output_path = clip_parquet_to_stations(path, stations, output_dir=var_output_dir)
        in_size = path.stat().st_size
        out_size = output_path.stat().st_size
        total_input += in_size
        total_output += out_size
        print(
            f"  saved {output_path} ({out_size/1024/1024:.2f} MB) from {in_size/1024/1024:.2f} MB input"
        )

    print(f"Total input size: {total_input/1024/1024:.2f} MB")
    print(f"Total output size: {total_output/1024/1024:.2f} MB")
    print(f"Compression factor: {(total_input/total_output) if total_output else 0:.2f}")


if __name__ == "__main__":
    main()
