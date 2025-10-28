from __future__ import annotations

import argparse
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster

COLORS = {
    "candidate": "green",
    "excluded": "gray",
}

DEFAULT_BOUNDS = (24.0, -125.0, 50.0, -66.0)  # Lower 48 approx


def load_inventory(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required_cols = {
        "station_id",
        "name",
        "latitude",
        "longitude",
        "has_tmax",
        "has_tmin",
        "has_prcp",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Inventory missing required columns: {missing}")
    return df


def filter_inventory(
    df: pd.DataFrame,
    *,
    min_datacoverage: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["network"] = df["station_id"].str[:3]
    desired_networks = {"USW", "USC"}

    has_all = df[["has_tmax", "has_tmin", "has_prcp"]].all(axis=1)
    in_network = df["network"].isin(desired_networks)
    meets = has_all & in_network

    if min_datacoverage is not None and "datacoverage" in df:
        meets &= df["datacoverage"].fillna(0) >= min_datacoverage

    if bounds is not None:
        min_lat, min_lon, max_lat, max_lon = bounds
        meets &= df["latitude"].between(min_lat, max_lat) & df["longitude"].between(min_lon, max_lon)

    candidates = df[meets]
    excluded = df[~meets]
    return candidates, excluded


def build_map(candidates: pd.DataFrame, excluded: pd.DataFrame) -> folium.Map:
    if candidates.empty and excluded.empty:
        return folium.Map(location=[39.5, -98.5], zoom_start=4, tiles="CartoDB positron")

    combined = pd.concat(
        [
            candidates.assign(coverage="candidate"),
            excluded.assign(coverage="excluded"),
        ],
        ignore_index=True,
    )
    center = [combined["latitude"].mean(), combined["longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")

    for coverage, label in (("candidate", "USW/USC complete"), ("excluded", "Other stations")):
        subset = combined[combined["coverage"] == coverage]
        if subset.empty:
            continue
        cluster = MarkerCluster(name=label, disableClusteringAtZoom=6)
        cluster.add_to(fmap)
        for _, row in subset.iterrows():
            popup = folium.Popup(
                folium.Html(
                    f"<b>{row['name']}</b><br/>ID: {row['station_id']}<br/>"
                    f"Network: {row['station_id'][:3]}<br/>"
                    f"TMAX/TMIN/PRCP: {row['has_tmax']}/{row['has_tmin']}/{row['has_prcp']}<br/>"
                    f"Datacoverage: {row.get('datacoverage', 'N/A')}",
                    script=True,
                ),
                max_width=250,
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4,
                color=COLORS.get(coverage, "blue"),
                fill=True,
                fill_opacity=0.9,
                popup=popup,
            ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive map highlighting candidate stations.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/stations/station_inventory.parquet"),
        help="Input station inventory parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stations/station_inventory_map.html"),
        help="Output HTML map path.",
    )
    parser.add_argument(
        "--min-datacoverage",
        type=float,
        default=0.9,
        help="Minimum datacoverage to accept (default: 0.9).",
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"),
        help="Optional bounding box; default filters to the lower 48 states.",
    )
    parser.add_argument(
        "--no-bounds",
        action="store_true",
        help="Disable bounding box filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Inventory file not found: {args.input}")

    df = load_inventory(args.input)
    bounds = None if args.no_bounds else (tuple(args.bounds) if args.bounds else DEFAULT_BOUNDS)
    candidates, excluded = filter_inventory(
        df,
        min_datacoverage=args.min_datacoverage,
        bounds=bounds,
    )
    fmap = build_map(candidates, excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(args.output)
    print(f"Map written to {args.output}")
    print(f"Candidate stations: {len(candidates)}")
    print(f"Excluded stations: {len(excluded)}")


if __name__ == "__main__":
    main()
