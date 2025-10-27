from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_folium import st_folium
    import folium
except ImportError:  # pragma: no cover
    st_folium = None
    folium = None

ICON_PATH = Path(__file__).resolve().parent / "forecast_accuracy_app.ico"
st.set_page_config(
    page_title="Forecast Accuracy App",
    layout="wide",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "📈",
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import load_settings
from src.ingest_current_forecast import get_current_gridpoint_forecast
from src.ingest_obs import get_ncei_daily_obs
from src.matcher import align_daily_pairs
from src.metrics import score_pairs
from src.station_lookup import select_nearest_station
from src.storage_readers import load_forecast_parquet


@st.cache_data(show_spinner=False)
def _fetch_station_candidates(lat: float, lon: float) -> list[dict]:
    stations_df = select_nearest_station(lat, lon, return_all=True)
    if isinstance(stations_df, pd.Series):
        stations_df = stations_df.to_frame().T
    if stations_df.empty:
        return []
    records = stations_df.to_dict(orient="records")
    for record in records:
        record.setdefault("station_id", record.get("id"))
    return records


@st.cache_data(show_spinner=False)
def _fetch_forecast(lat: float, lon: float, variables: tuple[str, ...]):
    return get_current_gridpoint_forecast(lat, lon, variables=variables)


@st.cache_data(show_spinner=False)
def _fetch_observations(station_id: str, start: date, end: date, variables: tuple[str, ...]):
    return get_ncei_daily_obs(station_id, start, end, variables=variables)


VAR_TITLES = {"TMAX": "Max Temperature", "TMIN": "Min Temperature", "PRCP": "Precipitation"}


@st.cache_data(show_spinner=False)
def _load_forecast_history(var: str, forecast_dir: str) -> pd.DataFrame:
    df = load_forecast_parquet(Path(forecast_dir), variables=[var])
    if df.empty:
        return df
    return df


def render_time_series(pairs: pd.DataFrame, variable: str):
    subset = pairs[pairs["var"] == variable]
    if subset.empty:
        st.info("No paired data available for the selected variable and dates.")
        return
    history = subset.sort_values("valid_date")
    display = pd.DataFrame(
        {
            "valid_date": history["valid_date"],
            "Forecast": history["f_value"],
            "Observed": history["o_value"],
        }
    ).set_index("valid_date")
    st.line_chart(display)


def render_lead_metrics(pairs: pd.DataFrame, variable: str):
    subset = pairs[pairs["var"] == variable]
    if subset.empty:
        st.info("No scoreable data for the selected variable.")
        return
    scores = score_pairs(subset, group_fields=["station_id", "var", "lead_hours"])
    scores = scores[scores["var"] == variable].sort_values("lead_hours")
    if scores.empty:
        st.info("No metrics computed for the selected variable.")
        return
    metric_chart = scores.set_index("lead_hours")[["mae", "rmse", "bias"]]
    st.line_chart(metric_chart)


def render_observation_chart(obs_df: pd.DataFrame, variable: str):
    subset = obs_df[obs_df["var"] == variable].copy()
    if subset.empty:
        st.info(f"No observations available for {variable}.")
        return
    subset["date"] = pd.to_datetime(subset["date"])
    subset.sort_values("date", inplace=True)
    unit = subset["unit"].dropna().iloc[0] if "unit" in subset.columns and not subset["unit"].dropna().empty else ""
    base_title = VAR_TITLES.get(variable, variable)
    y_title = f"{base_title} ({unit})" if unit else base_title
    date_axis = alt.Axis(format="%b %-d", labelAngle=-40, title="Date")
    chart = (
        alt.Chart(subset)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", axis=date_axis),
            y=alt.Y("value:Q", title=y_title),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("value:Q", title=f"{variable}", format=".2f"),
                alt.Tooltip("qc_flag:N", title="QC Flag"),
            ],
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def _prepare_point_history(
    var: str,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    settings,
) -> pd.DataFrame:
    history_df = _load_forecast_history(var, str(settings.data.forecast_dir))
    if history_df.empty:
        return history_df
    df = history_df.copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    mask = (df["valid_time"].dt.date >= start_date) & (df["valid_time"].dt.date <= end_date)
    df = df[mask]
    if df.empty or "latitude" not in df or "longitude" not in df:
        return df
    coords = (
        df[["grid_x", "grid_y", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates(subset=["grid_x", "grid_y"])
    )
    if coords.empty:
        return df
    coords["distance"] = np.sqrt((coords["latitude"] - lat) ** 2 + (coords["longitude"] - lon) ** 2)
    best = coords.sort_values("distance").iloc[0]
    gx = float(best["grid_x"])
    gy = float(best["grid_y"])
    mask = np.isclose(df["grid_x"].astype(float), gx) & np.isclose(df["grid_y"].astype(float), gy)
    return df[mask]


def render_history_chart(history_pairs: pd.DataFrame, variable: str, leads: Iterable[float]):
    subset = history_pairs[history_pairs["var"] == variable]
    if subset.empty:
        st.info("No historical forecast pairs available for the selected variable.")
        return
    if leads:
        subset = subset[subset["lead_hours"].isin(leads)]
        if subset.empty:
            st.info("No data for the chosen lead hours.")
            return
    observed = (
        subset.sort_values("valid_date")
        .drop_duplicates(subset=["valid_date"])
        .loc[:, ["valid_date", "o_value"]]
        .rename(columns={"o_value": "value"})
    )
    observed["series"] = "Observed"

    forecasts = subset.copy()
    forecasts["series"] = forecasts["lead_hours"].astype(int).astype(str) + "h lead"
    forecasts = forecasts.loc[:, ["valid_date", "series", "f_value"]].rename(columns={"f_value": "value"})

    chart_df = pd.concat(
        [
            observed.assign(series="Observed"),
            forecasts,
        ],
        ignore_index=True,
    )
    chart_df["valid_date"] = pd.to_datetime(chart_df["valid_date"])
    date_axis = alt.Axis(format="%b %-d", labelAngle=-40, title="Valid Date")
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("valid_date:T", axis=date_axis),
            y=alt.Y("value:Q", title=VAR_TITLES.get(variable, variable)),
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("valid_date:T", title="Valid Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
            ],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def main():
    settings = load_settings()
    st.title(settings.app.title)
    st.caption("Daily verification of NWS forecasts against NCEI observations.")

    if "selected_location" not in st.session_state:
        st.session_state["selected_location"] = {
            "lat": settings.app.default_lat,
            "lon": settings.app.default_lon,
        }
    if "selected_state" not in st.session_state:
        st.session_state["selected_state"] = settings.app.default_state
    if "fetch_trigger" not in st.session_state:
        st.session_state["fetch_trigger"] = False

    today = date.today()
    default_start = today - timedelta(days=14)

    with st.sidebar:
        st.header("Controls")
        use_map = st.checkbox(
            "Pick location on map",
            value=bool(st_folium),
            disabled=st_folium is None,
            help="Click on the map to update the forecast location." if st_folium else "Install streamlit-folium to enable the map picker.",
        )
        lat = st.number_input(
            "Latitude",
            value=float(st.session_state["selected_location"]["lat"]),
            format="%.4f",
            key="latitude_input",
        )
        lon = st.number_input(
            "Longitude",
            value=float(st.session_state["selected_location"]["lon"]),
            format="%.4f",
            key="longitude_input",
        )
        st.session_state["selected_location"] = {"lat": lat, "lon": lon}
        variables = st.multiselect(
            "Variables",
            options=settings.app.default_variables,
            default=list(settings.app.default_variables),
        )
        selected_variables = tuple(variables) if variables else settings.app.default_variables
        target_var = st.selectbox("Primary variable", options=list(selected_variables))
        start_date = st.date_input("Start date", value=default_start)
        end_date = st.date_input("End date", value=today)
        if st.button("Fetch & Verify", type="primary"):
            st.session_state["fetch_trigger"] = True

    if use_map and st_folium:
        map_col, info_col = st.columns([3, 1])
        with map_col:
            map_location = [
                st.session_state["selected_location"]["lat"],
                st.session_state["selected_location"]["lon"],
            ]
            map_zoom = 7
            folium_map = folium.Map(location=map_location, zoom_start=map_zoom, tiles="CartoDB positron")
            folium.Marker(
                map_location,
                tooltip="Selected location",
                draggable=False,
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(folium_map)
            folium.Circle(map_location, radius=50_000, color="#3186cc", fill=True, fill_opacity=0.2).add_to(folium_map)
            folium.LatLngPopup().add_to(folium_map)
            map_result = st_folium(
                folium_map,
                height=400,
                use_container_width=True,
                key="location_map",
            )
            click = None
            if map_result:
                click = map_result.get("last_clicked") or map_result.get("last_object_clicked")
            if click:
                lat_clicked = float(click["lat"])
                lon_clicked = float(click["lng"])
                current = st.session_state["selected_location"]
                if not (
                    abs(lat_clicked - current["lat"]) < 1e-6 and abs(lon_clicked - current["lon"]) < 1e-6
                ):
                    st.session_state["selected_location"] = {
                        "lat": lat_clicked,
                        "lon": lon_clicked,
                    }
                    st.rerun()
        with info_col:
            st.markdown(
                "Click anywhere on the map to move the forecast target. "
                "Manual latitude/longitude inputs stay available in the sidebar."
            )

    lat = float(st.session_state["selected_location"]["lat"])
    lon = float(st.session_state["selected_location"]["lon"])

    fetch_requested = st.session_state.get("fetch_trigger", False)

    if not fetch_requested:
        st.info("Select a location and press **Fetch & Verify** to load data.")
        return

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.session_state["fetch_trigger"] = False
        return

    with st.spinner("Locating candidate stations..."):
        try:
            station_candidates = _fetch_station_candidates(lat, lon)
        except Exception as exc:  # pragma: no cover
            st.error(f"Station lookup failed: {exc}")
            return

    if not station_candidates:
        st.error("No stations found within the search radius.")
        st.session_state["fetch_trigger"] = False
        return

    with st.spinner("Fetching current forecast..."):
        try:
            forecast_df = _fetch_forecast(lat, lon, selected_variables)
        except Exception as exc:  # pragma: no cover
            st.error(f"Forecast fetch failed: {exc}")
            return

    obs_df = pd.DataFrame()
    station = None
    for candidate in station_candidates[:10]:
        station_id = candidate.get("station_id") or candidate.get("id")
        try:
            obs_df = _fetch_observations(station_id, start_date, end_date, selected_variables)
        except Exception as exc:  # pragma: no cover
            st.warning(f"Observation fetch failed for {station_id}: {exc}")
            continue
        if not obs_df.empty:
            station = candidate
            break

    if station is None or obs_df.empty:
        st.warning("No observations available for nearby stations in the selected date range.")
        st.session_state["fetch_trigger"] = False
        return

    distance = station.get("distance_km")
    distance_text = f"{distance:.1f} km away" if isinstance(distance, (int, float)) else ""
    st.success(
        f"Using station: {station.get('name', station.get('station_id'))} "
        f"({station.get('station_id')}) {distance_text}"
    )

    pairs_df = align_daily_pairs(forecast_df, obs_df)

    history_forecast = _prepare_point_history(target_var, lat, lon, start_date, end_date, settings)
    history_pairs = pd.DataFrame()
    if not history_forecast.empty:
        history_pairs = align_daily_pairs(history_forecast, obs_df)

    current_tab, history_tab = st.tabs(["Current Forecast", "History"])

    with current_tab:
        st.subheader("Observed values from NCEI")
        for variable in selected_variables:
            st.markdown(f"**{variable} Observations**")
            render_observation_chart(obs_df, variable)

        if pairs_df.empty:
            st.info("Forecast data not available for this selection yet.")
        else:
            st.subheader("Forecast vs Observed")
            render_time_series(pairs_df, target_var)

            st.subheader("MAE/RMSE/Bias by Lead Hour")
            render_lead_metrics(pairs_df, target_var)

            st.dataframe(pairs_df.head(20))

    with history_tab:
        if history_forecast.empty or history_pairs.empty:
            st.info(
                "No historical forecast archives available for this location/date range yet. "
                "Run the NDFD ingest job and retry."
            )
        else:
            available_leads = sorted({int(l) for l in history_pairs["lead_hours"].unique()})
            default_leads = [lead for lead in available_leads if lead in (0, 24, 48)]
            if not default_leads:
                default_leads = available_leads[:3]
            selected_leads = st.multiselect(
                "Lead hours to display",
                options=available_leads,
                default=default_leads,
                key="history_leads",
            )
            leads_to_use = selected_leads or available_leads
            render_history_chart(history_pairs, target_var, leads_to_use)
            st.dataframe(history_pairs.head(50))

    st.session_state["fetch_trigger"] = False


if __name__ == "__main__":
    main()
