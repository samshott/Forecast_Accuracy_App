from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.config import load_settings
from src.ingest_current_forecast import get_current_gridpoint_forecast
from src.ingest_obs import get_ncei_daily_obs
from src.matcher import align_daily_pairs
from src.metrics import score_pairs
from src.station_lookup import select_nearest_station


@st.cache_data(show_spinner=False)
def _fetch_station(lat: float, lon: float):
    station = select_nearest_station(lat, lon)
    station_dict = station.to_dict()
    station_dict.setdefault("station_id", station_dict.get("id"))
    return station_dict


@st.cache_data(show_spinner=False)
def _fetch_forecast(lat: float, lon: float, variables: tuple[str, ...]):
    return get_current_gridpoint_forecast(lat, lon, variables=variables)


@st.cache_data(show_spinner=False)
def _fetch_observations(station_id: str, start: date, end: date, variables: tuple[str, ...]):
    return get_ncei_daily_obs(station_id, start, end, variables=variables)


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


def main():
    settings = load_settings()
    st.set_page_config(page_title=settings.app.title, layout="wide")
    st.title(settings.app.title)
    st.caption("Daily verification of NWS forecasts against NCEI observations.")

    today = date.today()
    default_start = today - timedelta(days=14)

    with st.sidebar:
        st.header("Controls")
        lat = st.number_input("Latitude", value=settings.app.default_lat, format="%.4f")
        lon = st.number_input("Longitude", value=settings.app.default_lon, format="%.4f")
        variables = st.multiselect(
            "Variables",
            options=settings.app.default_variables,
            default=list(settings.app.default_variables),
        )
        selected_variables = tuple(variables) if variables else settings.app.default_variables
        target_var = st.selectbox("Primary variable", options=list(selected_variables))
        start_date = st.date_input("Start date", value=default_start)
        end_date = st.date_input("End date", value=today)
        fetch_button = st.button("Fetch & Verify", type="primary")

    if not fetch_button:
        st.info("Select a location and press **Fetch & Verify** to load data.")
        return

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    with st.spinner("Locating station..."):
        try:
            station = _fetch_station(lat, lon)
        except Exception as exc:  # pragma: no cover
            st.error(f"Station lookup failed: {exc}")
            return

    st.success(f"Station selected: {station.get('name', station.get('station_id'))}")

    with st.spinner("Fetching current forecast..."):
        try:
            forecast_df = _fetch_forecast(lat, lon, selected_variables)
        except Exception as exc:  # pragma: no cover
            st.error(f"Forecast fetch failed: {exc}")
            return

    with st.spinner("Fetching observations..."):
        try:
            obs_df = _fetch_observations(station["station_id"], start_date, end_date, selected_variables)
        except Exception as exc:  # pragma: no cover
            st.error(f"Observation fetch failed: {exc}")
            return

    pairs_df = align_daily_pairs(forecast_df, obs_df)
    if pairs_df.empty:
        st.warning("No forecast/observation pairs matched. Try expanding the date range or location.")
        return

    st.subheader("Forecast vs Observed")
    render_time_series(pairs_df, target_var)

    st.subheader("MAE/RMSE/Bias by Lead Hour")
    render_lead_metrics(pairs_df, target_var)

    st.dataframe(pairs_df.head(20))


if __name__ == "__main__":
    main()
