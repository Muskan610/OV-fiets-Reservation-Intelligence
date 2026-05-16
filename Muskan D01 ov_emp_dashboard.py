"""Minimal OV-fiets operational inventory console with live model inference."""

from __future__ import annotations

import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from inventory_widgets import (
    build_historical_conditions,
    build_hourly_demand_index,
    build_station_region_map,
    render_allocation_widget,
    render_console_header,
    render_inventory_overview,
    render_pressure_banner,
    render_reservations_cancellations_chart,
    render_simulated_pickup_chart,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RETURNS_PATH = PROJECT_ROOT / "ov_fiets_with_returns.csv"
MODEL_PATH = PROJECT_ROOT / "forecast_demand_lgb.pkl"
ENCODERS_PATH = PROJECT_ROOT / "forecast_demand_encoders.pkl"
DEFAULT_STATION = "Gouda"

SEASON_DEFAULT_TEMP = {"Winter": 4.0, "Spring": 12.0, "Summer": 21.0, "Autumn": 11.0}


st.set_page_config(
    page_title="OV-fiets · Operations",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_returns(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_encoders(path: Path) -> dict:
    return joblib.load(path)


returns_df = load_returns(RETURNS_PATH)
model = load_model(MODEL_PATH)
encoders = load_encoders(ENCODERS_PATH)
hourly_index = build_hourly_demand_index(returns_df)
station_region_map = build_station_region_map(returns_df)
historical_conditions = build_historical_conditions(returns_df).set_index("hour")

stations = sorted(encoders["pickup_station"])
default_index = stations.index(DEFAULT_STATION) if DEFAULT_STATION in stations else 0

_min_date = datetime.date(2024, 1, 8)
_max_date = datetime.date(2024, 12, 30)


def _season_of(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


with st.sidebar:
    col_pad_l, col_logo, col_pad_r = st.columns([1, 2, 1])
    with col_logo:
        st.image("ov_fiets_logo.png", use_container_width=True)
    st.markdown("---")
    st.markdown("#### Station")
    selected_station = st.selectbox(
        "Station",
        stations,
        index=default_index,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### Date & time")
    simulated_date = st.date_input(
        "Date",
        value=datetime.date(2024, 11, 25),
        min_value=_min_date,
        max_value=_max_date,
        label_visibility="collapsed",
        help="History before 2024-01-08 is unavailable (168 h of lag required).",
    )

    _time_options = [f"{h:02d}:00" for h in range(24)]
    _selected_time = st.selectbox(
        "Time",
        _time_options,
        index=13,
        label_visibility="collapsed",
    )
    simulated_hour = int(_selected_time.split(":")[0])

    st.markdown("---")
    st.markdown("#### Re-allocation simulator slider")
    st.caption("Simulate re-allocation of bikes in reservation pool via this slider")
    bikes_available_now = st.slider(
        "Re-allocation simulator slider",
        min_value=0,
        max_value=20,
        value=12,
        label_visibility="collapsed",
    )
    st.markdown(
        '<div style="display:flex;justify-content:space-between;'
        'font-size:0.78rem;color:#7b8494;margin-top:-0.9rem;">'
        "<span>0</span><span>20</span></div>",
        unsafe_allow_html=True,
    )

    _hour_ts = pd.Timestamp(simulated_date) + pd.Timedelta(hours=int(simulated_hour))
    if _hour_ts in historical_conditions.index:
        weather = str(historical_conditions.at[_hour_ts, "weather_condition"])
        temperature_c = float(historical_conditions.at[_hour_ts, "temperature_celsius"])
    else:
        weather = "Clear"
        temperature_c = SEASON_DEFAULT_TEMP[_season_of(simulated_date.month)]
    if weather not in encoders["weather_condition"]:
        weather = encoders["weather_condition"][0]

    st.markdown("---")
    st.markdown(
        f"""
        <div class="ov-scope-card">
            <div class="ov-scope-label">Active scope</div>
            <div class="ov-scope-row"><span>Station</span><strong>{selected_station}</strong></div>
            <div class="ov-scope-row"><span>Timestamp</span><strong>{simulated_date} · {simulated_hour:02d}:00</strong></div>
            <div class="ov-scope-row"><span>Pool free</span><strong>{bikes_available_now} / 20</strong></div>
            <div class="ov-scope-row"><span>Recorded conditions</span><strong>{weather} · {temperature_c:.1f}°C</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_console_header(
    title="OV-fiets Employee Operations Dashboard",
    subtitle="Real-time inventory and 4-hour demand outlook for the operations desk",
    timestamp=f"{simulated_date} · {simulated_hour:02d}:00",
    station=selected_station,
)

snapshot = render_inventory_overview(
    returns_df=returns_df,
    model=model,
    encoders=encoders,
    hourly_index=hourly_index,
    station_region_map=station_region_map,
    selected_station=selected_station,
    simulated_hour=simulated_hour,
    simulated_date=simulated_date,
    bikes_available_now=bikes_available_now,
    weather=weather,
    temperature_c=temperature_c,
)

render_pressure_banner(snapshot)

render_allocation_widget(snapshot, simulated_hour=simulated_hour)

render_simulated_pickup_chart(selected_station, simulated_hour=simulated_hour)

render_reservations_cancellations_chart(
    returns_df=returns_df,
    selected_station=selected_station,
    simulated_date=simulated_date,
    simulated_hour=simulated_hour,
    forecast_next_4h=snapshot.bikes_reserved,
)
