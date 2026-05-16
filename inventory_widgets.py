"""Operational widgets for the OV-fiets employee inventory console."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd
import streamlit as st


TOTAL_CAPACITY = 100  # fixed total bikes per station for simulation
CANONICAL_PICKUP_CAPACITY = 80  # 80:20 reference; all other modes derive from this curve

NS_BLUE = "#0063d3"
NS_YELLOW = "#ffc917"
BG = "#f6f7f9"
PANEL = "#ffffff"
BORDER = "#dde2e8"
TEXT = "#243044"
MUTED = "#7b8494"
GREEN = "#21b36b"
ORANGE = "#f59e0b"
RED = "#ef4444"

RESERVATION_STATUSES = {"Reserved", "Confirmed", "Active", "Pending"}
CANCELLED_STATUSES = {"Cancelled", "Cancelled - Free", "Cancelled - Late", "No-show"}
COMPLETED_STATUS = "Completed"


@dataclass(frozen=True)
class InventorySnapshot:
    """Latest station values used by all operational widgets."""

    station: str
    latest_hour: pd.Timestamp | None
    total_available: int
    pickup_available: int
    reservation_available: int
    bikes_reserved: int
    already_taken: int
    reservation_pool: int
    live_count: int
    severity: str
    severity_ratio: float


def inject_styles() -> None:
    """Apply the light operational UI skin."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html {{
            font-size: 18px;
        }}

        html, body, .stApp, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}

        .stApp {{
            background: {BG};
            color: {TEXT};
        }}

        .block-container {{
            max-width: 1280px;
            padding-top: 4rem;
            padding-bottom: 4rem;
        }}

        section[data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid {BORDER};
        }}

        h1, h2, h3, p, label, span, div[data-testid="stMarkdownContainer"] {{
            color: {TEXT};
        }}

        .ov-header {{
            padding-top: 0.1rem;
            padding-bottom: 1.35rem;
            margin-bottom: 0.7rem;
            border-bottom: 1px solid {BORDER};
        }}

        .ov-header-row {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}

        .ov-eyebrow {{
            color: {NS_BLUE};
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }}

        .ov-title {{
            font-size: 1.95rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            color: #003b88;
            margin-top: 0.25rem;
        }}

        .ov-subtitle {{
            color: {MUTED};
            font-size: 0.98rem;
            margin-top: 0.3rem;
            max-width: 680px;
        }}

        .ov-header-meta {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
            flex-wrap: wrap;
        }}

        .ov-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.32rem 0.7rem;
            background: #ffffff;
            border: 1px solid {BORDER};
            border-radius: 999px;
            color: {TEXT};
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }}

        .ov-chip-quiet {{
            color: {MUTED};
            font-weight: 600;
        }}

        .ov-chip-dot {{
            width: 6px; height: 6px;
            border-radius: 50%;
            background: {NS_BLUE};
            display: inline-block;
        }}

        .ov-chip-live {{
            color: {GREEN};
            border-color: rgba(33,179,107,0.35);
            background: #f0faf5;
        }}

        .ov-chip-pulse {{
            width: 7px; height: 7px;
            border-radius: 50%;
            background: {GREEN};
            display: inline-block;
            box-shadow: 0 0 0 0 rgba(33,179,107,0.7);
            animation: ov-pulse 1.8s infinite;
        }}

        @keyframes ov-pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(33,179,107,0.6); }}
            70% {{ box-shadow: 0 0 0 8px rgba(33,179,107,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(33,179,107,0); }}
        }}

        .ov-scope-card {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.85rem 1rem 0.95rem 1rem;
        }}

        .ov-scope-label {{
            color: {NS_BLUE};
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }}

        .ov-scope-row {{
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            padding: 0.15rem 0;
            font-size: 0.85rem;
        }}

        .ov-scope-row span {{ color: {MUTED}; font-weight: 600; }}
        .ov-scope-row strong {{ color: #003b88; font-weight: 700; }}

        .ov-section-title {{
            color: {NS_BLUE};
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin: 1.85rem 0 0.75rem 0;
            border-left: 3px solid {NS_BLUE};
            padding-left: 0.65rem;
        }}

        .ov-card {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.07);
            padding: 1.3rem 1.4rem;
            height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
        }}

        .ov-card-label {{
            color: {MUTED};
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            line-height: 1.2;
        }}

        .ov-card-value {{
            color: #003b88;
            font-size: 2.35rem;
            line-height: 1;
            font-weight: 900;
            margin-top: 0.25rem;
        }}

        .ov-card-note {{
            color: #9aa3b2;
            font-size: 0.86rem;
            margin-top: 0.25rem;
        }}

        .ov-pressure {{
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.08);
            padding: 1.55rem 1.6rem;
            min-height: 245px;
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 2rem;
            align-items: center;
        }}

        .ov-pressure-kicker {{
            color: {MUTED};
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            line-height: 1.2;
        }}

        .ov-pressure-status {{
            color: var(--pressure);
            font-size: 1.7rem;
            font-weight: 950;
            line-height: 1.1;
            margin-top: 0.75rem;
        }}

        .ov-pressure-copy {{
            color: {TEXT};
            font-size: 0.9rem;
            margin-top: 0.8rem;
        }}

        .ov-pressure-numbers {{
            width: 218px;
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 1.25rem 1.2rem;
        }}

        .ov-kv {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: {TEXT};
            font-weight: 800;
            font-size: 0.92rem;
            padding: 0.22rem 0;
        }}

        .ov-kv span:first-child {{
            color: {MUTED};
            font-weight: 700;
        }}

        .ov-kv strong {{
            color: #003b88;
            font-weight: 900;
        }}

        .ov-mini .ov-kv {{
            font-size: 1rem;
            padding: 0;
        }}

        .ov-mini .ov-kv strong {{
            font-size: 1.35rem;
        }}

        .ov-pressure-numbers .ov-kv {{
            display: block;
            padding: 0 0 1.02rem 0;
        }}

        .ov-pressure-numbers .ov-kv strong {{
            display: block;
            color: #003b88;
            font-size: 1.0rem;
            font-weight: 850;
            line-height: 1.2;
            margin-top: 0.28rem;
        }}

        .ov-pressure-numbers .ov-kv:last-child {{
            padding-bottom: 0;
        }}

        .ov-pressure-numbers .ov-kv:last-child strong {{
            color: #003b88;
            font-size: 1.75rem;
            font-weight: 900;
            margin-top: 0.2rem;
        }}

        .ov-panel {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 9px;
            box-shadow: 0 1px 2px rgba(15,23,42,0.12);
            padding: 1.35rem;
        }}

        .ov-mini-title {{
            color: {MUTED};
            font-size: 0.78rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0;
        }}

        .ov-mini {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1.1rem 1.4rem;
            height: 150px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 0.35rem;
            margin-bottom: 0.8rem;
        }}

        .ov-alert {{
            border: 1px solid {BORDER};
            border-left: 6px solid var(--alert);
            background: #ffffff;
            border-radius: 9px;
            color: {TEXT};
            font-weight: 850;
            padding: 0.75rem 0.85rem;
            margin-top: 0.75rem;
        }}

        .ov-action-space {{
            height: 1.15rem;
        }}

        .ov-mode {{
            background: #edf2fc;
            border: 1px solid #d0daf0;
            border-radius: 10px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 0.8rem;
            height: 150px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .ov-mode-label {{
            color: {MUTED};
            font-size: 0.78rem;
            font-weight: 850;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}

        .ov-mode-value {{
            color: #003b88;
            font-size: 2.2rem;
            font-weight: 950;
            line-height: 1;
            margin: 0;
        }}

        .ov-mode-note {{
            color: {MUTED};
            font-size: 0.9rem;
            margin: 0;
        }}

        /* Strip unintended chrome from Streamlit's layout primitives */
        div[data-testid="stVerticalBlock"],
        div[data-testid="stHorizontalBlock"],
        div[data-testid="column"],
        div[data-testid="stElementContainer"] {{
            background: transparent !important;
            box-shadow: none !important;
        }}

        div.stButton > button {{
            background: #ffffff;
            border: 1px solid {BORDER};
            border-radius: 9px;
            color: #003b88;
            font-weight: 850;
            min-height: 3.55rem;
            box-shadow: 0 1px 2px rgba(15,23,42,0.08);
        }}

        div.stButton > button:hover {{
            border-color: {NS_BLUE};
            background: #f8fafc;
            color: #003b88;
        }}

        div.stButton > button[kind="primary"],
        div.stButton button[data-testid="stBaseButton-primary"] {{
            background: {NS_YELLOW};
            border-color: {NS_YELLOW};
            color: #003b88;
            font-weight: 950;
            box-shadow: none;
        }}

        div.stButton > button[kind="primary"]:hover,
        div.stButton button[data-testid="stBaseButton-primary"]:hover {{
            background: #ffd33d;
            border-color: #ffd33d;
            color: #003b88;
        }}

        /* Top toolbar */
        header[data-testid="stHeader"] {{
            background: {BG} !important;
        }}

        /* Sidebar widget inputs (selectbox, date, time, number) */
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-baseweb="input"],
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] {{
            background: {NS_BLUE} !important;
            color: #ffffff !important;
            border-color: {NS_BLUE} !important;
        }}

        section[data-testid="stSidebar"] [data-baseweb="select"] svg,
        section[data-testid="stSidebar"] [data-baseweb="input"] svg {{
            fill: #ffffff !important;
        }}

        section[data-testid="stSidebar"] input::placeholder {{
            color: rgba(255,255,255,0.6) !important;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


def build_station_options(returns_df: pd.DataFrame, predictions_df: pd.DataFrame) -> list[str]:
    """Create a station dropdown from the real data sources."""
    stations = set()
    if "pickup_station" in returns_df.columns:
        stations.update(returns_df["pickup_station"].dropna().astype(str).unique())
    if "station" in predictions_df.columns:
        stations.update(predictions_df["station"].dropna().astype(str).unique())
    return sorted(stations)


def render_console_header(
    title: str,
    subtitle: str,
    timestamp: str | None = None,
    station: str | None = None,
) -> None:
    inject_styles()
    chips = ""
    if station or timestamp:
        chip_items = []
        if station:
            chip_items.append(
                f'<span class="ov-chip"><span class="ov-chip-dot"></span>{escape(station)}</span>'
            )
        if timestamp:
            chip_items.append(f'<span class="ov-chip ov-chip-quiet">{escape(timestamp)}</span>')
        chip_items.append(
            '<span class="ov-chip ov-chip-live"><span class="ov-chip-pulse"></span>Live forecast</span>'
        )
        chips = f'<div class="ov-header-meta">{"".join(chip_items)}</div>'

    st.markdown(
        f"""
        <div class="ov-header">
            <div class="ov-header-row">
                <div>
                    <div class="ov-eyebrow">NS · Operations</div>
                    <div class="ov-title">{escape(title)}</div>
                    <div class="ov-subtitle">{escape(subtitle)}</div>
                </div>
                {chips}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _latest_prediction(predictions_df: pd.DataFrame, station: str) -> pd.Series | None:
    station_pred = predictions_df[predictions_df["station"].astype(str) == str(station)].copy()
    if station_pred.empty:
        return None
    station_pred["hour"] = pd.to_datetime(station_pred["hour"], errors="coerce")
    return station_pred.sort_values("hour").iloc[-1]


def _prediction_at(
    predictions_df: pd.DataFrame,
    station: str,
    sim_date: datetime.date,
    sim_hour: int,
) -> pd.Series | None:
    """Return the CSV row for the given station, date, and hour; None if not found."""
    rows = predictions_df[predictions_df["station"].astype(str) == str(station)].copy()
    if rows.empty:
        return None
    rows["hour"] = pd.to_datetime(rows["hour"], errors="coerce")
    mask = (rows["hour"].dt.date == sim_date) & (rows["hour"].dt.hour == sim_hour)
    matched = rows[mask]
    if matched.empty:
        day_rows = rows[rows["hour"].dt.date == sim_date]
        if day_rows.empty:
            return _latest_prediction(predictions_df, station)
        idx = (day_rows["hour"].dt.hour - sim_hour).abs().idxmin()
        return day_rows.loc[idx]
    return matched.iloc[-1]


def _station_returns(returns_df: pd.DataFrame, station: str) -> pd.DataFrame:
    if "pickup_station" not in returns_df.columns:
        return pd.DataFrame()
    return returns_df[returns_df["pickup_station"].astype(str) == str(station)].copy()


def _latest_window_returns(returns_df: pd.DataFrame, station: str) -> pd.DataFrame:
    station_df = _station_returns(returns_df, station)
    if station_df.empty or "reservation_date" not in station_df.columns:
        return station_df

    dates = pd.to_datetime(station_df["reservation_date"], errors="coerce")
    latest_date = dates.max()
    if pd.isna(latest_date):
        return station_df.tail(0)
    return station_df[dates.dt.date == latest_date.date()].copy()


def _count_reserved(window_df: pd.DataFrame) -> int:
    if window_df.empty or "rental_status" not in window_df.columns:
        return 0
    status = window_df["rental_status"].fillna("").astype(str)
    explicit = status.isin(RESERVATION_STATUSES).sum()
    if explicit:
        return int(explicit)
    active_reservations = ~status.isin(CANCELLED_STATUSES | {COMPLETED_STATUS})
    return int(active_reservations.sum())


def _count_in_use(window_df: pd.DataFrame) -> int:
    if window_df.empty or "rental_status" not in window_df.columns:
        return 0
    return int(window_df["rental_status"].fillna("").astype(str).eq(COMPLETED_STATUS).sum())


def _safe_int(value: object) -> int:
    if pd.isna(value):
        return 0
    return max(0, int(round(float(value))))


def _allocation_from_total(total_available: int, reservation_share: float = 0.20) -> tuple[int, int]:
    reservation = int(round(total_available * reservation_share))
    pickup = max(0, total_available - reservation)
    return pickup, reservation


def _active_reservation_share(station: str) -> float:
    key = f"allocation_{station}_active_pct"
    return float(st.session_state.get(key, 20)) / 100


def _simulate_pickup_baseline(station: str) -> pd.DataFrame:
    """Deterministic 24 h pickup-pool curve at the CANONICAL_PICKUP_CAPACITY (80).

    Departures are a property of demand, not pool size. All re-allocation modes derive
    from this single baseline so bikes are conserved when the operator shifts the split.
    """
    today = datetime.date.today().isoformat()
    seed = hash((station, today, CANONICAL_PICKUP_CAPACITY)) & 0xFFFF_FFFF
    rng = np.random.default_rng(seed)

    hours = np.arange(24)
    morning = np.exp(-0.5 * ((hours - 7.5) / 1.1) ** 2)
    evening = np.exp(-0.5 * ((hours - 17.5) / 1.1) ** 2)
    intensity = 0.55 * morning + 1.0 * evening + 0.04

    max_rate = CANONICAL_PICKUP_CAPACITY * 0.40
    departures = rng.poisson(intensity * max_rate).clip(0, CANONICAL_PICKUP_CAPACITY).astype(int)

    returns = np.zeros(24, dtype=int)
    for h, d in enumerate(departures):
        if d > 0:
            for delay in rng.integers(2, 5, size=int(d)):
                returns[(h + int(delay)) % 24] += 1

    available = np.empty(24, dtype=int)
    current = CANONICAL_PICKUP_CAPACITY
    for h in range(24):
        current = min(
            CANONICAL_PICKUP_CAPACITY,
            max(0, current + int(returns[h]) - int(departures[h])),
        )
        available[h] = current

    return pd.DataFrame({"Hour": hours, "Bikes Available": available})


def _pickup_curve_for_capacity(station: str, pickup_capacity: int) -> pd.DataFrame:
    """Derive a pickup curve at any capacity from the canonical baseline (bike-conserving)."""
    base = _simulate_pickup_baseline(station)
    shift = CANONICAL_PICKUP_CAPACITY - int(pickup_capacity)
    shifted = (base["Bikes Available"] - shift).clip(lower=0, upper=int(pickup_capacity))
    return pd.DataFrame({"Hour": base["Hour"].values, "Bikes Available": shifted.astype(int).values})


@st.cache_data(show_spinner=False)
def build_hourly_demand_index(returns_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-station hourly reservation counts + 24h/168h rolling means for lag features.

    Counts every booking regardless of rental_status (no-shows + cancellations included) —
    matches the training pipeline's demand definition. Rolling means here are unshifted
    (`s.rolling(24).mean()`); `_roll_value` looks up at `dt - 1h`, which yields the prior
    24h/168h window — equivalent to the training pipeline's `shift(1).rolling(N).mean()`.
    """
    if "pickup_station" not in returns_df.columns or "reservation_date" not in returns_df.columns:
        return {}

    df = returns_df.copy()
    df["rdt"] = pd.to_datetime(
        df["reservation_date"].astype(str) + " " + df["reservation_time"].astype(str),
        errors="coerce",
    )
    df = df[df["rdt"].notna()].copy()

    if df.empty:
        return {}

    start = df["rdt"].min().floor("h")
    end = df["rdt"].max().ceil("h")
    full_index = pd.date_range(start, end, freq="h")

    out: dict[str, pd.DataFrame] = {}
    df["hour_bucket"] = df["rdt"].dt.floor("h")
    grouped = df.groupby(["pickup_station", "hour_bucket"]).size()
    for station, station_series in grouped.groupby(level=0):
        s = station_series.droplevel(0).reindex(full_index, fill_value=0).astype(float)
        out[str(station)] = pd.DataFrame(
            {
                "demand": s,
                "roll_24h": s.rolling(24, min_periods=1).mean(),
                "roll_168h": s.rolling(168, min_periods=1).mean(),
            }
        )
    return out


@st.cache_data(show_spinner=False)
def build_historical_conditions(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Per-hour (weather, temperature) aggregated across stations — matches training pipeline.

    Training uses `raw.groupby('hour').agg(weather=first, temperature=mean)` so each hour has
    one regional weather and one mean temperature shared across all stations. The dashboard
    uses this to seed sidebar defaults so a fresh load on any historical timestamp matches
    the CSV-row prediction; operator can still override.
    """
    if "reservation_date" not in returns_df.columns:
        return pd.DataFrame(columns=["hour", "weather_condition", "temperature_celsius"])
    df = returns_df.copy()
    df["rdt"] = pd.to_datetime(
        df["reservation_date"].astype(str) + " " + df["reservation_time"].astype(str),
        errors="coerce",
    )
    df = df[df["rdt"].notna()].copy()
    df["hour"] = df["rdt"].dt.floor("h")
    out = (
        df.groupby("hour")
        .agg(
            weather_condition=("weather_condition", "first"),
            temperature_celsius=("temperature_celsius", "mean"),
        )
        .reset_index()
    )
    return out


@st.cache_data(show_spinner=False)
def build_station_region_map(returns_df: pd.DataFrame) -> dict[str, str]:
    """Station → region from the returns CSV (1:1 mapping)."""
    if "pickup_station" not in returns_df.columns or "station_region" not in returns_df.columns:
        return {}
    pairs = returns_df[["pickup_station", "station_region"]].dropna().drop_duplicates()
    return {str(r["pickup_station"]): str(r["station_region"]) for _, r in pairs.iterrows()}


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _lag_value(hourly_index: dict, station: str, dt: pd.Timestamp, hours_back: int) -> float:
    table = hourly_index.get(station)
    if table is None:
        return 0.0
    ts = dt - pd.Timedelta(hours=hours_back)
    if ts in table.index:
        return float(table.at[ts, "demand"])
    return 0.0


def _roll_value(hourly_index: dict, station: str, dt: pd.Timestamp, column: str) -> float:
    table = hourly_index.get(station)
    if table is None:
        return 0.0
    ts = dt - pd.Timedelta(hours=1)
    if ts in table.index:
        return float(table.at[ts, column])
    return 0.0


def predict_demand_live(
    model,
    encoders: dict,
    hourly_index: dict,
    station_region_map: dict,
    station: str,
    dt: pd.Timestamp,
    bikes_available_now: int,
    weather: str,
    temperature_c: float,
) -> tuple[float, dict]:
    """Build the 16-feature row at runtime and call the LightGBM model. Returns (pred, features)."""
    if station not in encoders.get("pickup_station", []):
        return 0.0, {}

    region = station_region_map.get(station, encoders["station_region"][0])
    season = _season_from_month(dt.month)
    weather_safe = weather if weather in encoders["weather_condition"] else encoders["weather_condition"][0]

    features = {
        "pickup_station": station,
        "station_region": region if region in encoders["station_region"] else encoders["station_region"][0],
        "weather_condition": weather_safe,
        "season": season,
        "hour_of_day": int(dt.hour),
        "day_of_week": int(dt.dayofweek),
        "month": int(dt.month),
        "is_weekend": int(dt.dayofweek >= 5),
        "is_peak_hour": int(dt.hour in {7, 8, 9, 17, 18, 19}),
        "temperature_celsius": float(temperature_c),
        "bikes_available_now": int(bikes_available_now),
        "lag_1h": _lag_value(hourly_index, station, dt, 1),
        "lag_24h": _lag_value(hourly_index, station, dt, 24),
        "lag_168h": _lag_value(hourly_index, station, dt, 168),
        "roll_24h_mean": _roll_value(hourly_index, station, dt, "roll_24h"),
        "roll_168h_mean": _roll_value(hourly_index, station, dt, "roll_168h"),
    }

    row = pd.DataFrame([features])[encoders["FEATURE_COLS"]]
    for col in encoders["CAT_COLS"]:
        row[col] = pd.Categorical(row[col], categories=encoders[col])
    pred = float(model.predict(row)[0])
    return max(0.0, pred), features


def build_inventory_snapshot(
    returns_df: pd.DataFrame,
    model,
    encoders: dict,
    hourly_index: dict,
    station_region_map: dict,
    selected_station: str,
    simulated_hour: int,
    simulated_date: datetime.date,
    bikes_available_now: int,
    weather: str,
    temperature_c: float,
) -> InventorySnapshot:
    """Compute the live snapshot via model.predict for the chosen scenario."""
    dt = pd.Timestamp(simulated_date) + pd.Timedelta(hours=int(simulated_hour))
    bikes_left = max(0, min(20, int(bikes_available_now)))

    demand_pred_raw, _features = predict_demand_live(
        model, encoders, hourly_index, station_region_map,
        selected_station, dt, bikes_left, weather, temperature_c,
    )
    bikes_reserved = int(round(demand_pred_raw))

    reservation_share = _active_reservation_share(selected_station)
    pickup_capacity = max(1, int(round(TOTAL_CAPACITY * (1 - reservation_share))))
    reservation_pool = TOTAL_CAPACITY - pickup_capacity

    already_taken = max(0, 20 - bikes_left)
    live_count = max(0, bikes_left + (reservation_pool - 20))
    reservation_available = max(0, live_count - bikes_reserved)
    severity, severity_ratio = _severity(demand_pred_raw, live_count, reservation_pool)

    sim_df = _pickup_curve_for_capacity(selected_station, pickup_capacity)
    pickup_available = int(sim_df.iloc[int(simulated_hour)]["Bikes Available"])

    return InventorySnapshot(
        station=selected_station,
        latest_hour=dt,
        total_available=bikes_left,
        pickup_available=pickup_available,
        reservation_available=reservation_available,
        bikes_reserved=bikes_reserved,
        already_taken=already_taken,
        reservation_pool=reservation_pool,
        live_count=live_count,
        severity=severity,
        severity_ratio=severity_ratio,
    )


def _metric_card(label: str, value: int, note: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="ov-card" style="--accent:{color};">
            <div class="ov-card-label">{escape(label)}</div>
            <div class="ov-card-value">{value:,}</div>
            <div class="ov-card-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_inventory_overview(
    returns_df: pd.DataFrame,
    model,
    encoders: dict,
    hourly_index: dict,
    station_region_map: dict,
    selected_station: str,
    simulated_hour: int,
    simulated_date: datetime.date,
    bikes_available_now: int,
    weather: str,
    temperature_c: float,
) -> InventorySnapshot:
    """Render the three compact inventory metric cards from a live-inferred snapshot."""
    inject_styles()
    snapshot = build_inventory_snapshot(
        returns_df, model, encoders, hourly_index, station_region_map,
        selected_station, simulated_hour, simulated_date,
        bikes_available_now, weather, temperature_c,
    )
    hour_text = (
        pd.Timestamp(snapshot.latest_hour).strftime("%Y-%m-%d %H:%M")
        if snapshot.latest_hour is not None and pd.notna(snapshot.latest_hour)
        else "latest snapshot"
    )

    pickup_pct = 100 - int(round(snapshot.reservation_pool))
    reservation_pct = int(round(snapshot.reservation_pool))
    reservation_note = (
        f"{snapshot.already_taken} already reserved"
    )

    st.markdown('<div class="ov-section-title">Live inventory</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        _metric_card("Pickup available", snapshot.pickup_available, f"As of {hour_text}", NS_BLUE)
    with cols[1]:
        _metric_card(
            "Reservation available",
            snapshot.live_count,
            reservation_note,
            NS_YELLOW,
        )
    with cols[2]:
        _metric_card("Forecast · next 4 h", snapshot.bikes_reserved, "Live model · LightGBM", ORANGE)
    return snapshot


def _severity(demand_pred: float, live_count: int, reservation_pool: int) -> tuple[str, float]:
    """Classify severity by projected % of pool remaining after 4-hour demand.

    projected_pct = (live_count - demand_pred) / reservation_pool
    Green  > 60% projected free
    Orange 10–60% projected free
    Red    < 10% projected free  (or live_count == 0)
    """
    if reservation_pool <= 0 or live_count <= 0:
        return "Red", 0.0
    projected_remaining = max(0.0, float(live_count) - float(demand_pred))
    pct = projected_remaining / float(reservation_pool)
    if pct >= 0.60:
        return "Green", pct
    if pct >= 0.10:
        return "Orange", pct
    return "Red", pct


def _severity_label(status: str) -> str:
    s = str(status).strip().lower()
    if s == "red":
        return "high"
    if s == "orange":
        return "medium"
    return "low"


def _pressure_theme(status: str) -> dict[str, str]:
    status = str(status).strip().lower()
    if status == "red":
        return {
            "color": RED,
            "bg": "rgba(239, 68, 68, 0.10)",
            "border": "rgba(239, 68, 68, 0.45)",
        }
    if status == "orange":
        return {
            "color": ORANGE,
            "bg": "rgba(245, 158, 11, 0.10)",
            "border": "rgba(245, 158, 11, 0.45)",
        }
    return {
        "color": GREEN,
        "bg": "rgba(52, 199, 89, 0.10)",
        "border": "rgba(52, 199, 89, 0.45)",
    }


def _pressure_message(status: str) -> str:
    status = str(status).strip().lower()
    if status == "red":
        return "Less than 10% of reservation bikes available. Re-allocate or refill immediately."
    if status == "orange":
        return "10–60% of reservation bikes available. Monitor closely and prepare to re-allocate."
    return "More than 60% of reservation bikes still available. No action required."


def render_pressure_banner(snapshot: InventorySnapshot) -> None:
    """Render the live ML pressure banner from the inventory snapshot."""
    inject_styles()

    st.markdown('<div class="ov-section-title">Demand outlook · next 4 hours</div>', unsafe_allow_html=True)

    if snapshot.latest_hour is None or pd.isna(snapshot.latest_hour):
        st.markdown(
            f"""
            <div class="ov-pressure" style="background:rgba(245,158,11,0.10); border:1px solid rgba(245,158,11,0.45);">
                <div>
                    <div class="ov-pressure-kicker">Forecast unavailable</div>
                    <div class="ov-pressure-status">Unavailable</div>
                    <div class="ov-pressure-copy">No forecast is available for this station at the chosen hour.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    theme = _pressure_theme(snapshot.severity)
    hour = pd.Timestamp(snapshot.latest_hour).strftime("%Y-%m-%d %H:%M")
    projected_remaining = max(0, snapshot.live_count - snapshot.bikes_reserved)
    avail_pct = (
        int(round(projected_remaining / snapshot.reservation_pool * 100))
        if snapshot.reservation_pool > 0
        else 0
    )

    st.markdown(
        f"""
        <div class="ov-pressure" style="background:{theme['bg']}; border:1px solid {theme['border']};">
            <div>
                <div class="ov-pressure-kicker">Reservation pool · {_severity_label(snapshot.severity)} pressure</div>
                <div class="ov-pressure-status" style="color:{theme['color']};">{avail_pct}% bikes are predicted to be available for reservation</div>
                <div class="ov-pressure-copy">Demand forecast for next 4 hours. To re-allocate use pool allocation below.</div>
            </div>
            <div class="ov-pressure-numbers">
                <div class="ov-kv"><span>Forecast hour</span><strong>{escape(hour)}</strong></div>
                <div class="ov-kv"><span>Pool capacity</span><strong>{snapshot.reservation_pool}</strong></div>
                <div class="ov-kv"><span>Already booked</span><strong>{snapshot.already_taken}</strong></div>
                <div class="ov-kv"><span>Forecast next 4 h</span><strong>{snapshot.bikes_reserved} bookings</strong></div>
                <div class="ov-kv"><span>Available now</span><strong>{snapshot.live_count}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _allocation_card(title: str, pickup: int, reservation: int) -> str:
    return f"""
    <div class="ov-mini">
        <div class="ov-mini-title">{escape(title)}</div>
        <div class="ov-kv"><span>Pickup</span><strong>{pickup:,}</strong></div>
        <div class="ov-kv"><span>Reservation</span><strong>{reservation:,}</strong></div>
    </div>
    """


def _allocation_alert(after_pickup: int, after_reservation: int, total: int) -> tuple[str, str] | None:
    pickup_floor = max(2, int(round(total * 0.15)))
    reservation_floor = max(1, int(round(total * 0.10)))
    if after_pickup <= pickup_floor:
        return RED, "Pickup inventory critically low."
    if after_reservation <= reservation_floor:
        return ORANGE, "Reservation inventory critically low."
    return None


def _allocation_state_at(
    reservation_pct: int,
    snapshot: InventorySnapshot,
    simulated_hour: int,
) -> tuple[int, int]:
    """Compute (pickup_available, reservation_live_remaining) for a hypothetical share.

    Symmetric, bike-conserving math: already-departed pickups and already-taken reservations
    are invariant under re-allocation; only the pool capacities move.
    """
    reservation_pool_new = int(round(TOTAL_CAPACITY * reservation_pct / 100))
    pickup_capacity_new = max(0, TOTAL_CAPACITY - reservation_pool_new)
    pickup_capacity_active = max(0, TOTAL_CAPACITY - snapshot.reservation_pool)
    already_departed_pickup = max(0, pickup_capacity_active - snapshot.pickup_available)
    pickup_available_new = max(0, pickup_capacity_new - already_departed_pickup)
    live_count_new = max(0, snapshot.total_available + (reservation_pool_new - 20))
    return pickup_available_new, live_count_new


def render_allocation_widget(snapshot: InventorySnapshot, simulated_hour: int) -> None:
    """Render the human-controlled allocation preview widget."""
    inject_styles()
    st.markdown(
        '<div class="ov-section-title">Pool allocation</div>',
        unsafe_allow_html=True,
    )

    key_prefix = f"allocation_{snapshot.station}"
    active_key = f"{key_prefix}_active_pct"
    pending_key = f"{key_prefix}_pending_pct"
    st.session_state.setdefault(active_key, 20)
    st.session_state.setdefault(pending_key, st.session_state[active_key])

    active_reservation_pct = int(st.session_state[active_key])
    active_pickup_pct = 100 - active_reservation_pct
    pending_reservation_pct = int(st.session_state[pending_key])
    pending_pickup_pct = 100 - pending_reservation_pct
    has_pending_change = pending_reservation_pct != active_reservation_pct

    after_pickup, after_reservation = _allocation_state_at(
        pending_reservation_pct, snapshot, simulated_hour
    )

    with st.container(border=True):
        current_left, current_right = st.columns([1, 1])
        with current_left:
            st.markdown(
                f"""
                <div class="ov-mode">
                    <div class="ov-mode-label">Active split · Pickup : Reservation</div>
                    <div class="ov-mode-value">{active_pickup_pct} : {active_reservation_pct}</div>
                    <div class="ov-mode-note">Currently in effect</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with current_right:
            st.markdown(
                _allocation_card(
                    "Available right now",
                    snapshot.pickup_available,
                    snapshot.live_count,
                ),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="ov-mini-title">Switch to</div>', unsafe_allow_html=True)
        preset_cols = st.columns(3)
        for col, (label, reservation_pct) in zip(preset_cols, [("80:20", 20), ("70:30", 30), ("60:40", 40)]):
            with col:
                if st.button(
                    label,
                    key=f"{key_prefix}_{label}",
                    type="primary" if reservation_pct == pending_reservation_pct else "secondary",
                    use_container_width=True,
                ):
                    st.session_state[pending_key] = reservation_pct
                    st.rerun()

        if has_pending_change:
            st.warning("Attention! This will change the bike pool allotment, review changes carefully below.")
            preview_left, preview_right = st.columns([1, 1])
            with preview_left:
                st.markdown(
                    f"""
                    <div class="ov-mode">
                        <div class="ov-mode-label">New split · Pickup : Reservation</div>
                        <div class="ov-mode-value">{pending_pickup_pct} : {pending_reservation_pct}</div>
                        <div class="ov-mode-note">Save to apply</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with preview_right:
                st.markdown(
                    _allocation_card(
                        "After re-allocation",
                        after_pickup,
                        after_reservation,
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="ov-action-space"></div>', unsafe_allow_html=True)
            cancel_col, save_col = st.columns([1, 1])
            with cancel_col:
                if st.button("Cancel", key=f"{key_prefix}_cancel", use_container_width=True):
                    st.session_state[pending_key] = st.session_state[active_key]
                    st.rerun()
            with save_col:
                if st.button(
                    "Save",
                    key=f"{key_prefix}_save",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state[active_key] = st.session_state[pending_key]
                    st.rerun()

    if has_pending_change:
        alert = _allocation_alert(after_pickup, after_reservation, TOTAL_CAPACITY)
        if alert:
            color, text = alert
            st.markdown(
                f"""
                <div class="ov-alert" style="--alert:{color};">
                    {escape(text)}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_simulated_pickup_chart(selected_station: str, simulated_hour: int | None = None) -> None:
    """Render the stochastic pickup-pool availability chart up to the simulated hour."""
    inject_styles()
    reservation_share = _active_reservation_share(selected_station)
    pickup_capacity = max(1, int(round(TOTAL_CAPACITY * (1 - reservation_share))))

    full_df = _pickup_curve_for_capacity(selected_station, pickup_capacity)
    current_hour = simulated_hour if simulated_hour is not None else datetime.datetime.now().hour
    current_hour = max(0, min(23, int(current_hour)))
    sim_df = full_df[full_df["Hour"] <= current_hour]

    st.markdown(
        '<div class="ov-section-title">Pickup activity · today</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Pool capacity {pickup_capacity} of {TOTAL_CAPACITY} · "
        f"shown through {current_hour:02d}:00"
    )

    spec = {
        "background": "white",
        "layer": [
            {
                "data": {"values": sim_df.to_dict(orient="records")},
                "mark": {
                    "type": "line",
                    "point": {"filled": True, "size": 60, "color": NS_BLUE},
                    "color": NS_BLUE,
                    "strokeWidth": 2.5,
                    "tooltip": True,
                },
                "encoding": {
                    "x": {
                        "field": "Hour",
                        "type": "quantitative",
                        "scale": {"domain": [0, 23]},
                        "axis": {"title": "Hour of day", "tickCount": 12, "labelColor": TEXT, "titleColor": TEXT},
                    },
                    "y": {
                        "field": "Bikes Available",
                        "type": "quantitative",
                        "scale": {"domain": [0, pickup_capacity]},
                        "axis": {"title": "Pickup bikes available", "labelColor": TEXT, "titleColor": TEXT},
                    },
                    "tooltip": [
                        {"field": "Hour", "type": "quantitative", "title": "Hour"},
                        {"field": "Bikes Available", "type": "quantitative", "title": "Bikes available (pickup)"},
                    ],
                },
            },
            {
                "data": {"values": [{"Hour": current_hour}]},
                "mark": {"type": "rule", "color": NS_YELLOW, "strokeWidth": 2, "strokeDash": [4, 3]},
                "encoding": {"x": {"field": "Hour", "type": "quantitative"}},
            },
        ],
        "height": 240,
        "config": _vega_axis_config(),
    }

    st.vega_lite_chart(spec, use_container_width=True)


@st.cache_data(show_spinner=False)
def build_status_timeline(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Per (station, day, hour) counts of Reservations (Completed) and Cancellations (no-show + cancelled).

    Cached once per session — the same long-form DataFrame is filtered per-render.
    """
    if "pickup_station" not in returns_df.columns:
        return pd.DataFrame(columns=["pickup_station", "date", "Hour", "Reservations", "Cancellations"])
    df = returns_df.copy()
    df["rdt"] = pd.to_datetime(
        df["reservation_date"].astype(str) + " " + df["reservation_time"].astype(str),
        errors="coerce",
    )
    df = df[df["rdt"].notna()].copy()
    df["date"] = df["rdt"].dt.date
    df["Hour"] = df["rdt"].dt.hour
    status = df["rental_status"].fillna("").astype(str)
    df["bucket"] = np.where(status == COMPLETED_STATUS, "Reservations", "Cancellations")
    wide = (
        df.groupby(["pickup_station", "date", "Hour", "bucket"]).size().unstack(fill_value=0).reset_index()
    )
    for col in ("Reservations", "Cancellations"):
        if col not in wide.columns:
            wide[col] = 0
    return wide[["pickup_station", "date", "Hour", "Reservations", "Cancellations"]]


RES_CANCEL_RANGE_OPTIONS = ("Today", "Past 7 days")


def render_reservations_cancellations_chart(
    returns_df: pd.DataFrame,
    selected_station: str,
    simulated_date: datetime.date,
    simulated_hour: int,
    forecast_next_4h: float | None = None,
) -> None:
    """Line plot of historical Reservations (Completed) vs Cancellations, with optional forecast overlay.

    Two display modes via inline dropdown:
      • Today        — 24 hourly points for `simulated_date`, past-only (hour ≤ simulated_hour),
                       plus a forward dashed segment showing the model's 4-hour cumulative forecast
                       distributed as a per-hour rate from hour+1 → hour+4.
      • Past 7 days  — 7 daily totals for the week ending on `simulated_date`.
    """
    inject_styles()
    timeline = build_status_timeline(returns_df)
    station_tl = timeline[timeline["pickup_station"] == selected_station]

    st.markdown(
        '<div class="ov-section-title">Booking activity</div>',
        unsafe_allow_html=True,
    )

    range_key = f"res_cancel_range_{selected_station}"
    col_label, col_select = st.columns([4, 2])
    with col_select:
        display_range = st.selectbox(
            "Display range",
            RES_CANCEL_RANGE_OPTIONS,
            index=0,
            key=range_key,
            label_visibility="collapsed",
        )

    if display_range == "Today":
        _render_res_cancel_today(
            station_tl, selected_station, simulated_date, simulated_hour, forecast_next_4h
        )
    else:
        _render_res_cancel_week(station_tl, selected_station, simulated_date)


def _vega_axis_config() -> dict:
    return {
        "background": "white",
        "view": {"stroke": "transparent", "fill": "white"},
        "axis": {
            "labelColor": TEXT,
            "titleColor": TEXT,
            "domainColor": TEXT,
            "tickColor": "#dde2e8",
            "gridColor": "#edf0f4",
            "labelFontSize": 11,
            "titleFontSize": 12,
        },
    }


def _render_res_cancel_today(
    station_tl: pd.DataFrame,
    selected_station: str,
    simulated_date: datetime.date,
    simulated_hour: int,
    forecast_next_4h: float | None,
) -> None:
    sub = station_tl[station_tl["date"] == simulated_date]
    full = pd.DataFrame({"Hour": range(24)})
    sub = full.merge(sub[["Hour", "Reservations", "Cancellations"]], on="Hour", how="left").fillna(0)
    current_hour = max(0, min(23, int(simulated_hour)))
    past = sub[sub["Hour"] <= current_hour].copy()

    long = past.melt(id_vars="Hour", value_vars=["Reservations", "Cancellations"],
                     var_name="Type", value_name="Count")
    long["Count"] = long["Count"].astype(int)

    # Forecast overlay: distribute the 4-hour cumulative as a per-hour rate over hours h+1..h+4.
    # Anchor the line at current_hour using the latest historical Reservations value so the dashed
    # segment continues directly out of the yellow current-hour rule.
    forecast_df = pd.DataFrame(columns=["Hour", "Forecast"])
    forecast_per_hour = 0.0
    forecast_end = current_hour
    if forecast_next_4h is not None and current_hour < 23:
        forecast_per_hour = max(0.0, float(forecast_next_4h)) / 4.0
        forecast_end = min(23, current_hour + 4)
        last_res = float(past[past["Hour"] == current_hour]["Reservations"].iloc[0]) if not past.empty else 0.0
        future_hours = list(range(current_hour, forecast_end + 1))
        y_values = [last_res] + [forecast_per_hour] * (len(future_hours) - 1)
        forecast_df = pd.DataFrame({"Hour": future_hours, "Forecast": y_values})

    total_res = int(past["Reservations"].sum())
    total_can = int(past["Cancellations"].sum())
    caption = (
        f"{selected_station} · {simulated_date} · history through {current_hour:02d}:00 · "
        f"{total_res} reservations, {total_can} cancellations"
    )
    if forecast_next_4h is not None and current_hour < 23:
        caption += (
            f" · forecast {float(forecast_next_4h):.0f} bookings in next 4 h "
            f"(~{forecast_per_hour:.1f}/h, {current_hour + 1:02d}:00 → {forecast_end:02d}:00)"
        )
    st.caption(caption)

    y_max_observed = max(past["Reservations"].max(), past["Cancellations"].max(), forecast_per_hour)
    y_max = max(1, int(round(float(y_max_observed))))

    layers = [
        {
            "data": {"values": long.to_dict(orient="records")},
            "mark": {"type": "line", "point": {"filled": True, "size": 60}, "strokeWidth": 2.5, "tooltip": True},
            "encoding": {
                "x": {
                    "field": "Hour",
                    "type": "quantitative",
                    "axis": {"title": "Hour of day", "labelColor": TEXT, "titleColor": TEXT, "tickCount": 12},
                    "scale": {"domain": [0, 23]},
                },
                "y": {
                    "field": "Count",
                    "type": "quantitative",
                    "axis": {"title": "Bookings per hour", "labelColor": TEXT, "titleColor": TEXT},
                    "scale": {"domain": [0, y_max + 1]},
                },
                "color": {
                    "field": "Type",
                    "type": "nominal",
                    "scale": {
                        "domain": ["Reservations", "Cancellations", "Forecast (next 4 h)"],
                        "range": [NS_BLUE, ORANGE, "#4A4A4A"],
                    },
                    "legend": {"title": None, "orient": "top", "labelColor": TEXT},
                },
                "tooltip": [
                    {"field": "Hour", "type": "quantitative", "title": "Hour"},
                    {"field": "Type", "type": "nominal"},
                    {"field": "Count", "type": "quantitative"},
                ],
            },
        },
        {
            "data": {"values": [{"Hour": current_hour}]},
            "mark": {"type": "rule", "color": NS_YELLOW, "strokeWidth": 2, "strokeDash": [4, 3]},
            "encoding": {"x": {"field": "Hour", "type": "quantitative"}},
        },
    ]

    if not forecast_df.empty:
        forecast_long = forecast_df.assign(Type="Forecast (next 4 h)").rename(
            columns={"Forecast": "Count"}
        )
        layers.append(
            {
                "data": {"values": forecast_long.to_dict(orient="records")},
                "mark": {
                    "type": "line",
                    "point": {"filled": False, "size": 70, "stroke": "#4A4A4A", "fill": "white", "strokeWidth": 2},
                    "strokeDash": [5, 4],
                    "strokeWidth": 2.5,
                    "color": "#4A4A4A",
                    "tooltip": True,
                },
                "encoding": {
                    "x": {"field": "Hour", "type": "quantitative", "scale": {"domain": [0, 23]}},
                    "y": {"field": "Count", "type": "quantitative"},
                    "color": {
                        "field": "Type",
                        "type": "nominal",
                        "legend": {"title": None, "orient": "top", "labelColor": TEXT},
                    },
                    "tooltip": [
                        {"field": "Hour", "type": "quantitative", "title": "Hour"},
                        {"field": "Type", "type": "nominal"},
                        {"field": "Count", "type": "quantitative", "title": "Forecast per hour"},
                    ],
                },
            }
        )

    spec = {
        "background": "white",
        "layer": layers,
        "height": 240,
        "config": _vega_axis_config(),
        "resolve": {"scale": {"color": "independent"}} if forecast_df.empty else {},
    }
    st.vega_lite_chart(spec, use_container_width=True)


def _render_res_cancel_week(
    station_tl: pd.DataFrame,
    selected_station: str,
    simulated_date: datetime.date,
) -> None:
    days = [simulated_date - datetime.timedelta(days=i) for i in range(6, -1, -1)]
    daily = (
        station_tl[station_tl["date"].isin(days)]
        .groupby("date")[["Reservations", "Cancellations"]]
        .sum()
        .reindex(days, fill_value=0)
        .reset_index()
    )
    daily["Day"] = daily["date"].apply(lambda d: pd.Timestamp(d).strftime("%a %m-%d"))

    long = daily.melt(id_vars="Day", value_vars=["Reservations", "Cancellations"],
                      var_name="Type", value_name="Count")
    long["Count"] = long["Count"].astype(int)

    total_res = int(daily["Reservations"].sum())
    total_can = int(daily["Cancellations"].sum())
    st.caption(
        f"{selected_station} · {days[0]} → {days[-1]} · "
        f"{total_res} reservations, {total_can} cancellations"
    )

    y_max = max(1, int(max(daily["Reservations"].max(), daily["Cancellations"].max())))

    spec = {
        "background": "white",
        "data": {"values": long.to_dict(orient="records")},
        "mark": {"type": "line", "point": {"filled": True, "size": 80}, "strokeWidth": 2.5, "tooltip": True},
        "encoding": {
            "x": {
                "field": "Day",
                "type": "ordinal",
                "axis": {"title": "Day", "labelColor": TEXT, "titleColor": TEXT, "labelAngle": 0},
                "scale": {"domain": daily["Day"].tolist()},
            },
            "y": {
                "field": "Count",
                "type": "quantitative",
                "axis": {"title": "Bookings", "labelColor": TEXT, "titleColor": TEXT},
                "scale": {"domain": [0, y_max + 1]},
            },
            "color": {
                "field": "Type",
                "type": "nominal",
                "scale": {"domain": ["Reservations", "Cancellations"], "range": [NS_BLUE, ORANGE]},
                "legend": {"title": None, "orient": "top", "labelColor": TEXT},
            },
            "tooltip": [
                {"field": "Day", "type": "ordinal"},
                {"field": "Type", "type": "nominal"},
                {"field": "Count", "type": "quantitative"},
            ],
        },
        "height": 240,
        "config": _vega_axis_config(),
    }
    st.vega_lite_chart(spec, use_container_width=True)
