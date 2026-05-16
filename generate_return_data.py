"""Generate OV-fiets return dataset.

Reads the synthetic source CSV, drops three noisy columns, and appends
simulated bike-return data for every Completed rental:
  - return_station        (= pickup_station; off-station returns not modelled)
  - return_datetime       (pickup_datetime + duration)
  - return_duration_minutes  (3-mode log-normal mixture, segment-weighted; 72 h cap)

Mixture: Commuter (~1 h) / Day-out (~4 h) / Multi-day (~28 h, capped at 72 h).
The 72 h cap matches OV-fiets policy: rentals up to 72 h carry no fine.

Non-completed rows (No-show, Cancelled - *) get NaN for all three.
Output: ov_fiets_with_returns.csv
"""

# %% Imports + config
import numpy as np
import pandas as pd

SOURCE_PATH  = "ov_fiets_synthetic_v2.csv"
OUT_PATH     = "ov_fiets_with_returns.csv"

DROP_COLS = [
    "bikes_available_pickup",
    "bikes_available_reservation",
    "reservation_availability_status",
]

RNG_SEED = 42

# Three-mode log-normal mixture: (median_minutes, sigma_log)
MODE_NAMES = ("commuter", "day_out", "multi_day")
MODE_MEDIAN_MIN = (60, 240, 1680)   # 1 h, 4 h, 28 h
MODE_SIGMA      = (0.50, 0.50, 0.45)

# P(mode | user_segment) as (P_commuter, P_day_out, P_multi_day) tuples summing to 1.
# Calibrated so the overall multi-day share lands near ~10 % of completed returns.
SEGMENT_MODE_WEIGHTS = {
    "Daily commuter":  (0.65, 0.33, 0.02),
    "Student":         (0.40, 0.55, 0.05),
    "Occasional user": (0.30, 0.62, 0.08),
    "Weekend rider":   (0.20, 0.68, 0.12),
    "Tourist":         (0.10, 0.65, 0.25),
}
DEFAULT_MODE_WEIGHTS = (0.25, 0.65, 0.10)

EBIKE_DURATION_MULT   = 1.20
WEEKEND_DURATION_MULT = 1.30

DURATION_MIN_MINUTES = 5
DURATION_MAX_MINUTES = 72 * 60  # 4320 (72 h no-fine cap)


# %% Load
print(f"Reading {SOURCE_PATH} ...")
df = pd.read_csv(SOURCE_PATH)
print(f"  shape: {df.shape}")
print("  rental_status counts:")
print(df["rental_status"].value_counts(dropna=False).to_string())


# %% Drop the three columns
missing = [c for c in DROP_COLS if c not in df.columns]
if missing:
    raise KeyError(f"Expected columns to drop are missing: {missing}")
df = df.drop(columns=DROP_COLS)
print(f"\nDropped {len(DROP_COLS)} columns -> shape now {df.shape}")


# %% Build pickup_datetime (actual ride start)
pickup_dt = pd.to_datetime(
    df["reservation_date"].astype(str) + " " + df["pickup_time"].astype(str),
    errors="coerce",
)
if pickup_dt.isna().any():
    n_bad = int(pickup_dt.isna().sum())
    raise ValueError(f"{n_bad} rows could not be parsed into pickup_datetime")


# %% Mask Completed rows (returns simulated only for these)
completed_mask = df["rental_status"].eq("Completed").to_numpy()
n_completed = int(completed_mask.sum())
print(f"\nCompleted rows to simulate returns for: {n_completed:,}")


# %% Sample mode label per Completed row (0=commuter, 1=day_out, 2=multi_day)
rng = np.random.default_rng(RNG_SEED)

segments_completed = df.loc[completed_mask, "user_segment"].fillna("__missing__").to_numpy()
weights_per_row = np.array(
    [SEGMENT_MODE_WEIGHTS.get(s, DEFAULT_MODE_WEIGHTS) for s in segments_completed],
    dtype=float,
)  # shape (n_completed, 3)

# Vectorised Categorical: draw a uniform u, return the first index where cumsum >= u
cum = np.cumsum(weights_per_row, axis=1)
u = rng.random(n_completed)[:, None]
mode_idx = (u < cum).argmax(axis=1)  # int in {0, 1, 2}

mode_share = np.bincount(mode_idx, minlength=3) / n_completed
print("  mode share:")
for name, share in zip(MODE_NAMES, mode_share):
    print(f"    {name:<10s} {share*100:5.2f} %")


# %% Sample base duration from each log-normal, then pick by mode_idx
# np.random.lognormal uses (mean, sigma) in log-space; mean = ln(median)
draws_per_mode = np.stack([
    rng.lognormal(mean=np.log(MODE_MEDIAN_MIN[i]),
                  sigma=MODE_SIGMA[i],
                  size=n_completed)
    for i in range(3)
])  # shape (3, n_completed)
duration = np.choose(mode_idx, draws_per_mode)


# %% Apply multiplicative adjustments + clip
bike_type_completed = df.loc[completed_mask, "bike_type"].to_numpy()
is_weekend_completed = df.loc[completed_mask, "is_weekend"].astype(bool).to_numpy()

duration = np.where(bike_type_completed == "E-bike", duration * EBIKE_DURATION_MULT, duration)
duration = np.where(is_weekend_completed, duration * WEEKEND_DURATION_MULT, duration)
duration = np.clip(duration, DURATION_MIN_MINUTES, DURATION_MAX_MINUTES)


# %% Compute return_datetime
pickup_dt_completed = pickup_dt[completed_mask]
return_dt_completed = pickup_dt_completed + pd.to_timedelta(duration, unit="m")


# %% Assign return columns into full DataFrame (NaN for non-Completed)
df["return_station"]          = pd.Series(pd.NA, index=df.index, dtype="object")
df["return_datetime"]         = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
df["return_duration_minutes"] = pd.Series(np.nan, index=df.index, dtype="float64")

df.loc[completed_mask, "return_station"]          = df.loc[completed_mask, "pickup_station"].to_numpy()
df.loc[completed_mask, "return_datetime"]         = return_dt_completed.to_numpy()
df.loc[completed_mask, "return_duration_minutes"] = duration


# %% Write + report
df.to_csv(OUT_PATH, index=False)

print(f"\nWrote {OUT_PATH}")
print(f"  shape: {df.shape}")
print(f"  non-null return_datetime: {df['return_datetime'].notna().sum():,} "
      f"(expected {n_completed:,})")

dur = df["return_duration_minutes"].dropna()
print("\nDuration summary (minutes):")
print(f"  mean   = {dur.mean():.1f}")
print(f"  median = {dur.median():.1f}")
print(f"  p75    = {dur.quantile(0.75):.1f}")
print(f"  p95    = {dur.quantile(0.95):.1f}")
print(f"  max    = {dur.max():.1f}")
print(f"  min    = {dur.min():.1f}")

over_24h = (dur > 24 * 60).mean() * 100
over_48h = (dur > 48 * 60).mean() * 100
at_cap   = (dur >= 72 * 60).mean() * 100
print(f"\nMulti-day calibration:")
print(f"  > 24 h: {over_24h:5.2f} %  (target ~10 %)")
print(f"  > 48 h: {over_48h:5.2f} %")
print(f"  = 72 h cap: {at_cap:5.2f} %")

print("\nMedian duration by user_segment (Completed only):")
print(
    df[df["rental_status"].eq("Completed")]
      .groupby("user_segment")["return_duration_minutes"]
      .median()
      .round(1)
      .to_string()
)

print("\nSample (Completed rows):")
print(
    df[df["rental_status"].eq("Completed")]
      .head(5)[["rental_id", "pickup_station", "return_station",
                "reservation_date", "pickup_time",
                "return_datetime", "return_duration_minutes"]]
      .to_string(index=False)
)
