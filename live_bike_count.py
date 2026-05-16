"""Live per-station bike count, sampled hourly.

Builds an event log from the returns dataset:
  - every Completed rental contributes a pickup event  (delta = -1 at pickup_datetime)
  - and a return event                                 (delta = +1 at return_datetime)

Every station starts the year with the same uniform inventory (STARTING_CAPACITY).
The live count is offset cumulative delta + STARTING_CAPACITY and is allowed to
go negative — a negative value means demand exceeded supply during that hour
(= stockout severity). The trajectory is sampled at every hour boundary and
written as a long-format CSV.

Output: ov_fiets_hourly_bike_count.csv
"""

# %% Imports + config
import numpy as np
import pandas as pd

SOURCE_PATH  = "ov_fiets_with_returns.csv"
OUT_PATH     = "ov_fiets_hourly_bike_count.csv"

STARTING_CAPACITY = 20  # uniform starting inventory at every station


# %% Load + restrict to Completed rentals
print(f"Reading {SOURCE_PATH} ...")
df = pd.read_csv(
    SOURCE_PATH,
    parse_dates=["return_datetime"],
)
print(f"  shape: {df.shape}")

completed = df[df["rental_status"].eq("Completed")].copy()
print(f"  Completed rentals: {len(completed):,}")

# pickup_datetime isn't stored in the CSV; reconstruct from date + pickup_time
completed["pickup_datetime"] = pd.to_datetime(
    completed["reservation_date"].astype(str) + " " + completed["pickup_time"].astype(str)
)


# %% Build event log (one row per pickup, one per return)
pickups = pd.DataFrame({
    "station": completed["pickup_station"].to_numpy(),
    "ts":      completed["pickup_datetime"].to_numpy(),
    "delta":   -1,
})
returns_df = pd.DataFrame({
    "station": completed["return_station"].to_numpy(),
    "ts":      completed["return_datetime"].to_numpy(),
    "delta":   1,
})
events = pd.concat([pickups, returns_df], ignore_index=True)
events = events.sort_values(["station", "ts"]).reset_index(drop=True)
print(f"\nEvents: {len(events):,}  (pickups={len(pickups):,}, returns={len(returns_df):,})")


# %% Cumulative delta + uniform starting capacity
events["cum_delta"] = events.groupby("station")["delta"].cumsum()
events["bikes_available"] = STARTING_CAPACITY + events["cum_delta"]
print(f"\nUniform starting capacity per station: {STARTING_CAPACITY}")


# %% Hourly grid across the full timeline
hour_min = events["ts"].min().floor("h")
hour_max = events["ts"].max().ceil("h")
hour_grid = pd.date_range(hour_min, hour_max, freq="h")
print(f"\nHourly grid: {len(hour_grid):,} hours from {hour_min} to {hour_max}")

stations = sorted(events["station"].unique())
print(f"Stations: {len(stations)}")


# %% Per-station merge_asof onto the grid (state at top-of-hour)
parts = []
grid_df = pd.DataFrame({"hour": hour_grid})

for station in stations:
    sub = (events.loc[events["station"] == station, ["ts", "bikes_available"]]
                  .sort_values("ts")
                  .reset_index(drop=True))
    merged = pd.merge_asof(
        grid_df,
        sub.rename(columns={"ts": "hour"}),
        on="hour",
        direction="backward",
        allow_exact_matches=True,
    )
    # Hours before the first event for this station: pre-fill with starting capacity
    merged["bikes_available"] = (merged["bikes_available"]
                                 .fillna(STARTING_CAPACITY)
                                 .astype(int))
    merged["station"] = station
    parts.append(merged)

hourly = (pd.concat(parts, ignore_index=True)
            [["station", "hour", "bikes_available"]]
            .sort_values(["station", "hour"])
            .reset_index(drop=True))


# %% Write + report
hourly.to_csv(OUT_PATH, index=False)
print(f"\nWrote {OUT_PATH}")
print(f"  shape: {hourly.shape}  "
      f"(expected ~{len(stations)} stations x {len(hour_grid)} hours = "
      f"{len(stations) * len(hour_grid):,})")

print("\nbikes_available summary across all (station, hour) cells:")
print(hourly["bikes_available"].describe().round(2).to_string())

agg = (hourly.groupby("station")["bikes_available"]
              .agg(["min", "max", "mean"])
              .round(2))
agg["stockout_hours"] = (
    hourly.assign(neg=lambda d: d["bikes_available"] < 0)
          .groupby("station")["neg"].sum().astype(int)
)
agg["stockout_pct"] = (agg["stockout_hours"] / len(hour_grid) * 100).round(2)

print("\nPer-station min / max / mean (sorted by min ascending — worst first):")
print(agg.sort_values("min").to_string())

n_neg = int((hourly["bikes_available"] < 0).sum())
print(f"\nTotal station-hours with negative count (stockout): "
      f"{n_neg:,} / {len(hourly):,}  ({n_neg / len(hourly) * 100:.2f}%)")
print(f"Stations that ever go negative: "
      f"{int((agg['min'] < 0).sum())} / {len(stations)}")

print("\nSample (Amsterdam Centraal, first 10 hours):")
sample_station = "Amsterdam Centraal" if "Amsterdam Centraal" in stations else stations[0]
print(
    hourly[hourly["station"] == sample_station]
      .head(10)[["station", "hour", "bikes_available"]]
      .to_string(index=False)
)
