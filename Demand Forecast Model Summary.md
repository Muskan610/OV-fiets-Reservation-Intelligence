# Demand Forecast Model Summary

The dashboard's live forecast comes from a LightGBM quantile regression model trained on a year of synthetic OV-fiets reservation data. This document explains the three-stage pipeline that produced it: simulating bike returns, computing hourly availability, and training the model.

---

## The 3-stage pipeline

```
ov_fiets_synthetic_v2.csv          ← source dataset (150,000 reservations, 2024)
        |
        v
[ generate_return_data.py ]        ← Stage 1: simulate bike return events
        |
        v
ov_fiets_with_returns.csv
        |
        +----------------------------+
        |                            |
        v                            v
[ live_bike_count.py ]      [ forecast_demand_lgb.py ]
        |                            ^
        v                            |
ov_fiets_hourly_bike_count.csv ------+
                                     |
                                     v
                       forecast_demand_lgb.pkl
                       forecast_demand_encoders.pkl
                       forecast_demand_predictions.csv
```

### How to retrain from scratch

Run in order from the project root using the `ds` conda environment:

```
"C:\Users\mbhat\anaconda3\envs\ds\python.exe" generate_return_data.py
"C:\Users\mbhat\anaconda3\envs\ds\python.exe" live_bike_count.py
"C:\Users\mbhat\anaconda3\envs\ds\python.exe" forecast_demand_lgb.py
```

Each script finishes in under a minute. Do not use base Anaconda Python — its dask version is too old for pandas 2.x.

---

## Stage 1 — `generate_return_data.py`

### What it does

Reads `ov_fiets_synthetic_v2.csv` and appends simulated bike-return information for every completed rental. Cancellations and no-shows get nothing — those bikes never left the rack.

### Columns it adds

| Column | Populated for | Meaning |
|---|---|---|
| `return_station` | Completed rentals only | Always equals `pickup_station` (no off-station returns) |
| `return_datetime` | Completed rentals only | `pickup_datetime + return_duration_minutes` |
| `return_duration_minutes` | Completed rentals only | Drawn from a 3-mode log-normal mixture; clipped to 5–4320 minutes |

Three columns from the source CSV are dropped because they were flagged as buggy: `bikes_available_pickup`, `bikes_available_reservation`, `reservation_availability_status`.

### How rental duration is simulated

Duration is drawn from a 3-mode log-normal mixture based on the rider's `user_segment`:

| Mode | Median | Purpose |
|---|---|---|
| Commuter | 60 min | Short station-to-office trips |
| Day out | 240 min | Leisure / errands |
| Multi-day | 1680 min (28 h) | Overnight and tourist stays |

Two multipliers are then applied: e-bikes get ×1.20, weekend rentals get ×1.30 (these compound). The final value is clipped to `[5, 4320]` minutes (5 min floor, 72 h cap — matching the real OV-fiets policy for fine-free returns).

The output is reproducible: `RNG_SEED = 42`.

---

## Stage 2 — `live_bike_count.py`

### What it does

Turns completed pickups and their simulated returns into an **hourly per-station bike-availability count**. Each station starts the year with `STARTING_CAPACITY = 20` bikes. Every pickup subtracts 1; every return adds 1. The running total is snapshotted at every hour boundary.

### Output

`ov_fiets_hourly_bike_count.csv` — 378,400 rows (43 stations × ~8,800 hours).

| Column | Meaning |
|---|---|
| `station` | Pickup station name |
| `hour` | Top-of-hour timestamp |
| `bikes_available` | Bikes on the rack at that moment (can go negative — that means stockout) |

### Key assumptions

- All returns happen at the pickup station (no off-station drops).
- Cancellations and no-shows do not affect the count.
- The count is a point-in-time snapshot at each hour, not an average.
- Negative values are allowed so stockout severity can be quantified.

---

## Stage 3 — `forecast_demand_lgb.py`

### What it does

For each `(station, hour)` cell, predicts the **total number of reservations in the next 4 hours** and converts it to a Green / Orange / Red pressure flag by comparing the forecast against the live `bikes_available` count.

### Pressure thresholds

```
ratio = predicted_4h_demand / bikes_available_now

ratio < 0.60              → Green  (comfortable headroom)
0.60 ≤ ratio < 0.90       → Orange (monitor, prepare to act)
ratio ≥ 0.90              → Red    (act now)
bikes_available_now == 0  → Red    (hard override)
```

### The 16 model features

#### Categorical (4)

| Feature | Values |
|---|---|
| `pickup_station` | 43 stations |
| `station_region` | 10 Dutch provinces |
| `weather_condition` | Clear / Cloudy / Light rain / Heavy rain / Windy / Snow |
| `season` | Winter / Spring / Summer / Autumn |

#### Numeric (12)

| Feature | Description |
|---|---|
| `hour_of_day` | 0–23 |
| `day_of_week` | 0 = Mon, 6 = Sun |
| `month` | 1–12 |
| `is_weekend` | 1 if Sat or Sun |
| `is_peak_hour` | 1 if hour ∈ {7, 8, 9, 17, 18, 19} |
| `temperature_celsius` | Per-hour mean temperature across stations |
| `bikes_available_now` | Live inventory at top of hour (from Stage 2) |
| `lag_1h` | Reservations at this station 1 hour ago |
| `lag_24h` | Reservations at this station 24 hours ago |
| `lag_168h` | Reservations at this station 168 hours ago (same hour last week) |
| `roll_24h_mean` | 24-hour rolling mean of demand (excluding current hour) |
| `roll_168h_mean` | 168-hour rolling mean of demand (excluding current hour) |

### Target variable

`demand_next_4h` = sum of reservations at that station in the 4 hours after the current one.

Why 4 hours? Per-hour demand is too sparse (mean ≈ 0.4 per station-hour) to produce meaningful pressure ratios against a 20-bike inventory. Aggregating to 4 hours pushes the mean to ≈ 1.6 and lets the ratios reach Orange and Red at realistic busy stations.

### Recall-first design

The model is configured to over-predict rather than under-predict, because a missed Red event (sending an operator home when a stockout is coming) is worse than a false alarm.

Two mechanisms enforce this:

1. **Quantile regression at α = 0.95** — the model targets the 95th percentile of demand, not the mean. It intentionally forecasts the high end of what could happen.
2. **5× sample weight for tail rows** — rows whose actual 4-hour demand is in the top 5% of training data are weighted 5× heavier, so the model pays extra attention to peak demand patterns.

The result is a deliberate positive bias of +2.27 on the test set. This is expected and desirable. Evaluate the model with quantile loss (pinball loss), not MAE.

### Train / test split

- Train: 2024-01-08 → 2024-10-31 (Jan-8 is the minimum date because `lag_168h` needs 168 hours of history before it)
- Test: 2024-11-01 → 2024-12-30

### Model performance on the test set

| Metric | Value |
|---|---|
| MAE | 2.32 |
| RMSE | 2.85 |
| Bias | +2.27 (intentional) |
| Correlation | 0.78 |
| Actual Red events | 89 |
| Predicted Red (true positives) | 65 — Red recall = **73%** |
| Predicted Orange (near-misses) | 23 — elevated-alert recall = **98.9%** |
| Missed Red (predicted Green) | 1 |
| Red precision | 24% (≈ 3 false alarms per true Red) |

Only 1 real Red event was silently downgraded to Green. The 3:1 false-alarm rate is an acceptable trade for the recall.

---

## Output artefacts

| File | What it is |
|---|---|
| `forecast_demand_lgb.pkl` | Trained `LGBMRegressor` (quantile α=0.95) |
| `forecast_demand_encoders.pkl` | Category orderings + feature column lists + CONFIG snapshot |
| `forecast_demand_predictions.csv` | Test-set predictions: station, hour, bikes_left, demand_actual, demand_pred, pressure flags |
| `forecast_demand_pressure_confusion.png` | 3×3 confusion matrix heatmap |
| `forecast_demand_per_station_metrics.csv` | Per-station MAE, Red rate, Red recall |
| `forecast_demand_lgb_importance.png` | Feature importance bar chart |

---

## Tuning reference

| Want to … | Change this |
|---|---|
| Change return-time distribution | `MODE_MEDIAN_MIN`, `MODE_SIGMA`, `SEGMENT_MODE_WEIGHTS` in `generate_return_data.py` |
| Change return-time cap | `DURATION_MAX_MINUTES` in `generate_return_data.py` |
| Change starting bike inventory | `STARTING_CAPACITY` in `live_bike_count.py` |
| Change forecast horizon | `FORECAST_WINDOW_HOURS` in `forecast_demand_lgb.py` |
| Push Red recall higher | Raise `QUANTILE_ALPHA` toward 0.99, or raise `TAIL_WEIGHT` |
| Trade Red precision back up | Lower `QUANTILE_ALPHA` toward 0.5 (mean prediction) |
| Change pressure thresholds | `PRESSURE_GREEN_MAX`, `PRESSURE_ORANGE_MAX` in `forecast_demand_lgb.py` |
