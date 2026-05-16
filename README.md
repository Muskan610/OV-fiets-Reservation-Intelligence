# OV-fiets Reservation Intelligence 
Two ML models built for NS's OV-fiets shared-bike reservation system: a per-user defaulter risk classifier (Block C) and a per-station demand forecast that drives a live employee operations dashboard (Block D).

---

## Project Overview

OV-fiets is NS's shared-bike service, available at train stations across the Netherlands. Customers can reserve a bike in advance, walk up and rent on the spot, or both — depending on station availability. This project explores two problems that arise from running a reservation system at scale:

1. **Who will not show up?** Some users cancel late or never collect their reserved bike. Predicting these defaulters in advance lets NS nudge them with a notification before they waste a reservation slot.
2. **When will demand spike?** Station staff need to know whether the current bike inventory can absorb expected demand in the next four hours, so they can re-allocate bikes between pickup and reservation pools before a stockout happens.

Because real OV-fiets reservation data is not publicly available (the feature was still being piloted during this project), a realistic synthetic dataset was created from publicly available ride-hailing data and used for both tracks.

---

## Defaulter Prediction

### What is a defaulter?

A defaulter is a user who either **no-shows** (does not collect a reserved bike) or **cancels late** (cancels within the window that prevents NS from re-offering the slot). Both outcomes waste capacity. NS already charges a fee, but the goal here is to intervene *before* the event — not after.

### Model

A supervised binary classification model trained on synthetic reservation history. Key input features include:

- Rolling user behaviour: `past_no_shows`, `past_cancellations`, `booking_frequency_monthly`, `user_tenure_days`
- Booking context: `user_segment` (commuter / tourist / student / weekend / occasional), `weather_condition`, `season`, `is_peak_hour`, `is_weekend`
- Reservation details: `station_region`, `bike_type`, `advance_booking_hours`

The model predicts, at reservation time, whether the booking is likely to result in a no-show or late cancellation.

### How it is used

This model runs **per user**, **per reservation**, inside the NS Travel Planner app — not in the employee dashboard. When a user with a high defaulter risk completes a booking, the system silently flags the reservation and triggers a push notification reminding the user of the cancellation policy and pickup deadline. Operations staff do not need to see individual user scores; the intervention happens entirely on the user-facing side.

### Artifacts

| File | Description |
|---|---|
| `Muskan D01 Supervised ML_ov_fiets_defaulter.ipynb` | Full model development notebook (EDA → feature engineering → training → evaluation) |
| `ovfiets_defaulter_prediction_model.pkl` | Trained classifier |
| `ovfiets_defaulter_scaler.pkl` | Feature scaler |
| `ovfiets_defaulter_feature_columns.pkl` | Ordered feature column list |
| `ovfiets_defaulter_preprocessing_metadata.pkl` | Preprocessing config |

**Figma Prototype** <br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/0b3f296c-3ad7-4810-ab51-ef2c74448734" width="700">
</p>

---

## Demand Forecast & Operations Dashboard

### Data pipeline

The demand model is built on top of a three-stage pipeline that turns raw reservation records into model-ready features.

```
ov_fiets_synthetic_v2.csv          ← 150,000 synthetic reservations, full year 2024
        |
        v
[ generate_return_data.py ]        Stage 1: simulate bike return events
        |                          (log-normal duration by user segment, e-bike ×1.20, weekend ×1.30)
        v
ov_fiets_with_returns.csv
        |
        +---------------------------+
        |                           |
        v                           v
[ live_bike_count.py ]     [ forecast_demand_lgb.py ]
        |                           ^
        v                           |
ov_fiets_hourly_bike_count.csv -----+
                                    |
                                    v
                      forecast_demand_lgb.pkl
                      forecast_demand_encoders.pkl
```

**Stage 1** appends simulated return events to each completed rental (cancellations and no-shows get nothing — those bikes never left the rack). Rental duration is drawn from a 3-mode log-normal mixture calibrated by user segment (commuter ≈ 60 min, day-out ≈ 240 min, multi-day ≈ 28 h).

**Stage 2** aggregates pickups and returns into an hourly per-station bike-availability count. Each of 43 stations starts the year with 20 bikes; every pickup subtracts 1, every return adds 1, snapshotted at each hour boundary.

**Stage 3** trains the forecast model. For each `(station, hour)` cell, the target is the total number of reservations in the next 4 hours. A 4-hour window is used because per-hour demand is too sparse (mean ≈ 0.4 per station-hour) to produce meaningful pressure ratios against a 20-bike pool.

### Model design

The model is an `LGBMRegressor` with **quantile objective at α = 0.95**, meaning it forecasts the high end of what is likely to happen rather than the average. An additional **5× sample weight** is applied to rows in the top 5% of demand, so the model pays extra attention to peak patterns.

This recall-first design reflects the asymmetric cost: a missed Red event (sending staff home when a stockout is coming) is worse than a false alarm that prompts a precautionary re-allocation.

**16 input features:** `pickup_station`, `station_region`, `weather_condition`, `season` (categorical); `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_peak_hour`, `temperature_celsius`, `bikes_available_now`, `lag_1h`, `lag_24h`, `lag_168h`, `roll_24h_mean`, `roll_168h_mean` (numeric).

**Test-set performance (Nov–Dec 2024):**

| Metric | Value |
|---|---|
| MAE | 2.32 |
| Bias | +2.27 (intentional) |
| Red recall | 73% (65 / 89 true Red events caught) |
| Elevated-alert recall | 98.9% (23 near-misses flagged as Orange) |
| Silent misses (Green when truly Red) | 1 |
| Red precision | 24% (~3 false alarms per true Red) |

### Pressure thresholds

```
ratio = forecast_4h_demand / bikes_available_now

ratio < 0.60         → Green  (low pressure — comfortable headroom)
0.60 ≤ ratio < 0.90  → Orange (medium pressure — monitor, prepare to act)
ratio ≥ 0.90         → Red    (high pressure — act now)
bikes_available == 0 → Red    (hard override)
```

### Operations Dashboard

The Streamlit dashboard (`Muskan D01 ov_emp_dashboard.py`) gives station operations staff a live view of inventory and a 4-hour demand outlook. The model runs inference **on every render** — there is no precomputed predictions file in the runtime path.

**What operators see:**
- Three inventory cards: bikes available for pickup, bikes available for reservation (live remaining), and the 4-hour demand forecast
- A colour-coded pressure banner (low / medium / high) with the severity ratio
- A simulated pickup chart (past hours only, up to the current simulated hour)
- A booking activity chart showing today's reservation and cancellation history alongside the 4-hour forecast
- A re-allocation widget with preset splits (80:20 / 70:30 / 60:40, pickup:reservation)

**Re-allocation mechanic:** The fleet is fixed at 100 bikes. The default split is 80 pickup / 20 reservation. Operators can shift bikes from pickup to reservation (but not the reverse — that would strand already-booked reservations). The 60:40 split is the operational floor. The pickup chart and inventory cards update coherently when a new split is previewed or committed.

**Sidebar controls:** station (43 options), simulation date (2024-01-08 to 2024-12-30), hour of day, pool-remaining slider (0–20), weather condition, temperature.
<br><br>
**Screenshots for demonstration**
<img width="1898" height="856" alt="image" src="https://github.com/user-attachments/assets/0035aaa1-c527-463e-afd0-c23aacc7674e" />
<img width="1907" height="853" alt="image" src="https://github.com/user-attachments/assets/20a81a63-7e91-40d4-aecc-29ba176c4e48" />


---

## Synthetic Dataset

Both models were trained on `ov_fiets_synthetic_v2.csv`, generated in `Block C Python scripts/Muskan D01 Synthetic data creation_ov_fiets_transformation_v2.ipynb`.

**Source:** Kaggle NCR (India) Uber ride-hailing dataset — 150,000 bookings remapped to 43 NS stations across 10 Dutch provinces.

**Key transformations applied:**
- User segment assignment (daily commuter, weekend rider, tourist, student, occasional)
- Correlation-aware rental outcome assignment (no-shows correlate with segment, prior history, weather)
- Rolling user history columns: `past_no_shows`, `past_cancellations`, `booking_frequency_monthly`, `user_tenure_days`
- Dutch weather patterns, seasonal temperature, NS station geography and region tagging
- Pricing model: €2 reservation fee, €2 late cancellation, €5 no-show

---

## Repository Structure

```
├── Muskan D01 ov_emp_dashboard.py         Entry point — Streamlit dashboard
├── inventory_widgets.py                   All widgets, ML inference helpers, CSS
│
├── generate_return_data.py                Pipeline Stage 1
├── live_bike_count.py                     Pipeline Stage 2
├── forecast_demand_lgb.py                 Pipeline Stage 3 (model training)
├── Muskan D01 forecast_demand_lgb.ipynb   Interactive version of Stage 3
│
├── forecast_demand_lgb.pkl                Trained demand model
├── forecast_demand_encoders.pkl           Encoders + feature schema
├── ov_fiets_with_returns.csv              Intermediate: reservations + returns
├── ov_fiets_hourly_bike_count.csv         Intermediate: hourly bike counts
├── ov_fiets_synthetic_v2.csv              Synthetic reservation dataset
│
├── Muskan D01 Supervised ML_ov_fiets_defaulter.ipynb   Block C model notebook
├── ovfiets_defaulter_prediction_model.pkl
├── ovfiets_defaulter_scaler.pkl
├── ovfiets_defaulter_feature_columns.pkl
├── ovfiets_defaulter_preprocessing_metadata.pkl
│
└── Block C Python scripts/
    ├── Muskan D01 Synthetic data creation_ov_fiets_transformation_v2.ipynb
    └── ncr_ride_bookings.csv              Original Kaggle source data
```

---

## How to Run

Install the required packages listed in the Tech Stack section, then activate your Python environment.

**Launch the dashboard:**
```
streamlit run "Muskan D01 ov_emp_dashboard.py"
```
Then open `http://localhost:8501`.

**Retrain the demand model from scratch:**
```
python generate_return_data.py
python live_bike_count.py
python forecast_demand_lgb.py
```
Each script finishes in under a minute. Run them in order from the project root.

**Defaulter model:** Open and run `Muskan D01 Supervised ML_ov_fiets_defaulter.ipynb` in Jupyter.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| ML | LightGBM 4.6, scikit-learn, joblib |
| Data | pandas 2.3, NumPy |
| Dashboard | Streamlit 1.50, Vega-Lite |
| Environment | Python 3.x, conda env `ds` (Windows) |
