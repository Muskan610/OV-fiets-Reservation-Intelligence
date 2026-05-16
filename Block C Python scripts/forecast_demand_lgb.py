"""Multi-hour demand forecast + live pressure flag (LightGBM).

For every (station, hour) cell on the test window, predict the cumulative
reservations during [hour+1, hour+1+FORECAST_WINDOW_HOURS) and turn that into
a Green / Orange / Red pressure flag against the live `bikes_available`
snapshot at the top of `hour`.

Default window: 4 hours -- captures a full operational shift's worth of
upcoming demand, so the ratio against current inventory is operationally
meaningful instead of dominated by noise.

Pressure rule:
    ratio = predicted_window_demand / bikes_left
    Green  : ratio < 0.60
    Orange : 0.60 <= ratio < 0.90
    Red    : ratio >= 0.90  (also Red if bikes_left == 0)

Artifacts (all under new_code/):
    forecast_demand_lgb.pkl, forecast_demand_encoders.pkl,
    forecast_demand_predictions.csv, forecast_demand_pressure_confusion.png,
    forecast_demand_per_station_metrics.csv, forecast_demand_lgb_importance.png
"""

# %% Imports + config
import itertools
import pickle
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

#PROJECT_ROOT    = Path("/Users/muskan/Desktop/CODE/MuskanDS/OV fiets")
#NEW_CODE        = PROJECT_ROOT / "new_code"
RETURNS_PATH    = NEW_CODE / "ov_fiets_with_returns.csv"
BIKE_COUNT_PATH = NEW_CODE / "ov_fiets_hourly_bike_count.csv"

CONFIG = {
    "TRAIN_END":             pd.Timestamp("2024-10-31 23:00:00"),
    "FORECAST_WINDOW_HOURS": 4,        # predict cumulative demand over the next N hours
    "PRESSURE_GREEN_MAX":    0.60,
    "PRESSURE_ORANGE_MAX":   0.90,
    "RANDOM_SEED":           42,
    # Recall-first: predict an upper conditional quantile so the pressure flag
    # over-calls peaks. 0.95 -> the model intentionally biases ~5% of the conditional
    # distribution into the prediction; tail events are not under-called.
    "QUANTILE_ALPHA":        0.95,
    # Extra weight on rows whose actual N-hour demand sits in the upper tail of the
    # train target -- amplifies the Red signal in the loss.
    "TAIL_QUANTILE":         0.95,
    "TAIL_WEIGHT":           5.0,
}

OUT_MODEL       = NEW_CODE / "forecast_demand_lgb.pkl"
OUT_ENC         = NEW_CODE / "forecast_demand_encoders.pkl"
OUT_PRED        = NEW_CODE / "forecast_demand_predictions.csv"
OUT_CONF        = NEW_CODE / "forecast_demand_pressure_confusion.png"
OUT_PER_STATION = NEW_CODE / "forecast_demand_per_station_metrics.csv"
OUT_FI          = NEW_CODE / "forecast_demand_lgb_importance.png"

CAT_COLS = ["pickup_station", "station_region", "weather_condition", "season"]


# %% Load data
print("Loading inputs ...")
raw = pd.read_csv(RETURNS_PATH)
raw["reservation_dt"] = pd.to_datetime(
    raw["reservation_date"].astype(str) + " " + raw["reservation_time"].astype(str),
    errors="coerce",
)
raw = raw.dropna(subset=["reservation_dt"]).copy()
raw["hour"] = raw["reservation_dt"].dt.floor("h")

bike_count = pd.read_csv(BIKE_COUNT_PATH, parse_dates=["hour"])
bike_count = bike_count.rename(columns={
    "station": "pickup_station",
    "bikes_available": "bikes_available_now",
})
print(f"  reservations: {len(raw):,}")
print(f"  bike-count rows: {len(bike_count):,}")


# %% Build (station x hour) demand grid
print("\nBuilding demand grid ...")
demand = (raw.groupby(["pickup_station", "hour"]).size()
            .rename("demand_count").reset_index())

stations  = sorted(raw["pickup_station"].unique())
hour_min  = raw["hour"].min()
hour_max  = raw["hour"].max()
all_hours = pd.date_range(hour_min, hour_max, freq="h")

grid = (pd.MultiIndex.from_product([stations, all_hours],
                                   names=["pickup_station", "hour"])
          .to_frame(index=False))
grid = grid.merge(demand, on=["pickup_station", "hour"], how="left")
grid["demand_count"] = grid["demand_count"].fillna(0).astype("int32")
print(f"  grid: {grid.shape[0]:,} rows ({len(stations)} stations x {len(all_hours)} hours)")

# Per-hour weather (applied to all stations -- weather is regional anyway).
weather_per_hour = (raw.groupby("hour")
                       .agg(weather_condition=("weather_condition", "first"),
                            temperature_celsius=("temperature_celsius", "mean"))
                       .reset_index())
grid = grid.merge(weather_per_hour, on="hour", how="left")
grid["weather_condition"]  = grid["weather_condition"].fillna("Clear")
grid["temperature_celsius"] = grid["temperature_celsius"].fillna(grid["temperature_celsius"].mean())

# Station -> region lookup.
station_region = raw.drop_duplicates("pickup_station")[["pickup_station", "station_region"]]
grid = grid.merge(station_region, on="pickup_station", how="left")


# %% Calendar features
ts = grid["hour"]
grid["hour_of_day"]  = ts.dt.hour.astype("int16")
grid["day_of_week"]  = ts.dt.dayofweek.astype("int16")
grid["month"]        = ts.dt.month.astype("int16")
grid["is_weekend"]   = grid["day_of_week"].isin([5, 6]).astype("int8")
grid["is_peak_hour"] = grid["hour_of_day"].isin([7, 8, 9, 17, 18, 19]).astype("int8")
grid["season"] = grid["month"].map({
    12: "Winter", 1: "Winter",  2: "Winter",
     3: "Spring", 4: "Spring",  5: "Spring",
     6: "Summer", 7: "Summer",  8: "Summer",
     9: "Autumn", 10: "Autumn", 11: "Autumn",
})


# %% Lag + rolling features (leakage-safe)
grid = grid.sort_values(["pickup_station", "hour"]).reset_index(drop=True)
gd = grid.groupby("pickup_station", sort=False)["demand_count"]
grid["lag_1h"]   = gd.shift(1)
grid["lag_24h"]  = gd.shift(24)
grid["lag_168h"] = gd.shift(168)
grid["roll_24h_mean"]  = grid.groupby("pickup_station")["demand_count"].transform(
    lambda s: s.shift(1).rolling(24,  min_periods=1).mean())
grid["roll_168h_mean"] = grid.groupby("pickup_station")["demand_count"].transform(
    lambda s: s.shift(1).rolling(168, min_periods=1).mean())


# %% Join bikes_available_now
grid = grid.merge(
    bike_count[["pickup_station", "hour", "bikes_available_now"]],
    on=["pickup_station", "hour"], how="left",
)


# %% Define forward-looking target: cumulative demand over the next N hours
W = CONFIG["FORECAST_WINDOW_HOURS"]
TARGET = f"demand_next_{W}h"

def fwd_window_sum(s, w):
    """Sum of s[t+1] + s[t+2] + ... + s[t+w]; last w rows become NaN."""
    out = sum(s.shift(-i) for i in range(1, w + 1))
    return out

grid[TARGET] = grid.groupby("pickup_station", sort=False)["demand_count"].transform(
    lambda s: fwd_window_sum(s, W)
)
print(f"\nForecast target: cumulative demand over next {W} h "
      f"(per station, mean = {grid[TARGET].mean():.3f})")

before = len(grid)
grid = grid.dropna(subset=[
    "lag_1h", "lag_24h", "lag_168h", "roll_24h_mean", "roll_168h_mean",
    TARGET, "bikes_available_now",
]).reset_index(drop=True)
print(f"  after dropping lag/target NaNs: {before:,} -> {len(grid):,}")


# %% Train/test split + categorical encoding
NUM_COLS = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "temperature_celsius", "bikes_available_now",
    "lag_1h", "lag_24h", "lag_168h", "roll_24h_mean", "roll_168h_mean",
]
FEATURE_COLS = CAT_COLS + NUM_COLS

train_df = grid[grid["hour"] <= CONFIG["TRAIN_END"]].copy()
test_df  = grid[grid["hour"] >  CONFIG["TRAIN_END"]].copy()

# Fit categorical orderings on train, apply to test for consistency.
for col in CAT_COLS:
    train_df[col] = train_df[col].astype("category")
    test_df[col]  = pd.Categorical(test_df[col], categories=train_df[col].cat.categories)

print(f"\nTrain: {len(train_df):,} rows, {train_df['hour'].min()} -> {train_df['hour'].max()}")
print(f"Test:  {len(test_df):,} rows, {test_df['hour'].min()} -> {test_df['hour'].max()}")
print(f"Target mean: train={train_df[TARGET].mean():.3f}, test={test_df[TARGET].mean():.3f}")


# %% LightGBM hyperparameter sweep (recall-first quantile regression)
X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET].astype(float)
X_test,  y_test  = test_df[FEATURE_COLS],  test_df[TARGET].astype(float)

# Sample weights -- boost rows in the upper tail of train demand. These are
# the rows that drive Red events and we want the loss to weigh them heavily.
tail_cutoff = float(np.quantile(y_train, CONFIG["TAIL_QUANTILE"]))
sample_weights_train = np.where(y_train >= tail_cutoff, CONFIG["TAIL_WEIGHT"], 1.0)
print(f"\nSample weighting: rows with demand >= {tail_cutoff:.1f} (top "
      f"{(1 - CONFIG['TAIL_QUANTILE']) * 100:.0f}%) get weight {CONFIG['TAIL_WEIGHT']}x; "
      f"{int((sample_weights_train > 1).sum()):,} of {len(y_train):,} rows tagged.")

base_params = {
    "objective":  "quantile",
    "alpha":      CONFIG["QUANTILE_ALPHA"],   # predict upper quantile (recall lever)
    "metric":     "quantile",
    "verbosity":  -1,
    "n_estimators": 600,
    "random_state": CONFIG["RANDOM_SEED"],
}
print(f"Objective: quantile (alpha={CONFIG['QUANTILE_ALPHA']}) -- expect positive bias by design.")

def quantile_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))

sweep = list(itertools.product([31, 63], [0.05, 0.10], [20, 50]))
print(f"\nSweeping {len(sweep)} combos (scoring by quantile loss; MAE shown for context):")
results = []
for nl, lr, mcs in sweep:
    params = dict(base_params, num_leaves=nl, learning_rate=lr, min_child_samples=mcs)
    m = lgb.LGBMRegressor(**params)
    m.fit(X_train, y_train,
          sample_weight=sample_weights_train,
          categorical_feature=CAT_COLS,
          eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    p   = m.predict(X_test)
    mae = mean_absolute_error(y_test, p)
    qll = quantile_loss(y_test.to_numpy(), p, CONFIG["QUANTILE_ALPHA"])
    results.append({"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs,
                    "mae": mae, "quantile_loss": qll})
    print(f"  nl={nl:2d} lr={lr:.2f} mcs={mcs:2d} -> qloss={qll:.4f}  MAE={mae:.4f}")

results_df = pd.DataFrame(results).sort_values("quantile_loss").reset_index(drop=True)
print(f"\nBest combo: {results_df.iloc[0].to_dict()}")


# %% Refit best on full train + final metrics
best = results_df.iloc[0]
final_params = dict(base_params,
                    num_leaves=int(best["num_leaves"]),
                    learning_rate=float(best["learning_rate"]),
                    min_child_samples=int(best["min_child_samples"]))
lgb_model = lgb.LGBMRegressor(**final_params)
lgb_model.fit(X_train, y_train,
              sample_weight=sample_weights_train,
              categorical_feature=CAT_COLS,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(30, verbose=False)])

y_pred = lgb_model.predict(X_test)
mae   = mean_absolute_error(y_test, y_pred)
rmse  = float(np.sqrt(mean_squared_error(y_test, y_pred)))
bias  = float((y_pred - y_test).mean())
corr  = float(np.corrcoef(y_test, y_pred)[0, 1])
print(f"\nTest metrics: MAE={mae:.4f}  RMSE={rmse:.4f}  bias={bias:+.4f}  corr={corr:.4f}")


# %% Feature importance plot
fi = pd.Series(lgb_model.feature_importances_, index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
fi.plot(kind="barh", ax=ax)
ax.set_title(f"LightGBM feature importance (next-{CONFIG['FORECAST_WINDOW_HOURS']}h demand)")
fig.tight_layout()
fig.savefig(OUT_FI, dpi=120)
plt.close(fig)


# %% Pressure flag derivation
def bucket_pressure(ratio_arr, bikes_left_arr,
                    green_max=CONFIG["PRESSURE_GREEN_MAX"],
                    orange_max=CONFIG["PRESSURE_ORANGE_MAX"]):
    return np.where(
        bikes_left_arr <= 0, "Red",
        np.where(ratio_arr >= orange_max, "Red",
        np.where(ratio_arr >= green_max,  "Orange", "Green")),
    )

bikes_left = test_df["bikes_available_now"].to_numpy()
safe_left  = np.maximum(bikes_left, 1)
ratio_actual = np.where(bikes_left > 0, y_test.to_numpy() / safe_left, np.inf)
ratio_pred   = np.where(bikes_left > 0, y_pred             / safe_left, np.inf)
pressure_actual = bucket_pressure(ratio_actual, bikes_left)
pressure_pred   = bucket_pressure(ratio_pred,   bikes_left)

LABELS = ["Green", "Orange", "Red"]
cm = confusion_matrix(pressure_actual, pressure_pred, labels=LABELS)
print("\nPressure confusion matrix (rows = actual, cols = predicted):")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

print("\nClassification report:")
print(classification_report(pressure_actual, pressure_pred, labels=LABELS, zero_division=0))


# %% Confusion matrix heatmap PNG
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(LABELS)
ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Pressure flag confusion (Green / Orange / Red)")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(OUT_CONF, dpi=120)
plt.close(fig)


# %% Per-station metrics
test_df = test_df.copy()
test_df["demand_pred"]     = y_pred
test_df["pressure_actual"] = pressure_actual
test_df["pressure_pred"]   = pressure_pred

station_key = test_df["pickup_station"].astype(str)
per_station = (test_df.assign(
        err=lambda d: (d["demand_pred"] - d[TARGET]).abs(),
        is_red_actual=lambda d: (d["pressure_actual"] == "Red").astype(int),
        is_red_pred=lambda d: (d["pressure_pred"] == "Red").astype(int),
        is_red_correct=lambda d: ((d["pressure_actual"] == "Red") & (d["pressure_pred"] == "Red")).astype(int),
    )
    .groupby(station_key)
    .agg(n=("err", "size"),
         mae=("err", "mean"),
         red_rate_actual=("is_red_actual", "mean"),
         red_rate_pred=("is_red_pred", "mean"),
         red_recall_num=("is_red_correct", "sum"),
         red_actual_num=("is_red_actual", "sum"))
    .assign(red_recall=lambda d: np.where(d["red_actual_num"] > 0,
                                          d["red_recall_num"] / d["red_actual_num"],
                                          np.nan))
    .drop(columns=["red_recall_num", "red_actual_num"])
    .sort_values("red_rate_actual", ascending=False))
per_station.to_csv(OUT_PER_STATION)
print("\nTop 10 most-pressured stations (by actual Red rate):")
print(per_station.head(10).round(4).to_string())


# %% Save model + encoders + predictions
with open(OUT_MODEL, "wb") as f:
    pickle.dump(lgb_model, f)

encoders = {col: list(train_df[col].cat.categories) for col in CAT_COLS}
encoders["FEATURE_COLS"] = FEATURE_COLS
encoders["CAT_COLS"]     = CAT_COLS
encoders["NUM_COLS"]     = NUM_COLS
encoders["CONFIG"]       = CONFIG
with open(OUT_ENC, "wb") as f:
    pickle.dump(encoders, f)

(test_df[["pickup_station", "hour", "bikes_available_now",
          TARGET, "demand_pred", "pressure_actual", "pressure_pred"]]
    .rename(columns={"pickup_station": "station",
                     "bikes_available_now": "bikes_left",
                     TARGET: "demand_actual"})
    .to_csv(OUT_PRED, index=False))
print(f"\nSaved artifacts under {NEW_CODE}/")


# %% Inference function (with demos)
def predict_window_demand(station, timestamp, bikes_left, weather, temperature,
                          lag_lookup, model=lgb_model, enc=encoders, cfg=CONFIG):
    """Predict cumulative reservations over the next FORECAST_WINDOW_HOURS and assign pressure."""
    h = pd.Timestamp(timestamp).floor("h")
    row = lag_lookup[(lag_lookup["pickup_station"] == station) & (lag_lookup["hour"] == h)]
    if len(row) == 0:
        raise KeyError(f"No lag context for {station} @ {h}")
    feat = {
        "pickup_station":       station,
        "station_region":       row["station_region"].iat[0],
        "weather_condition":    weather,
        "season":               row["season"].iat[0],
        "hour_of_day":          h.hour,
        "day_of_week":          h.dayofweek,
        "month":                h.month,
        "is_weekend":           int(h.dayofweek in (5, 6)),
        "is_peak_hour":         int(h.hour in (7, 8, 9, 17, 18, 19)),
        "temperature_celsius":  temperature,
        "bikes_available_now":  bikes_left,
        "lag_1h":               row["lag_1h"].iat[0],
        "lag_24h":              row["lag_24h"].iat[0],
        "lag_168h":             row["lag_168h"].iat[0],
        "roll_24h_mean":        row["roll_24h_mean"].iat[0],
        "roll_168h_mean":       row["roll_168h_mean"].iat[0],
    }
    X = pd.DataFrame([feat])
    for c in enc["CAT_COLS"]:
        X[c] = pd.Categorical(X[c], categories=enc[c])
    pred = float(model.predict(X[enc["FEATURE_COLS"]])[0])
    if bikes_left <= 0:
        flag, ratio = "Red", float("inf")
    else:
        ratio = pred / bikes_left
        flag = ("Red"    if ratio >= cfg["PRESSURE_ORANGE_MAX"]
                else "Orange" if ratio >= cfg["PRESSURE_GREEN_MAX"]
                else "Green")
    return {"demand_pred": pred, "ratio": ratio, "pressure": flag}

print(f"\nInference demos (5 random test rows; target = next {W} h cumulative demand):")
demo = test_df.sample(min(5, len(test_df)), random_state=42)
for _, r in demo.iterrows():
    out = predict_window_demand(
        station=str(r["pickup_station"]),
        timestamp=r["hour"],
        bikes_left=int(r["bikes_available_now"]),
        weather=str(r["weather_condition"]),
        temperature=float(r["temperature_celsius"]),
        lag_lookup=test_df,
    )
    print(f"  {str(r['pickup_station']):<22s} @ {r['hour']}  "
          f"bikes_left={int(r['bikes_available_now']):2d}  "
          f"actual={int(r[TARGET]):2d}  pred={out['demand_pred']:5.2f}  "
          f"flag_actual={r['pressure_actual']:<6s}  flag_pred={out['pressure']}")


# %% Brief summary
print("\n=== Summary ===")
print(f"  Best params: nl={final_params['num_leaves']} "
      f"lr={final_params['learning_rate']} mcs={final_params['min_child_samples']}")
print(f"  Test MAE = {mae:.4f}   RMSE = {rmse:.4f}   bias = {bias:+.4f}   corr = {corr:.4f}")
red_total      = int(cm[2, :].sum())
red_pred_total = int(cm[:, 2].sum())
red_correct    = int(cm[2, 2])
red_recall    = red_correct / red_total      if red_total      > 0 else float("nan")
red_precision = red_correct / red_pred_total if red_pred_total > 0 else float("nan")
print(f"  Pressure: Red recall = {red_recall:.4f}, Red precision = {red_precision:.4f}")
print(f"  Top-5 most-pressured stations (actual Red rate):")
for s, row in per_station.head(5).iterrows():
    print(f"    {s:<22s}  red_rate_actual={row['red_rate_actual']:.4f}  "
          f"red_recall={row['red_recall']:.4f}")
print(f"  Artifacts written under {NEW_CODE}/")
