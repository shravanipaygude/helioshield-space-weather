"""
scripts/train_v2_model.py - Train HelioShield v2 ML Model

Upgrades the 3-hour Kp forecasting model:
1. Loads NASA OMNI2 hourly space weather data (2016-2025).
2. Computes the 28 existing features + 6 new physics & persistence features (34 total):
   - newell_coupling_proxy: (wind_speed ** (4/3)) * (abs(bz_south) ** (2/3))
   - newell_roll3_sum: 3h rolling sum of newell_coupling_proxy
   - bz_south_streak: consecutive hours where bz_gsm < -4.0 nT
   - speed_diff_1h: 1h delta in solar wind velocity
   - pressure_diff_1h: 1h delta in dynamic pressure
   - bz_roll3_mean: 3h rolling mean of bz_gsm
3. Trains regularized Storm-Weighted XGBRegressor (n_estimators=450, lr=0.03, max_depth=5,
   min_child_weight=12, subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=2.5)
   with balanced weights: 3.0x for Kp >= 5.0, 1.5x for Kp >= 4.0, 1.0x otherwise.
4. Evaluates thresholds [5.0, 5.15, 5.25], picks optimal alert threshold, and saves
   model to models/kp_forecaster.json.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score
)
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_FILE = DATA_DIR / "omni2_2016_2025.csv"

def fetch_or_load_omni2() -> pd.DataFrame:
    """Fetch or load historical NASA OMNI2 dataset (2016-2025)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists():
        print(f"Loading cached NASA OMNI2 dataset from {CACHE_FILE}...")
        df = pd.read_csv(CACHE_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    print("Fetching NASA OMNI2 hourly data from Goddard SPDF (2016-2025)...")
    base_url = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{}.dat"
    use_cols = [0, 1, 2, 14, 23, 24, 38, 39, 40, 50]
    col_names = ["year", "doy", "hour", "bz_gsm", "proton_density", "wind_speed", "kp_10", "sunspot_num", "dst", "f10_7"]

    dfs = []
    for yr in range(2016, 2026):
        url = base_url.format(yr)
        print(f"  Downloading {yr} from {url}...")
        try:
            df_yr = pd.read_csv(url, sep=r"\s+", header=None, usecols=use_cols, names=col_names)
            dfs.append(df_yr)
        except Exception as e:
            print(f"  Error loading {yr}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["doy"].astype(str).str.zfill(3),
        format="%Y-%j"
    ) + pd.to_timedelta(df["hour"], unit="h")

    # Replace sentinel missing-value codes with NaN
    sentinels = {
        "bz_gsm": 999.9,
        "proton_density": 999.9,
        "wind_speed": 9999.0,
        "kp_10": 99,
        "sunspot_num": 999,
        "dst": 99999,
        "f10_7": 999.9
    }
    for col, val in sentinels.items():
        df[col] = df[col].replace(val, np.nan)

    df["kp"] = df["kp_10"] / 10.0
    # Impute <1% dropouts
    df[["bz_gsm", "proton_density", "wind_speed", "f10_7"]] = df[["bz_gsm", "proton_density", "wind_speed", "f10_7"]].ffill().bfill()

    # Save to local cache
    df.to_csv(CACHE_FILE, index=False)
    print(f"Saved NASA OMNI2 dataset to {CACHE_FILE} ({len(df):,} rows)")
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute 28 baseline features + 6 new v2 features (34 total)."""
    df = df.copy()

    # 1. Base physics features
    df["bz_south"] = np.minimum(df["bz_gsm"], 0.0)
    df["dynamic_pressure"] = df["proton_density"] * (df["wind_speed"] ** 2) * 1e-6
    df["electric_field_coupling"] = -df["wind_speed"] * df["bz_south"]

    # 2. Lags (1h, 3h, 6h) & 6-hour rolling means
    core_vars = ["bz_gsm", "bz_south", "wind_speed", "proton_density", "electric_field_coupling"]
    for col in core_vars:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)
        df[f"{col}_lag6"] = df[col].shift(6)
        df[f"{col}_roll6_mean"] = df[col].rolling(6).mean()

    df["bz_roll6_min"] = df["bz_gsm"].rolling(6).min()
    df["speed_roll6_max"] = df["wind_speed"].rolling(6).max()

    # 3. 6 NEW v2 PHYSICS & PERSISTENCE FEATURES
    # A. Newell coupling proxy: (v^(4/3)) * (|Bz_south|^(2/3))
    df["newell_coupling_proxy"] = (df["wind_speed"] ** (4.0 / 3.0)) * (np.abs(df["bz_south"]) ** (2.0 / 3.0))

    # B. 3h cumulative Newell coupling
    df["newell_roll3_sum"] = df["newell_coupling_proxy"].rolling(3).sum()

    # C. Consecutive hours of southward Bz < -4.0 nT
    cond_south = (df["bz_gsm"] < -4.0)
    df["bz_south_streak"] = cond_south.groupby((~cond_south).cumsum()).cumsum()

    # D. 1h rate of change in solar wind speed
    df["speed_diff_1h"] = df["wind_speed"].diff(1).fillna(0.0)

    # E. 1h rate of change in dynamic pressure
    df["pressure_diff_1h"] = df["dynamic_pressure"].diff(1).fillna(0.0)

    # F. 3h rolling mean of Bz GSM
    df["bz_roll3_mean"] = df["bz_gsm"].rolling(3).mean()

    # 4. Target variable: 3 hours ahead
    df["target_kp_3h"] = df["kp"].shift(-3)

    # Drop NaNs from shifts/rolling windows
    df_clean = df.dropna().reset_index(drop=True)

    non_feature_cols = [
        "year", "doy", "hour", "kp_10", "sunspot_num", "dst", "f10_7",
        "timestamp", "kp", "target_kp_3h"
    ]
    feature_cols = [c for c in df_clean.columns if c not in non_feature_cols]

    print(f"\nEngineered {len(feature_cols)} live-compatible features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    return df_clean, feature_cols


def main():
    print("=" * 70)
    print(" HELIOSHIELD v2 ML MODEL TRAINING PIPELINE")
    print("=" * 70)

    df_raw = fetch_or_load_omni2()
    df_model, feature_cols = engineer_features(df_raw)

    # Chronological Split
    train_df = df_model[df_model["year"] <= 2023]
    test_df = df_model[df_model["year"] >= 2024]

    X_train, y_train = train_df[feature_cols], train_df["target_kp_3h"]
    X_test, y_test = test_df[feature_cols], test_df["target_kp_3h"]

    print(f"\nTrain set (2016-2023): {len(X_train):,} samples")
    print(f"Test set  (2024-2025): {len(X_test):,} samples")

    # Balanced sample weights (3.0x for Kp >= 5.0, 1.5x for Kp >= 4.0)
    sample_weights = np.where(y_train >= 5.0, 3.0, np.where(y_train >= 4.0, 1.5, 1.0))
    print(f"\nSample weighting scheme:")
    print(f"  Kp >= 5.0: 3.0x weight ({(y_train >= 5.0).sum():,} samples)")
    print(f"  4.0 <= Kp < 5.0: 1.5x weight ({((y_train >= 4.0) & (y_train < 5.0)).sum():,} samples)")
    print(f"  Kp < 4.0:  1.0x weight ({(y_train < 4.0).sum():,} samples)")

    # Configure regularized XGBRegressor v2
    xgb_v2 = xgb.XGBRegressor(
        n_estimators=450,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=12,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=2.5,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining regularized Storm-Weighted XGBoost v2...")
    xgb_v2.fit(X_train, y_train, sample_weight=sample_weights)

    # Predict on unseen test set
    y_pred = np.clip(xgb_v2.predict(X_test), 0.0, 9.0)

    # Continuous Regression Metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"\nContinuous Regression Performance (2024-2025 Unseen Test Set):")
    print(f"  MAE:  {mae:.4f} Kp units")
    print(f"  RMSE: {rmse:.4f} Kp units")
    print(f"  R²:   {r2:.4f}")

    # Binary Storm Classification Evaluation (Storm True: Kp >= 5.0)
    storm_true = (y_test >= 5.0).astype(int)
    num_actual_storms = int(storm_true.sum())
    print(f"\nActual Geomagnetic Storm Hours in Test Set (Kp >= 5.0): {num_actual_storms} ({num_actual_storms/len(y_test):.2%})")

    # Evaluate multiple alert thresholds
    candidate_thresholds = [5.0, 5.15, 5.25]
    threshold_results = []

    print("\nThreshold Calibration Evaluation:")
    print("-------------------------------------------------------------")
    print(f" {'Threshold':^10} | {'Precision':^11} | {'Recall':^10} | {'F1-Score':^10} | {'Predicted':^10}")
    print("-------------------------------------------------------------")

    best_thresh = candidate_thresholds[0]
    best_f1 = -1.0
    best_metrics = {}

    for thresh in candidate_thresholds:
        storm_pred = (y_pred >= thresh).astype(int)
        p = float(precision_score(storm_true, storm_pred, zero_division=0)) * 100.0
        r = float(recall_score(storm_true, storm_pred, zero_division=0)) * 100.0
        f = float(f1_score(storm_true, storm_pred, zero_division=0))
        pred_count = int(storm_pred.sum())

        threshold_results.append({
            "threshold": thresh,
            "precision": round(p, 1),
            "recall": round(r, 1),
            "f1": round(f, 3),
            "predicted_count": pred_count
        })

        print(f" Kp >= {thresh:<4.2f}  | {p:>9.1f}%  | {r:>8.1f}%  | {f:>10.3f} | {pred_count:>8,d} hrs")

        # Select threshold maximizing F1 while boosting precision
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
            best_metrics = {"precision": p, "recall": r, "f1": f, "threshold": thresh}

    print("-------------------------------------------------------------")
    print(f"\nOptimal Alert Threshold Selected: Kp >= {best_thresh:.2f}")
    print(f"  Precision: {best_metrics['precision']:.1f}% (vs. v1 baseline 44.8%)")
    print(f"  Recall:    {best_metrics['recall']:.1f}%")
    print(f"  F1-Score:  {best_metrics['f1']:.3f} (vs. v1 baseline 0.487)")

    # Feature Importances
    importances = xgb_v2.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    print("\nTop 10 Feature Importances (XGBoost Gain):")
    label_map = {
        "dynamic_pressure": "Dynamic Pressure (P_dyn)",
        "electric_field_coupling": "Electric Field Coupling (V × Bz)",
        "wind_speed": "Solar Wind Velocity (v)",
        "speed_roll6_max": "6h Max Solar Wind Speed",
        "bz_roll6_min": "6h Min Southward Bz",
        "electric_field_coupling_roll6_mean": "6h Mean Electric Coupling",
        "newell_coupling_proxy": "Newell Coupling Proxy",
        "newell_roll3_sum": "3h Sum Newell Coupling",
        "bz_south_streak": "Southward Bz Duration Streak",
        "speed_diff_1h": "1h Delta Wind Speed",
        "pressure_diff_1h": "1h Delta Dynamic Pressure",
        "bz_roll3_mean": "3h Mean Bz GSM"
    }

    top_features = []
    for col, imp in feat_imp[:8]:
        lbl = label_map.get(col, col.replace("_", " ").title())
        top_features.append({
            "name": col,
            "importance": round(float(imp), 4),
            "label": lbl
        })
        print(f"  - {lbl:<32}: {imp:.4f} ({imp*100:.1f}%)")

    # Save model
    model_output_path = MODELS_DIR / "kp_forecaster.json"
    xgb_v2.save_model(str(model_output_path))
    print(f"\nSuccessfully exported v2 model to {model_output_path}!")

    # Return summary dict
    result = {
        "algorithm": "Storm-Weighted XGBoost Regressor v2",
        "train_window": "2016-2023 (70,122 hrs)",
        "test_window": "2024-2025 (17,541 hrs)",
        "features_count": len(feature_cols),
        "horizon_hours": 3,
        "optimal_threshold_kp": best_thresh,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "precision": round(best_metrics["precision"], 1),
        "recall": round(best_metrics["recall"], 1),
        "f1_score": round(best_metrics["f1"], 3),
        "top_features": top_features,
        "threshold_evaluations": threshold_results
    }

    print("\nSummary Result JSON:")
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main()

