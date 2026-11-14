"""
backend/telemetry.py - Real-Time NOAA RTSW Telemetry Processor for HelioShield.

Fetches live 1-minute NOAA Space Weather Prediction Center (SWPC) RTSW feeds:
- https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json
- https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json

Filters active satellite observations (active == True), resamples to 1-hour
intervals (bz_gsm, bt, wind_speed, proton_density), and computes the exact 28
physics, lag (1h, 3h, 6h), and 6h rolling features listed in models/feature_columns.json.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

logger = logging.getLogger("helioshield.telemetry")

# NOAA RTSW 1-minute JSON feed endpoints
MAG_1M_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
WIND_1M_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"

# NOAA 1-day fallback endpoints (in case 1-minute feeds undergo SWPC maintenance)
MAG_1DAY_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
PLASMA_1DAY_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"

USER_AGENT = "HelioShield-SpaceWeather/1.0 (Solar Storm & Disruption Forecaster)"

FEATURE_COLUMNS_FILE = Path(__file__).resolve().parent.parent / "models" / "feature_columns.json"


def load_feature_columns() -> List[str]:
    """Load canonical list of 28 feature columns from models/feature_columns.json."""
    if FEATURE_COLUMNS_FILE.exists():
        try:
            with open(FEATURE_COLUMNS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("feature_columns", [])
        except Exception as e:
            logger.warning("Could not read feature_columns.json: %s. Using default list.", e)

    return [
        "bz_gsm", "proton_density", "wind_speed", "bz_south", "dynamic_pressure",
        "electric_field_coupling", "bz_gsm_lag1", "bz_gsm_lag3", "bz_gsm_lag6",
        "bz_gsm_roll6_mean", "bz_south_lag1", "bz_south_lag3", "bz_south_lag6",
        "bz_south_roll6_mean", "wind_speed_lag1", "wind_speed_lag3", "wind_speed_lag6",
        "wind_speed_roll6_mean", "proton_density_lag1", "proton_density_lag3",
        "proton_density_lag6", "proton_density_roll6_mean", "electric_field_coupling_lag1",
        "electric_field_coupling_lag3", "electric_field_coupling_lag6",
        "electric_field_coupling_roll6_mean", "bz_roll6_min", "speed_roll6_max"
    ]


def _fetch_json(urls: List[str], timeout: int = 12) -> Optional[Any]:
    """Fetch JSON from primary URL with optional fallback URLs."""
    headers = {"User-Agent": USER_AGENT}
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200 and resp.text.strip():
                return resp.json()
            logger.warning("URL %s returned status %s", url, resp.status_code)
        except Exception as e:
            logger.warning("Failed fetching from %s: %s", url, e)
    return None


def _parse_noaa_data(data: Any, is_plasma: bool = False) -> pd.DataFrame:
    """Parse list-of-dicts (RTSW 1m) or list-of-lists (products 1-day) NOAA feeds."""
    if not data:
        return pd.DataFrame()

    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], list):
            # Header in row 0, data in rows 1..N
            df = pd.DataFrame(data[1:], columns=data[0])
        elif isinstance(data[0], dict):
            # Filter active == True when available
            active_records = [
                d for d in data
                if d.get("active") in (True, "true", "True", 1, "1")
            ]
            if active_records:
                df = pd.DataFrame(active_records)
            else:
                df = pd.DataFrame(data)
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Map proton_speed to wind_speed if in plasma feed
    if is_plasma:
        if "proton_speed" in df.columns and "wind_speed" not in df.columns:
            df["wind_speed"] = df["proton_speed"]
        if "density" in df.columns and "proton_density" not in df.columns:
            df["proton_density"] = df["density"]
        if "speed" in df.columns and "wind_speed" not in df.columns:
            df["wind_speed"] = df["speed"]

    # Parse time_tag
    if "time_tag" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp")
    else:
        return pd.DataFrame()

    return df


def generate_synthetic_telemetry(hours: int = 48) -> pd.DataFrame:
    """Generate realistic synthetic solar wind telemetry when NOAA feeds are offline."""
    logger.info("Generating realistic fallback telemetry for %d hours...", hours)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamps = [now - timedelta(hours=i) for i in range(hours - 1, -1, -1)]

    np.random.seed(42)
    # Realistic solar wind baseline: speed ~420 km/s, bz fluctuating around -1.5 nT, density ~4.5
    t = np.linspace(0, 4 * np.pi, hours)
    wind_speed = 420.0 + 35.0 * np.sin(t) + np.random.normal(0, 10, hours)
    bz_gsm = -1.2 + 3.0 * np.cos(t) + np.random.normal(0, 1.2, hours)
    bt = np.sqrt(bz_gsm**2 + 16.0) + np.random.normal(0, 0.5, hours)
    proton_density = 4.5 + 2.0 * np.sin(t * 1.5) + np.random.normal(0, 0.8, hours)
    proton_density = np.clip(proton_density, 0.5, 30.0)

    df = pd.DataFrame({
        "bz_gsm": bz_gsm,
        "bt": bt,
        "wind_speed": wind_speed,
        "proton_density": proton_density,
    }, index=pd.to_datetime(timestamps, utc=True))

    return df


def fetch_live_telemetry() -> Tuple[pd.DataFrame, bool]:
    """
    Fetch live 1-minute NOAA RTSW magnetic and wind feeds, filter active==True,
    and resample to 1-hour intervals.

    Returns:
        (df_hourly, is_live_feed)
    """
    mag_raw = _fetch_json([MAG_1M_URL, MAG_1DAY_URL])
    wind_raw = _fetch_json([WIND_1M_URL, PLASMA_1DAY_URL])

    df_mag = _parse_noaa_data(mag_raw, is_plasma=False)
    df_wind = _parse_noaa_data(wind_raw, is_plasma=True)

    is_live = True

    if df_mag.empty or df_wind.empty:
        logger.warning("NOAA RTSW live feed was empty or unreachable. Falling back to synthetic stream.")
        return generate_synthetic_telemetry(hours=48), False

    # Extract required columns and convert to numeric
    mag_cols = [c for c in ["bz_gsm", "bt"] if c in df_mag.columns]
    wind_cols = [c for c in ["wind_speed", "proton_density"] if c in df_wind.columns]

    for c in mag_cols:
        df_mag[c] = pd.to_numeric(df_mag[c], errors="coerce")
    for c in wind_cols:
        df_wind[c] = pd.to_numeric(df_wind[c], errors="coerce")

    # Resample to 1-hour intervals taking the mean of 1-minute active readings
    df_mag_1h = df_mag[mag_cols].resample("1h").mean()
    df_wind_1h = df_wind[wind_cols].resample("1h").mean()

    # Align on hourly timestamps
    df_hourly = pd.concat([df_mag_1h, df_wind_1h], axis=1)

    # Impute missing spacecraft dropouts (<1%) via time-series forward/backward fill
    df_hourly = df_hourly.ffill().bfill()

    # Ensure required base columns exist with physical defaults if missing
    if "bz_gsm" not in df_hourly.columns:
        df_hourly["bz_gsm"] = 0.0
    if "bt" not in df_hourly.columns:
        df_hourly["bt"] = np.abs(df_hourly["bz_gsm"]) + 2.0
    if "wind_speed" not in df_hourly.columns:
        df_hourly["wind_speed"] = 400.0
    if "proton_density" not in df_hourly.columns:
        df_hourly["proton_density"] = 4.0

    # If too few rows after resampling, pad with synthetic data
    if len(df_hourly) < 12:
        logger.info("Fewer than 12 hourly observations in live feed (%d rows). Blending with baseline.", len(df_hourly))
        synthetic = generate_synthetic_telemetry(hours=36)
        df_hourly = pd.concat([synthetic, df_hourly]).drop_duplicates().sort_index()
        df_hourly = df_hourly.ffill().bfill()

    return df_hourly, is_live


def compute_features(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the exact 28 physics, lag, and rolling features matching
    Phase 2 training schema in models/feature_columns.json.
    """
    df = df_hourly.copy()

    # 1. Physics-based coupling parameters
    # Southward magnetic component (only negative Bz couples directly into Earth's magnetosphere)
    df["bz_south"] = np.minimum(df["bz_gsm"], 0.0)

    # Solar wind dynamic pressure proxy: density * speed^2 * 1e-6 (nPa)
    df["dynamic_pressure"] = df["proton_density"] * (df["wind_speed"] ** 2) * 1e-6

    # Motional electric field coupling proxy: -wind_speed * bz_south (mV/m proxy)
    df["electric_field_coupling"] = -df["wind_speed"] * df["bz_south"]

    # 2. Lag features (1h, 3h, 6h ago) & 6-hour rolling averages
    lag_cols = ["bz_gsm", "bz_south", "wind_speed", "proton_density", "electric_field_coupling"]
    for col in lag_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)
        df[f"{col}_lag6"] = df[col].shift(6)
        df[f"{col}_roll6_mean"] = df[col].rolling(6, min_periods=1).mean()

    # 3. 6-hour rolling extremes
    df["bz_roll6_min"] = df["bz_gsm"].rolling(6, min_periods=1).min()
    df["speed_roll6_max"] = df["wind_speed"].rolling(6, min_periods=1).max()

    # Backward-fill the earliest edge rows so lag features are never NaN
    df = df.bfill().ffill()

    return df


def get_feature_vector(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract the latest 28-feature row for XGBoost inference and format
    the current telemetry reading.
    """
    feature_cols = load_feature_columns()

    # Verify all 28 feature columns exist
    missing_cols = [c for c in feature_cols if c not in df_features.columns]
    if missing_cols:
        logger.error("Missing expected feature columns: %s", missing_cols)
        for c in missing_cols:
            df_features[c] = 0.0

    X_latest = df_features[feature_cols].iloc[[-1]]
    latest_row = df_features.iloc[-1]
    latest_time = df_features.index[-1]

    iso_timestamp = latest_time.isoformat() if hasattr(latest_time, "isoformat") else str(latest_time)

    current_telemetry = {
        "timestamp": iso_timestamp,
        "bz_gsm": round(float(latest_row["bz_gsm"]), 2),
        "bt": round(float(latest_row.get("bt", abs(latest_row["bz_gsm"]) + 1.5)), 2),
        "wind_speed": round(float(latest_row["wind_speed"]), 1),
        "proton_density": round(float(latest_row["proton_density"]), 2),
        "dynamic_pressure": round(float(latest_row["dynamic_pressure"]), 3),
        "electric_field_coupling": round(float(latest_row["electric_field_coupling"]), 1),
        "bz_south": round(float(latest_row["bz_south"]), 2),
    }

    return X_latest, current_telemetry


def get_24h_history(df_features: pd.DataFrame, model=None) -> List[Dict[str, Any]]:
    """Format the most recent 24 hours of telemetry history for charting."""
    feature_cols = load_feature_columns()
    df_24h = df_features.tail(24)

    history = []
    for idx, row in df_24h.iterrows():
        iso_time = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)

        record = {
            "timestamp": iso_time,
            "bz_gsm": round(float(row["bz_gsm"]), 2),
            "bt": round(float(row.get("bt", abs(row["bz_gsm"]) + 1.5)), 2),
            "wind_speed": round(float(row["wind_speed"]), 1),
            "proton_density": round(float(row["proton_density"]), 2),
            "dynamic_pressure": round(float(row["dynamic_pressure"]), 3),
            "electric_field_coupling": round(float(row["electric_field_coupling"]), 1),
        }

        # If model is provided, compute historical estimate
        if model is not None:
            try:
                x_row = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
                if isinstance(model, xgb.Booster):
                    pred = float(model.predict(xgb.DMatrix(x_row))[0])
                else:
                    pred = float(model.predict(x_row)[0])
                record["kp_estimate"] = round(float(np.clip(pred, 0.0, 9.0)), 2)
            except Exception:
                record["kp_estimate"] = None
        else:
            record["kp_estimate"] = None

        history.append(record)

    return history
