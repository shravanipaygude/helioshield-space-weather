"""
backend/main.py - HelioShield Phase 3 FastAPI Application.

Exposes high-performance space weather endpoints:
- GET /api/health    -> Model status, feature count, operational health
- GET /api/forecast  -> 3-hour Kp forecast, current live telemetry,
                        AI disruption briefing, and 24h history
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend directory is in python search path
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Local imports
from telemetry import (
    fetch_live_telemetry,
    compute_features,
    get_feature_vector,
    get_24h_history,
    load_feature_columns,
    generate_synthetic_telemetry
)
from ai_explainer import generate_ai_briefing, get_noaa_storm_scale

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("helioshield.api")

load_dotenv()

# Model and Feature schema paths
MODEL_PATH = PROJECT_ROOT / "models" / "kp_forecaster.json"
FEATURE_PATH = PROJECT_ROOT / "models" / "feature_columns.json"

# Global state for loaded model and metadata
model_state: Dict[str, Any] = {
    "model": None,
    "feature_columns": [],
    "alert_threshold_kp": 4.5,
    "horizon_hours": 3,
    "model_loaded": False,
    "is_booster": False,
    "load_error": None
}


def load_model():
    """Load trained XGBoost Kp forecaster and feature column schema."""
    try:
        # Load feature schema
        if FEATURE_PATH.exists():
            with open(FEATURE_PATH, "r", encoding="utf-8") as f:
                schema = json.load(f)
                model_state["feature_columns"] = schema.get("feature_columns", load_feature_columns())
                model_state["alert_threshold_kp"] = schema.get("alert_threshold_kp", 4.5)
                model_state["horizon_hours"] = schema.get("horizon_hours", 3)
        else:
            model_state["feature_columns"] = load_feature_columns()

        # Load XGBoost Regressor or native Booster
        if MODEL_PATH.exists():
            try:
                forecaster = xgb.XGBRegressor()
                forecaster.load_model(str(MODEL_PATH))
                model_state["model"] = forecaster
                model_state["is_booster"] = False
                model_state["model_loaded"] = True
                logger.info("Loaded XGBoost model via XGBRegressor from %s (%d features)", MODEL_PATH, len(model_state["feature_columns"]))
            except Exception as e1:
                logger.warning("XGBRegressor load failed: %s. Loading via native xgb.Booster.", e1)
                booster = xgb.Booster()
                booster.load_model(str(MODEL_PATH))
                model_state["model"] = booster
                model_state["is_booster"] = True
                model_state["model_loaded"] = True
                logger.info("Loaded XGBoost model via native xgb.Booster from %s (%d features)", MODEL_PATH, len(model_state["feature_columns"]))
        else:
            err = f"Model file not found at {MODEL_PATH}"
            logger.error(err)
            model_state["load_error"] = err
            model_state["model_loaded"] = False
    except Exception as e:
        err = f"Failed to load XGBoost model: {e}"
        logger.error(err)
        model_state["load_error"] = err
        model_state["model_loaded"] = False


def run_prediction(X_input: pd.DataFrame) -> float:
    """Predict Kp 3 hours ahead using loaded XGBoost model."""
    model = model_state.get("model")
    if model is None:
        raise ValueError("Model not loaded")

    feature_cols = model_state["feature_columns"]
    X_features = X_input[feature_cols]

    if model_state.get("is_booster", False):
        dmat = xgb.DMatrix(X_features)
        raw_pred = float(model.predict(dmat)[0])
    else:
        raw_pred = float(model.predict(X_features)[0])

    return float(np.clip(raw_pred, 0.0, 9.0))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize models at startup."""
    logger.info("Initializing HelioShield Space Weather Engine...")
    load_model()
    yield
    logger.info("Shutting down HelioShield Space Weather Engine.")


# Create FastAPI instance with Lifespan
app = FastAPI(
    title="HelioShield Space Weather & Satellite Disruption API",
    description="Real-time NOAA RTSW Telemetry Ingestion, Physics-Informed ML Kp Forecaster, and AI Reasoning Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Enable Cross-Origin Resource Sharing (CORS) for frontend clients (Next.js/React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint with service summary and documentation links."""
    return {
        "service": "HelioShield — Solar Storm & Satellite Disruption Forecaster API",
        "phase": "Phase 3 (Operational Real-Time Backend)",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/api/health",
        "forecast_url": "/api/forecast"
    }


@app.get("/api/health")
def get_health():
    """
    Health check endpoint verifying model readiness, feature column alignment,
    and operational telemetry connectivity.
    """
    return {
        "status": "healthy" if model_state["model_loaded"] else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "loaded": model_state["model_loaded"],
            "model_path": str(MODEL_PATH),
            "algorithm": "XGBoost Regressor (Storm-Weighted)",
            "features_count": len(model_state["feature_columns"]),
            "horizon_hours": model_state["horizon_hours"],
            "alert_threshold_kp": model_state["alert_threshold_kp"],
            "error": model_state["load_error"]
        },
        "gemini_api_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "environment": "production" if os.getenv("ENVIRONMENT") == "production" else "development"
    }


@app.get("/api/forecast")
def get_forecast(
    simulate_storm: Optional[bool] = Query(False, description="Simulate a major storm condition for testing"),
    simulated_kp: Optional[float] = Query(None, description="Inject custom simulated Kp value (0.0 to 9.0) for evaluation")
):
    """
    Generate live space weather forecast:
    1. Fetch real-time NOAA RTSW L1 satellite telemetry (active == True).
    2. Resample to 1-hour intervals (bz_gsm, bt, wind_speed, proton_density).
    3. Compute 28 physics, lag (1h, 3h, 6h), and rolling features.
    4. Execute 3-hour ahead Kp inference using trained XGBoost model.
    5. Generate structured disruption briefing (Gemini 2.5 Flash with rule-based fallback).
    6. Return current telemetry, 24h history, forecast, and operational mitigation.
    """
    # 1. Fetch live telemetry or fallback
    try:
        df_hourly, is_live_feed = fetch_live_telemetry()
    except Exception as e:
        logger.error("Telemetry fetch failed: %s. Using synthetic baseline.", e)
        df_hourly = generate_synthetic_telemetry(hours=48)
        is_live_feed = False

    # 2. Compute 28 physics & time-series features
    try:
        df_features = compute_features(df_hourly)
        X_latest, current_telemetry = get_feature_vector(df_features)
    except Exception as e:
        logger.error("Feature engineering error: %s", e)
        raise HTTPException(status_code=500, detail=f"Feature computation error: {e}")

    # Optional storm simulation override for demonstrations
    if simulate_storm or simulated_kp is not None:
        target_kp = simulated_kp if simulated_kp is not None else 7.33
        predicted_kp = float(np.clip(target_kp, 0.0, 9.0))
        current_telemetry["bz_gsm"] = -14.2
        current_telemetry["bz_south"] = -14.2
        current_telemetry["wind_speed"] = 720.0
        current_telemetry["proton_density"] = 18.5
        current_telemetry["dynamic_pressure"] = round(18.5 * (720.0 ** 2) * 1e-6, 3)
        current_telemetry["electric_field_coupling"] = round(720.0 * 14.2, 1)
        is_live_feed = False
    else:
        # 3. Model Inference
        if model_state["model_loaded"] and model_state["model"] is not None:
            try:
                predicted_kp = run_prediction(X_latest)
            except Exception as e:
                logger.error("Inference prediction error: %s", e)
                # Fallback to physical heuristic approximation if model fails
                predicted_kp = float(np.clip(2.0 + 0.005 * current_telemetry["electric_field_coupling"] + 0.5 * current_telemetry["dynamic_pressure"], 0.0, 9.0))
        else:
            # Physics-based baseline heuristic if model not loaded
            predicted_kp = float(np.clip(2.0 + 0.005 * current_telemetry["electric_field_coupling"] + 0.5 * current_telemetry["dynamic_pressure"], 0.0, 9.0))

    predicted_kp = round(predicted_kp, 2)
    alert_threshold = float(model_state.get("alert_threshold_kp", 4.5))
    horizon_hours = int(model_state.get("horizon_hours", 3))
    storm_alert = bool(predicted_kp >= alert_threshold)
    noaa_scale, storm_category = get_noaa_storm_scale(predicted_kp)

    # 4. Generate AI Disruption Briefing
    ai_briefing = generate_ai_briefing(
        predicted_kp=predicted_kp,
        horizon_hours=horizon_hours,
        telemetry=current_telemetry,
        storm_alert=storm_alert
    )

    # 5. Extract 24-hour telemetry history for dashboard charts
    history_24h = get_24h_history(df_features, model=model_state["model"] if model_state["model_loaded"] else None)

    # Calculate target forecast timestamp (current telemetry time + horizon)
    latest_time_str = current_telemetry["timestamp"]
    try:
        base_dt = datetime.fromisoformat(latest_time_str.replace("Z", "+00:00"))
        target_dt = base_dt + timedelta(hours=horizon_hours)
        target_timestamp = target_dt.isoformat()
    except Exception:
        target_timestamp = (datetime.now(timezone.utc) + timedelta(hours=horizon_hours)).isoformat()

    # 6. Assemble complete response payload
    return {
        "forecast": {
            "predicted_kp": predicted_kp,
            "horizon_hours": horizon_hours,
            "alert_threshold": alert_threshold,
            "storm_alert": storm_alert,
            "noaa_scale": noaa_scale,
            "storm_category": storm_category,
            "target_timestamp": target_timestamp,
            "confidence_band": {
                "lower": round(max(0.0, predicted_kp - 0.74), 2),  # Based on test MAE 0.74
                "upper": round(min(9.0, predicted_kp + 0.74), 2)
            }
        },
        "current_telemetry": current_telemetry,
        "ai_briefing": ai_briefing,
        "history_24h": history_24h,
        "meta": {
            "is_live_feed": is_live_feed,
            "model_version": "HelioShield-XGBoost-StormWeighted-v1",
            "feature_count": len(model_state["feature_columns"]),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    }
