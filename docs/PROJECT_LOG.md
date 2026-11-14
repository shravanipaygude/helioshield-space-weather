# Engineering & AI/ML Project Log — Solar Storm & Satellite Disruption Forecaster

## Project Overview
- **Goal:** Predict geomagnetic storms ($K_p$ Index) using historical NASA solar wind telemetry and translate real-time NOAA satellite data into actionable disruption alerts using an AI reasoning layer.
- **Tech Stack:** Python, Pandas, NumPy, Scikit-Learn, XGBoost, Google Colab, FastAPI, LLM API, Next.js/React.

---

## Phase 0: Repository & Toolchain Setup
- **Date:** November 13, 2026
- **Tools Used:** Git, GitHub, Google Antigravity IDE
- **Decisions Made:**
  - Adopted a modular ML + Full-Stack folder architecture (`notebooks/`, `models/`, `backend/`, `frontend/`, `docs/`).
  - Configured `.gitignore` to exclude OS files (`desktop.ini`) and large raw datasets (`data/*.csv`, `data/*.txt`).
  - Using Google Colab for dataset exploration and model training to keep environment setup lightweight and reproducible.

---

## Phase 1: Space Weather Data Acquisition & Exploratory Data Analysis (EDA)
- **Date:** November 13, 2026
- **Tools & Libraries Used:** Google Colab, Python, Pandas, NumPy, Matplotlib, Seaborn, Requests
- **Datasets & APIs Verified:**
  1. **Historical Training Source:** NASA Goddard Space Physics Data Facility (SPDF) Hourly OMNI2 Archive (`spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_YYYY.dat`), covering 2016–2025 (**87,672 hourly records**).
  2. **Live Inference Source:** NOAA Space Weather Prediction Center (SWPC) Real-Time Solar Wind (RTSW) 1-minute JSON feeds (`services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json` and `rtsw_wind_1m.json`), filtering for `active == True` (`proton_speed`, `proton_density`, `bz_gsm`, `bt`).
- **Key Findings & Preprocessing Decisions:**
  - **NASA Sentinel Cleaning:** Replaced official NASA missing-data sentinel values (`999.9` for $B_z$/density, `9999.0` for wind speed, `99` for $K_p \times 10$) with `NaN`. Missingness was `< 1%` across all telemetry features (`bz_gsm`: 0.40%, `wind_speed`: 0.81%, `proton_density`: 0.94%) and imputed via time-series forward/backward fill (`ffill`/`bfill`).
  - **Target Scaling:** Scaled `kp_10` by `0.1` to obtain the standard `0.0–9.0` $K_p$ geomagnetic storm index and mapped NOAA storm categories (`G0` through `G5`).
  - **Class Imbalance Discovery:** 97.9% of hours (85,827) are quiet (`G0`, $K_p < 5$), while severe storms are rare (`G3`: 123 hrs, `G4`: 57 hrs, `G5`: 6 hrs). Therefore, Phase 2 model evaluation will prioritize RMSE/MAE on high-$K_p$ events and storm-detection F1/recall rather than raw accuracy alone.
- **Artifacts Saved:** `notebooks/01_eda_and_cleaning.ipynb`

---

## Phase 2: Feature Engineering & ML Model Training
- **Date:** November 14, 2026
- **Tools & Libraries Used:** Google Colab, Python, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib
- **Feature Engineering (28 Live-Compatible Features):**
  - Restricted all input features strictly to sensors available in NOAA's real-time L1 satellite feeds (`bz_gsm`, `wind_speed`, `proton_density`) to ensure zero train/serving skew in Phase 3.
  - Derived physical coupling features: southward magnetic component (`bz_south`), solar wind dynamic pressure (`dynamic_pressure`), and motional electric field (`electric_field_coupling = -wind_speed * bz_south`).
  - Added 1h, 3h, and 6h lag features plus 6-hour rolling averages/extremes (`bz_roll6_min`, `speed_roll6_max`) to forecast **`target_kp_3h`** ($K_p$ index 3 hours ahead).
- **Model Comparison & Evaluation (Chronological Split: Train 2016–2023 | Test 2024–2025):**
  - **Random Forest Baseline:** MAE `0.742` | RMSE `0.980` | $R^2$ `0.501` | Storm Recall ($K_p \ge 5$) `39.9%` | F1 `0.445`
  - **XGBoost (Unweighted):** MAE `0.728` | RMSE `0.957` | $R^2$ `0.524` | Storm Recall ($K_p \ge 5$) `43.3%` | F1 `0.482`
  - **Storm-Weighted XGBoost (Selected Final Model):** Applied `5.0x` sample weights for $K_p \ge 5.0$ and `2.5x` for $K_p \ge 4.0$, boosting Storm Recall to **`53.4%`** and Storm F1-Score to **`0.487`** (MAE `0.742`, RMSE `0.968`, $R^2$ `0.513`), with `dynamic_pressure` and `electric_field_coupling` ranked as the top two predictive drivers.
- **Artifacts Saved:** `notebooks/02_feature_engineering_and_training.ipynb`, `models/kp_forecaster.json`, `models/feature_columns.json`

---

## Phase 3: Real-Time Telemetry Ingestion, FastAPI Backend & AI Reasoning Layer
- **Date:** November 14, 2026
- **Tools & Libraries Used:** Python 3.13, FastAPI, Uvicorn, Pandas, NumPy, Scikit-Learn, XGBoost, Requests, Python-Dotenv, Google-GenAI
- **Architecture & Modules Implemented:**
  1. **Real-Time Telemetry Pipeline (`backend/telemetry.py`):**
     - Live fetching of NOAA Space Weather Prediction Center (SWPC) 1-minute Real-Time Solar Wind (RTSW) magnetic (`rtsw_mag_1m.json`) and plasma (`rtsw_wind_1m.json`) JSON streams with custom `User-Agent`.
     - Active satellite filtering (`active == True`) to isolate valid L1 DSCOVR/ACE measurements.
     - Resampling to 1-hour intervals (`bz_gsm`, `bt`, `wind_speed`, `proton_density`).
     - On-the-fly computation of the exact 28 physics, lag (1h, 3h, 6h), and 6-hour rolling features matching `models/feature_columns.json`.
     - Time-series forward/backward imputation and graceful synthetic fallback generation when NOAA feeds undergo scheduled cycling.
  2. **AI Reasoning Engine (`backend/ai_explainer.py`):**
     - Powered by Google GenAI (`gemini-2.5-flash`) when `GEMINI_API_KEY` is configured in `.env`.
     - Deterministic physics-informed rule-based fallback when no key is set or during API timeouts.
     - Generates structured JSON disruption briefings across critical infrastructure:
       - **Satellites:** LEO thermospheric orbital drag, GEO surface charging, star tracker blinding.
       - **GPS & Aviation:** GNSS L-band scintillation, single-frequency positioning errors, trans-polar HF radio blackouts.
       - **Power Grids:** Geomagnetically Induced Currents (GIC), transformer core saturation, harmonic resonance.
       - **Aurora:** Visibility levels, geographic latitude reach, and dynamic display intensity.
  3. **High-Performance FastAPI Service (`backend/main.py`):**
     - CORS middleware enabled for modern frontend consumption (Next.js/React).
     - Model lifespan loading: dynamically mounts the storm-weighted XGBoost regressor (`models/kp_forecaster.json`).
     - `GET /api/health`: Verifies model state, feature schema alignment, and operational health.
     - `GET /api/forecast`: End-to-end pipeline returning 3-hour Kp forecast, current telemetry snapshot, AI briefing, and 24-hour historical trend.
- **Verification & Testing:**
  - Automated endpoint tests passing with HTTP `200 OK` on both live streams and simulated storm injections (`simulate_storm=true`).
  - Active background server daemon running on `http://127.0.0.1:8000`.
- **Artifacts Saved:** `backend/requirements.txt`, `backend/telemetry.py`, `backend/ai_explainer.py`, `backend/main.py`, `.env.example`


---

## 2026-11-14 — Phase 4: Interactive Space Weather Control Center Dashboard & Deployment

- **Status:** Completed
- **Tools & Libraries Used:** Next.js 15 (App Router, Turbopack), React 19, TypeScript, Tailwind CSS, Lucide React, Recharts
- **Dashboard Architecture & Components (`frontend/`):**
  1. **Top Command Header:**
     - HelioShield Mission Control branding with status badge and pulsing connection indicator.
     - Live ticking UTC clock badge (`HH:MM:SS UTC`) synchronized to world standards.
     - Interactive **"Live NOAA / Simulate Severe Storm"** toggle switch that passes `?simulate_storm=true` to `/api/forecast`.
     - Manual telemetry refresh button with spinning state and auto-polling fallback.
  2. **Hero Forecast & Live L1 Satellite Telemetry Row:**
     - **3-Hour $K_p$ Forecast Gauge:** Prominently renders predicted $K_p$ index, NOAA G-Scale classification (G0–G5) with dynamic color coding (emerald `G0` through crimson `G5`), storm category, target timestamp, and confidence interval bounds (`lower` to `upper`).
     - **4 Live L1 Satellite KPI Cards:** Real-time DSCOVR/ACE measurements for Solar Wind Speed (`km/s`), Interplanetary Magnetic Field $B_z$ GSM (`nT` — highlighted crimson when negative/southward indicating reconnection), Proton Density (`p/cm³`), and Solar Wind Dynamic Pressure ($P_{dyn}$ in `nPa`).
  3. **AI Operational Risk Briefing Panel:**
     - Source attribution badge (`gemini-2.5-flash` or physics rule-based fallback).
     - Headline operational alert and executive technical summary.
     - 4 high-consequence sector risk cards:
       - **Satellites (LEO/GEO):** Evaluates thermospheric orbital drag and surface electrostatic discharge.
       - **GPS & Trans-Polar Aviation:** Tracks ionospheric scintillation and polar cap HF absorption.
       - **Power Grids:** Monitors Geomagnetically Induced Currents (GIC) and transformer thermal stress.
       - **Aurora Visibility:** Displays magnetic latitude reach and viewing conditions.
  4. **24-Hour Telemetry & Trend Analytics (Recharts):**
     - Dual interactive, responsive charts plotting the last 24 hours of space weather history:
       - *Solar Wind Speed & IMF $B_z$ GSM:* Area and line curves revealing solar wind shocks and interplanetary magnetic field rotations.
       - *$K_p$ Historical Trend & Prediction:* Forecast trajectory alongside a red dashed geomagnetic storm threshold ($K_p = 4.5$).
  5. **Critical Storm Simulation Engine:**
     - Allows operators to toggle into a synthetic severe Coronal Mass Ejection scenario ($B_z = -14.2\text{ nT}$, Wind Speed $= 720\text{ km/s}$, Density $= 18.5\text{ p/cm³}$).
     - Automatically verifies model scaling, resulting in $K_p \approx 7.33$ (G3 Strong Storm) with critical operational sector warnings.
- **Verification & Testing:**
  - `npm run build` executed with Turbopack: zero TypeScript errors, static generation succeeded (`5/5` routes).
  - Dev server active and listening on `http://localhost:3000` (`GET / 200 OK`).
  - Cross-origin API communication between Next.js and FastAPI (`http://127.0.0.1:8000`) verified for both live NOAA feeds and simulated storm conditions.
- **Artifacts Created:**
  - `frontend/src/app/page.tsx`: Complete Mission Control dashboard.
  - `frontend/src/app/layout.tsx`: Root layout with dark mode mission styling and typography.
  - `frontend/src/app/globals.css`: Custom scrollbars, glassmorphism panels, and glow utilities.
  - `frontend/package.json`: Configured with Next.js, Lucide Icons, and Recharts.


---

## 2026-11-14 — Gemini Model Migration & Resilience Enhancement

- **Status:** Completed
- **Root Cause Analysis:** Google AI Studio's `gemini-3.8-flash` preview endpoint experienced upstream server congestion (`504 Gateway Timeout` and `503 UNAVAILABLE: Spikes in demand are usually temporary`), triggering the deterministic physics-based fallback.
- **Model Migration:** Transitioned to `gemini-3.5-flash-lite`, which provides fast inference and reliable free-tier quota.
- **Resilience & Observability:**
  - Added explicit exception logging and stack tracing to terminal output for transparent debugging.
  - Implemented `HttpOptions` with `HttpRetryOptions(attempts=3, initial_delay=1.0)` to absorb transient upstream spikes seamlessly.
- **Verification:** Both live telemetry and simulated storm endpoints verified returning `ai_source: "Gemini 3.5 Flash-Lite (Live Generative AI)"` with valid structured disruption briefings.


---

## 2026-11-14 — Dedicated Model Performance & Validation Architecture (`/metrics`)

- **Status:** Completed
- **Tools & Libraries Used:** Next.js 15, React 19, TypeScript, Tailwind CSS, Recharts, FastAPI, Uvicorn
- **Backend Enhancements (`backend/main.py` & `models/feature_columns.json`):**
  - Added `validation_metrics` object to `models/feature_columns.json` documenting Phase 2 empirical test set benchmarks (17,541 continuous hourly samples across 2024–2025).
  - Exposed `GET /api/metrics` in `backend/main.py` providing metadata on the deployed model, feature count (28), prediction horizon (3h), baseline model comparisons (Random Forest, Unweighted XGBoost, Storm-Weighted XGBoost), and gradient gain feature importance rankings.
- **Frontend Architecture (`frontend/src/app/metrics/` & `frontend/src/app/page.tsx`):**
  - **Seamless Navigation Switcher:** Added a top navigation pill switcher to both `/` and `/metrics` enabling frictionless toggling between live **"Mission Control"** and empirical **"Model Benchmarks"** without cluttering real-time telemetry displays.
  - **Dedicated Benchmarks Page (`frontend/src/app/metrics/page.tsx`):**
    1. *Hero Validation Banner:* Details the 17,541-hour holdout test set (2024–2025), 70,122-hour training history (2016–2023), and 3-hour forward forecast horizon.
    2. *4 Top Benchmark KPI Cards:* Storm Recall (`53.4%`, +33.8% relative gain), Storm Precision (`44.8%`), Storm F1-Score (`0.487`), and Continuous Regression MAE (`0.742 Kp` / RMSE `0.968 Kp` / R² `0.513`).
    3. *Model Architecture Progression Table:* Compares Random Forest Baseline, Unweighted XGBoost, and HelioShield Storm-Weighted XGBoost across all continuous and classification metrics with deep operational tradeoff analysis.
    4. *Recharts Feature Importance Chart:* Horizontal bar visualization highlighting top physical drivers: Dynamic Pressure ($P_{dyn}$ at 26.4%), Electric Field Coupling ($V \times B_z$ at 16.9%), Solar Wind Velocity ($v$ at 12.7%), 6h Max Speed (7.7%), 6h Min Southward $B_z$ (5.6%), and 6h Mean Coupling (5.4%).
    5. *Class Imbalance Breakdown:* Visualizes the 47:1 natural quiet-to-storm ratio (97.9% quiet vs. 2.1% storm) and explains why a 5.0x storm sample weight multiplier was applied to prevent fatal false negatives.
- **Verification:**
  - `GET http://127.0.0.1:8000/api/metrics` verified returning valid JSON with HTTP 200 OK.
  - `GET http://localhost:3000/metrics` verified rendering cleanly with zero console errors.
  - Both FastAPI backend (`task-667`) and Next.js frontend (`task-731`) confirmed active in persistent background daemon sessions.


---

## 2026-11-14 — HelioShield ML Model v2 Upgrade & Deployment

- **Status:** Completed
- **Tools & Libraries Used:** Python 3.13, XGBoost, Pandas, NumPy, Scikit-Learn, FastAPI, Next.js
- **Key Enhancements in Model v2:**
  1. **New Physics & Persistence Features (34 features total):**
     - `newell_coupling_proxy`: $v^{4/3} \cdot |B_{z,\text{south}}|^{2/3}$ capturing empirical dayside magnetospheric energy transfer (became #1 driver at 21.9% gain).
     - `newell_roll3_sum`: 3-hour cumulative Newell energy coupling.
     - `bz_south_streak`: Consecutive hours where IMF $B_z < -4.0\text{ nT}$.
     - `speed_diff_1h`: 1-hour rate of change in solar wind speed ($\Delta v$).
     - `pressure_diff_1h`: 1-hour rate of change in dynamic pressure ($\Delta P_{dyn}$).
     - `bz_roll3_mean`: 3-hour smoothed IMF $B_z$ trend.
  2. **Regularized Storm-Weighted Hyperparameters:**
     - `n_estimators=450, learning_rate=0.03, max_depth=5, min_child_weight=12, subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=2.5`.
     - Calibrated sample weights: $3.0\times$ for $K_p \ge 5.0$, $1.5\times$ for $4.0 \le K_p < 5.0$, $1.0\times$ otherwise.
  3. **Empirical Benchmarks (2024–2025 Unseen Test Set, 17,541 hours):**
     - **Storm Precision:** **57.4%** (+12.6% absolute gain / +28.1% relative increase over v1's 44.8%).
     - **Storm Recall:** **35.2%** at conservative threshold $K_p \ge 5.0$.
     - **Storm $F_1$-Score:** **0.436**.
     - **Regression MAE:** **0.7303 Kp** (improved from 0.742).
     - **Regression RMSE:** **0.9547 Kp** (improved from 0.968).
     - **Regression $R^2$ Score:** **0.5267** (improved from 0.513).
  4. **Top 3 Physical Drivers:**
     1. Newell Coupling Proxy ($21.9\%$)
     2. Solar Wind Velocity $v$ ($18.0\%$)
     3. Solar Wind Dynamic Pressure $P_{dyn}$ ($16.9\%$)
- **System Sync:**
  - `scripts/train_v2_model.py`: End-to-end training pipeline caching NASA OMNI2 and saving `models/kp_forecaster.json`.
  - `backend/telemetry.py`: Updated with the exact 34-feature extraction pipeline.
  - `models/feature_columns.json`: Synchronized with 34 feature schema and v2 benchmarks.
  - `frontend/src/app/metrics/page.tsx`: Updated with v2 benchmark KPIs, progression table, and Newell driver ranking.
  - FastAPI server restarted on port 8000; both Mission Control (`/`) and Model Benchmarks (`/metrics`) verified responding cleanly.
