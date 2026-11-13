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