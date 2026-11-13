# HelioShield — Solar Storm & Satellite Disruption Forecaster

> **A hybrid Machine Learning & Applied AI space-weather platform** that forecasts geomagnetic storms ($K_p$ Index) from solar wind telemetry and translates real-time satellite feeds into plain-English disruption alerts for aviation, power grids, satellites, and aurora watchers.

---

## Why HelioShield?
Solar storms (Coronal Mass Ejections and high-speed solar wind streams) can trigger severe geomagnetic storms capable of degrading GPS accuracy, inducing power-grid currents, increasing Low-Earth Orbit (LEO) satellite drag, and disrupting high-frequency radio communications. 

Most space-weather dashboards dump raw physics charts ($B_z$, proton density, solar wind speed) that are hard to interpret. **HelioShield** bridges that gap with a two-stage intelligence pipeline:
1. **Classical ML Forecasting Engine:** Predicts upcoming geomagnetic storm intensity ($K_p$ Index) using historical solar wind and geomagnetic records.
2. **Applied AI Impact Explainer:** Translates live telemetry and model predictions into actionable, human-readable sector risk briefings.

---

## System Architecture

```text
[Historical NASA / GFZ Data] ──► [Google Colab: Cleaning & Feature Eng.] ──► [Trained XGBoost Model (.json/.pkl)]
                                                                                          │
[Live NOAA SWPC Satellite JSON] ───────────────────────► [FastAPI Backend] ◄──────────────┘
                                                                │
                                                  [LLM Risk Briefing Agent]
                                                                │
                                                 [React / Next.js Dashboard]

