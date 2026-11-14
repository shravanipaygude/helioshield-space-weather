"""
backend/ai_explainer.py - AI Reasoning Layer for HelioShield.

Translates real-time solar wind telemetry and ML Kp forecasts into actionable,
sector-specific operational disruption briefings.

Uses google-genai (gemini-2.5-flash) if GEMINI_API_KEY is configured in .env,
with a physics-informed rule-based fallback when no key is set or upon API errors.
Returns structured JSON briefings covering headline, summary, and sector assessments
for:
- Satellites (LEO orbital drag, GEO surface charging, attitude control)
- GPS & Aviation (GNSS scintillation, polar HF radio blackouts)
- Power Grids (GIC induction, transformer saturation, voltage stability)
- Aurora (visibility, latitude range, display activity)
"""

import json
import logging
import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

logger = logging.getLogger("helioshield.ai_explainer")


def get_noaa_storm_scale(kp: float) -> tuple[str, str]:
    """Map Kp index (0.0 - 9.0) to NOAA Space Weather Storm Scales."""
    if kp >= 9.0:
        return "G5", "Extreme Geomagnetic Superstorm"
    elif kp >= 8.0:
        return "G4", "Severe Geomagnetic Storm"
    elif kp >= 7.0:
        return "G3", "Strong Geomagnetic Storm"
    elif kp >= 6.0:
        return "G2", "Moderate Geomagnetic Storm"
    elif kp >= 5.0:
        return "G1", "Minor Geomagnetic Storm"
    elif kp >= 4.0:
        return "G0+", "Unsettled / Active Magnetosphere (Watch Threshold)"
    else:
        return "G0", "Quiet Magnetospheric Conditions"


def generate_rule_based_briefing(
    predicted_kp: float,
    horizon_hours: int,
    telemetry: Dict[str, Any],
    storm_alert: bool
) -> Dict[str, Any]:
    """
    Generate a high-fidelity, physics-informed space weather briefing
    without requiring external LLM API calls.
    """
    noaa_scale, storm_category = get_noaa_storm_scale(predicted_kp)
    bz = telemetry.get("bz_gsm", 0.0)
    bz_south = telemetry.get("bz_south", min(bz, 0.0))
    speed = telemetry.get("wind_speed", 400.0)
    density = telemetry.get("proton_density", 4.0)
    pressure = telemetry.get("dynamic_pressure", 2.0)
    coupling = telemetry.get("electric_field_coupling", 0.0)

    # Southward magnetic field diagnosis
    if bz < -5.0:
        bz_desc = f"Strong southward IMF orientation (Bz = {bz:.1f} nT) is driving aggressive magnetic reconnection at Earth's dayside magnetopause"
    elif bz < 0.0:
        bz_desc = f"Modest southward IMF component (Bz = {bz:.1f} nT) allows steady solar wind energy transfer into the magnetosphere"
    else:
        bz_desc = f"Northward IMF orientation (Bz = +{bz:.1f} nT) provides a protective magnetic shield, repelling major plasma injection"

    # Dynamic pressure diagnosis
    if pressure > 5.0:
        pressure_desc = f"Elevated dynamic pressure ({pressure:.2f} nPa) is compressing the magnetopause boundary towards geostationary orbits."
    else:
        pressure_desc = f"Solar wind dynamic pressure is nominal at {pressure:.2f} nPa."

    # Sector risk logic
    if predicted_kp >= 7.0:
        # G3 - G5
        headline = f"CRITICAL ALERT: {noaa_scale} {storm_category} Predicted Within {horizon_hours} Hours"
        summary = (
            f"HelioShield ML models forecast severe geomagnetic disturbance (Kp {predicted_kp:.2f}, {noaa_scale}) "
            f"in {horizon_hours} hours. {bz_desc}. {pressure_desc} "
            f"Critical defensive countermeasures recommended across orbital, grid, and navigation infrastructure."
        )
        sectors = {
            "satellites": {
                "risk_level": "Critical" if predicted_kp >= 8.0 else "High",
                "impact": (
                    f"Intense thermospheric heating will produce severe neutral atmospheric density surges at 300–600 km LEO, "
                    f"multiplying orbital drag up to 300%. High-energy electron injections threaten GEO spacecraft with "
                    f"deep internal dielectric charging and single-event latchups. Star tracker optical blinding likely."
                ),
                "recommendation": (
                    "Initiate orbital decay trajectory corrections; adjust collision avoidance covariance ellipsoids; "
                    "orient solar arrays to minimum drag configuration; place sensitive payload instruments in safe standby mode."
                )
            },
            "gps_aviation": {
                "risk_level": "Critical" if predicted_kp >= 8.0 else "High",
                "impact": (
                    f"Severe ionospheric total electron content (TEC) gradients will induce widespread L-band GNSS scintillation. "
                    f"Single-frequency GPS positioning errors may exceed 20–50 meters. Complete polar HF radio blackouts expected "
                    f"along trans-polar aviation airways due to energetic particle precipitation in the auroral oval."
                ),
                "recommendation": (
                    "Reroute trans-polar commercial flights south of 60° latitude to maintain line-of-sight VHF and satellite comms; "
                    "enforce inertial navigation system (INS) reversion protocols for precision approach aviation."
                )
            },
            "power_grids": {
                "risk_level": "Critical" if predicted_kp >= 8.0 else "High",
                "impact": (
                    f"Rapid rate of magnetic field variation (dB/dt) will induce large Geomagnetically Induced Currents (GICs) "
                    f"in high-latitude and coastal electrical transmission systems. Potential half-cycle transformer core saturation, "
                    f"elevated reactive power consumption, and protective relay false trips."
                ),
                "recommendation": (
                    "Power transmission operators should shed non-critical transformer loads, disconnect series capacitors at risk, "
                    "maximize dynamic reactive power reserves (SVC/STATCOM), and monitor transformer neutral ground currents continuously."
                )
            },
            "aurora": {
                "visibility": "Widespread / Exceptional",
                "latitude_range": "Down to 40°–45° geomagnetic latitude (Northern US, Central Europe, Southern Australia)",
                "activity": (
                    f"Historic auroral storm displays with fast-moving coronal curtains, multi-hued red/violet nitrogen emissions, "
                    f"and overhead pulsating corona visible well into mid-latitude population centers under clear dark skies."
                )
            }
        }

    elif predicted_kp >= 5.0:
        # G1 - G2
        headline = f"STORM WARNING: {noaa_scale} {storm_category} Expected in {horizon_hours} Hours"
        summary = (
            f"HelioShield predicts a {noaa_scale} storm event peaking at Kp {predicted_kp:.2f} within {horizon_hours} hours. "
            f"{bz_desc}. Solar wind stream velocity is {speed:.0f} km/s. "
            f"Operators should monitor satellite attitude telemetry and high-latitude transmission lines."
        )
        sectors = {
            "satellites": {
                "risk_level": "Moderate" if predicted_kp < 6.0 else "Elevated",
                "impact": (
                    f"Moderate expansion of upper thermosphere increases orbital drag on low-perigee LEO satellites (Starlink, ISS). "
                    f"Mild surface charging possible on nightside geostationary assets during substorm injections."
                ),
                "recommendation": (
                    "Update space situational awareness (SSA) ephemerides more frequently; verify reaction wheel saturation margins; "
                    "monitor telemetry downlink bit-error rates for anomalous single-event upsets."
                )
            },
            "gps_aviation": {
                "risk_level": "Moderate" if predicted_kp < 6.0 else "Elevated",
                "impact": (
                    f"Ionospheric scintillation and phase jitter on GPS L1/L2 frequencies across sub-auroral zones. "
                    f"HF aviation communication degradation over northern latitudes above 60°."
                ),
                "recommendation": (
                    "Utilize multi-constellation GNSS (GPS + Galileo) and satellite-based augmentation systems (WAAS/EGNOS) "
                    "to dampen pseudorange errors. Keep secondary satellite communication links on standby for polar routes."
                )
            },
            "power_grids": {
                "risk_level": "Moderate",
                "impact": (
                    f"Weak-to-moderate GIC currents may register on high-latitude transmission grids (e.g. Hydro-Québec, Nordics). "
                    f"Minor harmonic distortions and voltage fluctuations on long-haul EHV lines."
                ),
                "recommendation": (
                    "Dispatch grid engineers to inspect transformer neutral telemetry; maintain contingency generation spinning reserves; "
                    "verify SCADA stability margins."
                )
            },
            "aurora": {
                "visibility": "High / Active",
                "latitude_range": "50°–58° geomagnetic latitude (Canada, Alaska, Northern Tier US, Scotland, Scandinavia)",
                "activity": (
                    f"Dynamic green oxygen auroral arcs with bright westward-traveling surges and rayed bands active during peak substorms."
                )
            }
        }

    elif predicted_kp >= 4.0:
        # G0+ Unsettled / Elevated Watch
        headline = f"SPACE WEATHER WATCH: Unsettled Magnetosphere (Kp {predicted_kp:.2f}) Predicted"
        summary = (
            f"Magnetospheric activity is trending unsettled with Kp predicted to reach {predicted_kp:.2f} in {horizon_hours} hours. "
            f"{bz_desc}. Current solar wind speed is {speed:.0f} km/s. Conditions are below major storm thresholds but warrant baseline operational awareness."
        )
        sectors = {
            "satellites": {
                "risk_level": "Low",
                "impact": "Nominal orbit decay rates. Negligible spacecraft surface charging.",
                "recommendation": "Standard operational routines. Continue routine orbit determination tracking."
            },
            "gps_aviation": {
                "risk_level": "Low",
                "impact": "Localized minor ionospheric delays at high geomagnetic latitudes. Civil GPS accuracy within nominal ±3m.",
                "recommendation": "No commercial aviation restrictions. Dual-frequency GNSS operations nominal."
            },
            "power_grids": {
                "risk_level": "Low",
                "impact": "No significant ground-induced currents detected. Transformer neutrals stable.",
                "recommendation": "No mitigation required. System operating under normal contingency criteria."
            },
            "aurora": {
                "visibility": "Moderate",
                "latitude_range": "Poleward of 62° geomagnetic latitude (Northern Canada, Northern Scandinavia, Iceland)",
                "activity": "Faint green auroral glows and steady quiet arcs near the auroral oval."
            }
        }

    else:
        # G0 Quiet
        headline = f"NOMINAL CONDITIONS: Geomagnetically Quiet (Kp {predicted_kp:.2f})"
        summary = (
            f"Space weather remains quiet with 3-hour Kp forecast at {predicted_kp:.2f}. "
            f"Interplanetary magnetic field and solar wind plasma velocity ({speed:.0f} km/s) remain at tranquil baseline levels."
        )
        sectors = {
            "satellites": {
                "risk_level": "Low",
                "impact": "Minimal atmospheric drag; nominal orbital decay and attitude control.",
                "recommendation": "All spacecraft operating within baseline parameters."
            },
            "gps_aviation": {
                "risk_level": "Low",
                "impact": "Quiet ionosphere. Maximum satellite navigation precision globally.",
                "recommendation": "Standard navigation and communications in effect."
            },
            "power_grids": {
                "risk_level": "Low",
                "impact": "Zero geomagnetic disturbances on electrical transmission infrastructure.",
                "recommendation": "Nominal power grid operations."
            },
            "aurora": {
                "visibility": "Low",
                "latitude_range": "Confined to far polar latitudes (> 68°)",
                "activity": "Minimal auroral activity, faint diffuse oval emission."
            }
        }

    return {
        "headline": headline,
        "summary": summary,
        "sectors": sectors,
        "ai_source": "Physics-Based Operational Reasoner (Rule-Based Fallback)",
        "model_version": "HelioShield-RuleEngine-v1.0"
    }


def generate_ai_briefing(
    predicted_kp: float,
    horizon_hours: int,
    telemetry: Dict[str, Any],
    storm_alert: bool
) -> Dict[str, Any]:
    """
    Generate structured JSON space weather briefing using google-genai
    (gemini-2.5-flash) if GEMINI_API_KEY is available, or fall back to
    rule-based generator.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    noaa_scale, storm_category = get_noaa_storm_scale(predicted_kp)

    if not api_key:
        logger.info("No GEMINI_API_KEY found in environment. Using rule-based operational briefing.")
        return generate_rule_based_briefing(predicted_kp, horizon_hours, telemetry, storm_alert)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        bz_gsm = telemetry.get("bz_gsm", 0.0)
        bz_south = telemetry.get("bz_south", min(bz_gsm, 0.0))
        wind_speed = telemetry.get("wind_speed", 400.0)
        proton_density = telemetry.get("proton_density", 4.0)
        dynamic_pressure = telemetry.get("dynamic_pressure", 2.0)
        coupling = telemetry.get("electric_field_coupling", 0.0)

        prompt = f"""You are HelioShield's Operational Space Weather Forecaster.
Analyze this real-time space weather observation and ML model forecast to produce an authoritative, actionable disruption briefing for aerospace, navigation, and power utility engineers.

OBSERVED REAL-TIME SOLAR WIND METRICS:
- Interplanetary Magnetic Field Bz (GSM): {bz_gsm:.2f} nT (Southward component: {bz_south:.2f} nT)
- Solar Wind Velocity: {wind_speed:.1f} km/s
- Proton Density: {proton_density:.2f} p/cm³
- Solar Wind Dynamic Pressure: {dynamic_pressure:.3f} nPa
- Motional Electric Field Coupling: {coupling:.1f} mV/m

ML MODEL PREDICTION:
- Forecasted Kp Index (in {horizon_hours} hours): {predicted_kp:.2f}
- NOAA Geomagnetic Storm Scale: {noaa_scale} ({storm_category})
- Storm Alert Active: {storm_alert}

REQUIRED RESPONSE:
Output ONLY a single valid raw JSON object (strictly no markdown formatting, no ```json code fences) with the exact structure:
{{
  "headline": "<punchy, professional alert headline>",
  "summary": "<2-3 sentence overview covering solar wind physical coupling, forecasted storm timeline, and operational severity>",
  "sectors": {{
    "satellites": {{
      "risk_level": "Low" | "Moderate" | "Elevated" | "High" | "Critical",
      "impact": "<specific impact on LEO thermospheric drag, GEO surface charging, single-event latchups, star trackers>",
      "recommendation": "<actionable mitigation for spacecraft operators>"
    }},
    "gps_aviation": {{
      "risk_level": "Low" | "Moderate" | "Elevated" | "High" | "Critical",
      "impact": "<specific impact on GNSS L-band scintillation, positioning errors, polar HF aviation blackouts>",
      "recommendation": "<actionable mitigation for pilots and dispatchers>"
    }},
    "power_grids": {{
      "risk_level": "Low" | "Moderate" | "Elevated" | "High" | "Critical",
      "impact": "<specific impact on Geomagnetically Induced Currents (GICs), transformer core saturation, harmonics>",
      "recommendation": "<actionable mitigation for transmission system operators>"
    }},
    "aurora": {{
      "visibility": "Low" | "Moderate" | "High" | "Widespread",
      "latitude_range": "<expected geographic and geomagnetic latitudes of visibility>",
      "activity": "<description of auroral features and viewing conditions>"
    }}
  }}
}}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )

        response_text = response.text.strip()
        # Clean potential markdown wrapping if present
        if response_text.startswith("```"):
            response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
            response_text = re.sub(r"\s*```$", "", response_text)

        briefing = json.loads(response_text)
        briefing["ai_source"] = "Gemini 2.5 Flash (Live Generative AI)"
        briefing["model_version"] = "gemini-2.5-flash"

        # Validate core fields
        if "headline" in briefing and "summary" in briefing and "sectors" in briefing:
            return briefing

        logger.warning("Gemini response missing expected keys. Falling back to rule-based engine.")
        return generate_rule_based_briefing(predicted_kp, horizon_hours, telemetry, storm_alert)

    except Exception as e:
        logger.warning("Error invoking Gemini 2.5 Flash: %s. Using rule-based fallback.", e)
        fallback = generate_rule_based_briefing(predicted_kp, horizon_hours, telemetry, storm_alert)
        fallback["ai_error_note"] = f"Gemini invocation failed ({str(e)}). Physics-based rule engine was engaged."
        return fallback

