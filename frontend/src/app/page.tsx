'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import {
  ShieldAlert,
  Activity,
  Wind,
  Compass,
  Zap,
  Satellite,
  Plane,
  Eye,
  RefreshCw,
  AlertTriangle,
  Radio,
  Cpu,
  Clock,
  ChevronRight,
  TrendingUp,
  BarChart3
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend
} from 'recharts';

interface ForecastResponse {
  forecast: {
    predicted_kp: number;
    horizon_hours: number;
    alert_threshold: number;
    storm_alert: boolean;
    noaa_scale: string;
    storm_category: string;
    target_timestamp: string;
    confidence_band: {
      lower: number;
      upper: number;
    };
  };
  current_telemetry: {
    timestamp: string;
    bz_gsm: number;
    bt: number;
    wind_speed: number;
    proton_density: number;
    dynamic_pressure: number;
    electric_field_coupling: number;
    bz_south: number;
  };
  ai_briefing: {
    headline: string;
    summary: string;
    sectors: {
      satellites: {
        risk_level: string;
        impact: string;
        recommendation: string;
      };
      gps_aviation: {
        risk_level: string;
        impact: string;
        recommendation: string;
      };
      power_grids: {
        risk_level: string;
        impact: string;
        recommendation: string;
      };
      aurora: {
        visibility: string;
        latitude_range: string;
        activity: string;
      };
    };
    ai_source: string;
    model_version?: string;
  };
  history_24h: Array<{
    timestamp: string;
    bz_gsm: number;
    bt: number;
    wind_speed: number;
    proton_density: number;
    dynamic_pressure: number;
    electric_field_coupling: number;
    kp_estimate?: number | null;
  }>;
  meta: {
    is_live_feed: boolean;
    model_version: string;
    feature_count: number;
    generated_at: string;
  };
}

export default function SpaceWeatherDashboard() {
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [simulateStorm, setSimulateStorm] = useState<boolean>(false);
  const [utcTime, setUtcTime] = useState<string>('');
  const [isMounted, setIsMounted] = useState<boolean>(false);

  // Live UTC Clock
  useEffect(() => {
    setIsMounted(true);
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().replace('GMT', 'UTC'));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  // Fetch forecast data from FastAPI backend
  const fetchData = useCallback(async (isManualRefresh = false) => {
    if (isManualRefresh) setRefreshing(true);
    setError(null);
    try {
      const queryParam = simulateStorm ? '?simulate_storm=true' : '';
      const response = await fetch(`http://127.0.0.1:8000/api/forecast${queryParam}`, {
        cache: 'no-store'
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const json: ForecastResponse = await response.json();
      setData(json);
    } catch (err: unknown) {
      console.error('Failed to fetch space weather telemetry:', err);
      setError(err instanceof Error ? err.message : 'Unable to connect to HelioShield backend at http://127.0.0.1:8000');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [simulateStorm]);

  useEffect(() => {
    fetchData();
    // Auto-refresh telemetry every 60 seconds
    const interval = setInterval(() => {
      fetchData(false);
    }, 60000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Color helper for NOAA scales
  const getScaleTheme = (scale: string) => {
    switch (scale) {
      case 'G5':
        return {
          bg: 'bg-purple-950/60 border-purple-500/50 text-purple-300',
          badge: 'bg-purple-600 text-white',
          glow: 'shadow-[0_0_25px_rgba(168,85,247,0.35)]',
          bar: 'bg-gradient-to-r from-purple-500 to-pink-500'
        };
      case 'G4':
        return {
          bg: 'bg-red-950/60 border-red-500/50 text-red-300',
          badge: 'bg-red-600 text-white',
          glow: 'shadow-[0_0_25px_rgba(239,68,68,0.35)]',
          bar: 'bg-gradient-to-r from-red-600 to-rose-500'
        };
      case 'G3':
        return {
          bg: 'bg-rose-950/60 border-rose-500/50 text-rose-300',
          badge: 'bg-rose-500 text-white',
          glow: 'shadow-[0_0_20px_rgba(244,63,94,0.3)]',
          bar: 'bg-gradient-to-r from-orange-500 to-rose-500'
        };
      case 'G2':
        return {
          bg: 'bg-orange-950/50 border-orange-500/40 text-orange-300',
          badge: 'bg-orange-500 text-white',
          glow: 'shadow-[0_0_15px_rgba(249,115,22,0.25)]',
          bar: 'bg-gradient-to-r from-amber-500 to-orange-500'
        };
      case 'G1':
        return {
          bg: 'bg-amber-950/40 border-amber-500/40 text-amber-300',
          badge: 'bg-amber-500 text-slate-950 font-bold',
          glow: 'shadow-[0_0_15px_rgba(245,158,11,0.2)]',
          bar: 'bg-gradient-to-r from-yellow-500 to-amber-500'
        };
      default:
        return {
          bg: 'bg-emerald-950/30 border-emerald-500/30 text-emerald-300',
          badge: 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40',
          glow: 'shadow-[0_0_15px_rgba(16,185,129,0.15)]',
          bar: 'bg-gradient-to-r from-emerald-500 to-cyan-400'
        };
    }
  };

  const getRiskBadge = (level: string) => {
    const l = level.toLowerCase();
    if (l.includes('critical') || l.includes('extreme')) {
      return 'bg-purple-500/20 text-purple-400 border-purple-500/40';
    }
    if (l.includes('high') || l.includes('severe') || l.includes('widespread')) {
      return 'bg-rose-500/20 text-rose-400 border-rose-500/40';
    }
    if (l.includes('elevated') || l.includes('moderate')) {
      return 'bg-amber-500/20 text-amber-400 border-amber-500/40';
    }
    return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40';
  };

  const formatChartTime = (isoStr: string) => {
    try {
      const d = new Date(isoStr);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
    } catch {
      return isoStr.slice(11, 16);
    }
  };

  const formatTargetDate = (isoStr: string) => {
    try {
      const d = new Date(isoStr);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false }) + ' UTC';
    } catch {
      return isoStr;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Command Header */}
      <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Brand & Mission Title */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-indigo-600 to-purple-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 border border-cyan-400/30">
                <ShieldAlert className="w-5 h-5 text-white" />
              </div>
              <span className="absolute -top-1 -right-1 flex h-3 w-3">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${simulateStorm ? 'bg-rose-400' : 'bg-emerald-400'}`}></span>
                <span className={`relative inline-flex rounded-full h-3 w-3 ${simulateStorm ? 'bg-rose-500' : 'bg-emerald-500'}`}></span>
              </span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-slate-100 to-indigo-300">
                  HELIOSHIELD
                </h1>
                <span className="text-[10px] uppercase font-mono tracking-widest px-2 py-0.5 rounded bg-slate-800/90 text-cyan-400 border border-cyan-500/20">
                  v3.0-ML
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono tracking-tight">
                SPACE WEATHER MISSION CONTROL &bull; NOAA L1 SATELLITE RADAR
              </p>
            </div>
          </div>

          {/* Navigation Pill Switcher */}
          <nav aria-label="Dashboard views" className="flex items-center bg-slate-900/90 p-1 rounded-xl border border-slate-800 shadow-inner">
            <Link
              href="/"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm"
            >
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              Mission Control
            </Link>
            <Link
              href="/metrics"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors"
            >
              <BarChart3 className="w-3.5 h-3.5 text-slate-400" />
              Model Benchmarks
            </Link>
          </nav>

          {/* Center: Live UTC Clock & Telemetry Status */}
          <div className="flex items-center gap-3 text-xs font-mono">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
              <Clock className="w-3.5 h-3.5 text-cyan-400" />
              <span>{isMounted ? utcTime : 'SYNCING UTC...'}</span>
            </div>

            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
              <Radio className={`w-3.5 h-3.5 ${data?.meta.is_live_feed && !simulateStorm ? 'text-emerald-400 animate-pulse' : 'text-amber-400'}`} />
              <span className="text-slate-300">
                {simulateStorm ? 'SIMULATION INJECTION' : data?.meta.is_live_feed ? 'LIVE NOAA RTSW' : 'STANDBY CACHE'}
              </span>
            </div>
          </div>

          {/* Right: Controls & Interactive Mode Toggle */}
          <div className="flex items-center gap-3">
            {/* Mode Toggle: Live NOAA vs Simulate Storm */}
            <div className="flex items-center bg-slate-900 p-1 rounded-xl border border-slate-800">
              <button
                onClick={() => setSimulateStorm(false)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  !simulateStorm
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Live NOAA
              </button>
              <button
                onClick={() => setSimulateStorm(true)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                  simulateStorm
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm animate-pulse'
                    : 'text-slate-400 hover:text-rose-400'
                }`}
              >
                <AlertTriangle className="w-3 h-3" />
                Simulate Storm
              </button>
            </div>

            {/* Manual Refresh Button */}
            <button
              onClick={() => fetchData(true)}
              disabled={refreshing}
              className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 transition-all disabled:opacity-50"
              title="Refresh Telemetry"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin text-cyan-400' : ''}`} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Mission Control Grid */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Error Notification Banner */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-500/40 text-rose-200 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
              <div>
                <p className="font-semibold text-sm">Telemetry Connection Alert</p>
                <p className="text-xs text-rose-300/90">{error}</p>
              </div>
            </div>
            <button
              onClick={() => fetchData(true)}
              className="px-3 py-1 bg-rose-600/30 hover:bg-rose-600/50 text-xs rounded border border-rose-500/40 transition"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* Storm Scenario Banner */}
        {simulateStorm && (
          <div className="p-3.5 rounded-xl bg-gradient-to-r from-rose-950/80 via-amber-950/50 to-slate-900 border border-rose-500/40 flex items-center justify-between shadow-lg shadow-rose-950/30">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-rose-500/20 text-rose-400">
                <AlertTriangle className="w-5 h-5 animate-bounce" />
              </div>
              <div>
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-rose-400">
                  CRITICAL TEST SCENARIO ENGAGED
                </span>
                <p className="text-xs text-slate-300">
                  Simulating extreme Coronal Mass Ejection impact: Southward Bz = -14.2 nT, Wind Speed = 720 km/s, High Plasma Density = 18.5 p/cm³.
                </p>
              </div>
            </div>
            <button
              onClick={() => setSimulateStorm(false)}
              className="text-xs text-cyan-400 hover:underline font-mono"
            >
              Return to Live NOAA Feed &rarr;
            </button>
          </div>
        )}

        {/* Loading State Skeleton */}
        {loading && !data && (
          <div className="flex flex-col items-center justify-center py-24 space-y-4">
            <div className="relative w-16 h-16">
              <div className="w-16 h-16 rounded-full border-4 border-cyan-500/20 border-t-cyan-400 animate-spin"></div>
              <ShieldAlert className="w-8 h-8 text-cyan-400 absolute inset-0 m-auto animate-pulse" />
            </div>
            <p className="text-sm font-mono text-cyan-400 animate-pulse">
              INITIALIZING NOAA DSCOVR L1 STREAM &amp; XGBOOST MODEL...
            </p>
          </div>
        )}

        {data && (
          <>
            {/* HERO ROW: 3-Hour Kp Forecast Gauge + 4 Live L1 Telemetry Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left Column: 3-Hour Kp Forecast Gauge (5 cols) */}
              <div className={`lg:col-span-5 rounded-2xl p-6 border transition-all relative overflow-hidden flex flex-col justify-between ${getScaleTheme(data.forecast.noaa_scale).bg} ${getScaleTheme(data.forecast.noaa_scale).glow}`}>
                {/* Background Radar Watermark */}
                <div className="absolute -right-8 -bottom-8 opacity-5 pointer-events-none">
                  <ShieldAlert className="w-64 h-64 text-white" />
                </div>

                <div>
                  {/* Gauge Header */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-cyan-400" />
                      <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300">
                        {data.forecast.horizon_hours}-Hour Kp Forecast
                      </span>
                    </div>
                    <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-bold border ${getScaleTheme(data.forecast.noaa_scale).badge}`}>
                      {data.forecast.noaa_scale} STORM SCALE
                    </span>
                  </div>

                  {/* Main Metric Display */}
                  <div className="flex items-baseline gap-4 mb-2">
                    <span className="text-6xl font-extrabold font-mono tracking-tight text-white">
                      {data.forecast.predicted_kp.toFixed(2)}
                    </span>
                    <span className="text-sm text-slate-400 font-mono">
                      / 9.00 Kp
                    </span>
                  </div>

                  {/* Storm Category Text */}
                  <p className="text-sm font-semibold text-slate-200 mb-4">
                    {data.forecast.storm_category}
                  </p>

                  {/* Kp Visual Scale Progress Bar */}
                  <div className="space-y-1.5 mb-6">
                    <div className="flex justify-between text-[11px] font-mono text-slate-400">
                      <span>0.0 (Quiet)</span>
                      <span className="text-rose-400 font-bold">4.5 (Alert Threshold)</span>
                      <span>9.0 (G5 Extreme)</span>
                    </div>
                    <div className="relative w-full h-3 bg-slate-900/90 rounded-full overflow-hidden border border-slate-700/60">
                      {/* Gradient Fill */}
                      <div
                        className={`h-full transition-all duration-700 ${getScaleTheme(data.forecast.noaa_scale).bar}`}
                        style={{ width: `${Math.min(100, (data.forecast.predicted_kp / 9) * 100)}%` }}
                      ></div>
                      {/* Threshold Tick at 4.5/9.0 = 50% */}
                      <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-rose-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]"></div>
                    </div>
                  </div>
                </div>

                {/* Sub-Metadata Grid */}
                <div className="pt-4 border-t border-slate-800/80 grid grid-cols-2 gap-3 text-xs font-mono">
                  <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/60">
                    <span className="text-slate-400 block text-[10px]">Target Horizon</span>
                    <span className="text-slate-200 font-semibold">{formatTargetDate(data.forecast.target_timestamp)}</span>
                  </div>
                  <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/60">
                    <span className="text-slate-400 block text-[10px]">Confidence Band (MAE ±0.74)</span>
                    <span className="text-cyan-300 font-semibold">
                      {data.forecast.confidence_band.lower} – {data.forecast.confidence_band.upper} Kp
                    </span>
                  </div>
                </div>
              </div>

              {/* Right Column: 4 Live L1 Satellite Telemetry Cards (7 cols) */}
              <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* 1. Solar Wind Speed */}
                <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 flex flex-col justify-between hover:border-cyan-500/40 transition-all shadow-sm">
                  <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-xs font-mono uppercase tracking-wider">Solar Wind Speed</span>
                    <Wind className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div className="flex items-baseline gap-2 my-1">
                    <span className="text-3xl font-extrabold font-mono text-slate-100">
                      {data.current_telemetry.wind_speed.toFixed(1)}
                    </span>
                    <span className="text-xs text-slate-400 font-mono">km/s</span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-400 flex items-center justify-between pt-2 border-t border-slate-800/60">
                    <span>Status</span>
                    <span className={data.current_telemetry.wind_speed > 550 ? 'text-rose-400 font-semibold' : 'text-emerald-400 font-semibold'}>
                      {data.current_telemetry.wind_speed > 550 ? 'High-Speed Stream' : 'Nominal Velocity'}
                    </span>
                  </div>
                </div>

                {/* 2. IMF Bz GSM (Highlighted Red when Southward) */}
                <div className={`rounded-2xl p-5 border flex flex-col justify-between transition-all shadow-sm ${
                  data.current_telemetry.bz_gsm < 0
                    ? 'bg-rose-950/30 border-rose-500/50 shadow-[0_0_15px_rgba(244,63,94,0.15)]'
                    : 'bg-slate-900/60 border-slate-800/80 hover:border-cyan-500/40'
                }`}>
                  <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-xs font-mono uppercase tracking-wider">IMF Bz Magnetic Field</span>
                    <Compass className={`w-4 h-4 ${data.current_telemetry.bz_gsm < 0 ? 'text-rose-400 animate-pulse' : 'text-cyan-400'}`} />
                  </div>
                  <div className="flex items-baseline gap-2 my-1">
                    <span className={`text-3xl font-extrabold font-mono ${data.current_telemetry.bz_gsm < 0 ? 'text-rose-400' : 'text-cyan-300'}`}>
                      {data.current_telemetry.bz_gsm > 0 ? `+${data.current_telemetry.bz_gsm.toFixed(2)}` : data.current_telemetry.bz_gsm.toFixed(2)}
                    </span>
                    <span className="text-xs text-slate-400 font-mono">nT (GSM)</span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-400 flex items-center justify-between pt-2 border-t border-slate-800/60">
                    <span>Orientation</span>
                    <span className={data.current_telemetry.bz_gsm < 0 ? 'text-rose-400 font-bold' : 'text-cyan-400'}>
                      {data.current_telemetry.bz_gsm < 0 ? 'Southward (Reconnecting)' : 'Northward (Shielding)'}
                    </span>
                  </div>
                </div>

                {/* 3. Proton Density */}
                <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 flex flex-col justify-between hover:border-cyan-500/40 transition-all shadow-sm">
                  <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-xs font-mono uppercase tracking-wider">Proton Density</span>
                    <Activity className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div className="flex items-baseline gap-2 my-1">
                    <span className="text-3xl font-extrabold font-mono text-slate-100">
                      {data.current_telemetry.proton_density.toFixed(2)}
                    </span>
                    <span className="text-xs text-slate-400 font-mono">p/cm³</span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-400 flex items-center justify-between pt-2 border-t border-slate-800/60">
                    <span>Total Field Bt</span>
                    <span className="text-slate-300">{data.current_telemetry.bt.toFixed(2)} nT</span>
                  </div>
                </div>

                {/* 4. Solar Wind Dynamic Pressure */}
                <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 flex flex-col justify-between hover:border-cyan-500/40 transition-all shadow-sm">
                  <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-xs font-mono uppercase tracking-wider">Dynamic Pressure</span>
                    <Zap className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div className="flex items-baseline gap-2 my-1">
                    <span className="text-3xl font-extrabold font-mono text-slate-100">
                      {data.current_telemetry.dynamic_pressure.toFixed(3)}
                    </span>
                    <span className="text-xs text-slate-400 font-mono">nPa</span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-400 flex items-center justify-between pt-2 border-t border-slate-800/60">
                    <span>Electric Field Coupling</span>
                    <span className="text-slate-300">{data.current_telemetry.electric_field_coupling.toFixed(1)} mV/m</span>
                  </div>
                </div>
              </div>
            </div>

            {/* AI OPERATIONAL RISK BRIEFING PANEL */}
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 relative overflow-hidden space-y-6">
              {/* Header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                    <Cpu className="w-5 h-5" />
                  </div>
                  <div>
                    <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                      AI OPERATIONAL DISRUPTION BRIEFING
                    </h2>
                    <p className="text-xs text-slate-400 font-mono">
                      Domain-specific infrastructure impact assessment &amp; tactical countermeasures
                    </p>
                  </div>
                </div>

                {/* Source Pill */}
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-mono px-3 py-1 rounded-full bg-slate-800/90 text-cyan-300 border border-cyan-500/30 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping"></span>
                    {data.ai_briefing.ai_source}
                  </span>
                </div>
              </div>

              {/* Headline & Synthesis Overview */}
              <div className="bg-slate-950/70 rounded-xl p-4 border border-slate-800/80 space-y-2">
                <h3 className="text-sm font-bold text-cyan-300 tracking-wide flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 text-cyan-400" />
                  {data.ai_briefing.headline}
                </h3>
                <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                  {data.ai_briefing.summary}
                </p>
              </div>

              {/* 4 Sector Cards Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* 1. Satellites */}
                <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between hover:border-slate-700 transition">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-slate-200">
                        <Satellite className="w-4 h-4 text-indigo-400" />
                        <span className="text-xs font-bold uppercase tracking-wider">Satellites (LEO/GEO)</span>
                      </div>
                      <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${getRiskBadge(data.ai_briefing.sectors.satellites.risk_level)}`}>
                        {data.ai_briefing.sectors.satellites.risk_level}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Impact</span>
                      <p className="text-xs text-slate-300 leading-normal">{data.ai_briefing.sectors.satellites.impact}</p>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-slate-800/70">
                    <span className="text-[10px] font-mono uppercase text-cyan-400 block mb-1">Recommendation</span>
                    <p className="text-xs text-slate-400 leading-normal">{data.ai_briefing.sectors.satellites.recommendation}</p>
                  </div>
                </div>

                {/* 2. GPS & Aviation */}
                <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between hover:border-slate-700 transition">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-slate-200">
                        <Plane className="w-4 h-4 text-cyan-400" />
                        <span className="text-xs font-bold uppercase tracking-wider">GPS &amp; Aviation</span>
                      </div>
                      <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${getRiskBadge(data.ai_briefing.sectors.gps_aviation.risk_level)}`}>
                        {data.ai_briefing.sectors.gps_aviation.risk_level}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Impact</span>
                      <p className="text-xs text-slate-300 leading-normal">{data.ai_briefing.sectors.gps_aviation.impact}</p>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-slate-800/70">
                    <span className="text-[10px] font-mono uppercase text-cyan-400 block mb-1">Recommendation</span>
                    <p className="text-xs text-slate-400 leading-normal">{data.ai_briefing.sectors.gps_aviation.recommendation}</p>
                  </div>
                </div>

                {/* 3. Power Grids */}
                <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between hover:border-slate-700 transition">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-slate-200">
                        <Zap className="w-4 h-4 text-amber-400" />
                        <span className="text-xs font-bold uppercase tracking-wider">Power Grids (GIC)</span>
                      </div>
                      <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${getRiskBadge(data.ai_briefing.sectors.power_grids.risk_level)}`}>
                        {data.ai_briefing.sectors.power_grids.risk_level}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Impact</span>
                      <p className="text-xs text-slate-300 leading-normal">{data.ai_briefing.sectors.power_grids.impact}</p>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-slate-800/70">
                    <span className="text-[10px] font-mono uppercase text-cyan-400 block mb-1">Recommendation</span>
                    <p className="text-xs text-slate-400 leading-normal">{data.ai_briefing.sectors.power_grids.recommendation}</p>
                  </div>
                </div>

                {/* 4. Aurora */}
                <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between hover:border-slate-700 transition">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-slate-200">
                        <Eye className="w-4 h-4 text-emerald-400" />
                        <span className="text-xs font-bold uppercase tracking-wider">Aurora Oval</span>
                      </div>
                      <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${getRiskBadge(data.ai_briefing.sectors.aurora.visibility)}`}>
                        {data.ai_briefing.sectors.aurora.visibility}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Visibility Latitude</span>
                      <p className="text-xs text-slate-300 leading-normal font-mono">{data.ai_briefing.sectors.aurora.latitude_range}</p>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-slate-800/70">
                    <span className="text-[10px] font-mono uppercase text-cyan-400 block mb-1">Display Activity</span>
                    <p className="text-xs text-slate-400 leading-normal">{data.ai_briefing.sectors.aurora.activity}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* 24-HOUR TELEMETRY & KP TREND CHARTS (RECHARTS) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Chart 1: Solar Wind Speed & Bz IMF Over 24 Hours */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 flex flex-col space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Wind className="w-4 h-4 text-cyan-400" />
                    <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                      24h L1 Solar Wind Velocity &amp; IMF Bz
                    </h3>
                  </div>
                  <span className="text-[10px] font-mono text-slate-400">1-HOUR INTERVALS (UTC)</span>
                </div>

                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data.history_24h} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="speedGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="timestamp" tickFormatter={formatChartTime} stroke="#64748b" tick={{ fontSize: 10 }} />
                      <YAxis yAxisId="left" stroke="#06b6d4" tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                      <YAxis yAxisId="right" orientation="right" stroke="#f43f5e" tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '0.75rem', fontSize: '11px' }}
                        labelFormatter={(label) => `Time: ${formatChartTime(String(label))} UTC`}
                      />
                      <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: '11px' }} />
                      <Area yAxisId="left" type="monotone" dataKey="wind_speed" name="Speed (km/s)" stroke="#06b6d4" fillOpacity={1} fill="url(#speedGrad)" strokeWidth={2} />
                      <Line yAxisId="right" type="monotone" dataKey="bz_gsm" name="Bz GSM (nT)" stroke="#f43f5e" strokeWidth={2} dot={false} />
                      <ReferenceLine yAxisId="right" y={0} stroke="#94a3b8" strokeDasharray="2 2" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Chart 2: 24h Kp Geomagnetic Disturbance Index */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 flex flex-col space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-indigo-400" />
                    <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                      24h Geomagnetic Kp Index Trend
                    </h3>
                  </div>
                  <span className="text-[10px] font-mono text-rose-400 font-semibold">ALERT LINE = 4.5 Kp</span>
                </div>

                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data.history_24h} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="kpGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#6366f1" stopOpacity={0.5} />
                          <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="timestamp" tickFormatter={formatChartTime} stroke="#64748b" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#a5b4fc" tick={{ fontSize: 10 }} domain={[0, 9]} ticks={[0, 2, 4, 6, 8]} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '0.75rem', fontSize: '11px' }}
                        labelFormatter={(label) => `Time: ${formatChartTime(String(label))} UTC`}
                      />
                      <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: '11px' }} />
                      <Area type="monotone" dataKey="kp_estimate" name="Estimated Kp (0–9)" stroke="#818cf8" fillOpacity={1} fill="url(#kpGrad)" strokeWidth={2.5} />
                      <ReferenceLine y={4.5} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1.5} label={{ value: 'Storm Alert Threshold (Kp = 4.5)', position: 'insideTopRight', fill: '#ef4444', fontSize: 10 }} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 font-mono gap-2">
          <div className="flex items-center gap-2">
            <span>HELIO-SHIELD SPACE WEATHER ENGINE</span>
            <span>&bull;</span>
            <span>POWERED BY NASA OMNI2 &amp; NOAA SWPC</span>
          </div>
          <div>
            <span>MODEL: STORM-WEIGHTED XGBOOST (SC25 PEAK CALIBRATED)</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
