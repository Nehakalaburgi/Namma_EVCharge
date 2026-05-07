"""
NAMMA-EVCHARGE | Models
========================
Part A : Demand Forecasting  (Random Forest + simulated LSTM-style smoothing)
Part B : Dynamic Scheduling  (5 load-shift scenarios → Optimal Scenario selector)
Part C : Location Planning   (DBSCAN hotspot clustering)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error
from scipy.ndimage import gaussian_filter1d

from data_engine import (
    build_full_demand_dataset,
    build_zone_summary,
    build_all_charging_logs,
    ZONE_CONFIG,
)

np.random.seed(42)


# ═══════════════════════════════════════════
#  PART A  —  DEMAND FORECASTING
# ═══════════════════════════════════════════

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time and lag features for model training."""
    df = df.copy()
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_weekend"] = pd.to_datetime(df["datetime"]).dt.weekday >= 5
    df["is_peak"]    = ((df["hour"] >= 18) & (df["hour"] <= 22)).astype(int)
    df["is_offpeak"] = ((df["hour"] <= 6) | (df["hour"] >= 23)).astype(int)
    df["zone_enc"]   = pd.Categorical(df["zone"]).codes
    return df


def train_forecast_model(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest forecaster per zone.
    Returns a dict: {zone: (model, scaler, feature_cols, MAE)}
    """
    FEATURES = [
        "hour_sin", "hour_cos", "is_weekend", "is_peak",
        "is_offpeak", "feeder_capacity_kw",
    ]
    df = _engineer_features(df)
    zone_models = {}

    for zone in ZONE_CONFIG:
        zdf = df[df["zone"] == zone].copy().sort_values("datetime")
        X   = zdf[FEATURES].values
        y   = zdf["demand_kw"].values

        split = int(len(X) * 0.80)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        sc = MinMaxScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s  = sc.transform(X_test)

        rf = RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train_s, y_train)
        mae = mean_absolute_error(y_test, rf.predict(X_test_s))
        zone_models[zone] = (rf, sc, FEATURES, mae)

    return zone_models


def _lstm_smooth(series: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    """
    Simulated LSTM-style temporal smoothing using Gaussian filtering.
    Preserves trend while smoothing high-frequency noise — mimics the
    sequential memory effect of a recurrent network.
    """
    return gaussian_filter1d(series, sigma=sigma)


def forecast_next_day(zone_models: dict, target_date) -> pd.DataFrame:
    """
    Generate a 24-hour demand forecast for each zone on target_date.
    Applies LSTM-style smoothing for temporal coherence.
    """
    rows = []
    for zone, (rf, sc, features, mae) in zone_models.items():
        cfg = ZONE_CONFIG[zone]
        is_weekend = pd.Timestamp(target_date).weekday() >= 5

        for hour in range(24):
            row = {
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "is_weekend": int(is_weekend),
                "is_peak": int(18 <= hour <= 22),
                "is_offpeak": int(hour <= 6 or hour >= 23),
                "feeder_capacity_kw": cfg["feeder_capacity_kw"],
            }
            rows.append({"zone": zone, "hour": hour, **row, "mae": mae})

    pred_df = pd.DataFrame(rows)
    FEATURES = ["hour_sin", "hour_cos", "is_weekend", "is_peak", "is_offpeak", "feeder_capacity_kw"]
    results = []

    for zone, (rf, sc, features, mae) in zone_models.items():
        zdf = pred_df[pred_df["zone"] == zone].copy()
        X   = sc.transform(zdf[FEATURES].values)
        raw = rf.predict(X)
        smoothed = _lstm_smooth(raw, sigma=1.4)

        for i, hour in enumerate(range(24)):
            cfg = ZONE_CONFIG[zone]
            demand = max(0.0, smoothed[i])
            results.append({
                "zone": zone,
                "hour": hour,
                "forecast_demand_kw": round(demand, 1),
                "feeder_capacity_kw": cfg["feeder_capacity_kw"],
                "utilisation_pct": round(demand / cfg["feeder_capacity_kw"] * 100, 2),
                "exceeds_90pct": demand > cfg["feeder_capacity_kw"] * 0.90,
                "mae_kw": round(mae, 1),
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════
#  PART B  —  DYNAMIC SCHEDULING ENGINE
# ═══════════════════════════════════════════

SCENARIOS = {
    "S1_Flat_Incentive": {
        "description": "Flat 15% off-peak tariff incentive",
        "peak_shift_pct": 0.15,
        "offpeak_spread": "11PM-6AM",
        "implementation_complexity": "Low",
        "customer_disruption": "Minimal",
    },
    "S2_Smart_Timer": {
        "description": "Smart timer 20% load shift with V2G readiness",
        "peak_shift_pct": 0.20,
        "offpeak_spread": "11PM-5AM",
        "implementation_complexity": "Medium",
        "customer_disruption": "Low",
    },
    "S3_Dynamic_Pricing": {
        "description": "Real-time dynamic pricing shifting 25% load",
        "peak_shift_pct": 0.25,
        "offpeak_spread": "12AM-6AM",
        "implementation_complexity": "Medium",
        "customer_disruption": "Medium",
    },
    "S4_Mandatory_Defer": {
        "description": "Mandatory deferral of 30% non-urgent charging",
        "peak_shift_pct": 0.30,
        "offpeak_spread": "11PM-4AM",
        "implementation_complexity": "High",
        "customer_disruption": "High",
    },
    "S5_AI_Orchestrated": {
        "description": "AI-orchestrated 28% shift with zone-specific micro-schedules",
        "peak_shift_pct": 0.28,
        "offpeak_spread": "11PM-6AM",
        "implementation_complexity": "High",
        "customer_disruption": "Low",
    },
}


def _compute_grid_safety_score(
    forecast_kw: np.ndarray,
    shifted_kw: np.ndarray,
    capacity_kw: float,
) -> float:
    """
    Grid Safety Score (0–100):
      - 40 pts: Peak demand reduction ratio
      - 30 pts: Headroom maintained above 10% capacity
      - 20 pts: Even load distribution (low std dev)
      - 10 pts: No hour exceeds 90% utilisation
    """
    peak_orig    = np.max(forecast_kw)
    peak_shifted = np.max(shifted_kw)
    peak_score   = min(40, (peak_orig - peak_shifted) / peak_orig * 200)

    min_headroom  = (capacity_kw - peak_shifted) / capacity_kw
    headroom_score = min(30, max(0, min_headroom * 150))

    cv = np.std(shifted_kw) / (np.mean(shifted_kw) + 1e-6)
    dist_score = min(20, max(0, 20 - cv * 25))

    over_90 = np.sum(shifted_kw > capacity_kw * 0.90)
    safety_score = 10 if over_90 == 0 else max(0, 10 - over_90 * 2)

    return round(peak_score + headroom_score + dist_score + safety_score, 2)


def _apply_load_shift(
    forecast_kw: np.ndarray,
    shift_pct: float,
    capacity_kw: float,
) -> np.ndarray:
    """
    Shift `shift_pct` of peak-hour (18–22) demand into off-peak (23–06).
    Returns modified 24-hour load curve.
    """
    shifted = forecast_kw.copy()
    peak_hours    = list(range(18, 23))
    offpeak_hours = list(range(23, 24)) + list(range(0, 7))

    for h in peak_hours:
        shift_amount = shifted[h] * shift_pct
        shifted[h]  -= shift_amount
        # Distribute evenly across off-peak hours
        per_hour    = shift_amount / len(offpeak_hours)
        for oh in offpeak_hours:
            shifted[oh] += per_hour

    return np.clip(shifted, 0, capacity_kw * 0.98)


def simulate_all_scenarios(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each zone × scenario, compute shifted demand and Grid Safety Score.
    Returns detailed scenario comparison DataFrame.
    """
    records = []
    PEAK_HOURS    = list(range(18, 23))
    OFFPEAK_HOURS = list(range(23, 24)) + list(range(0, 7))

    for zone in ZONE_CONFIG:
        cfg       = ZONE_CONFIG[zone]
        capacity  = cfg["feeder_capacity_kw"]
        zdf       = forecast_df[forecast_df["zone"] == zone].sort_values("hour")
        base_load = zdf["forecast_demand_kw"].values  # 24 values

        for scenario_id, s_cfg in SCENARIOS.items():
            shifted = _apply_load_shift(base_load, s_cfg["peak_shift_pct"], capacity)
            gss     = _compute_grid_safety_score(base_load, shifted, capacity)

            peak_orig    = np.max(base_load[PEAK_HOURS])
            peak_shifted = np.max(shifted[PEAK_HOURS])
            reduction_pct = (peak_orig - peak_shifted) / peak_orig * 100

            offpeak_addition = np.sum(shifted[OFFPEAK_HOURS]) - np.sum(base_load[OFFPEAK_HOURS])

            records.append({
                "zone": zone,
                "scenario_id": scenario_id,
                "description": s_cfg["description"],
                "shift_pct": s_cfg["peak_shift_pct"] * 100,
                "peak_demand_orig_kw": round(peak_orig, 1),
                "peak_demand_shifted_kw": round(peak_shifted, 1),
                "peak_reduction_pct": round(reduction_pct, 1),
                "grid_safety_score": gss,
                "offpeak_addition_kw": round(offpeak_addition, 1),
                "implementation_complexity": s_cfg["implementation_complexity"],
                "customer_disruption": s_cfg["customer_disruption"],
                "meets_18pct_target": reduction_pct >= 18.0,
                "shifted_load_24h": shifted.tolist(),
            })

    return pd.DataFrame(records)


def select_optimal_scenario(scenario_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the optimal scenario per zone:
      1. Must meet ≥18% peak reduction target
      2. Maximise Grid Safety Score
      3. Prefer lower customer disruption if scores are within 5 pts
    Returns one row per zone with the winning scenario + NL explanation.
    """
    disruption_rank = {"Minimal": 1, "Low": 2, "Medium": 3, "High": 4}

    results = []
    for zone in ZONE_CONFIG:
        zdf = scenario_df[scenario_df["zone"] == zone].copy()
        eligible = zdf[zdf["meets_18pct_target"]].copy()

        if eligible.empty:
            # Fallback: best scorer regardless of target
            eligible = zdf.copy()

        eligible["disruption_rank"] = eligible["customer_disruption"].map(disruption_rank)
        eligible = eligible.sort_values(
            ["grid_safety_score", "disruption_rank"],
            ascending=[False, True],
        )
        best = eligible.iloc[0].copy()

        # Generate natural language explanation
        nl_explanation = _generate_nl_explanation(zone, best)
        best["nl_explanation"] = nl_explanation
        best["is_optimal"] = True
        results.append(best)

    return pd.DataFrame(results).reset_index(drop=True)


def _generate_nl_explanation(zone: str, row: pd.Series) -> str:
    """
    Produce a single-sentence plain-English rationale for the recommendation.
    """
    shift = round(row["shift_pct"])
    reduction = round(row["peak_reduction_pct"])
    gss = round(row["grid_safety_score"])
    scenario = row["description"]

    templates = [
        f"Shifting {shift}% of {zone}'s evening load via '{scenario}' reduces feeder stress "
        f"by {reduction}% at peak and raises the Grid Safety Score to {gss}/100, preventing "
        f"potential voltage irregularities in residential areas between 6–10 PM.",

        f"Implementing '{scenario}' in {zone} redistributes {shift}% of peak load to overnight "
        f"hours, cutting the 8 PM demand spike by {reduction}% and protecting feeder headroom "
        f"from breaching the 90% critical threshold.",

        f"A {shift}% managed load shift in {zone} through '{scenario}' achieves a {reduction}% "
        f"evening peak reduction (Grid Safety Score: {gss}/100), directly mitigating voltage "
        f"irregularities caused by spatial and temporal EV charging clustering.",
    ]
    # Pick template based on grid safety score tier
    if gss >= 75:
        return templates[0]
    elif gss >= 55:
        return templates[1]
    else:
        return templates[2]


def build_managed_schedule(optimal_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-zone 24-hour demand profile for the AI-Recommended schedule.
    """
    rows = []
    for _, opt in optimal_df.iterrows():
        zone = opt["zone"]
        shifted = opt["shifted_load_24h"]
        base_df = forecast_df[forecast_df["zone"] == zone].sort_values("hour")
        base_load = base_df["forecast_demand_kw"].values

        for hour in range(24):
            rows.append({
                "zone": zone,
                "hour": hour,
                "unmanaged_kw": round(base_load[hour], 1),
                "managed_kw": round(shifted[hour], 1),
                "reduction_kw": round(base_load[hour] - shifted[hour], 1),
                "feeder_capacity_kw": ZONE_CONFIG[zone]["feeder_capacity_kw"],
                "unmanaged_util_pct": round(base_load[hour] / ZONE_CONFIG[zone]["feeder_capacity_kw"] * 100, 2),
                "managed_util_pct": round(shifted[hour] / ZONE_CONFIG[zone]["feeder_capacity_kw"] * 100, 2),
                "exceeds_90pct_unmanaged": base_load[hour] > ZONE_CONFIG[zone]["feeder_capacity_kw"] * 0.90,
                "exceeds_90pct_managed": shifted[hour] > ZONE_CONFIG[zone]["feeder_capacity_kw"] * 0.90,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════
#  PART C  —  LOCATION PLANNING (DBSCAN)
# ═══════════════════════════════════════════

def run_hotspot_clustering(logs_df: pd.DataFrame, zone_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Apply DBSCAN to charging session coordinates to identify demand hotspots.
    Cross-reference with zone infrastructure data to flag underserved clusters.

    Returns hotspot DataFrame with: cluster_id, zone, centroid, demand density,
    nearest feeder distance, underservice score, and fact-sheet data.
    """
    results = []

    for zone in ZONE_CONFIG:
        cfg  = ZONE_CONFIG[zone]
        zlog = logs_df[logs_df["zone"] == zone][["lat", "lon"]].dropna()
        if len(zlog) < 10:
            continue

        # Normalise coordinates
        coords = zlog[["lat", "lon"]].values

        # DBSCAN: eps ≈ 0.5 km in degree-space, min_samples = 8
        db = DBSCAN(eps=0.006, min_samples=8, metric="euclidean")
        labels = db.fit_predict(coords)
        zlog = zlog.copy()
        zlog["cluster"] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        zsummary = zone_summary[zone_summary["zone"] == zone].iloc[0]

        # Build fact-sheet per cluster
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            cluster_pts = zlog[zlog["cluster"] == cluster_id]
            centroid_lat = cluster_pts["lat"].mean()
            centroid_lon = cluster_pts["lon"].mean()
            n_sessions   = len(cluster_pts)

            # Underservice score: high EV density + low station density → hotspot
            station_density = zsummary["station_density"]
            ev_per_sqkm     = zsummary["ev_per_sq_km"]
            underservice_score = round(
                min(10.0, (ev_per_sqkm / 200) * (1 / (station_density + 0.01)) * 2.5),
                2,
            )

            results.append({
                "zone": zone,
                "cluster_id": f"{zone[:3].upper()}-C{cluster_id:02d}",
                "centroid_lat": round(centroid_lat, 5),
                "centroid_lon": round(centroid_lon, 5),
                "session_count": n_sessions,
                "ev_density": cfg["ev_density"],
                "ev_growth_rate_pct": cfg["ev_growth_rate"] * 100,
                "existing_stations": cfg["existing_stations"],
                "station_density_per_sqkm": zsummary["station_density"],
                "ev_per_sq_km": zsummary["ev_per_sq_km"],
                "avg_distance_to_feeder_km": cfg["avg_distance_to_feeder_km"],
                "connectivity_score": cfg["connectivity_score"],
                "feeder_capacity_kw": cfg["feeder_capacity_kw"],
                "feeder_headroom_kw": cfg["feeder_headroom_kw"],
                "baseline_utilisation_pct": zsummary["baseline_utilisation_pct"],
                "underservice_score": underservice_score,
                "is_critical_hotspot": underservice_score >= 5.0,
                "recommended_new_stations": max(1, int(np.ceil(n_sessions / 60))),
                "n_clusters_in_zone": n_clusters,
            })

    df_hotspots = pd.DataFrame(results)

    # Attach natural language hotspot insight
    df_hotspots["hotspot_insight"] = df_hotspots.apply(_hotspot_insight, axis=1)

    return df_hotspots.sort_values("underservice_score", ascending=False).reset_index(drop=True)


def _hotspot_insight(row: pd.Series) -> str:
    """One-sentence NL insight for each hotspot cluster."""
    zone   = row["zone"]
    score  = row["underservice_score"]
    growth = row["ev_growth_rate_pct"]
    dist   = row["avg_distance_to_feeder_km"]
    recs   = row["recommended_new_stations"]

    if score >= 7:
        return (
            f"Cluster {row['cluster_id']} in {zone} is critically under-served: "
            f"with {growth:.0f}% annual EV growth and only {row['existing_stations']} stations across the zone, "
            f"adding {recs} charging point(s) here is urgent to prevent feeder saturation."
        )
    elif score >= 4:
        return (
            f"{zone}'s {row['cluster_id']} cluster shows elevated demand density {dist:.1f} km from the nearest feeder; "
            f"deploying {recs} additional station(s) will ease projected load by pre-distributing spatial clustering effects."
        )
    else:
        return (
            f"Cluster {row['cluster_id']} in {zone} is currently manageable, "
            f"but {growth:.0f}% EV growth rate warrants a medium-term plan for {recs} new charging point(s)."
        )


# ═══════════════════════════════════════════
#  CONVENIENCE RUNNER  (imports all in one)
# ═══════════════════════════════════════════

def run_full_pipeline(forecast_date=None):
    """
    Execute the full modelling pipeline and return a dict of all artefacts.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from datetime import date as dt_date
    if forecast_date is None:
        forecast_date = "2026-05-05"

    print("📦 Building datasets …")
    df_demand   = build_full_demand_dataset(days=7)
    df_zones    = build_zone_summary()
    df_logs     = build_all_charging_logs()

    print("🤖 Training forecast models …")
    zone_models = train_forecast_model(df_demand)

    print("🔮 Generating 24-hour forecast …")
    forecast_df = forecast_next_day(zone_models, forecast_date)

    print("⚙️  Simulating load-shift scenarios …")
    scenario_df = simulate_all_scenarios(forecast_df)

    print("✅ Selecting optimal scenarios …")
    optimal_df  = select_optimal_scenario(scenario_df)

    print("📅 Building managed schedule …")
    schedule_df = build_managed_schedule(optimal_df, forecast_df)

    print("📍 Running DBSCAN hotspot clustering …")
    hotspot_df  = run_hotspot_clustering(df_logs, df_zones)

    print("✔  Pipeline complete.")

    return {
        "df_demand":   df_demand,
        "df_zones":    df_zones,
        "df_logs":     df_logs,
        "zone_models": zone_models,
        "forecast_df": forecast_df,
        "scenario_df": scenario_df,
        "optimal_df":  optimal_df,
        "schedule_df": schedule_df,
        "hotspot_df":  hotspot_df,
    }


if __name__ == "__main__":
    artefacts = run_full_pipeline()
    print("\n=== Optimal Scenarios ===")
    cols = ["zone", "scenario_id", "peak_reduction_pct", "grid_safety_score", "nl_explanation"]
    print(artefacts["optimal_df"][cols].to_string(index=False))

    print("\n=== Top Hotspots ===")
    hcols = ["cluster_id", "zone", "underservice_score", "is_critical_hotspot", "recommended_new_stations"]
    print(artefacts["hotspot_df"][hcols].to_string(index=False))