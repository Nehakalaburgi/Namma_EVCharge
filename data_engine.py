"""
NAMMA-EVCHARGE | Data Engine
============================
Generates high-fidelity synthetic data for 5 Bengaluru zones based on
SAE J1772 charging standards and BESCOM feeder topology estimates.

Zones: Whitefield, Indiranagar, Electronic City, HSR Layout, Yelahanka
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# ZONE CONFIGURATION  (ground-truth parameters)
# ─────────────────────────────────────────────
ZONE_CONFIG = {
    "Whitefield": {
        "ev_density": 4200,          # registered EVs
        "feeder_capacity_kw": 8500,  # total feeder capacity
        "feeder_headroom_kw": 1800,  # available headroom (peak day)
        "existing_stations": 12,
        "area_sq_km": 38.2,
        "residential_ratio": 0.62,
        "commercial_ratio": 0.38,
        "ev_growth_rate": 0.34,      # YoY %
        "avg_distance_to_feeder_km": 1.2,
        "lat_center": 12.9698,
        "lon_center": 77.7499,
        "connectivity_score": 7.4,
    },
    "Indiranagar": {
        "ev_density": 3100,
        "feeder_capacity_kw": 6200,
        "feeder_headroom_kw": 900,
        "existing_stations": 18,
        "area_sq_km": 12.5,
        "residential_ratio": 0.55,
        "commercial_ratio": 0.45,
        "ev_growth_rate": 0.28,
        "avg_distance_to_feeder_km": 0.7,
        "lat_center": 12.9784,
        "lon_center": 77.6408,
        "connectivity_score": 8.9,
    },
    "Electronic City": {
        "ev_density": 5800,
        "feeder_capacity_kw": 12000,
        "feeder_headroom_kw": 3200,
        "existing_stations": 9,
        "area_sq_km": 52.4,
        "residential_ratio": 0.48,
        "commercial_ratio": 0.52,
        "ev_growth_rate": 0.41,
        "avg_distance_to_feeder_km": 1.9,
        "lat_center": 12.8452,
        "lon_center": 77.6602,
        "connectivity_score": 6.1,
    },
    "HSR Layout": {
        "ev_density": 2800,
        "feeder_capacity_kw": 5500,
        "feeder_headroom_kw": 700,
        "existing_stations": 14,
        "area_sq_km": 14.8,
        "residential_ratio": 0.70,
        "commercial_ratio": 0.30,
        "ev_growth_rate": 0.31,
        "avg_distance_to_feeder_km": 0.9,
        "lat_center": 12.9116,
        "lon_center": 77.6474,
        "connectivity_score": 8.2,
    },
    "Yelahanka": {
        "ev_density": 1900,
        "feeder_capacity_kw": 4800,
        "feeder_headroom_kw": 2100,
        "existing_stations": 5,
        "area_sq_km": 41.3,
        "residential_ratio": 0.75,
        "commercial_ratio": 0.25,
        "ev_growth_rate": 0.45,
        "avg_distance_to_feeder_km": 2.4,
        "lat_center": 13.1007,
        "lon_center": 77.5963,
        "connectivity_score": 5.8,
    },
}

# SAE J1772 Level profiles
SAE_J1772 = {
    "Level1_AC": {"power_kw": 1.4,  "typical_duration_hrs": 8.0},
    "Level2_AC": {"power_kw": 7.2,  "typical_duration_hrs": 3.5},
    "Level2_AC_Plus": {"power_kw": 11.5, "typical_duration_hrs": 2.2},
    "DC_Fast": {"power_kw": 50.0,  "typical_duration_hrs": 0.5},
}

CHARGER_MIX = {
    "Level1_AC":      0.20,
    "Level2_AC":      0.55,
    "Level2_AC_Plus": 0.18,
    "DC_Fast":        0.07,
}

np.random.seed(42)


# ─────────────────────────────────────────────
# HOURLY DEMAND CURVE SHAPES
# ─────────────────────────────────────────────

def _base_demand_curve(hour: int, zone: str) -> float:
    """
    Returns a multiplier (0–1) for charging demand at a given hour.
    Encodes the evening residential peak (18–22h) and an overnight trough.
    """
    config = ZONE_CONFIG[zone]
    res_ratio = config["residential_ratio"]
    com_ratio = config["commercial_ratio"]

    # Residential: peaks 18–22, trough 02–06
    res_curve = np.array([
        0.10, 0.08, 0.06, 0.05, 0.07, 0.12,   # 00–05
        0.18, 0.28, 0.35, 0.30, 0.25, 0.22,   # 06–11
        0.20, 0.18, 0.16, 0.20, 0.38, 0.75,   # 12–17
        0.95, 1.00, 0.90, 0.72, 0.50, 0.28,   # 18–23
    ])

    # Commercial: peaks 09–11, 14–17
    com_curve = np.array([
        0.05, 0.04, 0.03, 0.03, 0.04, 0.06,
        0.15, 0.35, 0.60, 0.80, 0.85, 0.70,
        0.55, 0.60, 0.75, 0.80, 0.60, 0.40,
        0.25, 0.18, 0.14, 0.10, 0.08, 0.06,
    ])

    return res_ratio * res_curve[hour] + com_ratio * com_curve[hour]


def generate_hourly_demand(zone: str, date: datetime) -> pd.DataFrame:
    """
    Generate one day of synthetic hourly EV charging demand (kW) for a zone.
    Includes stochastic noise and weekday/weekend modulation.
    """
    config = ZONE_CONFIG[zone]
    active_evs = config["ev_density"]
    capacity = config["feeder_capacity_kw"]

    # Weekend modifier: residential +15%, commercial -30%
    is_weekend = date.weekday() >= 5
    weekend_mod = 1.15 if is_weekend else 1.0

    # Active charging fraction (not all EVs charge each day)
    daily_charging_fraction = np.random.beta(3.5, 6.5)  # ~35% typical

    records = []
    for hour in range(24):
        base_mult = _base_demand_curve(hour, zone)
        noise = np.random.normal(1.0, 0.06)  # ±6% stochastic

        # Weighted average charger power draw
        avg_power = sum(
            CHARGER_MIX[lvl] * SAE_J1772[lvl]["power_kw"]
            for lvl in SAE_J1772
        )

        demand_kw = (
            active_evs
            * daily_charging_fraction
            * base_mult
            * weekend_mod
            * avg_power
            * noise
        )

        # Cap at feeder capacity (hard physical constraint)
        demand_kw = min(demand_kw, capacity * 0.98)

        utilisation_pct = demand_kw / capacity * 100

        records.append({
            "zone": zone,
            "datetime": date.replace(hour=hour, minute=0, second=0),
            "hour": hour,
            "demand_kw": round(demand_kw, 1),
            "feeder_capacity_kw": capacity,
            "feeder_headroom_kw": config["feeder_headroom_kw"],
            "utilisation_pct": round(utilisation_pct, 2),
            "is_peak_hour": 18 <= hour <= 22,
            "is_offpeak_hour": hour <= 6 or hour >= 23,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# CHARGING SESSION LOGS  (SAE J1772 compliant)
# ─────────────────────────────────────────────

def generate_charging_logs(zone: str, n_sessions: int = 500) -> pd.DataFrame:
    """
    Generate synthetic EV charging session logs for a zone.
    Each session has: start_time, charger_level, power_kw, energy_kwh,
    duration_hrs, soc_start (%), soc_end (%).
    """
    config = ZONE_CONFIG[zone]
    base_date = datetime(2026, 5, 5)
    sessions = []

    for _ in range(n_sessions):
        # Pick charger level probabilistically
        levels = list(CHARGER_MIX.keys())
        probs  = list(CHARGER_MIX.values())
        level  = np.random.choice(levels, p=probs)
        spec   = SAE_J1772[level]

        # Session start: weighted towards evening peak
# --- Cleaned and Fixed version for data_engine.py ---

# 1. Calculate the hour using a probability distribution from the base demand curve
        hour = int(np.random.choice(
            range(24),
            p=[_base_demand_curve(h, zone) / sum(_base_demand_curve(h2, zone) for h2 in range(24)) for h in range(24)]
        ))

# 2. Generate random minute and day offset as standard integers
        minute = int(np.random.randint(0, 60))
        day_offset = int(np.random.randint(0, 30)) 

# 3. Create the start datetime without component type errors
        start_dt = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)

        soc_start = np.random.uniform(15, 65)
        soc_end   = min(soc_start + np.random.uniform(20, 50), 100)
        energy    = (soc_end - soc_start) / 100 * 60  # assume 60 kWh battery
        duration  = energy / spec["power_kw"]

        sessions.append({
            "zone": zone,
            "session_id": f"{zone[:3].upper()}-{len(sessions):05d}",
            "start_datetime": start_dt,
            "charger_level": level,
            "power_kw": spec["power_kw"],
            "duration_hrs": round(duration, 2),
            "energy_kwh": round(energy, 2),
            "soc_start_pct": round(soc_start, 1),
            "soc_end_pct": round(soc_end, 1),
            "lat": config["lat_center"] + np.random.normal(0, 0.018),
            "lon": config["lon_center"] + np.random.normal(0, 0.018),
        })

    return pd.DataFrame(sessions)


# ─────────────────────────────────────────────
# MASTER DATASET BUILDERS
# ─────────────────────────────────────────────

def build_full_demand_dataset(days: int = 7) -> pd.DataFrame:
    """Build multi-day demand dataset across all zones."""
    base_date = datetime(2026, 5, 5)
    frames = []
    for day in range(days):
        dt = base_date + timedelta(days=day)
        for zone in ZONE_CONFIG:
            frames.append(generate_hourly_demand(zone, dt))
    df = pd.concat(frames, ignore_index=True)
    return df


def build_zone_summary() -> pd.DataFrame:
    """Zone-level summary DataFrame with infrastructure metadata."""
    rows = []
    for zone, cfg in ZONE_CONFIG.items():
        utilisation = (cfg["feeder_capacity_kw"] - cfg["feeder_headroom_kw"]) / cfg["feeder_capacity_kw"]
        rows.append({
            "zone": zone,
            "ev_density": cfg["ev_density"],
            "feeder_capacity_kw": cfg["feeder_capacity_kw"],
            "feeder_headroom_kw": cfg["feeder_headroom_kw"],
            "existing_stations": cfg["existing_stations"],
            "ev_growth_rate_pct": cfg["ev_growth_rate"] * 100,
            "area_sq_km": cfg["area_sq_km"],
            "ev_per_sq_km": round(cfg["ev_density"] / cfg["area_sq_km"], 1),
            "station_density": round(cfg["existing_stations"] / cfg["area_sq_km"], 3),
            "avg_distance_to_feeder_km": cfg["avg_distance_to_feeder_km"],
            "connectivity_score": cfg["connectivity_score"],
            "baseline_utilisation_pct": round(utilisation * 100, 1),
            "lat": cfg["lat_center"],
            "lon": cfg["lon_center"],
        })
    return pd.DataFrame(rows)


def build_all_charging_logs() -> pd.DataFrame:
    """Aggregate charging logs across all zones."""
    frames = [generate_charging_logs(z) for z in ZONE_CONFIG]
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# QUICK SMOKE-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_demand = build_full_demand_dataset(days=3)
    df_zones  = build_zone_summary()
    df_logs   = build_all_charging_logs()

    print("=== Demand Dataset ===")
    print(df_demand.head(10))
    print(f"\nShape: {df_demand.shape}")

    print("\n=== Zone Summary ===")
    print(df_zones.to_string(index=False))

    print(f"\n=== Charging Logs (sample) ===")
    print(df_logs.head(5))
    print(f"Total sessions: {len(df_logs)}")