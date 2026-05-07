"""
NAMMA-EVCHARGE | Streamlit Dashboard
=====================================
BESCOM Corporate theme — Blue & White
Tab 1 : Demand Forecast  (Unmanaged vs AI-Recommended + Grid Guardrail)
Tab 2 : Infrastructure Fact Sheets  (DBSCAN Hotspots)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from data_engine import ZONE_CONFIG

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="NAMMA-EVCHARGE | BESCOM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── BESCOM Corporate CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F0F4FA;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(175deg, #003d8f 0%, #0057c8 60%, #0070e0 100%);
    color: #ffffff;
  }
  [data-testid="stSidebar"] * { color: #ffffff !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label { color: #cce0ff !important; }

  /* Main header */
  .bescom-header {
    background: linear-gradient(90deg, #003d8f, #0057c8);
    padding: 18px 28px;
    border-radius: 10px;
    margin-bottom: 22px;
    display: flex; align-items: center; gap: 16px;
  }
  .bescom-logo-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.05em;
  }
  .bescom-sub {
    font-size: 0.82rem;
    color: #a8d0ff;
    letter-spacing: 0.08em;
  }

  /* KPI cards */
  .kpi-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 5px solid #0057c8;
    box-shadow: 0 2px 8px rgba(0,87,200,0.10);
  }
  .kpi-card.warning { border-left-color: #e53e3e; background: #fff5f5; }
  .kpi-card.success { border-left-color: #38a169; background: #f0fff4; }
  .kpi-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.0rem;
    font-weight: 700;
    color: #003d8f;
  }
  .kpi-label { font-size: 0.78rem; color: #4a5568; letter-spacing: 0.05em; }

  /* Section headers */
  .section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: #003d8f;
    border-bottom: 2px solid #0057c8;
    padding-bottom: 6px;
    margin-bottom: 16px;
  }

  /* Alert banner */
  .grid-alert {
    background: linear-gradient(90deg, #c53030, #e53e3e);
    border-radius: 8px;
    padding: 12px 18px;
    color: #fff;
    font-weight: 600;
    font-size: 0.92rem;
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 10px;
  }
  .grid-ok {
    background: linear-gradient(90deg, #276749, #38a169);
    border-radius: 8px;
    padding: 10px 18px;
    color: #fff;
    font-size: 0.88rem;
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 10px;
  }

  /* Fact sheet card */
  .fact-sheet {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #dbe4f0;
    box-shadow: 0 2px 12px rgba(0,87,200,0.08);
    margin-bottom: 16px;
  }
  .fact-sheet-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #003d8f;
  }
  .fact-tag {
    display: inline-block;
    background: #ebf4ff;
    color: #0057c8;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
  }
  .fact-tag.critical { background: #fff0f0; color: #c53030; }
  .nl-insight {
    background: #f0f7ff;
    border-left: 4px solid #0057c8;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    color: #2d3748;
    font-style: italic;
    margin-top: 12px;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 8px 8px 0 0;
    gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.0rem;
    font-weight: 600;
    color: #4a5568;
    padding: 10px 24px;
  }
  .stTabs [aria-selected="true"] {
    color: #003d8f !important;
    border-bottom: 3px solid #0057c8 !important;
  }

  /* Plotly chart container */
  .chart-wrap {
    background: #ffffff;
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 2px 8px rgba(0,87,200,0.08);
    margin-bottom: 20px;
  }
  div[data-testid="stMetricValue"] { font-family: 'Rajdhani', sans-serif; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  LOAD PIPELINE DATA (cached for performance)
# ══════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_pipeline():
    from models import run_full_pipeline
    return run_full_pipeline()


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bescom-header">
  <div>
    <div class="bescom-logo-text">⚡ NAMMA-EVCHARGE</div>
    <div class="bescom-sub">AI-POWERED EV LOAD MANAGEMENT SYSTEM &nbsp;|&nbsp; BESCOM &nbsp;|&nbsp; BENGALURU</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    st.markdown("---")

    ZONES = ["Whitefield", "Indiranagar", "Electronic City", "HSR Layout", "Yelahanka"]
    selected_zone = st.selectbox("📍 Select Zone", ZONES, index=0)
    st.markdown("---")
    st.markdown("**Forecast Date:** 05 May 2026")
    st.markdown("**Model:** RF + LSTM Smoothing")
    st.markdown("**Standards:** SAE J1772")
    st.markdown("---")
    st.markdown(
        "<small style='color:#a8d0ff'>© 2026 BESCOM Smart Grid Division<br>"
        "NAMMA-EVCHARGE v1.0 Prototype</small>",
        unsafe_allow_html=True,
    )

# ── Load data ───────────────────────────────────────────────────────────────
with st.spinner("🔄 Initialising AI pipeline — this takes ~20 seconds on first load …"):
    pipeline = load_pipeline()

schedule_df = pipeline["schedule_df"]
optimal_df  = pipeline["optimal_df"]
scenario_df = pipeline["scenario_df"]
hotspot_df  = pipeline["hotspot_df"]
df_zones    = pipeline["df_zones"]

# ── Global KPIs ─────────────────────────────────────────────────────────────
total_evs          = df_zones["ev_density"].sum()
avg_safety_score   = round(optimal_df["grid_safety_score"].mean(), 1)
critical_hotspots  = hotspot_df["is_critical_hotspot"].sum()
avg_peak_reduction = round(optimal_df["peak_reduction_pct"].mean(), 1)
zones_at_risk      = schedule_df[schedule_df["exceeds_90pct_unmanaged"]]["zone"].nunique()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{total_evs:,}</div>
    <div class="kpi-label">TOTAL EVs MONITORED</div></div>""", unsafe_allow_html=True)
with k2:
    card_cls = "warning" if zones_at_risk > 0 else "success"
    st.markdown(f"""<div class="kpi-card {card_cls}"><div class="kpi-value">{zones_at_risk}</div>
    <div class="kpi-label">ZONES AT FEEDER RISK</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card success"><div class="kpi-value">{avg_peak_reduction}%</div>
    <div class="kpi-label">AVG PEAK REDUCTION</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{avg_safety_score}</div>
    <div class="kpi-label">AVG GRID SAFETY SCORE</div></div>""", unsafe_allow_html=True)
with k5:
    card_cls = "warning" if critical_hotspots > 0 else "success"
    st.markdown(f"""<div class="kpi-card {card_cls}"><div class="kpi-value">{critical_hotspots}</div>
    <div class="kpi-label">CRITICAL HOTSPOTS</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════
tab1, tab2 = st.tabs(["📊  Demand Forecast & Scheduling", "🗺️  Infrastructure Fact Sheets"])


# ───────────────────────────────────────────────────────────────────────────
#  TAB 1 — DEMAND FORECAST
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f'<div class="section-title">Demand Forecast — {selected_zone}</div>', unsafe_allow_html=True)

    zone_sched  = schedule_df[schedule_df["zone"] == selected_zone].sort_values("hour")
    zone_opt    = optimal_df[optimal_df["zone"] == selected_zone].iloc[0]
    zone_scens  = scenario_df[scenario_df["zone"] == selected_zone]
    capacity    = zone_sched["feeder_capacity_kw"].iloc[0]
    headroom_90 = capacity * 0.90

    # ── Grid Guardrail ───────────────────────────────────────────────────
    unmanaged_breaches = zone_sched["exceeds_90pct_unmanaged"].sum()
    managed_breaches   = zone_sched["exceeds_90pct_managed"].sum()

    if unmanaged_breaches > 0:
        st.markdown(
            f'<div class="grid-alert">🔴 GRID GUARDRAIL: {selected_zone} — Unmanaged charging '
            f'exceeds 90% feeder capacity in <b>{unmanaged_breaches} hour(s)</b>. '
            f'Immediate load management required.</div>',
            unsafe_allow_html=True,
        )
    if managed_breaches == 0:
        st.markdown(
            f'<div class="grid-ok">✅ AI-Recommended schedule keeps {selected_zone} within '
            f'safe feeder limits for all 24 hours.</div>',
            unsafe_allow_html=True,
        )

    # ── NL Explanation ───────────────────────────────────────────────────
    st.info(f"💡 **AI Insight:** {zone_opt['nl_explanation']}")

    # ── Comparison Chart ─────────────────────────────────────────────────
    hours = zone_sched["hour"].tolist()
    hour_labels = [f"{h:02d}:00" for h in hours]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.70, 0.30],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Hourly EV Demand: Unmanaged vs AI-Recommended", "Load Reduction Achieved (kW)"],
    )

    # Capacity line
    fig.add_trace(go.Scatter(
        x=hour_labels, y=[capacity] * 24,
        name="Feeder Capacity", line=dict(color="#718096", dash="dot", width=1.5),
        mode="lines",
    ), row=1, col=1)

    # 90% guardrail
    fig.add_trace(go.Scatter(
        x=hour_labels, y=[headroom_90] * 24,
        name="90% Guardrail", line=dict(color="#e53e3e", dash="dash", width=2),
        fill="tonexty", fillcolor="rgba(229,62,62,0.06)",
        mode="lines",
    ), row=1, col=1)

    # Unmanaged fill
    fig.add_trace(go.Scatter(
        x=hour_labels, y=zone_sched["unmanaged_kw"].tolist(),
        name="Unmanaged Charging",
        line=dict(color="#e07b39", width=2.5),
        fill="tozeroy", fillcolor="rgba(224,123,57,0.12)",
        mode="lines+markers",
        marker=dict(size=5),
    ), row=1, col=1)

    # AI-Recommended fill
    fig.add_trace(go.Scatter(
        x=hour_labels, y=zone_sched["managed_kw"].tolist(),
        name="AI-Recommended Schedule",
        line=dict(color="#0057c8", width=2.8),
        fill="tozeroy", fillcolor="rgba(0,87,200,0.12)",
        mode="lines+markers",
        marker=dict(size=5),
    ), row=1, col=1)

    # Peak zone shading
    fig.add_vrect(
        x0="18:00", x1="22:00",
        fillcolor="rgba(229,62,62,0.07)",
        layer="below", line_width=0,
        annotation_text="Evening Peak (18–22h)", annotation_position="top left",
        annotation_font=dict(size=10, color="#c53030"),
    )
    fig.add_vrect(
        x0="23:00", x1="06:00",
        fillcolor="rgba(0,87,200,0.05)",
        layer="below", line_width=0,
        annotation_text="Off-Peak Window", annotation_position="top right",
        annotation_font=dict(size=10, color="#0057c8"),
    )

    # Bar chart: reduction
    reduction_vals = zone_sched["reduction_kw"].tolist()
    bar_colors = ["#38a169" if v >= 0 else "#e53e3e" for v in reduction_vals]
    fig.add_trace(go.Bar(
        x=hour_labels, y=reduction_vals,
        name="Load Shifted (kW)",
        marker_color=bar_colors,
        showlegend=True,
    ), row=2, col=1)

    fig.update_layout(
        height=560,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=30),
        xaxis2=dict(title="Hour of Day"),
        yaxis=dict(title="Demand (kW)", gridcolor="#e8edf5"),
        yaxis2=dict(title="Reduction (kW)", gridcolor="#e8edf5"),
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Scenario Comparison ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Load-Shift Scenario Comparison</div>', unsafe_allow_html=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=zone_scens["scenario_id"],
        y=zone_scens["grid_safety_score"],
        name="Grid Safety Score",
        marker=dict(
            color=zone_scens["grid_safety_score"],
            colorscale=[[0, "#f7b731"], [0.5, "#0057c8"], [1, "#003d8f"]],
            showscale=False,
        ),
        text=zone_scens["grid_safety_score"].round(1).astype(str),
        textposition="outside",
    ))

    # Highlight optimal
    opt_scenario = zone_opt["scenario_id"]
    for i, sid in enumerate(zone_scens["scenario_id"].tolist()):
        if sid == opt_scenario:
            fig2.add_shape(
                type="rect",
                x0=i - 0.45, x1=i + 0.45, y0=0, y1=zone_scens[zone_scens["scenario_id"] == sid]["grid_safety_score"].values[0] + 4,
                line=dict(color="#003d8f", width=2.5, dash="dot"),
                fillcolor="rgba(0,61,143,0.05)",
            )
            fig2.add_annotation(
                x=sid, y=zone_scens[zone_scens["scenario_id"] == sid]["grid_safety_score"].values[0] + 6,
                text="★ OPTIMAL", font=dict(color="#003d8f", size=11, family="Rajdhani"),
                showarrow=False,
            )

    # 18% target line on second axis
    fig2.add_trace(go.Scatter(
        x=zone_scens["scenario_id"],
        y=zone_scens["peak_reduction_pct"],
        name="Peak Reduction %",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#e53e3e", width=2, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))
    fig2.add_hline(y=18, line_dash="dot", line_color="#38a169", annotation_text="18% Target",
                   annotation_position="bottom right", yref="y2")

    fig2.update_layout(
        height=380,
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter", size=12),
        yaxis=dict(title="Grid Safety Score (0–100)", gridcolor="#e8edf5"),
        yaxis2=dict(title="Peak Reduction %", overlaying="y", side="right", gridcolor="#f0f4fa"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=60, t=40, b=40),
        xaxis=dict(title="Scenario"),
        barmode="overlay",
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Scenario Table ─────────────────────────────────────────────────────
    display_cols = [
        "scenario_id", "description", "shift_pct",
        "peak_reduction_pct", "grid_safety_score",
        "meets_18pct_target", "implementation_complexity", "customer_disruption",
    ]
    scen_display = zone_scens[display_cols].rename(columns={
        "scenario_id": "Scenario",
        "description": "Description",
        "shift_pct": "Shift %",
        "peak_reduction_pct": "Peak Reduction %",
        "grid_safety_score": "Safety Score",
        "meets_18pct_target": "Meets 18% Target",
        "implementation_complexity": "Complexity",
        "customer_disruption": "Disruption",
    })
    st.dataframe(
        scen_display.style
        .highlight_max(subset=["Safety Score"], color="#dbeafe")
        .map(lambda v: "background-color:#dcfce7" if v is True else
                            ("background-color:#fee2e2" if v is False else ""), subset=["Meets 18% Target"]),
        use_container_width=True, hide_index=True,
    )


# ───────────────────────────────────────────────────────────────────────────
#  TAB 2 — INFRASTRUCTURE FACT SHEETS
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">EV Infrastructure Hotspot Analysis — DBSCAN Clustering</div>',
                unsafe_allow_html=True)

    c_left, c_right = st.columns([1.1, 1])

    with c_left:
        # ── Map ─────────────────────────────────────────────────────────
        map_df = hotspot_df.copy()
        map_df["marker_size"] = map_df["underservice_score"] * 7 + 8
        map_df["color"] = map_df["is_critical_hotspot"].map({True: "#e53e3e", False: "#0057c8"})
        map_df["hover_text"] = map_df.apply(
            lambda r: f"<b>{r['cluster_id']}</b><br>Zone: {r['zone']}<br>"
                      f"Underservice Score: {r['underservice_score']:.1f}/10<br>"
                      f"Sessions: {r['session_count']}<br>"
                      f"Growth Rate: {r['ev_growth_rate_pct']:.0f}% YoY",
            axis=1,
        )

        fig_map = go.Figure()
        fig_map.add_trace(go.Scattermapbox(
            lat=map_df["centroid_lat"],
            lon=map_df["centroid_lon"],
            mode="markers+text",
            marker=dict(
                size=map_df["marker_size"],
                color=map_df["color"],
                opacity=0.85,
            ),
            text=map_df["cluster_id"],
            textposition="top right",
            textfont=dict(size=10, color="#003d8f"),
            hovertext=map_df["hover_text"],
            hoverinfo="text",
            name="Hotspot Clusters",
        ))

        fig_map.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=12.97, lon=77.64),
                zoom=10.2,
            ),
            height=460,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0.01, y=0.98),
        )
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Zone Infrastructure Bar ──────────────────────────────────────
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_zones["zone"],
            y=df_zones["feeder_headroom_kw"],
            name="Available Headroom (kW)",
            marker_color="#0057c8",
        ))
        fig_bar.add_trace(go.Bar(
            x=df_zones["zone"],
            y=df_zones["feeder_capacity_kw"] - df_zones["feeder_headroom_kw"],
            name="Used Capacity (kW)",
            marker_color="#a0c0e8",
        ))
        fig_bar.add_trace(go.Scatter(
            x=df_zones["zone"],
            y=df_zones["ev_density"] / 50,  # scaled for visibility
            name="EV Density (÷50)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#e53e3e", width=2),
            marker=dict(size=8),
        ))
        fig_bar.update_layout(
            barmode="stack",
            height=300,
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Inter", size=11),
            yaxis=dict(title="Feeder Capacity (kW)", gridcolor="#e8edf5"),
            yaxis2=dict(title="EV Density Proxy", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=50, r=60, t=30, b=40),
            title="Zone Feeder Capacity vs EV Density",
        )
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        # ── Fact Sheets ──────────────────────────────────────────────────
        st.markdown("#### 📋 Hotspot Fact Sheets")

        filter_zone = st.selectbox("Filter by Zone", ["All"] + list(ZONE_CONFIG.keys()), key="hs_zone")
        show_critical = st.checkbox("Show Critical Hotspots Only", value=False)

        disp_df = hotspot_df.copy()
        if filter_zone != "All":
            disp_df = disp_df[disp_df["zone"] == filter_zone]
        if show_critical:
            disp_df = disp_df[disp_df["is_critical_hotspot"]]

        if disp_df.empty:
            st.warning("No hotspots match the current filter.")
        else:
            for _, row in disp_df.iterrows():
                critical_tag = '<span class="fact-tag critical">⚠ CRITICAL</span>' if row["is_critical_hotspot"] else '<span class="fact-tag">STANDARD</span>'
                util_pct = row["baseline_utilisation_pct"]
                util_color = "#e53e3e" if util_pct >= 90 else ("#f6ad55" if util_pct >= 75 else "#38a169")

                # Grid guardrail per hotspot zone
                guardrail_html = ""
                if util_pct >= 90:
                    guardrail_html = f'<div style="background:#fff0f0;border-left:4px solid #e53e3e;padding:6px 10px;border-radius:0 6px 6px 0;font-size:0.80rem;color:#c53030;margin-top:8px;">🔴 <b>GRID GUARDRAIL:</b> {row["zone"]} projected demand exceeds 90% feeder capacity — immediate action required.</div>'

                connectivity_stars = "★" * int(row["connectivity_score"] / 2) + "☆" * (5 - int(row["connectivity_score"] / 2))

                st.markdown(f"""
<div class="fact-sheet">
  <div class="fact-sheet-title">{row['cluster_id']} — {row['zone']}</div>
  <div style="margin:6px 0 10px">{critical_tag}</div>
  <table style="width:100%;font-size:0.82rem;border-collapse:collapse">
    <tr><td style="padding:3px 0;color:#4a5568;width:55%">📈 EV Growth Rate</td>
        <td style="font-weight:600;color:#003d8f">{row['ev_growth_rate_pct']:.0f}% per year</td></tr>
    <tr><td style="padding:3px 0;color:#4a5568">🚗 Zone EV Count</td>
        <td style="font-weight:600;color:#003d8f">{row['ev_density']:,}</td></tr>
    <tr><td style="padding:3px 0;color:#4a5568">📍 Charging Sessions</td>
        <td style="font-weight:600;color:#003d8f">{row['session_count']}</td></tr>
    <tr><td style="padding:3px 0;color:#4a5568">🏠 Existing Stations</td>
        <td style="font-weight:600;color:#003d8f">{row['existing_stations']}</td></tr>
    <tr><td style="padding:3px 0;color:#4a5568">🔌 Nearest Feeder</td>
        <td style="font-weight:600;color:#003d8f">{row['avg_distance_to_feeder_km']:.1f} km</td></tr>
    <tr><td style="padding:3px 0;color:#4a5568">📡 Connectivity Score</td>
        <td style="font-weight:600;color:#e08c00">{connectivity_stars} ({row['connectivity_score']})</td></tr>
    <tr><td style="padding:3px 0;color:#4a5568">⚡ Feeder Utilisation</td>
        <td style="font-weight:600;color:{util_color}">{util_pct:.1f}% of capacity</td></tr>
    <tr><td style="padding:3px 0;color:#4a5568">🏗️ Underservice Score</td>
        <td style="font-weight:600;color:#c53030">{row['underservice_score']:.1f} / 10</td></tr>
    <tr><td style="padding:3px 0;color:#4a5568">✅ Recommended New Stations</td>
        <td style="font-weight:700;color:#276749">+{row['recommended_new_stations']}</td></tr>
  </table>
  {guardrail_html}
  <div class="nl-insight">💬 {row['hotspot_insight']}</div>
</div>
""", unsafe_allow_html=True)

    # ── Underservice Radar / Scatter ─────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:8px">Zone Infrastructure Intelligence</div>',
                unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)

    with cc1:
        # Scatter: EV density vs Feeder headroom
        fig_scatter = px.scatter(
            df_zones,
            x="feeder_headroom_kw",
            y="ev_density",
            size="ev_growth_rate_pct",
            color="baseline_utilisation_pct",
            color_continuous_scale=[[0, "#38a169"], [0.6, "#f6ad55"], [1, "#e53e3e"]],
            text="zone",
            title="EV Density vs Feeder Headroom",
            labels={
                "feeder_headroom_kw": "Available Headroom (kW)",
                "ev_density": "EV Registrations",
                "baseline_utilisation_pct": "Feeder Util %",
            },
            hover_data=["connectivity_score", "avg_distance_to_feeder_km"],
            size_max=40,
        )
        fig_scatter.update_traces(textposition="top center", marker=dict(opacity=0.85))
        fig_scatter.update_layout(
            height=320, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Inter", size=11), margin=dict(l=40, r=20, t=40, b=40),
        )
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cc2:
        # Connectivity + underservice radar
        zones_for_radar = list(ZONE_CONFIG.keys())
        radar_metrics   = {
            "EV Density": [ZONE_CONFIG[z]["ev_density"] / 600 for z in zones_for_radar],
            "Growth Rate": [ZONE_CONFIG[z]["ev_growth_rate"] * 200 for z in zones_for_radar],
            "Connectivity": [ZONE_CONFIG[z]["connectivity_score"] for z in zones_for_radar],
            "Feeder Risk":  [(1 - ZONE_CONFIG[z]["feeder_headroom_kw"] / ZONE_CONFIG[z]["feeder_capacity_kw"]) * 10
                             for z in zones_for_radar],
            "Station Gap":  [10 - min(10, ZONE_CONFIG[z]["existing_stations"] / 2) for z in zones_for_radar],
        }
        categories = list(radar_metrics.keys())
        fig_radar = go.Figure()
        colors = ["#003d8f", "#0057c8", "#e53e3e", "#38a169", "#e08c00"]
        for i, zone in enumerate(zones_for_radar):
            vals = [radar_metrics[c][i] for c in categories]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=zone,
                line=dict(color=colors[i], width=2),
                fillcolor=f"rgba{tuple(int(colors[i].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + (0.10,)}",
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor="#e8edf5")),
            height=320,
            paper_bgcolor="#ffffff",
            font=dict(family="Inter", size=11),
            title="Zone Infra Risk Radar",
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Full Hotspot Table ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Complete Hotspot Registry</div>', unsafe_allow_html=True)
    table_cols = [
        "cluster_id", "zone", "session_count", "ev_density", "ev_growth_rate_pct",
        "avg_distance_to_feeder_km", "connectivity_score", "baseline_utilisation_pct",
        "underservice_score", "recommended_new_stations", "is_critical_hotspot",
    ]
    tbl = hotspot_df[table_cols].rename(columns={
        "cluster_id": "Cluster", "zone": "Zone", "session_count": "Sessions",
        "ev_density": "EV Count", "ev_growth_rate_pct": "Growth % YoY",
        "avg_distance_to_feeder_km": "Feeder Dist (km)",
        "connectivity_score": "Connectivity", "baseline_utilisation_pct": "Feeder Util %",
        "underservice_score": "Underservice Score", "recommended_new_stations": "New Stations",
        "is_critical_hotspot": "Critical",
    })
    st.dataframe(
        tbl.style
        .background_gradient(subset=["Underservice Score"], cmap="RdYlGn_r")
        .map(lambda v: "background-color:#fee2e2;color:#c53030;font-weight:700"
                  if v is True else "", subset=["Critical"]),
        use_container_width=True,
        hide_index=True,
    )

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:1px solid #dbe4f0;margin-top:30px">
<div style="text-align:center;color:#718096;font-size:0.78rem;padding-bottom:14px">
  NAMMA-EVCHARGE v1.0 &nbsp;|&nbsp; Bengaluru Electricity Supply Company (BESCOM) &nbsp;|&nbsp;
  AI-Powered Smart Grid Division &nbsp;|&nbsp; Data: Synthetic (SAE J1772 Standard)
</div>
""", unsafe_allow_html=True)