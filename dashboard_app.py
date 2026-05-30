
# ============================================================
#  Smart Home ML Dashboard — Streamlit Application
#  Alexandria University | ECE Graduation Project
#  Dataset: Oct–Dec 2024 (90 days, 1-min resolution)
#  Run: streamlit run dashboard_app.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Home ML Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0f1117; }
  .block-container { padding-top: 1rem; }
  .stMetric { background: #1a1d27; border-radius: 8px;
              padding: 12px; border: 1px solid #2e3248; }
  .stMetric label { color: #8890a4 !important; font-size:12px; }
  .stMetric [data-testid="stMetricValue"] { color: #e8eaf0; font-size:26px; }
  .stMetric [data-testid="stMetricDelta"] { font-size:12px; }
  h1,h2,h3 { color: #e8eaf0; }
  div[data-testid="stAlert"] { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Feature sets ───────────────────────────────────────────────────
FEATURES_ENERGY = [
    "temperature_c","humidity_pct","occupancy",
    "power_lag_1","power_lag_5","power_lag_10","power_lag_30","power_lag_60",
    "temp_lag_1","temp_lag_30",
    "power_roll_mean_10","power_roll_mean_30","power_roll_mean_60",
    "power_roll_std_10","power_roll_std_60",
    "power_delta_1","power_delta_5","temp_delta_1",
    "hour_sin","hour_cos","dow_sin","dow_cos","is_weekend","temp_x_occupancy",
]
FEATURES_OCCUPANCY = [
    "motion_pir","door_open","rfid_scan","rfid_recent",
    "power_w","power_roll_mean_10","power_roll_mean_30",
    "temperature_c","humidity_pct",
    "hour_sin","hour_cos","dow_sin","dow_cos","is_weekend",
]
FEATURES_POWER_ANOMALY = [
    "power_w","current_a","power_lag_1","power_lag_5","power_lag_60",
    "power_roll_mean_10","power_roll_mean_60","power_roll_std_10","power_roll_std_60",
    "power_delta_1","power_delta_5","occupancy","temperature_c",
    "hour_sin","hour_cos","is_weekend",
]
FEATURES_GAS_ANOMALY = [
    "gas_level_adc","temperature_c","humidity_pct",
    "occupancy","hour_sin","hour_cos","is_weekend",
]

# ── Load data & models (cached) ────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet("smart_home_features.parquet")
    df["is_power_anomaly"] = (df["power_anomaly_label"] > 0).astype(int)
    df["is_gas_anomaly"]   = (df["gas_alert_label"]     > 0).astype(int)
    return df

@st.cache_resource
def load_models():
    return {
        "energy"       : joblib.load("models/xgb_energy_model.joblib"),
        "occupancy"    : joblib.load("models/rf_occupancy_model.joblib"),
        "power_anomaly": joblib.load("models/iso_power_anomaly_model.joblib"),
        "gas_anomaly"  : joblib.load("models/iso_gas_anomaly_model.joblib"),
    }

@st.cache_data
def compute_predictions(_df, _models):
    df = _df.copy()
    df["energy_pred"]      = _models["energy"].predict(df[FEATURES_ENERGY])
    df["occ_prob"]         = _models["occupancy"].predict_proba(df[FEATURES_OCCUPANCY])[:, 1]
    # FIX A/F: Store raw scores first, then NEGATE so positive = more anomalous
    # This makes scores intuitive: 0 = perfectly normal, higher = more anomalous
    df["power_anom_score"] = -_models["power_anomaly"].decision_function(df[FEATURES_POWER_ANOMALY])
    df["gas_anom_score"]   = -_models["gas_anomaly"].decision_function(df[FEATURES_GAS_ANOMALY])
    p_thresh = np.percentile(df.loc[df["is_power_anomaly"]==0, "power_anom_score"], 97)
    g_thresh = np.percentile(df.loc[df["is_gas_anomaly"]==0,   "gas_anom_score"],   97)
    df["power_alert"] = (df["power_anom_score"] > p_thresh).astype(int)
    df["gas_alert_f"] = (df["gas_anom_score"]   > g_thresh).astype(int)
    return df, p_thresh, g_thresh

# ── Plotly theme ───────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
    font=dict(color="#c8ccd8", family="monospace"),
    xaxis=dict(gridcolor="#2e3248", linecolor="#2e3248"),
    yaxis=dict(gridcolor="#2e3248", linecolor="#2e3248"),
    legend=dict(bgcolor="#1a1d27", bordercolor="#2e3248"),
    margin=dict(l=50, r=20, t=40, b=50),
)

# Helper: build layout merging custom yaxis on top of PLOTLY_LAYOUT
def _layout(**overrides):
    """Merge overrides into PLOTLY_LAYOUT without key conflicts."""
    base = dict(PLOTLY_LAYOUT)
    # Merge yaxis carefully: combine gridcolor/linecolor with any custom keys
    if "yaxis" in overrides:
        overrides["yaxis"] = {**PLOTLY_LAYOUT["yaxis"], **overrides["yaxis"]}
    if "xaxis" in overrides:
        overrides["xaxis"] = {**PLOTLY_LAYOUT["xaxis"], **overrides["xaxis"]}
    base.pop("yaxis", None)
    base.pop("xaxis", None)
    base.update(overrides)
    return base

# ── Load everything ────────────────────────────────────────────────
df_raw = load_data()
models = load_models()
df, POWER_THRESH, GAS_THRESH = compute_predictions(df_raw, models)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Smart Home ML")
    st.markdown("**Alexandria University**")
    st.markdown("ECE Graduation Project")
    st.divider()

    # FIX I: Dataset context moved to top of sidebar
    st.markdown("**Dataset**")
    st.caption(f"{df.shape[0]:,} readings")
    st.caption(f"{df.index.min().date()} → {df.index.max().date()}")
    st.caption("1-min resolution, 90 days")
    st.divider()

    # FIX 8: Date filter at top of sidebar, above navigation
    st.markdown("**Date filter**")
    date_range = st.date_input(
        "Select range",
        value=(df.index.min().date(), df.index.max().date()),
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
        label_visibility="collapsed",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        df_view = df.loc[str(date_range[0]):str(date_range[1])]
    else:
        df_view = df
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Overview",
        "⚡ Energy",
        "👤 Occupancy",
        "🚨 Anomaly Detection",
        "📈 Model Performance",
    ])

# ════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    # FIX C: Remove misleading "Live" — this is a historical dataset
    st.title("🏠 Smart Home — Dashboard Overview")
    st.caption("📅 Historical dataset: Oct–Dec 2024  ·  1-minute resolution  ·  90 days")

    latest = df_view.iloc[-1]

    # FIX 3: Anomaly banner with full explanation
    n_alerts = int(df_view["power_alert"].sum() + df_view["gas_alert_f"].sum())
    if n_alerts == 0:
        st.success("✅ All systems normal — no alerts in selected period")
    else:
        alert_pct = n_alerts / max(len(df_view), 1) * 100
        st.warning(
            f"⚠️ {n_alerts:,} anomaly readings detected in selected period "
            f"({alert_pct:.1f}% of readings) — flagged by Isolation Forest at the 97th-percentile "
            f"score threshold. Includes power spikes, outages, and unoccupied-on events."
        )

    # KPI cards
    st.subheader("Key Performance Indicators")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_kwh  = df_view["energy_kwh"].sum()
    waste_kwh  = df_view.loc[(df_view["occupancy"]==0) & (df_view["power_w"]>200), "energy_kwh"].sum()
    occ_pct    = df_view["occupancy"].mean() * 100
    avg_temp   = df_view["temperature_c"].mean()
    avg_humid  = df_view["humidity_pct"].mean()
    avg_moist  = df_view["soil_moisture_pct"].mean()

    c1.metric("⚡ Total Energy",   f"{total_kwh:.1f} kWh")
    # FIX 6: delta_color="inverse" so red arrow = bad (waste going up is bad)
    c2.metric("♻️ Wasted Energy",  f"{waste_kwh:.1f} kWh",
              f"↑ {waste_kwh/max(total_kwh,1)*100:.1f}% of total", delta_color="inverse")
    c3.metric("👤 Occupancy",      f"{occ_pct:.1f}%")
    c4.metric("🌡️ Avg Temp",       f"{avg_temp:.1f} °C")
    c5.metric("💧 Humidity",       f"{avg_humid:.0f}%")
    c6.metric("🌿 Soil Moisture",  f"{avg_moist:.1f}%")

    st.divider()
    # FIX C: "Last recorded reading" not "Current" to avoid live implication
    st.subheader("Last Recorded Reading")
    st.caption(f"Timestamp: {df_view.index[-1].strftime('%Y-%m-%d %H:%M')}  ·  End of dataset")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Power",       f"{latest['power_w']:.0f} W")
    s2.metric("Temperature", f"{latest['temperature_c']:.0f} °C")
    s3.metric("Gas Level",   f"{latest['gas_level_adc']:.0f} ADC")
    s4.metric("Occupancy",   "Home" if latest["occupancy"] else "Away")

    st.divider()
    # FIX 12: Explicit date range in subtitle
    last7  = df_view.tail(7*24*60)
    date_from = last7.index.min().strftime("%b %d")
    date_to   = last7.index.max().strftime("%b %d, %Y")
    st.subheader(f"7-Day Sensor Overview  ({date_from} – {date_to})")
    hourly = last7.resample("h").mean(numeric_only=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Power (W)", "Temperature (°C)  &  Humidity (%)", "Gas ADC"],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=hourly.index, y=hourly["power_w"],
        fill="tozeroy", fillcolor="rgba(129,199,132,0.15)",
        line=dict(color="#81c784", width=1.5), name="Power"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hourly.index, y=hourly["temperature_c"],
        line=dict(color="#ff8a65", width=1.5), name="Temp (°C)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=hourly.index, y=hourly["humidity_pct"],
        line=dict(color="#4fc3f7", width=1.2, dash="dash"), name="Humidity (%)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=hourly.index, y=hourly["gas_level_adc"],
        fill="tozeroy", fillcolor="rgba(255,241,118,0.1)",
        line=dict(color="#fff176", width=1), name="Gas ADC"), row=3, col=1)
    # FIX K: Add secondary y-axis label note in subtitle
    fig.update_yaxes(title_text="Watts", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="ADC", row=3, col=1)
    fig.update_layout(**_layout(height=480),
                      title_text=f"Last 7 Days — Hourly Averages ({date_from} – {date_to})")
    # FIX J: Hide Plotly toolbar
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ════════════════════════════════════════════════════════════════════
# PAGE: ENERGY
# ════════════════════════════════════════════════════════════════════
elif page == "⚡ Energy":
    st.title("⚡ Energy Analysis")

    daily = df_view.resample("D").agg(
        energy_kwh=("energy_kwh", "sum"),
        avg_power =("power_w",    "mean"),
        avg_temp  =("temperature_c", "mean"),
    ).round(3)
    trend = daily["energy_kwh"].rolling(7, center=True, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily.index, y=daily["energy_kwh"],
        name="Daily kWh", marker_color="rgba(129,199,132,0.6)",
        marker_line_color="#2e3248", marker_line_width=0.5))
    fig.add_trace(go.Scatter(x=trend.index, y=trend.values,
        name="7-day avg", line=dict(color="white", width=2.5)))
    fig.update_layout(**_layout(height=350,
        yaxis={"title": "kWh"}, bargap=0.1,
        title="Daily Energy Consumption (kWh)"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    col1, col2 = st.columns(2)

    with col1:
        last48 = df_view.tail(48*60).resample("15min").mean(numeric_only=True)
        date_48_from = last48.index.min().strftime("%b %d")
        date_48_to   = last48.index.max().strftime("%b %d, %Y")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=last48.index, y=last48["power_w"],
            name="Actual", line=dict(color="#c8ccd8", width=1.5)))
        fig2.add_trace(go.Scatter(x=last48.index, y=last48["energy_pred"],
            name="XGBoost Pred", line=dict(color="#81c784", width=1.5, dash="dash")))
        fig2.update_layout(**_layout(height=320,
            title=f"Actual vs Predicted — Last 48h ({date_48_from}–{date_48_to})"))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with col2:
        hp = df_view.groupby("hour").agg(mean=("power_w","mean"), std=("power_w","std")).reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=list(hp["hour"]) + list(hp["hour"][::-1]),
            y=list(hp["mean"]+hp["std"]) + list((hp["mean"]-hp["std"])[::-1]),
            fill="toself", fillcolor="rgba(129,199,132,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="±1 std", showlegend=True))
        fig3.add_trace(go.Scatter(x=hp["hour"], y=hp["mean"],
            line=dict(color="#81c784", width=2.5), name="Mean power",
            mode="lines+markers", marker=dict(size=5)))
        # FIX H: Connect hour 23 back to hour 0 for smooth 24-hr cycle display
        fig3.add_trace(go.Scatter(
            x=[hp["hour"].iloc[-1], hp["hour"].iloc[0]+24],
            y=[hp["mean"].iloc[-1], hp["mean"].iloc[0]],
            line=dict(color="#81c784", width=2.5, dash="dot"),
            showlegend=False, mode="lines"))
        fig3.update_layout(**_layout(height=320,
            xaxis={"title": "Hour of Day", "tickvals": list(range(0, 24, 2))},
            yaxis={"title": "Watts"},
            title="24-Hour Average Power Profile (all 90 days)"))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # Energy Waste Analysis
    st.subheader("Energy Waste Analysis")
    # FIX B: Show formula explaining the calculation
    st.caption(
        "ℹ️ Wasted energy = energy consumed while **unoccupied** and power > 200 W "
        "(standby threshold). Represents avoidable consumption over the selected period."
    )
    occ_mean   = df_view.loc[df_view["occupancy"]==1, "power_w"].mean()
    unocc_mean = df_view.loc[df_view["occupancy"]==0, "power_w"].mean()
    waste_est  = df_view.loc[(df_view["occupancy"]==0) & (df_view["power_w"]>200), "energy_kwh"].sum()
    total_kwh2 = df_view["energy_kwh"].sum()
    w1, w2, w3 = st.columns(3)
    w1.metric("Avg Power (Occupied)",   f"{occ_mean:.0f} W",
              "Active appliances + AC", delta_color="off")
    w2.metric("Avg Power (Unoccupied)", f"{unocc_mean:.0f} W",
              "Fridge + standby + AC", delta_color="off")
    w3.metric("Estimated Wasted Energy", f"{waste_est:.1f} kWh",
              f"↑ {waste_est/max(total_kwh2,1)*100:.1f}% of total", delta_color="inverse")

# ════════════════════════════════════════════════════════════════════
# PAGE: OCCUPANCY
# ════════════════════════════════════════════════════════════════════
elif page == "👤 Occupancy":
    st.title("👤 Occupancy Detection")

    col1, col2 = st.columns([3, 2])

    with col1:
        # FIX D: Reduce textfont size and remove annotations to fix overflow
        occ_hm = df_view.groupby(["day_of_week","hour"])["occupancy"].mean().unstack()
        day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        if len(occ_hm) == 7:
            occ_hm.index = day_labels
        # Use color-only heatmap — no text labels (they overflow)
        fig_hm = go.Figure(go.Heatmap(
            z=occ_hm.values * 100,
            x=list(range(24)),
            y=occ_hm.index.tolist(),
            colorscale=[[0,"#0f1117"],[0.3,"#1a237e"],[0.6,"#3949ab"],[1.0,"#e8eaf6"]],
            colorbar=dict(title="Occupancy %", ticksuffix="%"),
            hovertemplate="Hour %{x}:00  |  %{y}  |  %{z:.0f}% occupied<extra></extra>",
            zmin=0, zmax=100,
        ))
        fig_hm.update_layout(**_layout(height=340,
            xaxis={"title": "Hour of Day", "tickvals": list(range(0, 24, 2))},
            yaxis={"title": ""},
            title="Occupancy Rate (%) — Hour × Day of Week"))
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})
        st.caption("Hover over cells for exact values. Colour scale: dark = empty, light = occupied.")

    with col2:
        wkday = df_view[df_view["is_weekend"]==0].groupby("hour")["occupancy"].mean()*100
        wkend = df_view[df_view["is_weekend"]==1].groupby("hour")["occupancy"].mean()*100
        fig_wk = go.Figure()
        fig_wk.add_trace(go.Scatter(x=wkday.index, y=wkday.values,
            name="Weekday", line=dict(color="#4fc3f7", width=2),
            fill="tozeroy", fillcolor="rgba(79,195,247,0.1)"))
        fig_wk.add_trace(go.Scatter(x=wkend.index, y=wkend.values,
            name="Weekend", line=dict(color="#ce93d8", width=2),
            fill="tozeroy", fillcolor="rgba(206,147,216,0.1)"))
        fig_wk.update_layout(**_layout(height=340,
            xaxis={"title": "Hour of Day"},
            yaxis={"title": "Occupied (%)"},
            title="Weekday vs Weekend Profile"))
        st.plotly_chart(fig_wk, use_container_width=True, config={"displayModeBar": False})

    # Prediction Timeline — FIX 1: TypeError fixed via _layout helper
    st.subheader("Model Prediction Timeline")
    n_days = st.slider("Days to display", 1, 14, 5)
    timeline = df_view.tail(n_days*24*60).resample("15min").mean(numeric_only=True)
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(x=timeline.index, y=timeline["occupancy"],
        fill="tozeroy", fillcolor="rgba(200,200,216,0.25)",
        line=dict(color="#c8ccd8", width=1), name="Actual"))
    fig_tl.add_trace(go.Scatter(x=timeline.index, y=timeline["occ_prob"],
        fill="tozeroy", fillcolor="rgba(206,147,216,0.2)",
        line=dict(color="#ce93d8", width=1.5, dash="dash"), name="Model probability"))
    fig_tl.add_hline(y=0.5, line_dash="dot", line_color="#8890a4",
                     annotation_text="Decision threshold (0.5)")
    fig_tl.update_layout(**_layout(
        height=320,
        yaxis={"title": "Occupancy (0/1)", "range": [-0.05, 1.15]},
        title=f"Actual vs Predicted Occupancy — Last {n_days} Days (15-min avg)",
    ))
    st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": False})

# ════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    n_power = int(df_view["power_alert"].sum())
    n_gas   = int(df_view["gas_alert_f"].sum())
    a1, a2, a3 = st.columns(3)
    a1.metric("Power Anomalies", f"{n_power:,}")
    a2.metric("Gas Anomalies",   f"{n_gas:,}")
    a3.metric("Total Alerts",    f"{n_power+n_gas:,}")

    # FIX A: Scores are already positive (higher = more anomalous).
    # Threshold is also positive. Spikes naturally go UP — no axis reversal needed.
    # The chart now reads intuitively: calm baseline near 0, spikes above threshold.
    hourly_anom = df_view.resample("h").agg(
        power_anom_score=("power_anom_score", "max"),
        gas_anom_score  =("gas_anom_score",   "max"),
        power_w         =("power_w",          "mean"),
        power_alert     =("power_alert",      "max"),
        gas_alert_f     =("gas_alert_f",      "max"),
    )

    fig_sc = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Power Anomaly Score", "Gas Anomaly Score"],
        vertical_spacing=0.12)

    fig_sc.add_trace(go.Scatter(
        x=hourly_anom.index, y=hourly_anom["power_anom_score"],
        line=dict(color="#4fc3f7", width=0.8), name="Power score",
        fill="tozeroy", fillcolor="rgba(79,195,247,0.08)"), row=1, col=1)
    fig_sc.add_hline(y=POWER_THRESH, line_dash="dash", line_color="white",
                     annotation_text="Alert threshold", row=1, col=1)
    fig_sc.add_trace(go.Scatter(
        x=hourly_anom.index[hourly_anom["power_alert"]==1],
        y=hourly_anom.loc[hourly_anom["power_alert"]==1, "power_anom_score"],
        mode="markers", marker=dict(color="#ef5350", size=6),
        name="Alert"), row=1, col=1)

    fig_sc.add_trace(go.Scatter(
        x=hourly_anom.index, y=hourly_anom["gas_anom_score"],
        line=dict(color="#fff176", width=0.8), name="Gas score",
        fill="tozeroy", fillcolor="rgba(255,241,118,0.06)"), row=2, col=1)
    fig_sc.add_hline(y=GAS_THRESH, line_dash="dash", line_color="white",
                     annotation_text="Alert threshold", row=2, col=1)

    fig_sc.update_yaxes(title_text="Score (higher = more anomalous)", row=1, col=1)
    fig_sc.update_yaxes(title_text="Score (higher = more anomalous)", row=2, col=1)
    fig_sc.update_layout(**_layout(height=480,
        title="Anomaly Scores Over Time — spikes above threshold trigger alerts (hourly max)"))
    st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    # Alert Log
    st.subheader("Alert Log")
    st.caption(
        "ℹ️ Anomaly Score: higher value = more anomalous. "
        "Threshold = 97th percentile of normal scores. "
        "Type is based on power level and occupancy at time of alert."
    )

    alert_df = df_view[df_view["power_alert"]==1][[
        "power_w", "temperature_c", "occupancy", "power_anom_score"
    ]].copy()

    # FIX L: Improved alert type logic
    # Outage: power < 10W (clearly a loss of supply)
    # Waste:  occupancy=0 AND power 200–3000W (unoccupied but drawing significant load)
    # Fault:  everything else (unexpected spike — could be any power level when occupied)
    alert_df["Type"] = alert_df.apply(
        lambda r: "🔴 Outage" if r["power_w"] < 10
        else "🟡 Waste"  if (r["occupancy"] == 0 and r["power_w"] < 3000)
        else "🟠 Fault", axis=1)

    # FIX 4a: Round to 1 decimal (sensor accuracy ±15W — 4 decimals is misleading)
    alert_df["power_w"]        = alert_df["power_w"].round(1)
    alert_df["temperature_c"]  = alert_df["temperature_c"].round(1)
    # FIX F: Scores are already positive — round to 4 decimal places
    alert_df["power_anom_score"] = alert_df["power_anom_score"].round(4)
    # FIX 4b: Consistent Title Case column names + proper units
    alert_df["occupancy"] = alert_df["occupancy"].map({0: "No", 1: "Yes"})
    alert_df = alert_df.rename(columns={
        "power_w":          "Power (W)",
        "temperature_c":    "Temp (°C)",
        "occupancy":        "Occupied",
        "power_anom_score": "Anomaly Score",
    })
    # FIX E: "Timestamp" capitalised (index name)
    alert_df.index.name = "Timestamp"
    alert_df = alert_df[["Power (W)", "Temp (°C)", "Occupied", "Anomaly Score", "Type"]]
    st.dataframe(alert_df.tail(100), use_container_width=True, height=280)

# ════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error,
        f1_score, roc_auc_score, roc_curve,
    )

    split_idx = int(len(df) * 0.8)
    test_df   = df.iloc[split_idx:]

    mae_xgb   = mean_absolute_error(test_df["power_w"], test_df["energy_pred"])
    rmse_xgb  = float(np.sqrt(mean_squared_error(test_df["power_w"], test_df["energy_pred"])))
    ss_res    = float(np.sum((test_df["power_w"].values - test_df["energy_pred"].values)**2))
    ss_tot    = float(np.sum((test_df["power_w"].values - test_df["power_w"].mean())**2))
    r2_xgb    = float(1 - ss_res/ss_tot)
    mape_xgb  = float(np.mean(np.abs(
        (test_df["power_w"].values - test_df["energy_pred"].values) /
        (np.abs(test_df["power_w"].values) + 1e-8)
    )) * 100)

    occ_pred  = (test_df["occ_prob"] >= 0.5).astype(int)
    f1_occ    = float(f1_score(test_df["occupancy"], occ_pred))
    auc_occ   = float(roc_auc_score(test_df["occupancy"], test_df["occ_prob"]))

    # FIX 2: Power AUC corrected to 0.9088 (was hardcoded as 0.8749)
    # FIX 11: MAPE added to table (was missing)
    st.subheader("Results Summary")
    metrics_df = pd.DataFrame([
        {"Subsystem": "Energy Forecasting",  "Model": "XGBoost",          "Metric": "MAE",       "Value": f"{mae_xgb:.2f} W"},
        {"Subsystem": "Energy Forecasting",  "Model": "XGBoost",          "Metric": "RMSE",      "Value": f"{rmse_xgb:.2f} W"},
        {"Subsystem": "Energy Forecasting",  "Model": "XGBoost",          "Metric": "MAPE",      "Value": f"{mape_xgb:.2f}%"},
        {"Subsystem": "Energy Forecasting",  "Model": "XGBoost",          "Metric": "R²",        "Value": f"{r2_xgb:.4f}"},
        {"Subsystem": "Occupancy Detection", "Model": "Random Forest",    "Metric": "F1-Score",  "Value": f"{f1_occ:.4f}"},
        {"Subsystem": "Occupancy Detection", "Model": "Random Forest",    "Metric": "ROC-AUC",   "Value": f"{auc_occ:.4f}"},
        {"Subsystem": "Anomaly Detection",   "Model": "Isolation Forest", "Metric": "Power AUC", "Value": "0.9088"},
        {"Subsystem": "Anomaly Detection",   "Model": "Isolation Forest", "Metric": "Gas AUC",   "Value": "0.6833"},
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.caption(
        "Test set = last 20% of data chronologically (Dec 14–29, 2024). "
        "Energy & Occupancy metrics computed on held-out test split. "
        "Anomaly AUC from stratified evaluation (required due to extreme class imbalance: 161 anomalies in 129,540 readings)."
    )

    col1, col2 = st.columns(2)

    with col1:
        residuals = test_df["power_w"].values - test_df["energy_pred"].values
        fig_res = go.Figure(go.Histogram(
            x=residuals, nbinsx=80,
            marker_color="rgba(79,195,247,0.75)",
            marker_line_color="#0f1117", marker_line_width=0.3,
            name="Residuals"))
        fig_res.add_vline(x=0, line_color="white", line_width=2)
        fig_res.add_vline(x=float(residuals.mean()),
                          line_color="#ff8a65", line_dash="dash",
                          annotation_text=f"Mean={residuals.mean():.1f}W")
        fig_res.update_layout(**_layout(height=350,
            xaxis={"title": "Residual (W)"},
            yaxis={"title": "Count"},
            title=f"XGBoost Energy Residuals (MAE={mae_xgb:.1f}W, R²={r2_xgb:.4f})"))
        st.plotly_chart(fig_res, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fpr, tpr, _ = roc_curve(test_df["occupancy"], test_df["occ_prob"])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
            fill="tozeroy", fillcolor="rgba(206,147,216,0.1)",
            line=dict(color="#ce93d8", width=2.5),
            name=f"ROC AUC = {auc_occ:.4f}"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
            line=dict(color="#8890a4", width=1, dash="dash"),
            name="Random classifier"))
        fig_roc.update_layout(**_layout(height=350,
            xaxis={"title": "False Positive Rate"},
            yaxis={"title": "True Positive Rate"},
            title="Occupancy Detection — ROC Curve"))
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
