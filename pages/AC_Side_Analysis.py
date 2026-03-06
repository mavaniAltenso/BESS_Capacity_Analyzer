# In pages/AC_Side_Analysis.py

import streamlit as st
st.set_page_config(layout="wide", page_title="AC Analysis", page_icon="⚡")
from utils import check_login
check_login() # <-- This is the guard for the page

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
import re
from typing import Optional, Dict, List, Tuple, Set
from pathlib import Path
import io

def _trapz_compat(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x)
    elif hasattr(np, "trapz"):
        return np.trapz(y, x=x)
    else:
        raise AttributeError("Neither np.trapezoid nor np.trapz is available in this NumPy version.")
# Import shared functions
from utils import convert_mixed_numeric_columns, _sanitize_time_col, _check_cadence


# SECTION 1: AC DATA LOADING FUNCTION (Unchanged)


@st.cache_data
def load_hycon_hybrid_fast(uploaded_file,
                            sep=';',
                            encoding='latin1',
                            strip_trailing_hyphen=True,
                            parse_timestamp_utc=False):
    """
    Loads the AC-side CSV file from an uploaded file object.
    """
    file_buffer = io.StringIO(uploaded_file.getvalue().decode(encoding))
    row6 = row8 = None
    file_buffer.seek(0)
    for i, line in enumerate(file_buffer, start=1):
        if i == 6:
            row6 = line.rstrip('\n')
        elif i == 8:
            row8 = line.rstrip('\n')
            break
    if row6 is None or row8 is None:
        raise ValueError("File is too short or missing required header lines 6 and 8.")

    h6_parts = [p.strip() for p in row6.split(sep)]
    h8_parts = [p.strip() for p in row8.split(sep)]
    width = max(len(h6_parts), len(h8_parts))
    if len(h6_parts) < width: h6_parts += [''] * (width - len(h6_parts))
    if len(h8_parts) < width: h8_parts += [''] * (width - len(h8_parts))
    combined = [f"{a} {b}".strip() for a, b in zip(h6_parts, h8_parts)]
    combined = [pd.Series([c]).str.replace(r'\s+', ' ', regex=True).iloc[0].strip() for c in combined]
    if strip_trailing_hyphen:
        combined = [pd.Series([c]).str.replace(r'\s*-\s*$', '', regex=True).iloc[0] for c in combined]
    if combined: combined[0] = 'Timestamp'

    file_buffer.seek(0)
    df = pd.read_csv(file_buffer, sep=sep, skiprows=8, header=None, names=combined, encoding=encoding, low_memory=False)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=parse_timestamp_utc)
    return df


# SECTION 2.5: 3-STATE DETECTION FUNCTION (Unchanged)


@st.cache_data
def find_all_events(df_ac: pd.DataFrame, time_col: str, power_col: str, idle_threshold_kw: float, min_duration_s: float) -> List[Dict]:
    if df_ac is None: return []
    if power_col not in df_ac.columns or time_col not in df_ac.columns: 
        st.warning(f"Column '{power_col}' or '{time_col}' not found. Event detection failed.")
        return []
    d = df_ac[[time_col, power_col]].dropna().copy()
    if d.empty: return []
    d = _sanitize_time_col(d, time_col)
    if len(d) < 2: return []

    def get_state(power):
        if power > idle_threshold_kw: return 1  # Discharge
        elif power < -idle_threshold_kw: return -1 # Charge
        else: return 0  # Idle

    d['state'] = d[power_col].apply(get_state)
    d['state_block'] = (d['state'] != d['state'].shift(1)).cumsum()
    events = []
    for name, group in d.groupby('state_block'):
        if group.empty: continue
        block_state = group['state'].iloc[0]
        if block_state == 0: continue
        start_time = group[time_col].iloc[0]
        end_time = group[time_col].iloc[-1]
        if (end_time - start_time).total_seconds() >= min_duration_s:
            events.append({"type": "Discharge" if block_state == 1 else "Charge", "start": start_time, "end": end_time})
    return sorted(events, key=lambda x: x['start'])


# SECTION 3: AC-SIDE ANALYSIS FUNCTIONS (MERGED KPI LOGIC)


def compute_nominal_from_poi_plotly(
    df: pd.DataFrame, discharge_start, discharge_end, 
    time_col: str = "Timestamp", power_col: str = "PoiPwrAt kW", title: str = "AC Power Analysis (Plotly)",
    drop_duplicate_timestamps: bool = True, charge_start: Optional[str] = None, charge_end: Optional[str] = None,
    sampling_seconds: Optional[float] = None, warn_irregular: bool = True,
    rte_min_charge_kWh: float = 0.01, soc_col: Optional[str] = None,
    P_nom_charge_kW: Optional[float] = None, tol_charge_pct: Optional[float] = None,
    P_nom_discharge_kW: Optional[float] = None, tol_discharge_pct: Optional[float] = None
) -> Dict:
    
    ts_start_raw = pd.to_datetime(discharge_start, errors="coerce")
    ts_end_raw = pd.to_datetime(discharge_end, errors="coerce")
    if pd.isna(ts_start_raw) or pd.isna(ts_end_raw): raise ValueError("Invalid discharge times.")
    if ts_end_raw <= ts_start_raw: raise ValueError("discharge_end must be after discharge_start")
    c_start_raw = pd.to_datetime(charge_start, errors="coerce")
    c_end_raw = pd.to_datetime(charge_end, errors="coerce")

    all_times = [ts_start_raw, ts_end_raw]
    has_charge = pd.notna(c_start_raw) and pd.notna(c_end_raw)
    if has_charge: all_times.extend([c_start_raw, c_end_raw])
    plot_start, plot_end = min(all_times), max(all_times)

    ts_start, ts_end = ts_start_raw, ts_end_raw
    c_start, c_end = c_start_raw, c_end_raw

    if time_col not in df.columns or power_col not in df.columns: raise KeyError(f"Missing columns.")
    warnings_list = []
    cols_to_keep = {time_col, power_col}
    if soc_col and soc_col in df.columns: cols_to_keep.add(soc_col)
    elif soc_col: warnings_list.append(f"SOC column '{soc_col}' not found.")
    
    d = df[list(cols_to_keep)].copy()
    d = _sanitize_time_col(d, time_col)
    if drop_duplicate_timestamps: d = d.groupby(time_col, as_index=False).mean()
    if d.empty: raise ValueError("No valid data.")

    fig_soc = go.Figure()
    if soc_col and soc_col in d.columns:
        fig_soc.add_trace(go.Scatter(x=d[time_col], y=d[soc_col], mode="lines", name="SOC (%)", line=dict(color="#FF5733", width=2)))
        if has_charge: 
            fig_soc.add_vrect(x0=c_start, x1=c_end, fillcolor="green", opacity=0.15, layer="below", line_width=0, name="Charge Window")
        fig_soc.add_vrect(x0=ts_start, x1=ts_end, fillcolor="red", opacity=0.15, layer="below", line_width=0, name="Discharge Window")
        
        fig_soc.update_layout(
            title=f"System SOC ({soc_col})", 
            xaxis_title="Time", 
            yaxis_title="SOC (%)", 
            template="plotly_white", 
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
        )
    else: 
        fig_soc.update_layout(title="SOC Plot (Not Available)")

    dPlot = d[(d[time_col] >= plot_start) & (d[time_col] <= plot_end)].copy()
    if dPlot.empty: raise ValueError("No data in plot window.")

    # --- KPI Calculation ---
    # This section now uses the Nominal Discharge parameters by default
    dKPI = d[(d[time_col] >= ts_start) & (d[time_col] <= ts_end)][[time_col, power_col]].copy()
    if dKPI.empty or len(dKPI) < 2: raise ValueError("Insufficient data in discharge window.")
    
    dKPI["dt_s"] = dKPI[time_col].diff().dt.total_seconds()
    first_dt = (dKPI.iloc[1][time_col] - dKPI.iloc[0][time_col]).total_seconds()
    dKPI.iloc[0, dKPI.columns.get_loc("dt_s")] = first_dt
    dKPI = dKPI[dKPI["dt_s"] > 0].copy() 

    dKPI["E_kWh_slice"] = dKPI[power_col] * (dKPI["dt_s"] / 3600.0)
    actual_energy_kWh = dKPI["E_kWh_slice"].sum()

    # --- MODIFIED: Use P_nom_discharge_kW and tol_discharge_pct for KPI ---
    if P_nom_discharge_kW is None or tol_discharge_pct is None:
        raise ValueError("Nominal Discharge Power and Tolerance must be set for KPI calculation.")
        
    P_nom_kW_kpi = float(P_nom_discharge_kW)
    tol_pct_kpi = float(tol_discharge_pct)
    
    band_low = P_nom_kW_kpi * (1 - tol_pct_kpi / 100.0)
    band_high = P_nom_kW_kpi * (1 + tol_pct_kpi / 100.0)
    # --- END MODIFICATION ---
    
    dKPI["in_band"] = (dKPI[power_col] >= band_low) & (dKPI[power_col] <= band_high)
    
    shapes = []
    
    kpi_segs, in_seg, acc_s, seg_start = [], False, 0.0, None
    for i, row in dKPI.iterrows():
        if row["in_band"]:
            if not in_seg: in_seg, seg_start, acc_s = True, row[time_col], row["dt_s"]
            else: acc_s += row["dt_s"]
        elif in_seg: kpi_segs.append((seg_start, row[time_col], acc_s)); in_seg = False
    if in_seg: kpi_segs.append((seg_start, dKPI.iloc[-1][time_col], acc_s))
    
    for s, e, _ in kpi_segs: 
        shapes.append(dict(type="rect", x0=s, x1=e, y0=band_low, y1=band_high, 
                           fillcolor="rgba(44, 160, 44, 0.2)", # Green
                           line_width=0, layer="below"))
    
    longest_s = max([s for *_, s in kpi_segs], default=0.0)

    # --- RTE Calculations ---
    E_ch_kWh, E_dis_kWh, RTE_pct, E_ch_nom, E_dis_nom, RTE_nom = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    ch_nom_start_ts, ch_nom_end_ts = pd.NaT, pd.NaT
    dis_nom_start_ts, dis_nom_end_ts = pd.NaT, pd.NaT
    df_calc_poi_egymtr = None

    if has_charge:
        cols_for_ce = [time_col, power_col]
        if soc_col in d.columns: cols_for_ce.append(soc_col)
        d_ce = d[(d[time_col] >= c_start) & (d[time_col] <= ts_end)][cols_for_ce].copy()

        if not d_ce.empty and len(d_ce) > 1:
             d_ce["dt_s"] = d_ce[time_col].diff().dt.total_seconds()
             first_dt_ce = (d_ce.iloc[1][time_col] - d_ce.iloc[0][time_col]).total_seconds()
             d_ce.iloc[0, d_ce.columns.get_loc("dt_s")] = first_dt_ce
             d_ce = d_ce[d_ce["dt_s"] > 0].copy()
             d_ce["Calc-PoiEgy"] = d_ce[power_col] * (d_ce["dt_s"] / 3600.0)
             d_ce["Calc-PoiEgyMtr"] = d_ce["Calc-PoiEgy"].cumsum()
             cols_for_df_calc = [time_col, "Calc-PoiEgy", "Calc-PoiEgyMtr"]
             if soc_col in d_ce.columns: cols_for_df_calc.append(soc_col)
             df_calc_poi_egymtr = d_ce[cols_for_df_calc]

        dd = d[[time_col, power_col]].copy()
        # --- MODIFIED: Removed discharge_positive logic ---
        dd["P"] = dd[power_col] 
        d_ch = dd[(dd[time_col] >= c_start) & (dd[time_col] <= c_end)].copy()
        d_dis = dd[(dd[time_col] >= ts_start) & (dd[time_col] <= ts_end)].copy()

        for sub in [d_ch, d_dis]:
             if not sub.empty and len(sub) > 1:
                 sub["dt_s"] = sub[time_col].diff().dt.total_seconds()
                 first_dt_sub = (sub.iloc[1][time_col] - sub.iloc[0][time_col]).total_seconds()
                 sub.iloc[0, sub.columns.get_loc("dt_s")] = first_dt_sub
                 sub = sub[sub["dt_s"] > 0]

        P_ch_star = (-d_ch["P"]).clip(lower=0.0).to_numpy() if not d_ch.empty else np.array([])
        P_dis_star = d_dis["P"].clip(lower=0.0).to_numpy() if not d_dis.empty else np.array([])
        
        if not d_ch.empty:
            E_ch_kWh = _trapz_compat(P_ch_star, x=d_ch[time_col].astype("int64") / 1e9) / 3600.0
        if not d_dis.empty:
            E_dis_kWh = _trapz_compat(P_dis_star, x=d_dis[time_col].astype("int64") / 1e9) / 3600.0
        if not np.isnan(E_ch_kWh) and E_ch_kWh > rte_min_charge_kWh: RTE_pct = 100.0 * E_dis_kWh / E_ch_kWh
        else: warnings_list.append(f"Full E_charge ({E_ch_kWh:.2f} kWh) is below min threshold for RTE calc.")

        # --- Nominal RTE ---
        if P_nom_charge_kW is not None and tol_charge_pct is not None and not d_ch.empty:
            P_c, tol_c = float(P_nom_charge_kW), float(tol_charge_pct)/100.0
            ch_band_low, ch_band_high = P_c - abs(P_c)*tol_c, P_c + abs(P_c)*tol_c
            
            d_ch['in_nom_band'] = (d_ch["P"] >= ch_band_low) & (d_ch["P"] <= ch_band_high)
            d_ch_nom = d_ch[d_ch['in_nom_band']]

            ch_segs, in_seg, acc_s, seg_start = [], False, 0.0, None
            for i, row in d_ch.iterrows():
                if row["in_nom_band"]:
                    if not in_seg: in_seg, seg_start, acc_s = True, row[time_col], row["dt_s"]
                    else: acc_s += row["dt_s"]
                elif in_seg: ch_segs.append((seg_start, row[time_col], acc_s)); in_seg = False
            if in_seg: ch_segs.append((seg_start, d_ch.iloc[-1][time_col], acc_s))

            for s, e, _ in ch_segs: 
                shapes.append(dict(type="rect", x0=s, x1=e, y0=ch_band_low, y1=ch_band_high, 
                                   fillcolor="rgba(148, 103, 189, 0.2)", # Purple
                                   line_width=0, layer="below"))
            
            if not d_ch_nom.empty: 
                E_ch_nom = _trapz_compat((-d_ch_nom["P"]).clip(lower=0).to_numpy(),x=d_ch_nom[time_col].astype("int64") / 1e9) / 3600.0
                ch_nom_start_ts = d_ch_nom[time_col].min()
                ch_nom_end_ts = d_ch_nom[time_col].max()
            else: warnings_list.append(f"No charge data found in Nominal Band [{ch_band_low:.0f}, {ch_band_high:.0f}] kW.")

        if P_nom_discharge_kW is not None and tol_discharge_pct is not None and not d_dis.empty:
            P_d, tol_d = float(P_nom_discharge_kW), float(tol_discharge_pct)/100.0
            dis_band_low, dis_band_high = P_d - abs(P_d)*tol_d, P_d + abs(P_d)*tol_d
            d_dis_nom = d_dis[(d_dis["P"] >= dis_band_low) & (d_dis["P"] <= dis_band_high)]
            if not d_dis_nom.empty: 
                E_dis_nom = np._trapz_compat(d_dis_nom["P"].clip(lower=0).to_numpy(),x=d_dis_nom[time_col].astype("int64") / 1e9) / 3600.0
                dis_nom_start_ts = d_dis_nom[time_col].min()
                dis_nom_end_ts = d_dis_nom[time_col].max()
            else: warnings_list.append(f"No discharge data found in Nominal Band [{dis_band_low:.0f}, {dis_band_high:.0f}] kW.")
            
        if not np.isnan(E_ch_nom) and E_ch_nom > rte_min_charge_kWh: RTE_nom = 100.0 * E_dis_nom / E_ch_nom
        else: warnings_list.append(f"Nominal E_charge ({E_ch_nom:.2f} kWh) is below min threshold for RTE calc.")

    # --- Plotting ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dPlot[time_col], y=dPlot[power_col], mode="lines", name=f"{power_col}", line=dict(color="#1f77b4", width=2)))
    x_min, x_max = dPlot[time_col].min(), dPlot[time_col].max()
    # --- MODIFIED: Use the KPI band values for the trace ---
    fig.add_trace(go.Scatter(x=[x_min, x_max], y=[band_low, band_low], mode="lines", name=f"Discharge Band (±{tol_pct_kpi:.0f}%)", line=dict(color="#2ca02c", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=[x_min, x_max], y=[band_high, band_high], mode="lines", line=dict(color="#2ca02c", width=1.5, dash="dash"), showlegend=False))

    if has_charge and P_nom_charge_kW is not None and tol_charge_pct is not None:
        P_c, tol_c = float(P_nom_charge_kW), float(tol_charge_pct)/100.0
        fig.add_trace(go.Scatter(x=[x_min, x_max], y=[P_c - abs(P_c)*tol_c]*2, mode="lines", name=f"Charge Band (±{tol_charge_pct:.0f}%)", line=dict(color="#9467bd", width=1.5, dash="dash")))
        fig.add_trace(go.Scatter(x=[x_min, x_max], y=[P_c + abs(P_c)*tol_c]*2, mode="lines", line=dict(color="#9467bd", width=1.5, dash="dash"), showlegend=False))

    fig.update_layout(shapes=shapes)
    
    fig.update_layout(
        title=title, 
        xaxis_title="Time", 
        yaxis_title="Power (kW)", 
        template="plotly_white", 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    )

    # --- New Summary Table ---
    metrics, values = [], []
    metrics.extend(["--- OVERALL SETTINGS ---", "Total Analysis Start", "Total Analysis End", "Sampling Interval (s)"])
    values.extend(["", plot_start.strftime('%Y-%m-%d %H:%M:%S'), plot_end.strftime('%Y-%m-%d %H:%M:%S'), sampling_seconds])

    # --- MODIFIED: Merged KPI results ---
    metrics.extend(["--- DISCHARGE TEST RESULTS ---", "Discharge Window Start", "Discharge Window End", "Nominal Power (kW)", "Tolerance (%)", "Total Actual Energy (kWh)", "Continuous In-Band Time (min)"])
    values.extend(["", ts_start.strftime('%H:%M:%S'), ts_end.strftime('%H:%M:%S'), P_nom_kW_kpi, tol_pct_kpi, round(actual_energy_kWh, 3), round(longest_s/60.0, 3)])

    if has_charge:
        metrics.extend(["--- FULL-CYCLE RTE ---", "Charge Start Time", "Charge End Time", "Discharge Start Time", "Discharge End Time", "Total Energy IN (kWh)", "Total Energy OUT (kWh)", "Full-Cycle RTE (%)"])
        values.extend(["", c_start.strftime('%H:%M:%S'), c_end.strftime('%H:%M:%S'), ts_start.strftime('%H:%M:%S'), ts_end.strftime('%H:%M:%S'), round(E_ch_kWh, 3) if not np.isnan(E_ch_kWh) else None, round(E_dis_kWh, 3) if not np.isnan(E_dis_kWh) else None, round(RTE_pct, 3) if not np.isnan(RTE_pct) else None])

        metrics.extend([
            "--- NOMINAL POWER RTE ---",
            "Charge Band",
            "Nominal Charge Start",
            "Nominal Charge End",
            "Filtered Energy IN (kWh)",
            "Discharge Band",
            "Nominal Discharge Start",
            "Nominal Discharge End",
            "Filtered Energy OUT (kWh)",
            "Nominal Power RTE (%)"
        ])
        values.extend([
            "", # Separator
            f"[{P_nom_charge_kW}kW ±{tol_charge_pct}%]",
            ch_nom_start_ts.strftime('%H:%M:%S') if pd.notna(ch_nom_start_ts) else None,
            ch_nom_end_ts.strftime('%H:%M:%S') if pd.notna(ch_nom_end_ts) else None,
            round(E_ch_nom, 3) if not np.isnan(E_ch_nom) else None,
            f"[{P_nom_discharge_kW}kW ±{tol_discharge_pct}%]",
            dis_nom_start_ts.strftime('%H:%M:%S') if pd.notna(dis_nom_start_ts) else None,
            dis_nom_end_ts.strftime('%H:%M:%S') if pd.notna(dis_nom_end_ts) else None,
            round(E_dis_nom, 3) if not np.isnan(E_dis_nom) else None,
            round(RTE_nom, 3) if not np.isnan(RTE_nom) else None
        ])

    return {"summary_table": pd.DataFrame({"Metric": metrics, "Value": values}), "figure": fig, "figure_soc": fig_soc, "df_calc_poi_egymtr": df_calc_poi_egymtr, "warnings": warnings_list}


def plot_calc_poi_egymtr(df_energy: pd.DataFrame, 
                         time_col: str = "Timestamp", 
                         soc_col: str = "SocAvg %", 
                         title: str = "AC-Side Cumulative Energy & SOC") -> go.Figure:
    if df_energy is None or df_energy.empty: 
        return go.Figure().update_layout(title="No data to plot.")
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_energy[time_col], y=df_energy["Calc-PoiEgyMtr"], 
        mode="lines", name="Cumulative Energy (kWh)", 
        line=dict(color="#FF5733", width=2.5)
    ))
    
    yaxis2_config = dict(overlaying="y", side="right", showgrid=False, visible=False) 
    
    if soc_col in df_energy.columns:
        fig.add_trace(go.Scatter(
            x=df_energy[time_col], y=df_energy[soc_col], 
            mode="lines", name="SOC (%)", 
            line=dict(color="#337AFF", width=1.5, dash="solid"), 
            yaxis="y2"
        ))
        yaxis2_config = dict(
            title=dict(text="SOC (%)", font=dict(color="#337AFF")),
            tickfont=dict(color="#337AFF"),
            overlaying="y", 
            side="right", 
            showgrid=False
        )
    else:
        title = "AC-Side Cumulative Energy"

    fig.update_layout(
        title=title, xaxis_title="Time",
        yaxis=dict(
            title=dict(text="Cumulative Energy (kWh)", font=dict(color="#FF5733")),
            tickfont=dict(color="#FF5733")
        ),
        yaxis2=yaxis2_config,
        template="plotly_white", 
        hovermode="x unified", 
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    )
    return fig


# SECTION 4: WORKFLOW APPLICATION (STATE MANAGEMENT FIXED)

st.title("⚡ AC-Side Capacity & RTE Analysis")

from utils import check_login
check_login() 

STATE_INIT, STATE_PROCESSING, STATE_CHOOSE_ANALYSIS, STATE_CONFIGURE, STATE_RUNNING, STATE_RESULTS = "INIT", "PROC", "CHOOSE", "CONFIG", "RUN", "RES"
if 'ac_workflow_state' not in st.session_state: 
    st.session_state.ac_workflow_state = STATE_INIT
if 'ac_analysis_config' not in st.session_state: 
    st.session_state.ac_analysis_config = {}

def format_event(event):
    start_str, end_str = event['start'].strftime('%H:%M'), event['end'].strftime('%H:%M')
    duration_min = (event['end'] - event['start']).total_seconds() / 60.0
    return f"[{event['type']}] {start_str} → {end_str} ({duration_min:.0f} min)"

st.sidebar.header("AC Analysis Configuration")
uploaded_file = st.sidebar.file_uploader("Upload AC Data File (HyCon)", type=["csv"], key="ac_uploader")

# --- MODIFIED: Robust state logic ---
if uploaded_file is None:
    if 'ac_last_file_id' not in st.session_state or st.session_state.ac_last_file_id is None:
        st.info("Upload your AC-side (HyCon) file to begin.")
        st.session_state.ac_workflow_state = STATE_INIT
        st.session_state.ac_df = None
        st.session_state.ac_results = None
    
elif uploaded_file.file_id != st.session_state.get('ac_last_file_id'): 
    st.session_state.ac_last_file_id = uploaded_file.file_id
    st.session_state.ac_workflow_state = STATE_PROCESSING
    st.session_state.ac_results = None 
    st.rerun()

elif st.session_state.ac_workflow_state == STATE_INIT and 'ac_df' not in st.session_state:
    st.session_state.ac_workflow_state = STATE_PROCESSING
    st.rerun()
# --- END MODIFICATION ---


if st.session_state.ac_workflow_state == STATE_PROCESSING:
    with st.spinner("Processing AC file..."):
        try:
            df_ac = load_hycon_hybrid_fast(uploaded_file)
            st.session_state.ac_df = convert_mixed_numeric_columns(df_ac, exclude={'Timestamp'}, verbose=False)
            st.session_state.ac_workflow_state = STATE_CHOOSE_ANALYSIS
            st.rerun()
        except Exception as e: 
            st.error(f"Error processing file: {e}"); 
            st.session_state.ac_workflow_state = STATE_INIT; st.stop()

# --- MODIFIED: ALL UI elements are now INSIDE this block ---
if 'ac_df' in st.session_state and st.session_state.ac_df is not None:
    df_cols = st.session_state.ac_df.columns.to_list()
    
    def get_index(cols, candidates, default=0):
        for cand in candidates:
            if cand in cols: return cols.index(cand)
        return default

    st.sidebar.subheader("Column Selection")
    st.sidebar.selectbox("Timestamp Column", options=df_cols, 
                         index=get_index(df_cols, ["Timestamp"]), 
                         key="ac_time_col")
    st.sidebar.selectbox("Power Column", options=df_cols, 
                         index=get_index(df_cols, ["PoiPwrAt kW"], 1), 
                         key="ac_power_col")
    
    soc_options = [None] + df_cols
    soc_idx = get_index(soc_options, ["SocAvg %"])
    st.sidebar.selectbox("SOC Column (optional)", options=soc_options, 
                         index=soc_idx, 
                         key="ac_soc_col")

    st.sidebar.subheader("Event Auto-Detection")
    st.sidebar.number_input("Idle Threshold (kW)", value=100.0, key="ac_idle_kw")
    st.sidebar.number_input("Min Event Duration (s)", value=600.0, key="ac_min_duration")

    if st.session_state.ac_workflow_state in [STATE_CONFIGURE, STATE_RUNNING, STATE_RESULTS]:
        st.sidebar.subheader("Test Time Windows")
        config = st.session_state.ac_analysis_config
        
        # --- THIS BLOCK IS NOW CORRECTLY PLACED ---
        if config.get("analysis_type") == "MANUAL":
            st.sidebar.info("Manual Mode")
            
            try:
                time_col = st.session_state.ac_time_col
                first_ts = st.session_state.ac_df[time_col].iloc[0]
                d_default = first_ts.date()
                t_default = first_ts.time()
            except Exception:
                d_default = pd.to_datetime("today").date()
                t_default = pd.to_datetime("12:00").time()

            c1, c2 = st.sidebar.columns(2)
            with c1: 
                ch_s_d = st.date_input("Charge Start", d_default, key="ac_ch_s_d")
                ch_s_t = st.time_input("Time", t_default, key='ac_ch_s_t')
            with c2: 
                ch_e_d = st.date_input("Charge End", d_default, key="ac_ch_e_d")
                ch_e_t = st.time_input("Time", t_default, key='ac_ch_e_t')
            c3, c4 = st.sidebar.columns(2)
            with c3: 
                dis_s_d = st.date_input("Discharge Start", d_default, key="ac_dis_s_d")
                dis_s_t = st.time_input("Time", t_default, key='ac_dis_s_t')
            with c4: 
                dis_e_d = st.date_input("Discharge End", d_default, key="ac_dis_e_d")
                dis_e_t = st.time_input("Time", t_default, key='ac_dis_e_t')
            
            st.session_state.ac_analysis_config.update({
                'charge_start': pd.Timestamp(f"{st.session_state.ac_ch_s_d} {st.session_state.ac_ch_s_t}"), 
                'charge_end': pd.Timestamp(f"{st.session_state.ac_ch_e_d} {st.session_state.ac_ch_e_t}"), 
                'discharge_start': pd.Timestamp(f"{st.session_state.ac_dis_s_d} {st.session_state.ac_dis_s_t}"), 
                'discharge_end': pd.Timestamp(f"{st.session_state.ac_dis_e_d} {st.session_state.ac_dis_e_t}")
            })
        else:
            st.sidebar.success("Auto-Detected Times Active")
        # --- END OF MOVED BLOCK ---

        st.sidebar.subheader("RTE & KPI Settings")
        st.sidebar.number_input("Nominal Power (kW)", value=24500.0, key="ac_p_nom_shared",
                                help="The absolute nominal power. E.g., 24500. This will be used as +24500 for discharge and -24500 for charge.")
        
        c5, c6 = st.sidebar.columns(2)
        with c5:
             st.sidebar.number_input("Nominal Charge Tol. (%)", value=10.0, key="ac_tol_charge_pct")
        with c6:
             st.sidebar.number_input("Nominal Discharge Tol. (%)", value=1.0, key="ac_tol_discharge_pct")

        st.sidebar.number_input("Sampling Interval (s)", min_value=1, value=1, key="ac_sampling_seconds")
        
        if st.sidebar.button("Run Analysis", type="primary", use_container_width=True): 
            st.session_state.ac_workflow_state = STATE_RUNNING
            st.rerun()
        if st.session_state.ac_workflow_state == STATE_RESULTS and st.sidebar.button("Reset Analysis", use_container_width=True): 
            st.session_state.ac_workflow_state = STATE_CHOOSE_ANALYSIS
            st.session_state.ac_results = None
            st.session_state.ac_results_summary = None 
            st.rerun()


    if st.session_state.ac_workflow_state == STATE_CHOOSE_ANALYSIS:
        with st.expander("Show Data Preview (First 10 Rows)"):
            st.dataframe(st.session_state.ac_df.head(10), use_container_width=True)
        
        all_events = find_all_events(st.session_state.ac_df, 
                                     st.session_state.ac_time_col, 
                                     st.session_state.ac_power_col, 
                                     st.session_state.ac_idle_kw, 
                                     st.session_state.ac_min_duration)
        dis_events, ch_events = [e for e in all_events if e['type'] == 'Discharge'], [e for e in all_events if e['type'] == 'Charge']
        
        st.subheader("Choose Analysis Type")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("#### ⚙️ Select charging & discharging cycle 🔋")
                if not dis_events or not ch_events: st.warning("Need both charge & discharge.")
                else:
                    sel_ch = st.selectbox("1. Charge Event", [format_event(e) for e in ch_events], key="rte_ch")
                    sel_dis_rte = st.selectbox("2. Discharge Event", [format_event(e) for e in dis_events], key="rte_dis")
                    if st.button("Evaluate RTE", use_container_width=True):
                        ch_ev = next(e for e in ch_events if format_event(e) == sel_ch)
                        dis_ev = next(e for e in dis_events if format_event(e) == sel_dis_rte)
                        st.session_state.master_charge_start = ch_ev['start']
                        st.session_state.master_charge_end = ch_ev['end']
                        st.session_state.master_discharge_start = dis_ev['start']
                        st.session_state.master_discharge_end = dis_ev['end']
                        st.session_state.ac_analysis_config = {"analysis_type": "RTE", "discharge_start": dis_ev['start'], "discharge_end": dis_ev['end'], "charge_start": ch_ev['start'], "charge_end": ch_ev['end']}
                        st.session_state.ac_workflow_state = STATE_CONFIGURE; st.rerun()
        with c2:
            with st.container(border=True):
                st.markdown("#### ⌨️ Manual Input")
                st.info("Manually enter all start/end times.")
                if st.button("Evaluate Manual", use_container_width=True):
                    st.session_state.master_charge_start, st.session_state.master_charge_end = None, None
                    st.session_state.master_discharge_start, st.session_state.master_discharge_end = None, None
                    st.session_state.ac_analysis_config = {"analysis_type": "MANUAL"}
                    st.session_state.ac_workflow_state = STATE_CONFIGURE; st.rerun()

    if st.session_state.ac_workflow_state == STATE_CONFIGURE:
        st.success("Settings loaded. Please review sidebar and click 'Run Analysis'.")
        if st.session_state.ac_analysis_config.get("analysis_type") == "MANUAL":
            st.session_state.master_charge_start = st.session_state.ac_analysis_config.get('charge_start')
            st.session_state.master_charge_end = st.session_state.ac_analysis_config.get('charge_end')
            st.session_state.master_discharge_start = st.session_state.ac_analysis_config.get('discharge_start')
            st.session_state.master_discharge_end = st.session_state.ac_analysis_config.get('discharge_end')
            
        with st.expander("Data Diagnostics"): convert_mixed_numeric_columns(st.session_state.ac_df, exclude={st.session_state.ac_time_col}, verbose=True)

    if st.session_state.ac_workflow_state == STATE_RUNNING:
        with st.spinner("Analyzing AC Side..."):
            try:
                cfg = st.session_state.ac_analysis_config
                p_nom_shared = st.session_state.ac_p_nom_shared
                
                final_cfg = {
                    "discharge_start": cfg.get("discharge_start"),
                    "discharge_end": cfg.get("discharge_end"),
                    "charge_start": cfg.get("charge_start"),
                    "charge_end": cfg.get("charge_end"),
                    "P_nom_charge_kW": -abs(p_nom_shared),
                    "tol_charge_pct": st.session_state.ac_tol_charge_pct,
                    "P_nom_discharge_kW": abs(p_nom_shared),
                    "tol_discharge_pct": st.session_state.ac_tol_discharge_pct,
                    "sampling_seconds": st.session_state.ac_sampling_seconds,
                    "time_col": st.session_state.ac_time_col,
                    "power_col": st.session_state.ac_power_col,
                    "soc_col": st.session_state.ac_soc_col
                }
                
                results = compute_nominal_from_poi_plotly(st.session_state.ac_df, **final_cfg)
                st.session_state.ac_results = results
                st.session_state.ac_results_summary = results.get("summary_table")
                st.session_state.ac_workflow_state = STATE_RESULTS; st.rerun()
            except Exception as e: 
                st.error(f"Analysis failed: {e}")
                st.exception(e)
                st.session_state.ac_workflow_state = STATE_CONFIGURE

    if st.session_state.ac_workflow_state == STATE_RESULTS:
        res = st.session_state.ac_results
        if res:
            st.subheader("AC Analysis Results")
            st.dataframe(res['summary_table'], hide_index=True, use_container_width=True)
            
            t1, t2, t3 = st.tabs(["Nominal delivered AC Power", "Cumulative Energy & SOC", "SOC Plot"])
            
            with t1: 
                st.plotly_chart(res['figure'], use_container_width=True)
                st.download_button("Download Power Plot (HTML)", res['figure'].to_html(), "power_plot.html", "text/html")
            
            with t2: 
                fig_cum = plot_calc_poi_egymtr(res['df_calc_poi_egymtr'], st.session_state.ac_time_col, soc_col=st.session_state.ac_soc_col)
                st.plotly_chart(fig_cum, use_container_width=True)
                st.download_button("Download Energy/SOC Plot (HTML)", fig_cum.to_html(), "energy_soc_plot.html", "text/html")
            
            with t3: 
                st.plotly_chart(res['figure_soc'], use_container_width=True)
                st.download_button("Download Full SOC Plot (HTML)", res['figure_soc'].to_html(), "soc_plot.html", "text/html")

            if res['warnings']:
                st.subheader("Analysis Warnings")
                for w in res['warnings']: st.warning(w)
        else:
            st.error("No results found. An unknown error may have occurred.")
            st.session_state.ac_workflow_state = STATE_CHOOSE_ANALYSIS