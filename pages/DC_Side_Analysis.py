import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
import re
from typing import Optional, Dict, List, Tuple, Set
from pathlib import Path
import io

# Import shared functions
from utils import convert_mixed_numeric_columns, _sanitize_time_col, _check_cadence

# =======================================================================
# SECTION 1: DC DATA LOADING FUNCTION
# =======================================================================

@st.cache_data
def load_and_prep_dc_data(uploaded_file, sep=';', dayfirst=False) -> pd.DataFrame:
    """
    Loads and prepares the DC-side CSV file from an uploaded file object.
    """
    df = pd.read_csv(uploaded_file, sep=sep, dtype=str, engine='python')
    df.columns = df.columns.str.strip()
    required_time_cols = ['Date', 'Time']
    if not all(col in df.columns for col in required_time_cols):
         raise ValueError(f"DC File must contain {required_time_cols} columns.")

    df['Date'] = df['Date'].str.strip()
    df['Time'] = df['Time'].str.strip()
    
    if 'TZ' in df.columns:
        df['TZ'] = df['TZ'].astype(str).str.strip()
        def extract_tz_hours(tz_str):
            if pd.isna(tz_str): return 0
            m = re.search(r'([+-]?\d{1,3})', tz_str)
            if not m: return 0
            try: return int(m.group(1))
            except: return 0
        df['TZ_offset_h'] = df['TZ'].apply(extract_tz_hours)
    else:
        df['TZ_offset_h'] = 0

    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce', dayfirst=dayfirst)
    mask = df['Datetime'].notna()
    df.loc[mask, 'Datetime'] = df.loc[mask, 'Datetime'] + pd.to_timedelta(df.loc[mask, 'TZ_offset_h'], unit='h')
    cols_to_drop = ['Date', 'Time', 'TZ', 'TZ_offset_h']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.dropna(subset=['Datetime'])
    df = df.set_index('Datetime').sort_index()
    return df

# =======================================================================
# SECTION 3: DC ANALYZER CLASS
# =======================================================================

class DcCapacityTestAnalyzer:
    def __init__(self, master_config: dict, df_dc: pd.DataFrame):
        self.config = master_config
        self.df_dc = df_dc.copy()
        self.dfs_by_device = None
        self.dc_rte_summary = None
        self.dc_rte_system_totals = None
        self.dc_system_cumulative_energy = None
        self.dc_system_soc = None

    def run_analysis(self):
        with st.spinner("Preparing DC data..."):
            self._clean_and_partition_dc_df()
        with st.spinner("Running DC RTE analysis..."):
            self._run_dc_rte_analysis()
        with st.spinner("Running DC Energy & SOC analysis..."):
             self._run_dc_cumulative_energy_analysis()
             self._run_dc_soc_analysis()

    def _clean_and_partition_dc_df(self):
        dc_device_col = self.config['dc_device_col']
        cols_to_convert = [self.config['dc_power_col'], self.config['dc_soc_col']]
        cols_to_convert = [c for c in cols_to_convert if c in self.df_dc.columns and c is not None]
        
        if cols_to_convert:
             temp_df = self.df_dc[cols_to_convert].copy()
             temp_df = convert_mixed_numeric_columns(temp_df, verbose=False)
             for c in cols_to_convert:
                 self.df_dc[c] = temp_df[c]

        if dc_device_col not in self.df_dc.columns:
             raise KeyError(f"Device column '{dc_device_col}' not found.")

        self.dfs_by_device = {dev: g for dev, g in self.df_dc.groupby(dc_device_col)}

    def _run_dc_rte_analysis(self):
        rows = []
        t_ch_s, t_ch_e = self.config['charge_start'], self.config['charge_end']
        t_dis_s, t_dis_e = self.config['discharge_start'], self.config['discharge_end']
        P_COL = self.config['dc_power_col']
        
        for dev, d in self.dfs_by_device.items():
            if P_COL not in d.columns: continue
            dd = d.sort_index().copy()
            scale = 1000.0 if self.config.get('dc_is_power_in_watts', False) else 1.0
            
            dd[P_COL] = pd.to_numeric(dd[P_COL], errors='coerce')
            P = dd[P_COL].fillna(0.0).to_numpy() / scale

            if not self.config.get('dc_discharge_positive', True): P = -P
            dd["P"] = P

            dd["dt_s"] = dd.index.to_series().diff().dt.total_seconds().fillna(method='bfill')
            dd = dd[dd["dt_s"] > 0]

            d_ch = dd[(dd.index >= t_ch_s) & (dd.index <= t_ch_e)]
            d_dis = dd[(dd.index >= t_dis_s) & (dd.index <= t_dis_e)]

            E_ch, E_dis = 0.0, 0.0
            if not d_ch.empty:
                 E_ch = np.trapz((-d_ch["P"]).clip(lower=0), x=d_ch.index.astype(np.int64)/1e9) / 3600.0
            if not d_dis.empty:
                 E_dis = np.trapz((d_dis["P"]).clip(lower=0), x=d_dis.index.astype(np.int64)/1e9) / 3600.0

            eta = (E_dis / E_ch) if E_ch > self.config.get('rte_min_charge_kwh', 0.01) else np.nan
            rows.append({"Device": dev, "E_in": E_ch, "E_out": E_dis, "RTE": eta})

        self.dc_rte_summary = pd.DataFrame(rows).sort_values("Device")
        self.dc_rte_system_totals = {
            "Total_E_in": self.dc_rte_summary["E_in"].sum(),
            "Total_E_out": self.dc_rte_summary["E_out"].sum(),
            "System_RTE": (self.dc_rte_summary["E_out"].sum() / self.dc_rte_summary["E_in"].sum()) if self.dc_rte_summary["E_in"].sum() > 0 else np.nan
        }

    def _run_dc_cumulative_energy_analysis(self):
        P_COL = self.config['dc_power_col']
        if P_COL not in self.df_dc.columns: return
        
        scale = 1000.0 if self.config.get('dc_is_power_in_watts', False) else 1.0
        
        self.df_dc[P_COL] = pd.to_numeric(self.df_dc[P_COL], errors='coerce')
        
        power_wide = self.df_dc.pivot_table(index='Datetime', columns=self.config['dc_device_col'], values=P_COL, aggfunc='first')
        power_wide = power_wide.fillna(0.0).astype(float) / scale
        
        if not self.config.get('dc_discharge_positive', True):
             power_wide = -power_wide
             
        P_system = power_wide.sum(axis=1).sort_index()
        
        t_s = P_system.index.astype(np.int64) / 1e9
        e_cum_joules = np.concatenate([[0], \
             np.cumsum(0.5 * (P_system.values[:-1] + P_system.values[1:]) * np.diff(t_s))])
        
        self.dc_system_cumulative_energy = pd.Series(data=e_cum_joules/3600.0, index=P_system.index)

    def _run_dc_soc_analysis(self):
        SOC_COL = self.config['dc_soc_col']
        if SOC_COL not in self.df_dc.columns or SOC_COL is None: return
        
        self.df_dc[SOC_COL] = pd.to_numeric(self.df_dc[SOC_COL], errors='coerce')
        
        soc_wide = self.df_dc.pivot_table(index='Datetime', columns=self.config['dc_device_col'], values=SOC_COL, aggfunc='first')
        soc_wide = soc_wide.astype(float)
        
        if not self.config.get('dc_is_soc_percent', True):
             soc_wide = soc_wide * 100.0
             
        self.dc_system_soc = soc_wide.mean(axis=1).sort_index()

# =======================================================================
# SECTION 4: PLOTTING FUNCTIONS
# =======================================================================

def get_dc_efficiency_bar_plot(analyzer: DcCapacityTestAnalyzer) -> go.Figure:
    fig = go.Figure()
    summ = analyzer.dc_rte_summary
    if summ is None or summ.empty: return fig.update_layout(title="No data.")

    fig.add_trace(go.Bar(
        x=summ['Device'], y=summ['RTE']*100, name='Device RTE',
        text=(summ['RTE']*100).apply(lambda x: f"{x:.1f}%"), textposition='auto',
        marker_color='#2ca02c'
    ))

    sys_rte = analyzer.dc_rte_system_totals.get('System_RTE')
    if pd.notna(sys_rte):
        fig.add_hline(y=sys_rte*100, line_dash="dash", line_color="red", 
                      annotation_text=f"System Avg: {sys_rte*100:.1f}%", annotation_position="bottom right")

    fig.update_layout(
        title="Per-Device DC Efficiency (RTE)", xaxis_title="Device ID", yaxis_title="RTE (%)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    )
    return fig

def get_dc_energy_plot(analyzer: DcCapacityTestAnalyzer) -> go.Figure:
    fig = go.Figure()
    e_data = analyzer.dc_system_cumulative_energy
    if e_data is None or e_data.empty: return fig.update_layout(title="No energy data.")
    
    fig.add_trace(go.Scatter(
        x=e_data.index, y=e_data.values, mode='lines', name='System DC Energy (Net)',
        line=dict(color='#1f77b4', width=2.5)
    ))
    fig.update_layout(
        title="System Cumulative Net DC Energy", xaxis_title="Time", yaxis_title="Energy (kWh)",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    )
    return fig

def get_dc_soc_plot(analyzer: DcCapacityTestAnalyzer) -> go.Figure:
    fig = go.Figure()
    s_data = analyzer.dc_system_soc
    if s_data is None or s_data.empty: return fig.update_layout(title="No SOC data.")
    
    fig.add_trace(go.Scatter(
        x=s_data.index, y=s_data.values, mode='lines', name='Avg System SOC',
        line=dict(color='#2ca02c', width=2.5, dash="solid")
    ))
    fig.update_layout(
        title="Average System SOC", xaxis_title="Time", yaxis_title="SOC (%)",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    )
    return fig

# =======================================================================
# SECTION 5: STREAMLIT APP (STATE MANAGEMENT FIXED)
# =======================================================================

st.title("🔋 DC-Side Capacity & RTE Analysis")

if 'dc_analyzer' not in st.session_state: st.session_state.dc_analyzer = None
if 'dc_df' not in st.session_state: st.session_state.dc_df = None
if 'dc_last_file_id' not in st.session_state: st.session_state.dc_last_file_id = None

st.sidebar.header("DC Analysis Configuration")

if 'master_charge_start' not in st.session_state or st.session_state.master_charge_start is None:
    st.error("Please run an AC-Side Analysis first to set the master time windows.")
    st.info("Navigate to the `AC_Side_Analysis` page from the sidebar to begin.")
    st.stop()
else:
    st.sidebar.success("Using time windows from AC Analysis:")
    st.sidebar.markdown(f"""
    * **Charge:** `{st.session_state.master_charge_start.strftime('%H:%M:%S')}` to `{st.session_state.master_charge_end.strftime('%H:%M:%S')}`
    * **Discharge:** `{st.session_state.master_discharge_start.strftime('%H:%M:%S')}` to `{st.session_state.master_discharge_end.strftime('%H:%M:%S')}`
    """)

uploaded_file = st.sidebar.file_uploader("Upload DC Data File (CSV)", type=["csv"], key="dc_uploader")

# --- MODIFIED: Robust state logic ---
if uploaded_file is None:
    # If uploader is empty, only reset if we have no file ID saved
    if 'dc_last_file_id' not in st.session_state or st.session_state.dc_last_file_id is None:
        st.info("Upload your DC-side (MVPS) file to continue.")
        st.session_state.dc_last_file_id = None
        st.session_state.dc_df = None
        st.session_state.dc_analyzer = None
    # If it's empty but we have a file ID, user is on another page.
    # We do *nothing* and let the rest of the script use the data in st.session_state.
    
elif uploaded_file.file_id != st.session_state.get('dc_last_file_id'):
    # This is a NEW file upload
    st.session_state.dc_last_file_id = uploaded_file.file_id
    st.session_state.dc_df = None # Force re-load
    st.session_state.dc_analyzer = None # Clear old results
    st.rerun()

elif 'dc_df' not in st.session_state or st.session_state.dc_df is None:
    # This catches the first run after upload
    with st.spinner("Loading DC data file..."):
        try:
            st.session_state.dc_df = load_and_prep_dc_data(uploaded_file)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.stop()
# --- END MODIFICATION ---

if 'dc_df' in st.session_state and st.session_state.dc_df is not None:
    df = st.session_state.dc_df
    all_cols = df.columns.tolist()

    with st.expander("Show Data Preview (First 10 Rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    st.sidebar.subheader("Column Selection")
    
    def get_idx(cols, candidates, default=0):
        # Find first candidate that exists in cols
        for cand in candidates:
            if cand in cols:
                # Find index of this candidate
                try: return cols.index(cand)
                except ValueError: continue
        return default # Fallback

    st.sidebar.selectbox("Device ID Column", all_cols, index=get_idx(all_cols, ["Device", "Cluster", "String"]), key="dc_device_col")
    st.sidebar.selectbox("DC Power Column", all_cols, index=get_idx(all_cols, ["DcTotWatt", "Power", "DC_Power"]), key="dc_power_col")
    
    soc_options = [None] + all_cols
    soc_idx = get_idx(soc_options, ["Bat.SOCTot", "SOC", "StateOfCharge"])
    st.sidebar.selectbox("SOC Column", soc_options, index=soc_idx, key="dc_soc_col")

    st.sidebar.subheader("Settings")
    st.sidebar.checkbox("Power is in Watts (will convert to kW)", True, key="dc_is_watts")
    st.sidebar.checkbox("Discharge is Positive Value", True, key="dc_dis_pos")
    st.sidebar.checkbox("SOC is already % (0-100)", True, key="dc_soc_is_pct")

    if st.sidebar.button("Run DC Analysis", type="primary", use_container_width=True):
        config = {
            "charge_start": st.session_state.master_charge_start,
            "charge_end": st.session_state.master_charge_end,
            "discharge_start": st.session_state.master_discharge_start,
            "discharge_end": st.session_state.master_discharge_end,
            "dc_device_col": st.session_state.dc_device_col, 
            "dc_power_col": st.session_state.dc_power_col, 
            "dc_soc_col": st.session_state.dc_soc_col,
            "dc_is_power_in_watts": st.session_state.dc_is_watts, 
            "dc_discharge_positive": st.session_state.dc_dis_pos, 
            "dc_is_soc_percent": st.session_state.dc_soc_is_pct,
            "rte_min_charge_kwh": 0.1 
        }
        
        st.session_state.dc_analyzer = DcCapacityTestAnalyzer(config, df)
        st.session_state.dc_analyzer.run_analysis()
        
        st.session_state.dc_results_totals = st.session_state.dc_analyzer.dc_rte_system_totals
        st.session_state.dc_rte_summary = st.session_state.dc_analyzer.dc_rte_summary
        st.rerun()

if 'dc_analyzer' in st.session_state and st.session_state.dc_analyzer:
    an = st.session_state.dc_analyzer
    totals = an.dc_rte_system_totals
    
    st.divider()
    st.header("DC Analysis Results")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total DC Energy IN", f"{totals['Total_E_in']:,.1f} kWh")
    m2.metric("Total DC Energy OUT", f"{totals['Total_E_out']:,.1f} kWh")
    rte_val = totals['System_RTE']*100 if pd.notna(totals['System_RTE']) else 0
    m3.metric("System DC RTE", f"{rte_val:.2f} %")

    t1, t2, t3 = st.tabs(["Efficiency by Device", "System Cumulative Energy", "Avg System SOC"])
    with t1:
        st.plotly_chart(get_dc_efficiency_bar_plot(an), use_container_width=True)
        st.dataframe(an.dc_rte_summary.style.format({"E_in": "{:,.1f}", "E_out": "{:,.1f}", "RTE": "{:.2%}"}), use_container_width=True)
    with t2:
        st.plotly_chart(get_dc_energy_plot(an), use_container_width=True)
    with t3:
        st.plotly_chart(get_dc_soc_plot(an), use_container_width=True)