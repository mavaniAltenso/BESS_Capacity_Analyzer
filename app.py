import streamlit as st
import pandas as pd

# =======================================================================
# MAIN APP HOMEPAGE
# =======================================================================

# st.set_page_config() can only be called once, and it must be in the main app.py file
st.set_page_config(
    layout="wide",
    page_title="BESS Test Analyzer",
    page_icon="🔋"
)

# Initialize session state keys for sharing data between pages
if 'master_charge_start' not in st.session_state:
    st.session_state.master_charge_start = None
if 'master_charge_end' not in st.session_state:
    st.session_state.master_charge_end = None
if 'master_discharge_start' not in st.session_state:
    st.session_state.master_discharge_start = None
if 'master_discharge_end' not in st.session_state:
    st.session_state.master_discharge_end = None
    
if 'ac_results_summary' not in st.session_state:
    st.session_state.ac_results_summary = None
if 'dc_results_totals' not in st.session_state:
    st.session_state.dc_results_totals = None
if 'dc_rte_summary' not in st.session_state:
    st.session_state.dc_rte_summary = None
    

# --- Homepage Content ---
st.title("🔋 BESS Capacity & RTE Test Analyzer")
st.info("Welcome! Please select an analysis from the sidebar to begin.")

st.markdown("""
### How to use this tool:

1.  **Start with `1_⚡_AC_Side_Analysis`:**
    * Upload your AC-side (HyCon) data file.
    * The app will auto-detect charge/discharge events.
    * Select the events you wish to analyze. This will set the "master time windows" for the whole app.
    * Run the AC analysis.

2.  **Go to `2_🔋_DC_Side_Analysis`:**
    * The app will automatically use the time windows you selected from the AC analysis.
    * Upload your DC-side (MVPS) data file and select the correct columns.
    * Run the DC analysis.

3.  **View `3_📊_System_Summary`:**
    * This page will automatically show a combined report comparing AC and DC results, including inverter efficiencies.
""")