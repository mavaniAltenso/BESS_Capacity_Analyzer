import streamlit as st
import pandas as pd


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
### Tool Guide:

1.  **Start with `⚡_AC_Side_Analysis`:**
    * Upload your AC-side (HyCon) data file.
    * The app will auto-detect charge/discharge events based on Idle power and min time duration.
    * Select the events you wish to analyze. This will set the "Time windows" for the whole app.
    * Run the AC analysis.

2.  **Go to `🔋_DC_Side_Analysis`:**
    * The app will automatically use the time windows you selected from the AC analysis.
    * Upload your DC-side (Hymon) data file and select the correct columns for power.
    * Run the DC analysis.

3.  **View `📊_System_Summary`:**
    * This page will show a combined report comparing AC and DC results, including inverter efficiencies.
""")