import streamlit as st
import pandas as pd
from utils import check_login # Import the shared function

# =======================================================================
# 1. PAGE CONFIGURATION AND LOGIN CHECK
# =======================================================================

# Set the page config for the *main app* (after login)
st.set_page_config(
    layout="wide",
    page_title="BESS Test Analyzer",
    page_icon="🔋"
)

# --- THIS IS THE FIX ---
# Run the login check at the top of the script.
# The check_login function will handle stopping the script if not logged in.
check_login()

# =======================================================================
# 2. MAIN APPLICATION (This code only runs if login is successful)
# =======================================================================
    
# --- Initialize session state keys for sharing data between pages ---
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
st.info(f"Welcome, **{st.session_state.get('username', 'user')}**! Please select an analysis from the sidebar to begin.")

st.markdown("""
### How to use this tool:

1.  **Start with `AC_Side_Analysis`:**
    * Upload your AC-side (HyCon) data file.
    * The app will auto-detect charge/discharge events.
    * Select the events you wish to analyze. This will set the "master time windows" for the whole app.
    * Run the AC analysis.

2.  **Go to `DC_Side_Analysis`:**
    * The app will automatically use the time windows you selected from the AC analysis.
    * Upload your DC-side (MVPS) data file and select the correct columns.
    * Run the DC analysis.

3.  **View `System_Summary`:**
    * This page will automatically show a combined report comparing AC and DC results, including inverter efficiencies.
""")