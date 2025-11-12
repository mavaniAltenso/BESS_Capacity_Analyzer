import streamlit as st
import pandas as pd
from utils import check_login # Import the shared function

# 1. PAGE CONFIGURATION AND LOGIN CHECK

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

# 2. MAIN APPLICATION (This code only runs if login is successful)
    
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

# --- NEW: Added explanation expander ---
with st.expander("How Calculations Work"):
    st.markdown(r"""
    The application calculates energy (in kWh) by integrating power (in kW) over time (in hours).
    
    Let $P(t)$ be the power at time $t$. The total energy $E$ from a start time $T_{start}$ to an end time $T_{end}$ is the integral:
    
    $$
    E = \int_{T_{start}}^{T_{end}} P(t) \,dt
    $$
    
    The app performs this calculation numerically using the **Trapezoidal Rule** (`np.trapz`) on the time-series data for the highest accuracy.
    
    ---
    
    ### Full-Cycle RTE Calculations
    
    This method uses the **full "zero-to-zero" event windows** for calculations. Let $P_{AC}(t)$ be the AC power and $P_{DC,total}(t)$ be the total DC power.
    
    * **AC Energy In ($E_{AC,in}$):** The total energy measured at the AC-side during the **charge window** (where $P_{AC}(t)$ is negative).
        $$
        E_{AC,in} = \int_{T_{charge,start}}^{T_{charge,end}} |\min(0, P_{AC}(t))| \,dt
        $$
    
    * **AC Energy Out ($E_{AC,out}$):** The total energy measured at the AC-side during the **discharge window** (where $P_{AC}(t)$ is positive).
        $$
        E_{AC,out} = \int_{T_{discharge,start}}^{T_{discharge,end}} \max(0, P_{AC}(t)) \,dt
        $$
    
    * **DC Energy In ($E_{DC,in}$):** The sum of all DC device power during the **charge window**.
        $$
        E_{DC,in} = \int_{T_{charge,start}}^{T_{charge,end}} |\min(0, P_{DC,total}(t))| \,dt
        $$
        
    * **DC Energy Out ($E_{DC,out}$):** The sum of all DC device power during the **discharge window**.
        $$
        E_{DC,out} = \int_{T_{discharge,start}}^{T_{discharge,end}} \max(0, P_{DC,total}(t)) \,dt
        $$
        
    ### Nominal Power RTE Calculations
    
    This method uses the *exact same formulas* as above, but it first **filters the data**. The integrals are only performed on time slices where the power $P(t)$ is inside the user-defined "Nominal" tolerance band.
    """)
# --- END NEW SECTION ---