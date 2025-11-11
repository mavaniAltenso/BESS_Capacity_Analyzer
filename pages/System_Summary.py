import streamlit as st
import pandas as pd
import numpy as np

st.title("📊 System Summary & Inverter Efficiency")


ac_summary = st.session_state.get('ac_results_summary')
dc_totals = st.session_state.get('dc_results_totals')

if not isinstance(ac_summary, pd.DataFrame) or not isinstance(dc_totals, dict):
    st.error("Analysis Incomplete")
    st.warning("Please run both the AC-Side and DC-Side analyses first.")
    
    if not isinstance(ac_summary, pd.DataFrame):
        st.info("Missing: AC-Side Results. Go to `AC_Side_Analysis`.")
    if not isinstance(dc_totals, dict):
        st.info("Missing: DC-Side Results. Go to `DC_Side_Analysis`.")
    st.stop()


# 2. EXTRACT DATA & CALCULATE INVERTER EFFICIENCY


try:
    # --- Helper to find value in the AC summary table ---
    def get_ac_val(metric_name):
        try:
            # Find the row where Metric matches, then get the Value
            return float(ac_summary.loc[ac_summary['Metric'] == metric_name, 'Value'].iloc[0])
        except (IndexError, TypeError, ValueError):
            st.warning(f"Could not find or parse metric: {metric_name} from AC results.")
            return np.nan

    # --- Get AC Values ---
    E_ac_in = get_ac_val("Total Energy IN (kWh)")
    E_ac_out = get_ac_val("Total Energy OUT (kWh)")
    RTE_ac = get_ac_val("Full-Cycle RTE (%)")
    
    # --- Get DC Values ---
    E_dc_in = dc_totals.get('Total_E_in', np.nan)
    E_dc_out = dc_totals.get('Total_E_out', np.nan)
    RTE_dc_raw = dc_totals.get('System_RTE', np.nan)
    RTE_dc = RTE_dc_raw * 100.0 if pd.notna(RTE_dc_raw) else np.nan
    
    # --- Calculate Inverter/PCS Efficiencies ---
    inv_eff_charge = (E_dc_in / E_ac_in) * 100.0 if E_ac_in > 0 else np.nan
    inv_eff_discharge = (E_ac_out / E_dc_out) * 100.0 if E_dc_out > 0 else np.nan
    
    # --- Calculate Losses ---
    loss_charge_kwh = E_ac_in - E_dc_in
    loss_discharge_kwh = E_dc_out - E_ac_out
    loss_dc_roundtrip_kwh = E_dc_in - E_dc_out
    loss_total_kwh = E_ac_in - E_ac_out


    # 3. DISPLAY SUMMARY TABLES

    st.subheader("System Efficiency Summary")
    
    summary_data = {
        "Metric": [
            "AC Round-Trip Efficiency (RTE)",
            "DC Round-Trip Efficiency (RTE)",
            "Inverter Charge Efficiency",
            "Inverter Discharge Efficiency"
        ],
        "Value": [
            f"{RTE_ac:.2f} %" if pd.notna(RTE_ac) else "N/A",
            f"{RTE_dc:.2f} %" if pd.notna(RTE_dc) else "N/A",
            f"{inv_eff_charge:.2f} %" if pd.notna(inv_eff_charge) else "N/A",
            f"{inv_eff_discharge:.2f} %" if pd.notna(inv_eff_discharge) else "N/A"
        ],
        "Calculation": [
            "E_ac_out / E_ac_in",
            "E_dc_out / E_dc_in",
            "E_dc_in / E_ac_in",
            "E_ac_out / E_dc_out"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    st.subheader("Energy & Loss Summary (kWh)")

    loss_data = {
        "Stage": ["AC (Grid)", "Inverter (Charge)", "DC (Battery)", "Inverter (Discharge)", "AC (Grid)"],
        "Energy (kWh)": [
            f"{E_ac_in:,.1f}" if pd.notna(E_ac_in) else "N/A",
            "",
            f"{E_dc_in:,.1f}" if pd.notna(E_dc_in) else "N/A",
            f"{E_dc_out:,.1f}" if pd.notna(E_dc_out) else "N/A",
            f"{E_ac_out:,.1f}" if pd.notna(E_ac_out) else "N/A"
        ],
        "Loss (kWh)": [
            "",
            f"{loss_charge_kwh:,.1f}" if pd.notna(loss_charge_kwh) else "N/A",
            f"{loss_dc_roundtrip_kwh:,.1f}" if pd.notna(loss_dc_roundtrip_kwh) else "N/A",
            f"{loss_discharge_kwh:,.1f}" if pd.notna(loss_discharge_kwh) else "N/A",
            ""
        ],
        "Description": [
            "Energy IN from Grid",
            "Losses during AC-to-DC conversion",
            "Energy IN to Battery",
            "Energy OUT from Battery",
            "Energy OUT to Grid"
        ]
    }
    st.dataframe(pd.DataFrame(loss_data), hide_index=True, use_container_width=True)
    st.metric("Total System Losses (Round-Trip)", f"{loss_total_kwh:,.1f} kWh" if pd.notna(loss_total_kwh) else "N/A")

except Exception as e:
    st.error("Failed to build summary. An error occurred.")
    st.exception(e)