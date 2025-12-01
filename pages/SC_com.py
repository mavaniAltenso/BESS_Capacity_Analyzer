import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# 1. PAGE CONFIG & AUTHENTICATION
st.set_page_config(layout="wide", page_title="SC Com Validation")

from utils import check_login
check_login() 

# 2. DATA LOADER (Cached)

@st.cache_data
def load_tool_scc_csv(uploaded_file, drop_ms_option=False):
    try:
        uploaded_file.seek(0)
        header_rows_indices = [9, 10]
        data_skip_rows = 11 
        has_ms_column = True 
        
        # Read Headers
        header_rows = pd.read_csv(
            uploaded_file, encoding="latin1", sep=";", engine="python",
            skiprows=header_rows_indices[0], nrows=len(header_rows_indices), header=None
        )
        combined_headers = header_rows.fillna("").astype(str).agg(" ".join)
        combined_headers = combined_headers.str.replace(r"\s+", " ", regex=True).str.replace("-", "").str.strip()
        num_expected_cols = len(combined_headers)

        # Read Data
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file, encoding="latin1", sep=";", engine="python",
            skiprows=data_skip_rows, header=None, on_bad_lines="skip",
            usecols=range(num_expected_cols) 
        )
        df.columns = combined_headers

        # Unique Columns
        def make_unique(cols):
            seen = {}
            result = []
            for col in cols:
                col = col.strip()
                if col in seen:
                    seen[col] += 1
                    result.append(f"{col}.{seen[col]}")
                else:
                    seen[col] = 0
                    result.append(col)
            return result

        df.columns = make_unique(df.columns)
        df.rename(columns={df.columns[0]: "Timestamp"}, inplace=True)

        if has_ms_column and drop_ms_option and len(df.columns) > 1 and df.columns[1].lower().strip() == "ms":
            df.drop(columns=[df.columns[1]], inplace=True)

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        
        # Clean Numeric Data
        data_cols = [col for col in df.columns if col != 'Timestamp']
        for col in data_cols:
            try:
                col_data_cleaned = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(col_data_cleaned)
            except ValueError:
                pass 
                    
        df = df.set_index("Timestamp")
        return df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# 3. ANALYSIS LOGIC

def run_energy_validation(df, t_start, t_end, deadband_kw):
    # Map columns
    col_ac_power = 'InvMs.TotW kW'
    col_dc_power = 'DcMs.TotWatt kW'
    cnt_ac_out = 'Cnt.TotAcWhOut MWh' 
    cnt_ac_in  = 'Cnt.TotAcWhIn MWh'  
    cnt_dc_in  = 'Cnt.TotDcWhIn MWh'  
    cnt_dc_out = 'Cnt.TotDcWhOut MWh' 

    # Check if columns exist
    required_cols = [col_ac_power, col_dc_power, cnt_ac_out, cnt_ac_in, cnt_dc_in, cnt_dc_out]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        return None, None

    # Slice Data
    mask = (df.index >= pd.to_datetime(t_start)) & (df.index <= pd.to_datetime(t_end))
    df_win = df.loc[mask].copy()
    
    if df_win.empty:
        return None, None

    # Time Delta
    df_win['delta_h'] = df_win.index.to_series().diff().dt.total_seconds().div(3600).fillna(0)

    # Integration (With Deadband)
    ac_clean = df_win[col_ac_power].copy()
    if deadband_kw > 0:
        ac_clean[ac_clean.abs() < deadband_kw] = 0
        
    calc_ac_disch = (ac_clean.clip(lower=0) * df_win['delta_h']).sum()
    calc_ac_charg = (ac_clean.clip(upper=0).abs() * df_win['delta_h']).sum()
    
    # Save Cumulative curves
    df_win['Cum_AC_Disch'] = (ac_clean.clip(lower=0) * df_win['delta_h']).cumsum()
    df_win['Cum_AC_Charg'] = (ac_clean.clip(upper=0).abs() * df_win['delta_h']).cumsum()

    dc_clean = df_win[col_dc_power].copy()
    if deadband_kw > 0:
        dc_clean[dc_clean.abs() < deadband_kw] = 0

    calc_dc_disch = (dc_clean.clip(lower=0) * df_win['delta_h']).sum()
    calc_dc_charg = (dc_clean.clip(upper=0).abs() * df_win['delta_h']).sum()

    df_win['Cum_DC_Disch'] = (dc_clean.clip(lower=0) * df_win['delta_h']).cumsum()
    df_win['Cum_DC_Charg'] = (dc_clean.clip(upper=0).abs() * df_win['delta_h']).cumsum()

    # Counters (Delta)
    def get_counter_delta(col_name):
        if col_name in df_win.columns:
            return (df_win[col_name].iloc[-1] - df_win[col_name].iloc[0]) * 1000
        return 0.0

    count_ac_disch = get_counter_delta(cnt_ac_out) # Grid Output
    count_ac_charg = get_counter_delta(cnt_ac_in)  # Grid Input
    
    # NOTE: Swapped logic for DC Counters to match Inverter Physics
    count_dc_charg = get_counter_delta(cnt_dc_out) # Energy TO Battery
    count_dc_disch = get_counter_delta(cnt_dc_in)  # Energy FROM Battery

    # --- EFFICIENCIES ---
    
    # 1. Inverter Discharge Eff (AC Out / DC In)
    eff_disch = (count_ac_disch / count_dc_disch * 100) if count_dc_disch > 0 else 0
    
    # 2. Inverter Charge Eff (DC Out / AC In)
    eff_charg = (count_dc_charg / count_ac_charg * 100) if count_ac_charg > 0 else 0
    
    # 3. Battery Internal Eff (DC From Batt / DC To Batt)
    # Note: Using Discharged / Charged
    eff_batt  = (count_dc_disch / count_dc_charg * 100) if count_dc_charg > 0 else 0
    
    # 4. System Round Trip (AC Out / AC In)
    eff_rte   = (count_ac_disch / count_ac_charg * 100) if count_ac_charg > 0 else 0

    results = {
        "ac_disch_calc": calc_ac_disch, "ac_disch_cnt": count_ac_disch,
        "ac_charg_calc": calc_ac_charg, "ac_charg_cnt": count_ac_charg,
        "dc_disch_calc": calc_dc_disch, "dc_disch_cnt": count_dc_disch,
        "dc_charg_calc": calc_dc_charg, "dc_charg_cnt": count_dc_charg,
        "eff_disch": eff_disch,
        "eff_charg": eff_charg,
        "eff_batt": eff_batt,
        "eff_rte": eff_rte
    }
    
    return results, df_win


# 4. UI MAIN PAGE

st.title("Inverter Efficiency Analysis")
st.markdown("Independent validation of Inverter counters vs. Calculated Energy.")

st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload SCC CSV", type=["csv"], key="scc_uploader")

if uploaded_file is not None:
    df = load_tool_scc_csv(uploaded_file, drop_ms_option=True)
    
    if df is not None:
        # --- WHOLE FILE PRE-CALCULATION ---
        df['delta_h'] = df.index.to_series().diff().dt.total_seconds().div(3600).fillna(0)
        
        # Check if basic columns exist before plotting
        if 'InvMs.TotW kW' in df.columns:
            p_disch = df['InvMs.TotW kW'].clip(lower=0)
            p_charg = df['InvMs.TotW kW'].clip(upper=0).abs()
            df['Full_Cum_Discharge'] = (p_disch * df['delta_h']).cumsum()
            df['Full_Cum_Charge'] = (p_charg * df['delta_h']).cumsum()

            # --- SECTION 1: VISUALIZE FIRST ---
            st.subheader("Inspect Data & Select Window")
            st.info("Use the charts below to identify correct cycle.")

            tab_view1, tab_view2 = st.tabs(["Active Power & SoC", "Cumulative Energy"])

            with tab_view1:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=df.index, y=df['InvMs.TotW kW'], name="Active Power (kW)", line=dict(color='blue', width=1)), secondary_y=False)
                if 'Bat.SOCTot %' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['Bat.SOCTot %'], name="SoC (%)", line=dict(color='green', width=2, dash='dot')), secondary_y=True)
                    fig.update_yaxes(title_text="State of Charge (%)", secondary_y=True, range=[0, 105])
                fig.update_layout(title_text="Active Power & SoC", hovermode="x unified", height=500)
                st.plotly_chart(fig, use_container_width=True)

            with tab_view2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df.index, y=df['Full_Cum_Discharge'], name="Total Discharged (Out)", line=dict(color='purple', width=2)))
                fig2.add_trace(go.Scatter(x=df.index, y=df['Full_Cum_Charge'], name="Total Charged (In)", line=dict(color='orange', width=2)))
                fig2.update_layout(title_text="Cumulative Energy (Entire File)", yaxis_title="kWh", hovermode="x unified", height=500)
                st.plotly_chart(fig2, use_container_width=True)
                tot_out = df['Full_Cum_Discharge'].iloc[-1]
                tot_in = df['Full_Cum_Charge'].iloc[-1]
                st.caption(f"**File Totals:** Input: {tot_in:,.2f} kWh | Output: {tot_out:,.2f} kWh")

            # --- SECTION 2: DEFINE WINDOW (GRID LAYOUT) ---
            st.subheader("Calculation Window")
            
            min_date = df.index.min().to_pydatetime()
            max_date = df.index.max().to_pydatetime()

            c_date, c_time = st.columns(2)
            
            with c_date:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key="sc_start_d")
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key="sc_end_d")
                
            with c_time:
                start_time = st.time_input("Start Time", value=min_date.time(), step=60, key="sc_start_t")
                end_time = st.time_input("End Time", value=max_date.time(), step=60, key="sc_end_t")

            start_val = pd.to_datetime(f"{start_date} {start_time}")
            end_val = pd.to_datetime(f"{end_date} {end_time}")

            if start_val >= end_val:
                st.error("'Start Time' must be earlier than 'End Time'.")
            else:
                st.markdown("") 
                c_dead, c_btn = st.columns(2)
                with c_dead:
                    deadband = st.number_input("Deadband Filter (kW)", value=0.0, step=0.1, help="Filters noise below this value.", key="sc_deadband")
                
                with c_btn:
                    st.write("") # Spacer
                    run_btn = st.button("Analyze Selected Window", type="primary", use_container_width=True, key="sc_run_btn")

                # --- SECTION 3: RESULTS ---
                if run_btn:
                    results, df_win = run_energy_validation(df, start_val, end_val, deadband)
                    
                    if results:
                        st.divider()
                        st.title("Results")
                        
                        tab1, tab2 = st.tabs(["Efficiency Report", "Cumulative Plots"])

                        with tab1:
                            # 4 METRICS LAYOUT
                            k1, k2, k3, k4 = st.columns(4)
                            k1.metric("Discharge Eff (Inv)", f"{results['eff_disch']:.2f}%", help="AC Out / DC In")
                            k2.metric("Charge Eff (Inv)", f"{results['eff_charg']:.2f}%", help="DC Out / AC In")
                            k3.metric("Battery Eff (Chem)", f"{results['eff_batt']:.2f}%", help="DC From Batt / DC To Batt")
                            k4.metric("System RTE (Total)", f"{results['eff_rte']:.2f}%", help="AC Out / AC In")

                            st.subheader("Detailed Counter Check")
                            
                            # TABLE WITHOUT DIFF COLUMN
                            comp_data = {
                                "Metric": ["AC Discharge", "AC Charge", "DC Discharge", "DC Charge"],
                                "Calculated (kWh)": [results['ac_disch_calc'], results['ac_charg_calc'], results['dc_disch_calc'], results['dc_charg_calc']],
                                "Counter (kWh)": [results['ac_disch_cnt'], results['ac_charg_cnt'], results['dc_disch_cnt'], results['dc_charg_cnt']]
                            }
                            df_comp = pd.DataFrame(comp_data)
                            
                            st.dataframe(
                                df_comp.style.format({
                                    "Calculated (kWh)": "{:,.2f}",
                                    "Counter (kWh)": "{:,.2f}"
                                }),
                                use_container_width=True
                            )

                        with tab2:
                            st.subheader("Energy Accumulation (Visual Proof)")
                            fig_cum, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                            
                            ax1.plot(df_win.index, df_win['Cum_DC_Disch'], label='DC Input (From Batt)', color='purple', linestyle='--')
                            ax1.plot(df_win.index, df_win['Cum_AC_Disch'], label='AC Output (To Grid)', color='blue')
                            ax1.fill_between(df_win.index, df_win['Cum_AC_Disch'], df_win['Cum_DC_Disch'], color='gray', alpha=0.2, label='Inverter Loss')
                            ax1.set_title("Discharge Phase")
                            ax1.set_ylabel("kWh")
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            ax2.plot(df_win.index, df_win['Cum_AC_Charg'], label='AC Input (From Grid)', color='orange', linestyle='--')
                            ax2.plot(df_win.index, df_win['Cum_DC_Charg'], label='DC Output (To Batt)', color='green')
                            ax2.fill_between(df_win.index, df_win['Cum_DC_Charg'], df_win['Cum_AC_Charg'], color='gray', alpha=0.2, label='Inverter Loss')
                            ax2.set_title("Charge Phase")
                            ax2.set_ylabel("kWh")
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            
                            st.pyplot(fig_cum)
                            
                    else:
                        st.error("No valid data found in selected window or missing columns.")
        else:
             st.error("File missing required power columns (InvMs.TotW kW).")