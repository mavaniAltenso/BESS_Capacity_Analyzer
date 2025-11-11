# In utils.py

import streamlit as st
import numpy as np
import pandas as pd
import re
from typing import Optional

# =======================================================================
# NEW: PASSWORD FUNCTION (Corrected)
# =======================================================================
def check_login():
    """
    Checks if user is logged in. 
    If not, draws login form and stops the page execution.
    """
    if st.session_state.get("logged_in"):
        return True

    # --- We removed st.set_page_config() from here ---
    
    st.title("🔋 BESS Analyzer Login")
    st.write("Please enter your credentials to access the application.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if "credentials" not in st.secrets or "users" not in st.secrets["credentials"]:
                st.error("Secrets not configured correctly. Contact administrator.")
                st.stop() # Stop execution

            users_dict = st.secrets["credentials"]["users"]
            
            if username in users_dict and users_dict[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()  # Rerun the script to show the main app
            else:
                st.error("Incorrect username or password")

    st.stop()
    return False

# =======================================================================
# SHARED HELPER FUNCTIONS (No changes below this line)
# =======================================================================
def _sanitize_time_col(d: pd.DataFrame, time_col: str) -> pd.DataFrame:
    d = d.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)
    return d

def _check_cadence(dt_s: pd.Series, expected_seconds: Optional[float], rtol: float = 0.02, atol: float = 0.5) -> dict:
    x = dt_s.dropna().to_numpy(dtype=float)
    x = x[x > 0]
    if x.size == 0: return dict(is_regular=False, dt_median=np.nan, dt_p95=np.nan, frac_off=1.0)
    dt_median = float(np.median(x))
    dt_p95 = float(np.quantile(x, 0.95))
    if expected_seconds is None or np.isnan(expected_seconds):
        tol = max(abs(dt_median) * rtol, atol)
        frac_off = float((np.abs(x - dt_median) > tol).mean())
        is_regular = frac_off <= 0.05
    else:
        tol = max(abs(expected_seconds) * rtol, atol)
        frac_off = float((np.abs(x - expected_seconds) > tol).mean())
        is_regular = (frac_off <= 0.05) and (abs(dt_median - expected_seconds) <= tol)
    return dict(is_regular=is_regular, dt_median=dt_median, dt_p95=dt_p95, frac_off=frac_off)

def _strip_spaces(s: str) -> str:
    if not isinstance(s, str): return s
    return s.replace('\u00A0', '').replace('\u202F', '').replace(' ', '').strip()

def _classify_value(s: str):
    if s is None or s == '': return 'other'
    has_comma = ',' in s
    has_dot = '.' in s
    if has_comma and has_dot: return 'EU' if s.rfind(',') > s.rfind('.') else 'US'
    if has_comma: return 'comma_only'
    if has_dot: return 'dot_only'
    if re.fullmatch(r'[+-]?\d+', s): return 'int'
    return 'other'

def _convert_value(s: str, preference: str):
    if s is None or (isinstance(s, float) and pd.isna(s)): return np.nan
    if not isinstance(s, str): return s
    s0 = _strip_spaces(s)
    if s0 == '' or s0.lower() in ('nan', 'none', 'null'): return np.nan
    kind = _classify_value(s0)
    if kind == 'EU':
        try: return float(s0.replace('.', '').replace(',', '.'))
        except: return np.nan
    if kind == 'US':
        try: return float(s0.replace(',', ''))
        except: return np.nan
    if kind == 'comma_only':
        if preference == 'EU':
            try: return float(s0.replace(',', '.'))
            except: return np.nan
        if preference == 'US':
            try: return float(s0.replace(',', ''))
            except: return np.nan
        last_grp = s0.split(',')[-1]
        if last_grp.isdigit() and len(last_grp) == 3 and len(s0.split(',')) >= 2:
             try: return float(s0.replace(',', ''))
             except: return np.nan
        try: return float(s0.replace(',', '.'))
        except: return np.nan
    if kind == 'dot_only':
        if preference == 'US':
             try: return float(s0)
             except: return np.nan
        if preference == 'EU':
             try: return float(s0.replace('.', ''))
             except: return np.nan
        last_grp = s0.split('.')[-1]
        if last_grp.isdigit() and len(last_grp) == 3 and len(s0.split('.')) >= 2:
             try: return float(s0.replace('.', ''))
             except: return np.nan
        try: return float(s0)
        except: return np.nan
    if kind == 'int':
        try: return float(s0)
        except: return np.nan
    return np.nan

@st.cache_data
def convert_mixed_numeric_columns(df_in: pd.DataFrame, exclude: set = None, verbose: bool = True) -> pd.DataFrame:
    df_out = df_in.copy()
    exclude = set() if exclude is None else set(exclude)
    diagnostics = {}
    for col in df_out.columns:
        if col in exclude or pd.api.types.is_numeric_dtype(df_out[col]): continue
        s = df_out[col].astype(str)
        if not s.str.contains(r'\d', regex=True).any(): continue
        s_clean = s.map(_strip_spaces)
        kinds = s_clean.map(_classify_value)
        eu_votes = int((kinds == 'EU').sum())
        us_votes = int((kinds == 'US').sum())
        preference = 'EU' if eu_votes > us_votes else ('US' if us_votes > eu_votes else None)
        converted = s_clean.map(lambda x: _convert_value(x, preference))
        if (np.isfinite(converted).sum() / max(len(converted), 1)) < 0.1:
             diagnostics[col] = "Skipped (low valid ratio)"
             continue
        df_out[col] = pd.Series(converted, index=df_out.index, dtype="Float64")
        diagnostics[col] = f"Converted (pref={preference})"
    
    if verbose and diagnostics:
        for c, info in diagnostics.items(): st.text(f"- {c}: {info}")
    return df_out