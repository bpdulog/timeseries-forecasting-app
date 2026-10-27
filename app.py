import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Time Series Forecasting & Diagnostic Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Time Series Forecasting & Diagnostic Suite")
st.caption("Econometric Box-Jenkins Methodology (Identification • Estimation • Diagnostics • Multi-Step Forecasting)")

# ==========================================
# 1. SIDEBAR: DATA INGESTION & CONFIG
# ==========================================
st.sidebar.header("📁 1. Data Ingestion")

data_source = st.sidebar.radio(
    "Choose Data Source:",
    ("Use Built-in Sample (US Steel Prices)", "Upload CSV / Excel File"),
)

df_raw = None

if data_source == "Use Built-in Sample (US Steel Prices)":
    try:
        df_raw = pd.read_csv("sample_data.csv")
    except Exception:
        # Generate inline fallback if file not in working dir
        np.random.seed(42)
        dates = pd.date_range(start="2005-01-01", periods=180, freq="MS")
        trend = np.linspace(400, 750, 180)
        seasonal = 45 * np.sin(2 * np.pi * np.arange(180) / 12) + 20 * np.cos(2 * np.pi * np.arange(180) / 12)
        noise = np.random.normal(0, 25, 180)
        df_raw = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Steel_Price_USD": np.round(trend + seasonal + noise, 2)})
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")

if df_raw is None:
    st.info("👈 Please select a data source or upload a CSV file from the sidebar to begin.")
    st.stop()

# Column Selection
cols = list(df_raw.columns)
date_col = st.sidebar.selectbox("Date / Timestamp Column", options=cols, index=0)
numeric_cols = [c for c in cols if c != date_col and pd.to_numeric(df_raw[c], errors="coerce").notnull().sum() > 5]

if not numeric_cols:
    st.error("No numeric columns found in the dataset.")
    st.stop()

target_col = st.sidebar.selectbox("Target Series (Y)", options=numeric_cols, index=0)

# Preprocessing & Frequency
st.sidebar.header("⚙️ 2. Frequency & Transformations")
seasonality_choice = st.sidebar.selectbox(
    "Series Frequency / Seasonality (s)",
    options=["Monthly (s=12)", "Quarterly (s=4)", "Daily (s=7)", "Weekly (s=52)", "Annual / Non-Seasonal (s=1)"],
    index=0,
)

freq_map = {
    "Monthly (s=12)": (12, "MS"),
    "Quarterly (s=4)": (4, "QS"),
    "Daily (s=7)": (7, "D"),
    "Weekly (s=52)": (52, "W"),
    "Annual / Non-Seasonal (s=1)": (1, None),
}
seasonal_period, default_freq = freq_map[seasonality_choice]

# Clean series
df = df_raw[[date_col, target_col]].copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(by=date_col)
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

missing_count = df[target_col].isnull().sum()
if missing_count > 0:
    st.sidebar.warning(f"{missing_count} missing value(s) detected.")
    impute_method = st.sidebar.selectbox("Imputation Method", ["Linear Interpolation", "Forward Fill", "Drop Missing"])
    if impute_method == "Linear Interpolation":
        df[target_col] = df[target_col].interpolate(method="linear").bfill()
    elif impute_method == "Forward Fill":
        df[target_col] = df[target_col].ffill().bfill()
    else:
        df = df.dropna(subset=[target_col])

# Variance Stabilization (Log Transform)
apply_log = st.sidebar.checkbox("Apply Natural Log Transform (log(Y))", value=False, help="Useful if variance grows proportionally with series level")

series_raw = df.set_index(date_col)[target_col]
if apply_log:
    if (series_raw <= 0).any():
        st.sidebar.error("Cannot apply log transform to non-positive values.")
        st.stop()
    y_work = np.log(series_raw)
    series_label = f"log({target_col})"
else:
    y_work = series_raw.copy()
    series_label = target_col

N = len(y_work)
if N < 15:
    st.error(f"Series length is too short ({N} observations). At least 15 observations are recommended.")
    st.stop()

# A fitted model belongs to the exact data and transformation used to fit it.
data_signature = (tuple(pd.util.hash_pandas_object(y_work, index=True)), apply_log, seasonal_period)
if st.session_state.get("data_signature") != data_signature:
    for key in ("best_arima_order", "best_arima_model", "manual_model", "manual_name", "exp_model", "exp_name", "active_engine"):
        st.session_state.pop(key, None)
    st.session_state["data_signature"] = data_signature

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
st.subheader("1. Exploratory Data Analysis & Summary")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Observations (T)", N)
m2.metric("Start Date", series_raw.index.min().strftime("%Y-%m-%d"))
m3.metric("End Date", series_raw.index.max().strftime("%Y-%m-%d"))
m4.metric("Mean", f"{series_raw.mean():.2f}")
m5.metric("Median", f"{series_raw.median():.2f}")
m6.metric("Std Dev", f"{series_raw.std():.2f}")

# Time series plot
fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(x=y_work.index, y=y_work.values, mode="lines+markers", name=series_label, line=dict(color="#1f77b4", width=2)))
fig_raw.update_layout(
    title=f"Time Series Plot: {series_label}",
    xaxis_title="Date",
    yaxis_title=series_label,
    template="plotly_white",
    height=380,
    margin=dict(l=40, r=40, t=40, b=40),
)
st.plotly_chart(fig_raw, use_container_width=True)

# Classical Decomposition
with st.expander("🔍 View Time Series Decomposition (Trend, Seasonal, Remainder)"):
    if seasonal_period > 1 and N >= 2 * seasonal_period:
        decomp_model = st.radio("Decomposition Type", ["additive", "multiplicative"], horizontal=True)
        try:
            # Multiplicative requires positive values
            decomp_data = series_raw if (decomp_model == "multiplicative" and (series_raw > 0).all()) else y_work
            decomp = seasonal_decompose(decomp_data, model=decomp_model, period=seasonal_period)
            
            fig_decomp = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Trend Component", "Seasonal Component", "Residual / Irregular"])
            fig_decomp.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend.values, name="Trend", line=dict(color="#2ca02c")), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal.values, name="Seasonal", line=dict(color="#ff7f0e")), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid.values, mode="markers", name="Residual", marker=dict(color="#d62728", size=4)), row=3, col=1)
            fig_decomp.update_layout(height=520, template="plotly_white", showlegend=False, margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_decomp, use_container_width=True)
        except Exception as ex:
            st.info(f"Decomposition notice: {ex}")
    else:
        st.info("Decomposition requires seasonal period > 1 and at least 2 full seasonal cycles.")

# ==========================================
# 3. STATIONARITY & DIFFERENCING (BOX-JENKINS STEP 1)
# ==========================================
st.markdown("---")
st.subheader("2. Covariance Stationarity & Unit Root Diagnostics (Box-Jenkins Step 1)")

col_stat_l, col_stat_r = st.columns([1, 1])

with col_stat_l:
    st.markdown("##### Augmented Dickey-Fuller (ADF) Test on Working Series")
    try:
        adf_res = adfuller(y_work.dropna(), autolag="AIC")
        adf_stat, adf_p, adf_crit = adf_res[0], adf_res[1], adf_res[4]
        
        st.write(f"**ADF Test Statistic:** `{adf_stat:.4f}`")
        st.write(f"**p-value:** `{adf_p:.4e}`")
        st.write(f"**Critical Values:** 1%: `{adf_crit['1%']:.3f}`, 5%: `{adf_crit['5%']:.3f}`, 10%: `{adf_crit['10%']:.3f}`")
        
        if adf_p <= 0.05:
            st.success("✅ Reject the unit-root null at the 5% significance level. This test alone does not establish covariance stationarity.")
        else:
            st.warning("⚠️ Fail to reject the unit-root null at the 5% significance level. Consider differencing and other diagnostics.")
    except Exception as ex:
        st.error(f"ADF test failed: {ex}")

with col_stat_r:
    st.markdown("##### Differencing Controls")
    diff_d = st.slider("Regular Differencing Order (d)", min_value=0, max_value=2, value=0 if (locals().get('adf_p', 1.0) <= 0.05) else 1)
    diff_D = 0
    if seasonal_period > 1:
        diff_D = st.slider(f"Seasonal Differencing Order (D, lag={seasonal_period})", min_value=0, max_value=1, value=0)
    
    # Apply differencing for inspection
    y_diff = y_work.copy()
    if diff_d > 0:
        for _ in range(diff_d):
            y_diff = y_diff.diff()
    if diff_D > 0 and seasonal_period > 1:
        y_diff = y_diff.diff(seasonal_period)
    y_diff_clean = y_diff.dropna()

    if (diff_d > 0 or diff_D > 0) and len(y_diff_clean) > 10:
        try:
            adf_diff = adfuller(y_diff_clean, autolag="AIC")
            st.write(f"**Differenced Series ADF p-value:** `{adf_diff[1]:.4e}`")
            st.info("The ADF test assesses evidence against a unit root; it does not prove stationarity.")
        except ValueError as ex:
            st.info(f"Differenced-series ADF test unavailable: {ex}")

# Differenced plot
if st.session_state.get("model_differencing") != (diff_d, diff_D):
    for key in ("best_arima_order", "best_arima_model", "manual_model", "manual_name", "exp_model", "exp_name", "active_engine"):
        st.session_state.pop(key, None)
    st.session_state["model_differencing"] = (diff_d, diff_D)

if diff_d > 0 or diff_D > 0:
    fig_diff = go.Figure()
    fig_diff.add_trace(go.Scatter(x=y_diff_clean.index, y=y_diff_clean.values, mode="lines", name=f"Diff(d={diff_d}, D={diff_D})", line=dict(color="#9467bd")))
    fig_diff.update_layout(title=f"Differenced Series (d={diff_d}, D={diff_D})", template="plotly_white", height=280, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_diff, use_container_width=True)

# ==========================================
# 4. MODEL IDENTIFICATION (ACF & PACF)
# ==========================================
st.markdown("---")
st.subheader("3. Model Identification: ACF & PACF Analysis (Box-Jenkins Step 2)")

max_lags = min(36, len(y_diff_clean) // 2 - 1)
if max_lags >= 4:
    acf_vals = acf(y_diff_clean, nlags=max_lags)
    pacf_vals = pacf(y_diff_clean, nlags=max_lags, method="ols")
    conf_bound = 1.96 / np.sqrt(len(y_diff_clean))
    lags = np.arange(len(acf_vals))
    
    fig_corr = make_subplots(rows=1, cols=2, subplot_titles=["Autocorrelation Function (ACF)", "Partial Autocorrelation Function (PACF)"])
    
    # ACF
    fig_corr.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF", marker_color="#1f77b4"), row=1, col=1)
    fig_corr.add_trace(go.Scatter(x=[0, max_lags], y=[conf_bound, conf_bound], mode="lines", line=dict(dash="dash", color="red"), name="+95% Bound"), row=1, col=1)
    fig_corr.add_trace(go.Scatter(x=[0, max_lags], y=[-conf_bound, -conf_bound], mode="lines", line=dict(dash="dash", color="red"), name="-95% Bound"), row=1, col=1)
    
    # PACF
    fig_corr.add_trace(go.Bar(x=lags, y=pacf_vals, name="PACF", marker_color="#ff7f0e"), row=1, col=2)
    fig_corr.add_trace(go.Scatter(x=[0, max_lags], y=[conf_bound, conf_bound], mode="lines", line=dict(dash="dash", color="red"), showlegend=False), row=1, col=2)
    fig_corr.add_trace(go.Scatter(x=[0, max_lags], y=[-conf_bound, -conf_bound], mode="lines", line=dict(dash="dash", color="red"), showlegend=False), row=1, col=2)
    
    fig_corr.update_layout(height=350, template="plotly_white", showlegend=False, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.info("💡 **Identification Rules**: PACF cutoff at lag $p$ + decaying ACF $\\rightarrow$ **AR($p$)** | ACF cutoff at lag $q$ + decaying PACF $\\rightarrow$ **MA($q$)** | Spikes at seasonal lags ($s, 2s$) indicate seasonal orders $(P, Q)_s$.")

# ==========================================
# 5. MODEL ESTIMATION & COMPARISON
# ==========================================
st.markdown("---")
st.subheader("4. Model Estimation & Diagnostics (Box-Jenkins Steps 3 & 4)")

tab_arima, tab_exp = st.tabs(["🎯 ARIMA / SARIMA Engine", "📉 Exponential Smoothing Baselines"])

fitted_model = None
model_name = ""

with tab_arima:
    col_ar_mode, col_ar_cfg = st.columns([1, 2])
    
    with col_ar_mode:
        arima_mode = st.radio("Model Selection Strategy", ["Automated AIC Grid Search", "Manual Parameter Specification"])
        trend_option = st.selectbox("Deterministic Term", ["c (constant/drift)", "n (no constant)"])
        trend_val = "c" if "constant" in trend_option else "n"
    
    with col_ar_cfg:
        if arima_mode == "Automated AIC Grid Search":
            st.markdown("##### Auto-Search Configuration")
            max_p = st.number_input("Max AR (p)", 0, 4, 2)
            max_q = st.number_input("Max MA (q)", 0, 4, 2)
            search_d = diff_d
            
            run_search = st.button("🚀 Run Grid Search & Fit Best Model", type="primary")
            
            search_config = (max_p, max_q, search_d, trend_val)
            if run_search or "best_arima_order" not in st.session_state or st.session_state.get("search_config") != search_config:
                best_aic = float("inf")
                best_order = (1, search_d, 1)
                best_res = None
                
                with st.spinner("Searching optimal (p, d, q) specifications..."):
                    for p_i in range(max_p + 1):
                        for q_i in range(max_q + 1):
                            try:
                                candidate = SARIMAX(
                                    y_work,
                                    order=(p_i, search_d, q_i),
                                    trend=trend_val,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False)
                                if candidate.aic < best_aic:
                                    best_aic = candidate.aic
                                    best_order = (p_i, search_d, q_i)
                                    best_res = candidate
                            except Exception:
                                continue
                st.session_state["best_arima_order"] = best_order
                st.session_state["best_arima_model"] = best_res
                st.session_state["search_config"] = search_config
                st.session_state["active_engine"] = "ARIMA"
                
            if st.session_state.get("best_arima_model") is not None and st.session_state.get("active_engine") == "ARIMA":
                best_order = st.session_state["best_arima_order"]
                fitted_model = st.session_state["best_arima_model"]
                model_name = f"ARIMA{best_order}"
                st.success(f"🏆 Best Specification: **ARIMA{best_order}** with AIC: `{fitted_model.aic:.2f}` | BIC: `{fitted_model.bic:.2f}`")
        else:
            st.markdown("##### Manual Specification")
            c_p, c_d, c_q = st.columns(3)
            p_val = c_p.number_input("p (AR)", 0, 5, 1)
            d_val = c_d.number_input("d (Diff)", 0, 2, diff_d)
            q_val = c_q.number_input("q (MA)", 0, 5, 1)
            
            include_seasonal = st.checkbox("Include Seasonal SARIMA component", value=(seasonal_period > 1))
            P_val, D_val, Q_val = 0, 0, 0
            if include_seasonal:
                cs_p, cs_d, cs_q = st.columns(3)
                P_val = cs_p.number_input("P (Seasonal AR)", 0, 2, 1)
                D_val = cs_d.number_input("D (Seasonal Diff)", 0, 1, 0)
                Q_val = cs_q.number_input("Q (Seasonal MA)", 0, 2, 0)
                
            if st.button("Fit Manual SARIMA Model"):
                try:
                    fitted_model = SARIMAX(
                        y_work,
                        order=(p_val, d_val, q_val),
                        seasonal_order=(P_val, D_val, Q_val, seasonal_period) if include_seasonal else (0, 0, 0, 0),
                        trend=trend_val,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                    st.session_state["manual_model"] = fitted_model
                    st.session_state["manual_name"] = f"SARIMA({p_val},{d_val},{q_val})x({P_val},{D_val},{Q_val})[{seasonal_period}]"
                    st.session_state["active_engine"] = "MANUAL_ARIMA"
                except Exception as ex:
                    st.error(f"Fitting error: {ex}")
            
            if st.session_state.get("active_engine") == "MANUAL_ARIMA":
                fitted_model = st.session_state.get("manual_model")
                model_name = st.session_state.get("manual_name")

with tab_exp:
    exp_method = st.selectbox(
        "Exponential Smoothing Model",
        ["Simple Exponential Smoothing (SES)", "Holt's Linear Trend", "Holt-Winters Seasonal Additive", "Holt-Winters Seasonal Multiplicative"],
    )
    if st.button("Fit Exponential Smoothing Model"):
        try:
            if "Simple" in exp_method:
                fitted_model = SimpleExpSmoothing(y_work).fit()
                model_name = "Simple Exponential Smoothing (SES)"
            elif "Holt's Linear" in exp_method:
                fitted_model = Holt(y_work).fit()
                model_name = "Holt's Linear Trend"
            elif "Additive" in exp_method:
                fitted_model = ExponentialSmoothing(y_work, trend="add", seasonal="add", seasonal_periods=seasonal_period).fit()
                model_name = "Holt-Winters (Additive)"
            else:
                fitted_model = ExponentialSmoothing(y_work, trend="add", seasonal="mul", seasonal_periods=seasonal_period).fit()
                model_name = "Holt-Winters (Multiplicative)"
            st.session_state["exp_model"] = fitted_model
            st.session_state["exp_name"] = model_name
            st.session_state["active_engine"] = "EXP"
        except Exception as ex:
            st.error(f"Exponential smoothing fit error: {ex}")
            
    if st.session_state.get("active_engine") == "EXP":
        fitted_model = st.session_state.get("exp_model")
        model_name = st.session_state.get("exp_name")

# Fallback: auto-fit ARIMA(1, d, 1) if nothing chosen yet
if fitted_model is None:
    try:
        fitted_model = SARIMAX(y_work, order=(1, diff_d, 1), trend="c").fit(disp=False)
        model_name = f"Default ARIMA(1,{diff_d},1)"
    except Exception:
        fitted_model = SimpleExpSmoothing(y_work).fit()
        model_name = "Default SES"

# ==========================================
# 6. RESIDUAL DIAGNOSTICS SUITE (STEP 4)
# ==========================================
st.markdown("##### Residual Diagnostics Suite")

residuals = pd.Series(fitted_model.resid, index=y_work.index).dropna()
# Remove initial zero or unstable diff residuals
if len(residuals) > 5:
    residuals = residuals.iloc[max(diff_d, 1):]

col_diag1, col_diag2 = st.columns(2)

with col_diag1:
    # Standardized residuals
    std_res = (residuals - residuals.mean()) / residuals.std()
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=std_res.index, y=std_res.values, mode="lines", name="Std Residuals", line=dict(color="#2ca02c")))
    fig_res.add_hline(y=0, line_dash="dash", line_color="black")
    fig_res.add_hline(y=2, line_dash="dot", line_color="red")
    fig_res.add_hline(y=-2, line_dash="dot", line_color="red")
    fig_res.update_layout(title="Standardized Residuals (Mean 0, Homoscedastic Bounds)", template="plotly_white", height=280, margin=dict(l=30, r=30, t=35, b=30))
    st.plotly_chart(fig_res, use_container_width=True)

with col_diag2:
    # Ljung-Box Test
    test_lags = [l for l in [6, 12, 18, 24] if l < len(residuals) // 2]
    if test_lags:
        lb_df = acorr_ljungbox(residuals, lags=test_lags, return_df=True)
        st.markdown("**Ljung-Box White Noise Portmanteau Test**")
        st.dataframe(lb_df.style.format({"lb_stat": "{:.3f}", "lb_pvalue": "{:.4e}"}), use_container_width=True)
        min_p = lb_df["lb_pvalue"].min()
        if min_p > 0.05:
            st.success("✅ **Residuals are White Noise:** No significant serial autocorrelation remaining (p > 0.05).")
        else:
            st.warning("⚠️ **Autocorrelation Detected:** Some residual autocorrelation remains. Consider higher AR/MA terms.")

# ==========================================
# 7. MULTI-STEP FORECASTING & EXPORT (STEP 5)
# ==========================================
st.markdown("---")
st.subheader("5. Multi-Step Forecasting & Prediction Cones (Box-Jenkins Step 5)")

col_fc1, col_fc2 = st.columns([1, 3])

with col_fc1:
    h = st.slider("Forecast Horizon (h periods ahead)", min_value=1, max_value=36, value=12)
    export_csv = st.empty()

# Calculate forecasts and intervals
last_date = y_work.index[-1]
if default_freq:
    future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(default_freq), periods=h, freq=default_freq)
else:
    # Fallback to inferred step
    step = (y_work.index[-1] - y_work.index[0]) / (len(y_work) - 1)
    future_dates = [last_date + (i + 1) * step for i in range(h)]

if hasattr(fitted_model, "get_forecast"):
    fc_res = fitted_model.get_forecast(steps=h)
    point_fc = fc_res.predicted_mean
    ci_80 = fc_res.conf_int(alpha=0.20)
    ci_95 = fc_res.conf_int(alpha=0.05)
    
    lo_80 = ci_80.iloc[:, 0].values
    hi_80 = ci_80.iloc[:, 1].values
    lo_95 = ci_95.iloc[:, 0].values
    hi_95 = ci_95.iloc[:, 1].values
    mean_fc = point_fc.values
else:
    # Holt-Winters / SES fallback
    st.caption("Smoothing intervals are approximate residual-based bands, not model-calibrated prediction intervals.")
    mean_fc = fitted_model.forecast(h).values
    # Empirical standard error expansion
    res_std = residuals.std()
    horizons = np.sqrt(np.arange(1, h + 1))
    lo_80 = mean_fc - 1.28 * res_std * horizons
    hi_80 = mean_fc + 1.28 * res_std * horizons
    lo_95 = mean_fc - 1.96 * res_std * horizons
    hi_95 = mean_fc + 1.96 * res_std * horizons

# Back-transform if log was applied
if apply_log:
    final_point = np.exp(mean_fc)
    final_lo_80 = np.exp(lo_80)
    final_hi_80 = np.exp(hi_80)
    final_lo_95 = np.exp(lo_95)
    final_hi_95 = np.exp(hi_95)
    plot_history_y = series_raw.values
    plot_fitted_y = np.exp(fitted_model.fittedvalues)
else:
    final_point = mean_fc
    final_lo_80 = lo_80
    final_hi_80 = hi_80
    final_lo_95 = lo_95
    final_hi_95 = hi_95
    plot_history_y = y_work.values
    plot_fitted_y = fitted_model.fittedvalues

# Forecast Visualization
fig_fc = go.Figure()

# Historical
fig_fc.add_trace(go.Scatter(x=series_raw.index, y=plot_history_y, mode="lines", name="Historical Observations", line=dict(color="#1f77b4", width=2)))

# In-sample fit
fig_fc.add_trace(go.Scatter(x=series_raw.index, y=plot_fitted_y, mode="lines", name="In-Sample Model Fit", line=dict(color="#7f7f7f", width=1, dash="dot")))

# 95% Confidence Band
fig_fc.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates)[::-1],
    y=list(final_hi_95) + list(final_lo_95)[::-1],
    fill="toself",
    fillcolor="rgba(255, 127, 14, 0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="95% Prediction Interval",
))

# 80% Confidence Band
fig_fc.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates)[::-1],
    y=list(final_hi_80) + list(final_lo_80)[::-1],
    fill="toself",
    fillcolor="rgba(255, 127, 14, 0.3)",
    line=dict(color="rgba(255,255,255,0)"),
    name="80% Prediction Interval",
))

# Point Forecast Line
fig_fc.add_trace(go.Scatter(
    x=future_dates,
    y=final_point,
    mode="lines+markers",
    name=f"Forecast ({model_name})",
    line=dict(color="#ff7f0e", width=3),
    marker=dict(size=6),
))

fig_fc.update_layout(
    title=f"Multi-Step Ahead Point Forecast & Uncertainty Cones ({model_name})",
    xaxis_title="Date",
    yaxis_title=target_col,
    template="plotly_white",
    height=480,
    hovermode="x unified",
    margin=dict(l=40, r=40, t=50, b=40),
)

with col_fc2:
    st.plotly_chart(fig_fc, use_container_width=True)

# Forecast Table & Download
df_forecast = pd.DataFrame({
    "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
    "Point_Forecast": np.round(final_point, 3),
    "Lo_80": np.round(final_lo_80, 3),
    "Hi_80": np.round(final_hi_80, 3),
    "Lo_95": np.round(final_lo_95, 3),
    "Hi_95": np.round(final_hi_95, 3),
})

st.markdown("##### Out-of-Sample Forecast Table")
st.dataframe(df_forecast, use_container_width=True)

csv_data = df_forecast.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Forecast CSV",
    data=csv_data,
    file_name="forecast_results.csv",
    mime="text/csv",
)
