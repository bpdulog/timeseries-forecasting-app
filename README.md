# 📈 Time Series Econometric Forecasting & Diagnostic Web Suite

An interactive time series forecasting web application following the **Box-Jenkins econometric methodology** (as taught in CMU Tepper 45-912 *Business Forecasting with Time Series Models* by Prof. Fallaw Sowell).

This project runs **both as a native local Streamlit application** and **100% client-side in the browser on GitHub Pages** with zero backend server via [Stlite](https://github.com/whitphx/stlite) and Pyodide WebAssembly.

---

## 🚀 Features

1. **Flexible Data Ingestion**:
   - Upload any `.csv` or `.xlsx` file.
   - Built-in monthly commodity sample dataset (**US Hot Rolled Steel Prices**) ready for instant testing.
   - Column selector for date/timestamp and target series $Y_t$.
   - User-selected frequency (monthly, quarterly, daily, weekly, or non-seasonal).
   - Missing data imputation (linear interpolation, forward-fill, or removal).

2. **Exploratory Data Analysis & Transformations**:
   - Time plots and descriptive summary statistics (Min, Q1, Median, Mean, Q3, Max, Std Dev).
   - Optional variance stabilization via **Natural Log Transformation** ($\log(Y_t)$) with automated back-transformation ($\exp(\hat{Y})$) for final forecasts and intervals.
   - Classical Additive and Multiplicative time series decomposition (Trend, Seasonality, Remainder).

3. **Covariance Stationarity & Differencing (Box-Jenkins Step 1)**:
   - **Augmented Dickey-Fuller (ADF) Unit Root Test** with test statistics, p-values, critical thresholds, and stationarity verdict.
   - Differencing controls for regular first difference ($d=1, 2$) and seasonal difference ($D=1$).
   - Live plot and ADF re-test of the differenced series $\Delta Y_t$.

4. **Model Identification (Box-Jenkins Step 2)**:
   - Sample **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** up to 36 lags.
   - Overlaid 95% Bartlett confidence limits ($\pm 1.96 / \sqrt{T}$).
   - Clear identification rules for AR($p$), MA($q$), and ARMA processes.

5. **Model Estimation & Comparison (Box-Jenkins Step 3)**:
   - **ARIMA / SARIMA Engine**:
     - *Automated Mode*: Grid search across $(p, d, q)$ candidate combinations minimizing AIC and BIC.
     - *Manual Mode*: Custom $(p, d, q) \times (P, D, Q)_s$ parameters with constant/drift selection.
     - Complete econometric output table: Coefficients, standard errors, z-stats, p-values, Log-Likelihood, AIC, and BIC.
   - **Exponential Smoothing Baselines**:
     - Simple Exponential Smoothing (SES)
     - Holt's Linear Trend
     - Holt-Winters Additive & Multiplicative Seasonality

6. **Residual Diagnostic Suite (Box-Jenkins Step 4)**:
   - Standardized residual plot with mean-zero and 2-sigma homoscedastic bounds.
   - **Ljung-Box White Noise Portmanteau Test** across lags 6, 12, 18 to verify absence of serial autocorrelation.

7. **Multi-Step Horizon Forecasting (Box-Jenkins Step 5)**:
   - Horizon slider ($h$ periods ahead).
   - **80% and 95% prediction intervals** from ARIMA models; smoothing models use approximate residual-based bands.
   - Interactive Plotly chart combining historical series, in-sample fitted curve, point forecast, and confidence ribbons.
   - Out-of-sample forecast data table.
   - One-click **CSV Download** of all forecasted values and intervals.

---

## 🛠️ Option 1: Running Locally with Python & Streamlit

```bash
# Navigate to the project directory
cd C:\Users\Bryan\timeseries-forecasting-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🌐 Option 2: Deploying to GitHub Pages (Zero-Server Web App)

Because this repository contains `index.html` powered by Stlite (Pyodide WebAssembly), it runs completely inside static hosting with **no server needed**:

1. Create a new repository on GitHub (e.g. `timeseries-forecaster`).
2. Push this folder to GitHub:
   ```bash
   cd C:\Users\Bryan\timeseries-forecasting-app
   git init
   git add .
   git commit -m "Initial commit of Time Series Forecasting Suite"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/timeseries-forecaster.git
   git push -u origin main
   ```
3. In your GitHub repository:
   - Go to **Settings** $\rightarrow$ **Pages**.
   - Under **Build and deployment** $\rightarrow$ **Source**, choose **Deploy from a branch**.
   - Select branch: `main` and folder: `/ (root)`.
   - Click **Save**.
4. Within 1–2 minutes, your web application will be live at:
   `https://<YOUR_GITHUB_USERNAME>.github.io/timeseries-forecaster/`

---

## 🧪 Testing the WebAssembly Version Locally

To test `index.html` in your browser before publishing to GitHub Pages:

```bash
cd C:\Users\Bryan\timeseries-forecasting-app
python -m http.server 8000
```
Then visit `http://localhost:8000` in your web browser.
