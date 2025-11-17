


# app.py (Streamlit)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("Portfolio Analysis & Markowitz Optimization (TRY)")

# -------------------------
# PARAMETERS
# -------------------------
tick_index = "XU030.IS"
tick_gold = "IAU"
tick_silver = "SLV"
tick_spy = "SPY"
tick_fx = "TRY=X"
# -------------------------
# FIXED DOWNLOAD PERIOD
# -------------------------
period = "3y"  # fixed 3-year download period to ensure full 1-year analysis
# period = st.sidebar.selectbox("Download period", options=["2y", "3y", "5y"], index=1)
TRADING_DAYS = 252

# default RF
default_rf = 36.0

st.sidebar.header("User inputs")
notional = st.sidebar.number_input("Portfolio notional (TRY) for allocation amounts", value=1_000_000, step=10000)
rf_input = st.sidebar.number_input("Risk-free annual return (%)", value=float(default_rf), step=0.5)

# Download data
st.write("Downloading adjusted prices and FX...")
symbols = [tick_index, tick_gold, tick_silver, tick_spy, tick_fx]
df_raw = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]
df_raw = df_raw.rename(columns={
    tick_index: "BIST30",
    tick_gold: "GoldUSD",
    tick_silver: "SilverUSD",
    tick_spy: "SPYUSD",
    tick_fx: "USDTRY"
})
df_raw = df_raw.dropna(how="any")  # aligned days

# Convert USD assets to TRY immediately
df_try = pd.DataFrame(index=df_raw.index)
df_try["BIST30"] = df_raw["BIST30"]
df_try["GoldTRY"]   = df_raw["GoldUSD"]  * df_raw["USDTRY"]
df_try["SilverTRY"] = df_raw["SilverUSD"]* df_raw["USDTRY"]
df_try["SPYTRY"]    = df_raw["SPYUSD"]   * df_raw["USDTRY"]
df_try = df_try.dropna(how="any")

# last 252 days
prices = df_try.tail(TRADING_DAYS)
if prices.shape[0] < 100:
    st.error("Not enough data in the selected period. Increase 'Download period'.")
    st.stop()

# historical metrics
returns_1y_pct = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
daily_returns = prices.pct_change().dropna()
annual_vol_pct = daily_returns.std() * np.sqrt(TRADING_DAYS) * 100
cov_annual = daily_returns.cov() * TRADING_DAYS
corr = daily_returns.corr()

# display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1-year total return (%)")
    st.dataframe(returns_1y_pct.round(3).to_frame("1Y Return (%)"))
with col2:
    st.subheader("Annualized volatility (%)")
    st.dataframe(annual_vol_pct.round(3).to_frame("Annual Vol (%)"))
with col3:
    st.subheader("Covariance (annual)")
    st.dataframe(cov_annual.round(6))

# correlation heatmap
st.subheader("Correlation heatmap (1-year daily returns)")
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
st.pyplot(fig)

# normalized price plot
st.subheader("Normalized price (1 = first date of 1-year window)")
norm = prices / prices.iloc[0]
fig, ax = plt.subplots(figsize=(10,4))
norm.plot(ax=ax)
ax.set_xlabel("Date"); ax.set_ylabel("Normalized price")
st.pyplot(fig)

# -------------------------
# USER-EDITABLE EXPECTED RETURNS (annual, %) - Streamlit only
# -------------------------
st.subheader("Expected annual returns (user inputs)")
st.write("If you leave them unchanged, defaults equal 1-year total returns (%)")

# Use 1-year total return (P_end/P_start -1)*100 as default
returns_1y_pct = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

user_expected = {}
assets = list(prices.columns)
col_inputs = st.columns(len(assets))
for i, asset in enumerate(assets):
    default_val = float(returns_1y_pct[asset].round(2))
    user_expected[asset] = col_inputs[i].number_input(f"{asset} (%)", value=default_val, step=0.1)

# Risk-free
rf_annual = rf_input / 100.0

# Build mu vector (annual) using user inputs (converted to fraction)
mu_annual_user = pd.Series({a: user_expected[a]/100.0 for a in assets})

# Extend to include RF
mu_ext = mu_annual_user.copy()
mu_ext["RF"] = rf_annual

# Covariance extended
cov_ext = cov_annual.copy()
cov_ext["RF"] = 0.0
cov_ext.loc["RF"] = 0.0

# -------------------------
# Optimization: Monte Carlo + exact max Sharpe using SLSQP
# -------------------------
st.subheader("Optimization & Efficient Frontier")

# Monte Carlo scatter for visualization
# num_rand = st.sidebar.slider("Monte Carlo portfolios", 5000, 1000, 20000, step=1000)
# Fixed number of Monte Carlo portfolios
num_rand = 10000
res_mc = np.zeros((3, num_rand))
w_list = []
for i in range(num_rand):
    w = np.random.random(len(mu_ext))
    w /= w.sum()
    w_list.append(w)
    r = np.dot(w, mu_ext.values)
    vol = np.sqrt(w @ cov_ext.values @ w)
    sharpe = (r - rf_annual) / vol if vol>0 else np.nan
    res_mc[0,i] = r
    res_mc[1,i] = vol
    res_mc[2,i] = sharpe

# exact max Sharpe
n = len(mu_ext)
bounds = tuple((0.0, 1.0) for _ in range(n))
cons = ({'type':'eq', 'fun': lambda w: np.sum(w)-1.0})
def neg_sharpe(w):
    r = float(np.dot(w, mu_ext.values))
    vol = float(np.sqrt(w @ cov_ext.values @ w))
    return - (r - rf_annual) / vol if vol>0 else 1e6

x0 = np.array([1.0/n]*n)
opt = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons, method='SLSQP')
if not opt.success:
    st.error("Sharpe optimization failed: " + opt.message)
w_sharpe = opt.x
ret_sharpe = float(np.dot(w_sharpe, mu_ext.values))
vol_sharpe = float(np.sqrt(w_sharpe @ cov_ext.values @ w_sharpe))
sharpe_val = (ret_sharpe - rf_annual) / vol_sharpe

st.write("Max-Sharpe portfolio (weights):")
weights_df = pd.DataFrame({"Asset": mu_ext.index, "Weight": w_sharpe}).set_index("Asset")
st.dataframe(weights_df.style.format({"Weight":"{:.4f}"}))

st.write(f"Portfolio expected annual return: {ret_sharpe:.2%}, vol: {vol_sharpe:.2%}, Sharpe: {sharpe_val:.4f}")

# Efficient frontier (min vol for target returns)
target_returns = np.linspace(mu_ext.min(), mu_ext.max(), 40)
frontier_vols = []
for tr in target_returns:
    def fun(w): return float(w @ cov_ext.values @ w)
    cons_local = (
        {'type':'eq', 'fun': lambda w: np.sum(w)-1.0},
        {'type':'eq', 'fun': lambda w: float(np.dot(w, mu_ext.values)) - tr}
    )
    try:
        res_t = minimize(fun, x0, bounds=bounds, constraints=cons_local, method='SLSQP')
        if res_t.success:
            frontier_vols.append(np.sqrt(res_t.fun))
        else:
            frontier_vols.append(np.nan)
    except Exception:
        frontier_vols.append(np.nan)

# Plot efficient frontier + Monte Carlo scatter + assets + max sharpe
fig, ax = plt.subplots(figsize=(8,5))
sc = ax.scatter(res_mc[1,:], res_mc[0,:], c=res_mc[2,:], cmap="viridis", alpha=0.4, s=8)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Sharpe")
ax.plot(frontier_vols, target_returns, color='red', lw=2, label="Efficient frontier")
# asset points
for i, name in enumerate(mu_ext.index):
    vol_i = np.sqrt(cov_ext.values[i,i])
    ret_i = mu_ext.values[i]
    ax.scatter(vol_i, ret_i, marker='o', s=80)
    ax.text(vol_i, ret_i, "  "+name)
ax.scatter(vol_sharpe, ret_sharpe, marker='*', color='black', s=200, label='Max Sharpe')
ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Annualized Return")
ax.legend()
st.pyplot(fig)

# Pie chart with allocation amounts
st.subheader("Optimal allocation (Max Sharpe) - Pie chart & amounts")
alloc_amounts = w_sharpe * notional
alloc_df = pd.DataFrame({"Weight": w_sharpe, "Allocation_TL": alloc_amounts}, index=mu_ext.index)
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.pie(w_sharpe, labels=mu_ext.index, autopct="%1.1f%%", startangle=90)
ax2.set_title("Max-Sharpe Allocation Weights")
st.pyplot(fig2)
st.dataframe(alloc_df.style.format({"Weight":"{:.4f}", "Allocation_TL":"{:.2f}"}))

# Portfolio time series: build daily returns including RF and compute cumulative
daily_with_rf = daily_returns.copy()
daily_rf = (1 + rf_annual)**(1.0/TRADING_DAYS) - 1.0
daily_with_rf["RF"] = daily_rf
daily_with_rf = daily_with_rf[mu_ext.index]  # align
port_daily = daily_with_rf @ w_sharpe
port_cum = (1+port_daily).cumprod()
assets_cum = (1 + daily_with_rf).cumprod()

st.subheader("Asset cumulative (normalized) vs Max-Sharpe portfolio")
fig3, ax3 = plt.subplots(figsize=(10,5))
for col in assets_cum.columns:
    ax3.plot(assets_cum.index, assets_cum[col], label=col, alpha=0.65)
ax3.plot(port_cum.index, port_cum, label="Max-Sharpe Portfolio", color='black', lw=3)
ax3.set_ylabel("Cumulative index (normalized)")
ax3.legend()
st.pyplot(fig3)

st.success("Optimization complete. Change expected returns or RF on the left to recompute.")
