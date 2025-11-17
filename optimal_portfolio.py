import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# Streamlit App Title
# ----------------------------------------------------
st.title("📈 Multi-Asset Portfolio Analysis & Optimization (TRY Based)")

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
assets = {
    "BIST30": "XU030.IS",
    "Gold": "IAU",
    "Silver": "SLV",
    "SP500": "SPY",
}
ticker_fx = "TRY=X"  # USD/TRY

period = "2y"

st.subheader("Downloading price data...")

# ----------------------------------------------------
# DOWNLOAD DATA
# ----------------------------------------------------
tickers = list(assets.values()) + [ticker_fx]
raw = yf.download(tickers, period=period, auto_adjust=True)["Close"]
raw = raw.dropna()

# Split data
fx = raw[ticker_fx]
data_usd = raw.drop(columns=[ticker_fx])

# Convert USD ETFs to TRY
data_try = data_usd.multiply(fx, axis=0)

# Clean column names
data_try.columns = list(assets.keys())

st.success("Data downloaded successfully.")

# ----------------------------------------------------
# LAST 1 YEAR DATA
# ----------------------------------------------------
prices_1y = data_try.tail(252)

returns_1y = (prices_1y.iloc[-1] / prices_1y.iloc[0] - 1) * 100
vol_annual = prices_1y.pct_change().std() * np.sqrt(252) * 100
cov_matrix = prices_1y.pct_change().cov() * 252

# ----------------------------------------------------
# Display tables
# ----------------------------------------------------
st.subheader("📊 One-Year Return (%)")
st.dataframe(returns_1y.round(2).to_frame("1Y Return (%)"))

st.subheader("📉 Annualized Volatility (%)")
st.dataframe(vol_annual.round(2).to_frame("Volatility (%)"))

st.subheader("📘 Covariance Matrix")
st.dataframe(cov_matrix.round(4))

# ----------------------------------------------------
# CORRELATION HEATMAP
# ----------------------------------------------------
st.subheader("🔗 Correlation Heatmap")

corr = prices_1y.pct_change().corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ----------------------------------------------------
# NORMALIZED PERFORMANCE PLOT
# ----------------------------------------------------
st.subheader("📈 1-Year Normalized Price Performance")

norm = prices_1y / prices_1y.iloc[0]
fig, ax = plt.subplots(figsize=(8, 4))
norm.plot(ax=ax)
ax.set_title("Normalized Performance (1 = first day)")
st.pyplot(fig)

# ----------------------------------------------------
# USER INPUT FOR EXPECTED RETURNS
# ----------------------------------------------------
st.subheader("✏️ Enter Your Expected Annual Returns (%)")

user_returns = {}
default_rf = 36.0  # Risk-free rate default

for asset in assets.keys():
    user_returns[asset] = st.number_input(
        f"Expected annual return for {asset} (%)",
        value=float(returns_1y[asset].round(2)),
        step=0.5
    )

rf_rate = st.number_input("Risk-Free Rate (%)", value=default_rf, step=0.5)

exp_returns = pd.Series(user_returns)

# ----------------------------------------------------
# MARKOWITZ OPTIMIZATION
# ----------------------------------------------------
st.subheader("🏦 Portfolio Optimization (Markowitz)")

returns_daily = prices_1y.pct_change().dropna()

cov = returns_daily.cov() * 252
mean_returns = exp_returns / 100

def portfolio_metrics(weights):
    weights = np.array(weights)
    port_return = np.sum(weights * mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe = (port_return - rf_rate/100) / port_vol
    return port_return, port_vol, sharpe

# Generate random portfolios
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    w = np.random.random(len(assets))
    w /= np.sum(w)
    weights_record.append(w)

    r, vol, s = portfolio_metrics(w)
    results[0, i] = r
    results[1, i] = vol
    results[2, i] = s

# Get max Sharpe portfolio
max_sharpe_idx = np.argmax(results[2])
max_sharpe_weights = weights_record[max_sharpe_idx]
max_sharpe_ret, max_sharpe_vol, _ = portfolio_metrics(max_sharpe_weights)

# ----------------------------------------------------
# Efficient Frontier Plot
# ----------------------------------------------------
st.subheader("📉 Efficient Frontier")

fig, ax = plt.subplots(figsize=(8, 4))
scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap="viridis")
ax.set_xlabel("Volatility")
ax.set_ylabel("Expected Return")
ax.scatter(max_sharpe_vol, max_sharpe_ret, color="red", s=80, label="Max Sharpe")
ax.legend()
st.pyplot(fig)

# ----------------------------------------------------
# PIE CHART OF OPTIMAL PORTFOLIO
# ----------------------------------------------------
st.subheader("🥧 Optimal Portfolio Allocation (Max Sharpe Ratio)")

fig, ax = plt.subplots()
ax.pie(max_sharpe_weights, labels=assets.keys(), autopct="%1.1f%%")
st.pyplot(fig)

weights_df = pd.DataFrame(
    max_sharpe_weights,
    index=assets.keys(),
    columns=["Weight"]
)
st.dataframe(weights_df)

# ----------------------------------------------------
# PLOT INDIVIDUAL ASSETS + MAX SHARPE PORTFOLIO VALUE
# ----------------------------------------------------
st.subheader("📈 Asset Prices vs. Max Sharpe Portfolio Performance")

portfolio_series = (prices_1y.pct_change().fillna(0) @ max_sharpe_weights).add(1).cumprod()

fig, ax = plt.subplots(figsize=(8, 4))
(norm * 100).plot(ax=ax, alpha=0.5)
(portfolio_series * 100).plot(ax=ax, linewidth=3, label="Max Sharpe Portfolio")
ax.legend()
st.pyplot(fig)

st.success("App ready! All results generated dynamically.")
