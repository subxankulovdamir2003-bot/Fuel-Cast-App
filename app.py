"""
FuelCast - "Should I Fill Up?"
OPIM 5671 Team Project 1, Group 9

Run locally:   streamlit run app.py
Model: ARIMA(3,1,0) + month effects on weekly U.S. regular gasoline (EIA),
       with prediction intervals scaled by recent forecast errors.
Same model as Sections 7-11 of the notebook.
"""
import io
import warnings
import urllib.request

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

EIA_URL = "https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls"
BACKUP_CSV = "gas_weekly.csv"     # snapshot shipped with the app in case EIA is unreachable
Z80 = norm.ppf(0.90)
VOL_WINDOW = 8

st.set_page_config(page_title="FuelCast", page_icon="⛽", layout="centered")


# ------------------------------------------------------------------ data + model
@st.cache_data(ttl=6 * 3600)
def load_gas():
    try:
        raw = urllib.request.urlopen(EIA_URL, timeout=15).read()
        df = pd.read_excel(io.BytesIO(raw), sheet_name="Data 3", header=2).iloc[:, :2]
        source = "EIA (live)"
    except Exception:
        df = pd.read_csv(BACKUP_CSV, parse_dates=["Date"])
        source = "saved copy of EIA data"
    df.columns = ["Date", "Gas"]
    s = df.dropna().set_index("Date")["Gas"].loc["2005-01-01":]
    return s, source


def month_dummies(dates):
    m = pd.DatetimeIndex(dates).month
    return pd.DataFrame({f"m{k}": (m == k).astype(float) for k in range(1, 13)}, index=dates)


@st.cache_data(ttl=6 * 3600)
def forecast(gas):
    chg = gas.diff().dropna()
    fit = ARIMA(chg, exog=month_dummies(chg.index), order=(3, 0, 0), trend="n").fit()
    last_date, last = gas.index[-1], gas.iloc[-1]
    future = pd.DatetimeIndex([last_date + pd.Timedelta(weeks=1), last_date + pd.Timedelta(weeks=2)])
    fc = fit.get_forecast(2, exog=month_dummies(future))
    point = last + np.cumsum(fc.predicted_mean.values)
    recent = fit.resid.iloc[-VOL_WINDOW:]
    scale = float(np.clip(np.sqrt(np.mean(recent ** 2)) / np.sqrt(fit.params["sigma2"]), 0.75, 3.0))
    se = fc.se_mean.values * scale
    p_up = float(norm.cdf((point[0] - last) / se[0]))
    return dict(last_date=last_date, last=last, future=future, point=point, se=se, scale=scale, p_up=p_up)


def recommendation(p_up, fuel_left, low=0.40, high=0.60):
    if p_up >= high:
        return "FILL UP NOW", "Prices are more likely to go up than down next week."
    if p_up <= low:
        if fuel_left <= 0.25:
            return "BUY A LITTLE NOW", "Prices look likely to dip, but your tank is low. Buy enough for this week and fill up next week."
        return "WAIT", "Prices are more likely to dip next week, and you have enough fuel to wait."
    return "PARTIALLY FILL", "The forecast doesn't point clearly either way. Splitting the purchase limits the downside."


COLORS = {"FILL UP NOW": "#2E7D4F", "PARTIALLY FILL": "#2B6CB0", "WAIT": "#1B2E4A", "BUY A LITTLE NOW": "#2B6CB0"}

# ------------------------------------------------------------------ page
gas, source = load_gas()
f = forecast(gas)

st.title("⛽ FuelCast")
st.caption("Should I fill up now or wait? A 1-2 week gas price forecast turned into a decision.")

with st.sidebar:
    st.header("Your situation")
    local_price = st.number_input("Price at your station ($/gal)", 1.0, 10.0, float(round(f["last"], 2)), 0.01,
                                  help="Defaults to the national average. Your local price moves with it, so we apply the forecast % change.")
    tank = st.slider("Tank size (gal)", 8, 30, 14)
    fuel_left = st.slider("Fuel left in tank", 0.0, 1.0, 0.30, 0.05, format="%.2f")
    miles = st.slider("Miles you drive per week", 0, 1500, 250, 25)
    mpg = st.slider("Your car's MPG", 10, 60, 28)
    st.divider()
    fleet_gal = st.number_input("Running a fleet? Gallons per week (optional)", 0, 1_000_000, 0, 100)

pct = f["point"] / f["last"] - 1
lo = (f["point"] - Z80 * f["se"]) / f["last"] - 1
hi = (f["point"] + Z80 * f["se"]) / f["last"] - 1
local_fc, local_lo, local_hi = local_price * (1 + pct), local_price * (1 + lo), local_price * (1 + hi)

label, why = recommendation(f["p_up"], fuel_left)
need = tank * (1 - fuel_left)
week_gal = miles / mpg if mpg else 0
buy_now = {"FILL UP NOW": need, "PARTIALLY FILL": need / 2, "WAIT": 0.0,
           "BUY A LITTLE NOW": min(need, week_gal)}[label]

st.markdown(
    f"<div style='background:{COLORS[label]};color:white;padding:18px 22px;border-radius:10px;margin:8px 0 4px 0'>"
    f"<div style='font-size:0.85rem;opacity:0.8'>Recommendation for this week</div>"
    f"<div style='font-size:2rem;font-weight:700;letter-spacing:0.5px'>{label}</div>"
    f"<div style='font-size:1rem;margin-top:4px'>{why}</div></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Chance price is higher next week", f"{f['p_up']:.0%}")
c2.metric("Your price next week (forecast)", f"${local_fc[0]:.2f}", f"{(local_fc[0] - local_price) * 100:+.1f}¢",
          delta_color="inverse")
c3.metric("Buy now", f"{buy_now:.1f} gal")

diff = (local_fc[0] - local_price) * need
if abs(diff) < 0.05:
    diff_txt = "Filling now or next week works out about the same for a full tank."
elif diff > 0:
    diff_txt = f"Waiting a week would likely cost about \\${diff:.2f} more for a full tank."
else:
    diff_txt = f"Waiting a week would likely save about \\${-diff:.2f} on a full tank."
st.caption(f"You need {need:.1f} gal to fill up. 80% range for next week at your station: "
           f"\\${local_lo[0]:.2f} to \\${local_hi[0]:.2f}. {diff_txt}")

# chart
hist = gas.iloc[-26:]
scale_local = local_price / f["last"]
hist_df = pd.DataFrame({"Date": hist.index, "Price": hist.values * scale_local, "Series": "Actual"})
fc_df = pd.DataFrame({"Date": [f["last_date"], *f["future"]], "Price": [local_price, *local_fc],
                      "Low": [local_price, *local_lo], "High": [local_price, *local_hi], "Series": "Forecast"})
band = alt.Chart(fc_df).mark_area(opacity=0.2, color="#2E7D4F").encode(x="Date:T", y=alt.Y("Low:Q", scale=alt.Scale(zero=False)), y2="High:Q")
lines = alt.Chart(pd.concat([hist_df, fc_df[["Date", "Price", "Series"]]])).mark_line(point=True).encode(
    x=alt.X("Date:T", title=None), y=alt.Y("Price:Q", title="$ per gallon (your station)", scale=alt.Scale(zero=False)),
    color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=["#1B2E4A", "#2E7D4F"]),
                    legend=alt.Legend(orient="bottom", title=None)))
st.altair_chart(band + lines, width="stretch")

st.subheader("2-week fuel budget")
b1, b2 = st.columns(2)
b1.metric("Expected cost, next 2 weeks", f"${week_gal * local_fc.sum():.2f}")
b2.metric("80% range", f"\\${week_gal * local_lo.sum():.0f} to \\${week_gal * local_hi.sum():.0f}")
st.caption(f"Based on {week_gal:.1f} gal/week ({miles} miles at {mpg} MPG).")

if fleet_gal:
    st.subheader("Fleet view")
    delta = (f["point"][0] - f["last"]) * fleet_gal
    k1, k2 = st.columns(2)
    k1.metric("Expected cost, next 2 weeks", f"${fleet_gal * local_fc.sum():,.0f}")
    k2.metric("Waiting a week to buy this week's volume",
              f"{'saves' if delta < 0 else 'costs'} \\${abs(delta):,.0f}")
    st.caption(f"80% range for the 2-week cost: \\${fleet_gal * local_lo.sum():,.0f} to \\${fleet_gal * local_hi.sum():,.0f}.")

with st.expander("How this works"):
    st.markdown(f"""
- **Data:** weekly U.S. average regular gasoline price from the EIA, 2005 onward ({source}, latest week {f['last_date']:%b %d, %Y}: ${f['last']:.3f}).
- **Model:** ARIMA(3,1,0) with month effects (spring run-up, fall decline). In our test (Jan 2024 to Sep 2026) it beat the "same as last week" forecast by about 10% at 1 week and 15% at 2 weeks.
- **Uncertainty:** the range is widened when recent forecast errors have been large. Right now it is scaled by **{f['scale']:.2f}x**.
- **Decision rule:** fill up if the chance of a price increase is 60% or more, wait if it is 40% or less, otherwise split the purchase. Never "wait" on a tank that is a quarter full or less.
- **Backtest:** following the rule saved about 0.8¢/gal versus always filling right away, a bit over 40% of what perfect foresight would save. Small for one car, meaningful for a fleet.
- **Limits:** national average, weekly updates, and no model sees sudden shocks coming.
""")

st.caption("OPIM 5671 Group 9: Damir Sukhankulov, Jonathan Trippett, Ates Boyaci, Luke Bauer, Priyanka Ravikumar")
