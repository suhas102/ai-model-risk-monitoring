"""
P5 - AI risk and resilience assessment (demand-forecasting model)
------------------------------------------------------------------
System under assessment: an ML model that forecasts weekly demand per product to drive stock allocation.
Question: how can this model fail, how would we KNOW, and what is the fallback so the business keeps running?

Part A  Demonstrate the main failure mode (data/concept drift) and a monitoring + fallback design.
Part B  Produce a risk register and resilience controls for the AI system (CSV + used in the README).

DATA: synthetic weekly demand for a fictional product range. The "world change" at week 105
(price inflation + changed promo response + new product mix) is planted so drift can be detected.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import ks_2samp

OUT = "."
rng = np.random.default_rng(11)
T, DRIFT_AT = 156, 105
weeks = pd.date_range("2023-01-02", periods=T, freq="W-MON")
woy = weeks.isocalendar().week.to_numpy()
post = np.arange(T) >= DRIFT_AT

price_idx = 100 + rng.normal(0, 3, T) + np.where(post, 18, 0)                       # inflation shock
promo = (rng.random(T) < 0.2).astype(int)
comp_price_idx = 100 + rng.normal(0, 4, T) + np.where(post, 4, 0)
season = np.sin(2 * np.pi * woy / 52) + 0.8 * ((woy >= 47) & (woy <= 52))
new_mix = np.where(post, rng.beta(5, 2, T), rng.beta(2, 5, T))                      # share of newly listed lines
promo_lift = np.where(post, 2, 22)                                                  # concept drift: promo works less
demand = (200 + 35 * season - 1.6 * (price_idx - 100) + 0.7 * (comp_price_idx - price_idx)
          + promo_lift * promo + 70 * new_mix * post + rng.normal(0, 6, T))
df = pd.DataFrame(dict(week=weeks, price_idx=price_idx, promo=promo, comp_price_idx=comp_price_idx,
                       season=season, new_mix=new_mix, demand=demand))
df["lag1"] = df.demand.shift(1)
df = df.dropna().reset_index(drop=True)
feats = ["price_idx", "promo", "comp_price_idx", "season", "new_mix", "lag1"]
TRAIN_END = 60
train = df.iloc[:TRAIN_END]
model = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, random_state=0).fit(train[feats], train.demand)
df["pred"] = model.predict(df[feats])
# fallback = simple, transparent challenger model re-fitted weekly on recent data (adapts quickly, easy to explain)
from sklearn.linear_model import Ridge
fb = np.full(len(df), np.nan)
for i in range(TRAIN_END - 20, len(df)):                      # simple challenger re-fitted weekly on the last 26 weeks only
    lo = max(0, i - 26)
    fb[i] = Ridge(alpha=1.0).fit(df.iloc[lo:i][feats], df.iloc[lo:i].demand).predict(df.iloc[[i]][feats])[0]
df["seasonal_naive"] = pd.Series(fb).fillna(df.lag1)
df["abs_err_model"] = (df.pred - df.demand).abs()
df["abs_err_fallback"] = (df.seasonal_naive - df.demand).abs()
df["ape_model"] = df.abs_err_model / df.demand * 100

# ------------------------------------------------------------ monitoring metrics
def psi(ref, cur, bins=5):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-3
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-3
    return float(np.sum((c - r) * np.log(c / r)))

WINDOW = 20
mon_rows = []
for i in range(TRAIN_END + WINDOW, len(df)):
    cur = df.iloc[i - WINDOW:i]
    rec = dict(week=df.week[i], rolling_mape=cur.ape_model.mean(),
               rolling_mae_model=cur.abs_err_model.mean(), rolling_mae_fallback=cur.abs_err_fallback.mean())
    pv = {}
    for f in ["price_idx", "new_mix", "comp_price_idx"]:
        rec[f"psi_{f}"] = psi(train[f].to_numpy(), cur[f].to_numpy())
        pv[f] = ks_2samp(train[f].to_numpy(), cur[f].to_numpy()).pvalue
    rec["max_psi"] = max(rec[k] for k in rec if k.startswith("psi_"))
    rec["min_ks_p"] = min(pv.values()); rec["n_features_drifting"] = sum(p < 0.001 for p in pv.values())
    mon_rows.append(rec)
mon = pd.DataFrame(mon_rows)
# alert rules (documented thresholds - industry rule of thumb: PSI 0.1 watch, 0.25 act)
# NOTE: classic PSI thresholds (0.1 / 0.25) assume large samples. With a 20-week window the PSI of two
# IDENTICAL distributions is already ~0.25 by chance, so alerts use a two-sample KS test per input feature instead.
# AMBER = any input shifted at p<0.01 ; RED = 2+ inputs shifted at p<0.001, or lagging error metric breached.
PSI_WATCH, PSI_ACT, MAPE_ACT = 0.10, 0.25, 6.0
mon["status"] = np.where((mon.n_features_drifting >= 2) | (mon.rolling_mape > MAPE_ACT), "RED - switch to fallback + retrain",
                  np.where(mon.min_ks_p < 0.01, "AMBER - investigate", "green"))
mon.to_csv(f"{OUT}/monitoring_log.csv", index=False)

first_amber = mon[mon.status.str.startswith("AMBER") | mon.status.str.startswith("RED")].week.min()
first_red = mon[mon.status.str.startswith("RED")].week.min()
drift_week = df.week[df.index[df.week >= weeks[DRIFT_AT]][0]]

# ------------------------------------------------------------ policy simulation: unmonitored vs monitored-with-fallback
df["use_fallback"] = df.week.isin(mon[mon.status.str.startswith("RED")].week)
df["err_unmonitored"] = df.abs_err_model
df["err_monitored"] = np.where(df.use_fallback, df.abs_err_fallback, df.abs_err_model)
post_eval = df[df.week >= first_red] if pd.notna(first_red) else df.iloc[0:0]
mae_un = post_eval.err_unmonitored.mean(); mae_mon = post_eval.err_monitored.mean()
mae_pre = df[(df.index >= TRAIN_END) & (df.week < drift_week)].abs_err_model.mean()

# retraining on recent data (the remediation) - shown for the report
recent = df[(df.week >= first_red)].iloc[:12] if pd.notna(first_red) else df.iloc[:12]
retrain_idx = df.index[df.week == recent.week.iloc[-1]][0] if len(recent) else TRAIN_END
m2 = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, random_state=0).fit(
        df.iloc[max(0, retrain_idx - 40):retrain_idx][feats], df.iloc[max(0, retrain_idx - 40):retrain_idx].demand)
after = df.iloc[retrain_idx:]
mae_retrained = mean_absolute_error(after.demand, m2.predict(after[feats])) if len(after) else np.nan
mae_old_after = after.abs_err_model.mean() if len(after) else np.nan

# ------------------------------------------------------------ charts
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
ax[0, 0].plot(df.week, df.demand, color="k", lw=1.2, label="Actual demand")
ax[0, 0].plot(df.week, df.pred, color="#e07b00", lw=1.2, label="Model forecast")
ax[0, 0].axvline(weeks[TRAIN_END], color="grey", ls=":"); ax[0, 0].text(weeks[TRAIN_END], df.demand.max(), " train/test", fontsize=8)
ax[0, 0].axvline(drift_week, color="#c62828", ls="--"); ax[0, 0].text(drift_week, df.demand.min() + 5, " world changes", color="#c62828", fontsize=8)
ax[0, 0].legend(); ax[0, 0].set_title("Forecast vs actual")

ax[0, 1].semilogy(mon.week, mon.min_ks_p.clip(lower=1e-12), color="#6a1b9a", label="min KS p-value across inputs")
ax[0, 1].axhline(0.01, color="#f9a825", ls="--", label="AMBER p<0.01"); ax[0, 1].axhline(0.001, color="#c62828", ls="--", label="RED p<0.001")
ax[0, 1].axvline(drift_week, color="#c62828", ls="--", alpha=.5)
ax[0, 1].set_title("Input drift test (lower = more drift)"); ax[0, 1].legend(fontsize=8)

ax[1, 0].plot(mon.week, mon.rolling_mape, color="#1f3b73", label="Rolling 20wk MAPE (needs actuals - lagging)")
ax[1, 0].axhline(MAPE_ACT, color="#c62828", ls="--"); ax[1, 0].axvline(drift_week, color="#c62828", ls="--", alpha=.5)
ax[1, 0].set_title("Performance decay (% error)"); ax[1, 0].legend()

lab = ["Model alone\n(no monitoring)", "Monitored\n+ fallback", "Retrained GBM\n(same model type)"]
vals = [mae_un, mae_mon, mae_retrained]
ax[1, 1].bar(lab, vals, color=["#c62828", "#2e7d32", "#1f3b73"])
for i, v in enumerate(vals): ax[1, 1].text(i, v + .3, f"{v:.1f}", ha="center")
ax[1, 1].set_title("Mean abs error after first RED alert (units/week)")
fig.suptitle("AI risk & resilience: demand-forecast model drift, detection and fallback (synthetic)", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/ai_drift_monitoring.png", dpi=140); plt.close()

# ------------------------------------------------------------ Part B: risk register
reg = pd.DataFrame([
 # id, risk, cause, impact, L, I, control, KRI, owner
 ("AI-01", "Data / concept drift degrades forecast", "Price shocks, new ranges, changed promo response", "Over/under-stock, markdown cost, lost sales", 4, 4, "PSI + rolling MAPE monitoring; documented thresholds; scheduled retrain; fallback model", "Max PSI; 12-wk MAPE; days since retrain", "Head of Demand Planning"),
 ("AI-02", "Silent model failure (stale or null inputs)", "Upstream feed breaks or schema changes", "Plausible but wrong forecasts used for allocation", 3, 5, "Input validation and freshness checks; hard-stop on missing features; alert to on-call", "Feature null rate; feed latency; failed validation count", "Data Engineering Lead"),
 ("AI-03", "Third-party model / API outage or vendor change", "Hosted model or cloud region unavailable; model version retired", "Forecasts unavailable at peak; planners revert to manual", 2, 5, "Warm fallback (4-week-average) in own environment; vendor SLAs and exit plan; version pinning", "Fallback invocation count; vendor SLA breaches", "Tech Resilience / Procurement"),
 ("AI-04", "Over-reliance on model output (automation bias)", "Planners stop challenging forecasts", "Errors propagate unchecked into orders", 3, 4, "Human-in-the-loop for large changes; override logging; forecast-vs-plan challenge in S&OP", "% overrides; share of orders above tolerance auto-approved", "Supply Chain Director"),
 ("AI-05", "Bias / unfair outcome in allocation across stores or regions", "Training data under-represents smaller stores", "Persistent under-supply to some customers/colleagues", 3, 3, "Segment-level error reporting; fairness review at each release", "MAPE gap between best and worst segment", "Head of Demand Planning"),
 ("AI-06", "Weak change control on model releases", "Retrain deployed without testing or sign-off", "Regression in accuracy or behaviour", 3, 4, "Model registry; champion-challenger testing; approval gate; rollback plan", "Releases without sign-off; rollback time", "Tech Risk Partner"),
 ("AI-07", "Data privacy / security exposure", "Training data contains personal data; model or data exfiltrated", "Regulatory breach, reputational harm", 2, 5, "Data minimisation; access controls; DPIA; logging", "Open DPIA actions; access review exceptions", "DPO / CISO"),
 ("AI-08", "Unexplainable decisions block investigation", "Black-box model, no explanation logging", "Slow root cause after incident; weak audit trail", 3, 3, "Feature importance logging; reason codes for large moves; versioned data snapshots", "% forecasts with explanation stored", "Data Science Lead"),
], columns=["id", "risk", "cause", "impact", "likelihood_1_5", "impact_1_5", "key_controls", "key_risk_indicators", "suggested_owner"])
reg["inherent_score"] = reg.likelihood_1_5 * reg.impact_1_5
reg["rating"] = pd.cut(reg.inherent_score, [0, 7, 14, 25], labels=["Low", "Medium", "High"])
reg = reg.sort_values("inherent_score", ascending=False)
reg.to_csv(f"{OUT}/ai_risk_register.csv", index=False)

print(f"Drift planted at: {drift_week.date()}")
print(f"First AMBER/RED alert: {pd.Timestamp(first_amber).date()}  | first RED: {pd.Timestamp(first_red).date()}")
print(f"MAE before drift (test period): {mae_pre:.1f}")
print(f"MAE after first RED alert - model alone {mae_un:.1f} | monitored+fallback {mae_mon:.1f} | retrained {mae_retrained:.1f} (old model on same weeks {mae_old_after:.1f})")
print("\nRegister (top):\n", reg[["id", "risk", "inherent_score", "rating"]].to_string(index=False))
