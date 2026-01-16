"""
Reproduces Appendix X: Volatility-Managed Portfolios

Input:
  - data/predictions.parquet
    Required columns:
      date, permno, ret, y,
      pred_<arch>_<L>_<optimizer>

Output:
  - paper_figs_appendix/*.pdf
  - paper_figs_appendix/*.tex
  - paper_figs_appendix/*.csv
"""

import numpy as np
import pyarrow as pa
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt

import os
from matplotlib.lines import Line2D

OUTDIR = "paper_figs_appendix"
os.makedirs(OUTDIR, exist_ok=True)

## TODO: change source
print(os.listdir())
df_pl = pl.read_parquet("sp500_vol_forecasts_2000_2024.parquet")
df = df_pl.to_pandas()
df["date"] = pd.to_datetime(df["date"])

MODELS = [
    "lasso_0.05", "ols",
    "pred_cnn_100_muon", "pred_cnn_100_adam", "pred_cnn_100_sgd",
    "pred_lstm_100_muon", "pred_lstm_100_adam", "pred_lstm_100_sgd",
    "pred_transformer_100_muon", "pred_transformer_100_adam", "pred_transformer_100_sgd",
    "pred_mlp_100_muon", "pred_mlp_100_adam", "pred_mlp_100_sgd"
]

# Rolling windows
ROLL_TURN = 126   # ~6 months
ROLL_TURN_LONG = 252  # ~1 year
ANN = 252

# ----------------------------
# Helpers (consistent with your notebook)
# ----------------------------
def assign_quintile(x: pd.Series) -> pd.Series:
    # Q1 = lowest sig_hat, Q5 = highest sig_hat
    return pd.qcut(x, 5, labels=False, duplicates="drop") + 1

def sharpe_ann(r: pd.Series, ann: int = ANN) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    mu = r.mean() * ann
    sd = r.std() * np.sqrt(ann)
    return mu / sd if sd > 0 else np.nan

def ann_ret(r: pd.Series, ann: int = ANN) -> float:
    r = r.dropna()
    return (r.mean() * ann) if len(r) else np.nan

def ann_vol(r: pd.Series, ann: int = ANN) -> float:
    r = r.dropna()
    return (r.std() * np.sqrt(ann)) if len(r) else np.nan

def max_drawdown(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    return dd.min()

def turnover_equal_weight_quintile(tmp: pd.DataFrame, q_target: int = 1):
    """
    Exact turnover for equal-weight portfolio holding all names in quintile q_target each day.
    turnover_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|
    where w_{i,t} = 1/n_t if i in C_t else 0.
    Computes turnover from overlaps without forming full weights matrix.

    Returns: (avg_turnover, turnover_series)
    """
    g = (
        tmp.loc[tmp["q"] == q_target]
        .groupby("date")["permno"]
        .apply(lambda s: set(s.dropna().astype(int)))
        .sort_index()
    )
    dates = g.index
    if len(dates) < 2:
        return np.nan, pd.Series(dtype=float)

    to, to_dates = [], []
    prev_set = g.iloc[0]
    prev_n = len(prev_set)

    for d, cur_set in g.iloc[1:].items():
        cur_n = len(cur_set)
        if prev_n == 0 or cur_n == 0:
            prev_set, prev_n = cur_set, cur_n
            continue
        k = len(prev_set.intersection(cur_set))
        sum_abs = k * abs(1/cur_n - 1/prev_n) + (cur_n - k) * (1/cur_n) + (prev_n - k) * (1/prev_n)
        turnover_t = 0.5 * sum_abs
        to.append(turnover_t)
        to_dates.append(d)
        prev_set, prev_n = cur_set, cur_n

    to_series = pd.Series(to, index=pd.to_datetime(to_dates), name=f"turnover_Q{q_target}")
    return to_series.mean(), to_series

def model_family(m: str) -> str:
    if m in ["ols", "lasso_0.05"]:
        return "Linear"
    if "cnn" in m:
        return "CNN"
    if "lstm" in m:
        return "LSTM"
    if "mlp" in m:
        return "MLP"
    if 'transformer' in m:
        return 'TRANSFORMER'
    if m == "ground_truth":
        return "Ground truth"
    return "Other"

def model_optimizer(m: str) -> str:
    if "adam" in m:
        return "Adam"
    if "sgd" in m:
        return "SGD"
    if "muon" in m:
        return "Muon"
    return "NA"

def pretty_label(m: str) -> str:
    # Shorter legend labels
    m2 = m.replace("pred_", "").replace("_100_", " ").replace("_", " ")
    m2 = m2.replace("lasso 0.05", "LASSO").replace("ols", "OLS").replace("ground truth", "Truth")
    return m2


# ----------------------------
# Build quintile returns + turnover time series per model
# ----------------------------
qret_ts = {}         # model -> DataFrame (date x quintiles)
turnover_ts = {}     # model -> dict("Q1".."Q5") -> Series turnover_t
summ_rows = []       # for tables/frontier

for m in MODELS:
    if m == "ground_truth":
        cols = ["date", "permno", "ret", "y"]
        tmp = df[cols].dropna().copy()
        tmp["sig_hat"] = np.sqrt(np.exp(tmp["y"]))  # uses realized y as "prediction"
    else:
        if m not in df.columns:
            continue
        cols = ["date", "permno", "ret", "y", m]
        tmp = df[cols].dropna().copy()
        tmp["sig_hat"] = np.sqrt(np.exp(tmp[m]))  # matches your notebook logic

    # Quintiles each day
    tmp["q"] = tmp.groupby("date")["sig_hat"].transform(assign_quintile)

    # Equal-weight quintile returns
    qret_m = (
        tmp.groupby(["date", "q"])["ret"].mean()
        .unstack("q").sort_index()
    )
    qret_m.columns = [f"Q{int(c)}" for c in qret_m.columns]
    qret_ts[m] = qret_m

    # Turnover series for each quintile
    turnover_ts[m] = {}
    for q in [1, 2, 3, 4, 5]:
        avg_to, to_series = turnover_equal_weight_quintile(tmp, q_target=q)
        turnover_ts[m][f"Q{q}"] = to_series

    # Summary rows for Q1 and Q5
    for q in [1, 5]:
        qname = f"Q{q}"
        r = qret_m.get(qname)
        row = {
            "model": m,
            "family": model_family(m),
            "optimizer": model_optimizer(m),
            "quintile": qname,
            "ann_ret": ann_ret(r),
            "ann_vol": ann_vol(r),
            "sharpe": sharpe_ann(r),
            "max_dd": max_drawdown(r),
            "turnover": turnover_ts[m][qname].mean() if len(turnover_ts[m][qname]) else np.nan
        }
        summ_rows.append(row)

summary = pd.DataFrame(summ_rows).dropna(subset=["turnover", "sharpe"])
summary = summary.sort_values(["quintile", "turnover"])

# ----------------------------
# (A) PAPER TABLE: summary stats (Q1 and Q5)
# ----------------------------
# Keep the table compact: show all models OR a selected subset.
# Here: all available models, but you can filter below.
summary_table = summary[["model", "family", "optimizer", "quintile", "ann_ret", "ann_vol", "sharpe", "max_dd", "turnover"]].copy()

# Optional: filter specific models
# keep = ["ols","lasso_0.05","pred_cnn_100_adam","pred_cnn_100_sgd","pred_cnn_100_muon",
#         "pred_lstm_100_adam","pred_lstm_100_sgd","pred_lstm_100_muon",
#         "pred_mlp_100_adam","pred_mlp_100_sgd","pred_mlp_100_muon"]
# summary_table = summary_table[summary_table["model"].isin(keep)]

summary_csv = os.path.join(OUTDIR, "table_appendix_vol_sorted_summary.csv")
summary_table.to_csv(summary_csv, index=False)

# Also export a LaTeX table (booktabs-friendly)
def df_to_latex(df_, path):
    df2 = df_.copy()
    for c in ["ann_ret","ann_vol","sharpe","max_dd","turnover"]:
        df2[c] = df2[c].astype(float).round(3)
    latex = df2.to_latex(index=False, escape=True)
    with open(path, "w") as f:
        f.write(latex)

summary_tex = os.path.join(OUTDIR, "table_appendix_vol_sorted_summary.tex")
df_to_latex(summary_table, summary_tex)

print("Saved:", summary_csv)
print("Saved:", summary_tex)

# ----------------------------
# (B) FIGURE: Sharpe–Turnover frontier (Q1 and Q5)
# ----------------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "axes.linewidth": 1.0,
})

family_colors = {
    "Linear": "tab:blue",
    "CNN": "tab:orange",
    "LSTM": "tab:green",
    "TRANSFORMER": "tab:red",
    "MLP": "tab:purple",
    "Ground truth": "tab:brown"
}

optimizer_shapes = {"Adam": "s", "SGD": "o", "Muon": "^","NA": "X"}

def plot_frontier_ax(ax, frontier_df, title: str):
    # Scatter
    for _, row in frontier_df.iterrows():
        ax.scatter(
            row["turnover"],
            row["sharpe"],
            color=family_colors.get(row["family"], "tab:gray"),
            marker=optimizer_shapes.get(row["optimizer"], "o"),
            s=110,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
            zorder=3
        )

    ax.set_title(title)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

# Prepare data
frontier_q1 = summary[summary["quintile"] == "Q1"].copy()
frontier_q5 = summary[summary["quintile"] == "Q5"].copy()

# Two panels
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

plot_frontier_ax(axes[0], frontier_q1, "Q1: Low-volatility portfolio")
plot_frontier_ax(axes[1], frontier_q5, "Q5: High-volatility portfolio")

# Shared labels
fig.supxlabel("Average daily turnover", y=-0.02)
fig.supylabel("Annualized Sharpe", x=-0.02)

# One shared legend (family + optimizer)
# Only include families/optimizers that actually appear, to avoid clutter.
families_present = list(dict.fromkeys(summary["family"].dropna().tolist()))
optimizers_present = list(dict.fromkeys(summary["optimizer"].dropna().tolist()))

family_legend = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=family_colors.get(f, "tab:gray"),
           markeredgecolor="white", markeredgewidth=0.9,
           label=f, markersize=10)
    for f in family_colors.keys() if f in families_present
]

optimizer_legend = [
    Line2D([0], [0], marker=optimizer_shapes.get(o, "o"), color="black",
           linestyle="None", label=o, markersize=10)
    for o in optimizer_shapes.keys() if o in optimizers_present
]

handles = family_legend + optimizer_legend
labels = [h.get_label() for h in handles]

# Put legend below panels
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.18)
)

# Save
outpath = os.path.join(OUTDIR, "fig_frontier_Q1Q5_two_panel.pdf")
fig.savefig(outpath, bbox_inches="tight")
plt.close(fig)
print("Saved:", outpath)

# ----------------------------
# (C) FIGURE: Rolling turnover by quintile, split by optimizer
# ----------------------------
crises = {
    "GFC": ("2007-08-01", "2009-06-30"),
    "Euro Debt": ("2011-05-01", "2012-07-31"),
    "COVID": ("2020-02-15", "2020-12-31"),
    "Inflation Shock": ("2022-01-01", "2022-10-31"),
}

def plot_turnover_panels(optimizer: str, models: list, fname: str):
    qs = ["Q1", "Q3", "Q5"]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12.8, 3.6), sharey=True, constrained_layout=True)

    for ax, q in zip(axes, qs):
        for m in models:
            if m not in turnover_ts:
                continue
            s = turnover_ts[m][q].rolling(ROLL_TURN).mean()
            s.plot(ax=ax, label=pretty_label(m))

        for start, end in crises.values():
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="grey", alpha=0.15, zorder=0)

        ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
        ax.set_title(q)
        ax.set_xlabel("Date")

    axes[0].set_ylabel("Turnover (rolling 6m)")
    axes[0].legend(title="Model", loc="upper left", frameon=False)

    fig.suptitle(f"Rolling 6-Month Turnover by Quintile ({optimizer} Optimizer)", y=1.02)
    outpath = os.path.join(OUTDIR, fname)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath)

plot_turnover_panels("Adam",
    ["pred_cnn_100_adam", "pred_lstm_100_adam", "pred_mlp_100_adam", "pred_transformer_100_adam", "lasso_0.05", "ols"],
    "fig_turnover_panels_adam.pdf"
)
plot_turnover_panels("SGD",
    ["pred_cnn_100_sgd", "pred_lstm_100_sgd", "pred_mlp_100_sgd", "pred_transformer_100_sgd", "lasso_0.05", "ols"],
    "fig_turnover_panels_sgd.pdf"
)
plot_turnover_panels("Muon",
    ["pred_cnn_100_muon", "pred_lstm_100_muon", "pred_mlp_100_muon", "pred_transformer_100_muon", "lasso_0.05", "ols"],
    "fig_turnover_panels_muon.pdf"
)

# Optional: average turnover by optimizer (Q1), rolling 1y (your notebook has this) :contentReference[oaicite:2]{index=2}
optimizer_groups = {
    "Adam": ["pred_cnn_100_adam","pred_lstm_100_adam","pred_mlp_100_adam","pred_transformer_100_adam"],
    "SGD": ["pred_cnn_100_sgd","pred_lstm_100_sgd","pred_mlp_100_sgd","pred_transformer_100_sgd"],
    "Muon": ["pred_cnn_100_muon","pred_lstm_100_muon","pred_mlp_100_muon","pred_transformer_100_muon"],
    "Linear": ["ols","lasso_0.05"]
}

opt_turn = {}
Q = "Q1"
for opt, ms in optimizer_groups.items():
    ts = pd.concat([turnover_ts[m][Q] for m in ms if m in turnover_ts], axis=1)
    opt_turn[opt] = ts.mean(axis=1)

fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
for opt, series in opt_turn.items():
    series.rolling(ROLL_TURN_LONG).mean().plot(ax=ax, label=opt)
for start, end in crises.values():
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="grey", alpha=0.15, zorder=0)
ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
ax.set_title("Rolling 1-Year Turnover of Volatility Rankings (Q1)")
ax.set_ylabel("Turnover")
ax.set_xlabel("Date")
ax.legend(title="Optimizer", frameon=False)
outpath = os.path.join(OUTDIR, "fig_turnover_optimizer_avg_Q1.pdf")
fig.savefig(outpath, bbox_inches="tight")
plt.close(fig)
print("Saved:", outpath)



