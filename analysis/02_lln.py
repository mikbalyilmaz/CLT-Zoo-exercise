"""Analysis 2: the Law of Large Numbers on the sliders of version 1, where it holds and where it fails.

Why: the README of version 1 promised a running-mean panel and normal curves that are "automatically
disabled when the population mean or variance is infinite", but the notebook had neither. Its Pareto slider
(alpha from 0.8 to 10) covers all three regimes of the LLN with one family: finite variance (alpha > 2),
finite mean but infinite variance (1 < alpha <= 2) and infinite mean (alpha <= 1).

Theory (xm = 1): for alpha > 2 the CLT gives |Xbar_n - mu| of order n^(-1/2); for 1 < alpha < 2 the sum is in
the domain of attraction of a stable law of index alpha, so |Xbar_n - mu| is of order n^(1/alpha - 1), slower
than n^(-1/2); for alpha < 1, S_n / n^(1/alpha) has a stable limit, so Xbar_n itself grows like n^(1/alpha - 1).

Design: Pareto(xm = 1) with alpha = 2.5 (the preset of version 1), 1.5 and 0.8 (the smallest slider value).
(a) One running mean of 10^5 draws each (seed 42). (b) Over M = 4,000 independent paths: the median of
|Xbar_n - mu| (alpha = 2.5 and 1.5) or of Xbar_n (alpha = 0.8) at 21 values of n from 10 to 10^5; the slope of
its logarithm against log n, fitted for n from 10^3 to 10^5 and separately in each decade of n, is compared
with the exponents above.
"""

import math

import numpy as np

from common import save_table
from cltzoo import ZOO, running_mean, running_mean_quantiles
from cltzoo.style import BLUE, INK, MUTED, end_label, fit_log_axis, num, save_figure, source_note, use_style

d = ZOO["Pareto"]
ALPHAS = (2.5, 1.5, 0.8)
NMAX = 100_000
CHECK = np.unique(np.round(np.logspace(1, 5, 21)).astype(int))
M = 4000

rows, curves, paths = [], {}, {}
for i, a in enumerate(ALPHAS):
    p = {"xm": 1.0, "alpha": a}
    m = d.moments(**p)
    mu = m["mean"]
    paths[a] = running_mean(d.sample(np.random.default_rng(42), NMAX, **p))
    finite = math.isfinite(mu)
    q = running_mean_quantiles(d, p, CHECK, M, np.random.default_rng(100 + i), q=(0.25, 0.5, 0.75),
                               center=mu if finite else 0.0, absolute=finite)
    curves[a] = q
    sel = CHECK >= 1000
    slope = float(np.polyfit(np.log(CHECK[sel]), np.log(q[1][sel]), 1)[0])
    theory = -0.5 if a > 2 else 1 / a - 1
    decades = {}
    for lo in (10, 100, 1000, 10_000):        # local slope in each decade of n
        s_ = (CHECK >= lo) & (CHECK <= 10 * lo)
        decades[f"slope_{lo}_{10 * lo}"] = float(np.polyfit(np.log(CHECK[s_]), np.log(q[1][s_]), 1)[0])
    row = {"alpha": a, "mean": mu, "variance": m["var"], "quantity": "median |Xbar_n - mu|" if finite else "median Xbar_n",
           "slope_fitted": slope, "slope_theory": theory, "median_at_1e5": float(q[1][-1]),
           "path_final_value": float(paths[a][-1]), **decades}
    if a > 2:   # CLT: median |Xbar_n - mu| -> 0.6745 sigma / sqrt(n)
        row["clt_prediction_at_1e5"] = 0.6744897501960817 * math.sqrt(m["var"] / NMAX)
    rows.append(row)
    print(row)
save_table(rows, "table4_lln_rates")
save_table([{"n": int(n), **{f"alpha_{a}_median": float(curves[a][1][j]) for a in ALPHAS}}
            for j, n in enumerate(CHECK)], "table5_lln_medians")

use_style()
COLORS = {2.5: INK, 1.5: BLUE, 0.8: MUTED}


def draw():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2.35), constrained_layout=True)
    ax = axes[0]
    n = np.arange(1, NMAX + 1)
    for a in ALPHAS:
        ax.plot(n, paths[a], color=COLORS[a], lw=1.0)
        mu = d.moments(xm=1.0, alpha=a)["mean"]
        if math.isfinite(mu):
            ax.axhline(mu, color=COLORS[a], lw=0.8, ls=(0, (1, 1.4)))
        end_label(ax, n[-1], paths[a][-1], rf"$\alpha$ = {a}", COLORS[a])
    fit_log_axis(ax, "x", [1, NMAX])
    fit_log_axis(ax, "y", np.concatenate([paths[a] for a in ALPHAS]))
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"Running mean $\bar X_n$")
    ax.set_title(r"(a) One path each; dotted: $\mu$")
    ax = axes[1]
    for a in ALPHAS:
        q = curves[a]
        ax.fill_between(CHECK, q[0], q[2], color=COLORS[a], alpha=0.15, lw=0)
        ax.plot(CHECK, q[1], color=COLORS[a], lw=1.2, marker="o", ms=2)
        ref = q[1][-1] * (CHECK / CHECK[-1]) ** (-0.5 if a > 2 else 1 / a - 1)
        ax.plot(CHECK, ref, color=COLORS[a], lw=0.7, ls=(0, (1, 1.4)))
        end_label(ax, CHECK[-1], q[1][-1], rf"$\alpha$ = {a}", COLORS[a])
    fit_log_axis(ax, "x", CHECK, every=2)
    allq = np.concatenate([np.concatenate(curves[a]) for a in ALPHAS])
    fit_log_axis(ax, "y", np.append(allq, 100 * allq.max()), every=2)    # room above the curves for the note
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$|\bar X_n - \mu|$, or $\bar X_n$ for $\alpha$ = 0.8")
    ax.set_title(r"(b) Median over 4,000 paths")
    lines = [f"{a}: {num(r['slope_fitted'], '{:.3f}')} ({num(r['slope_theory'], '{:.3f}')})"
             for a, r in ((r_["alpha"], r_) for r_ in rows)]
    source_note(ax, "Slope, fitted (theory)\n" + "\n".join(rf"$\alpha$ = {t}" for t in lines), y=0.98, va="top")
    return fig


save_figure("fig2_lln", draw, lambda: "Figure 2. The LLN for Pareto(1, alpha) at three points of the slider of version 1")
