"""Analysis 3: the Central Limit Theorem across the zoo, and the five discrete members exactly.

Why: the README of version 1 promised the sampling distribution of the mean and the standardized mean with a
normal curve for every member of the zoo. Part A draws them. For the five discrete members (Bernoulli,
Binomial, Poisson, Geometric, Negative Binomial) the law of the sum is known exactly, so the distance to the
normal distribution can be computed without simulation. A sum of integers is itself an integer, so its
distribution function jumps; this sets a floor of order 1/sqrt(N) that no amount of averaging removes, and
it is what the continuity correction is for. Part B measures both, exactly.

Design:
* Part A (Table 6, Figure 3): every preset, N = 30, M = 20,000 sample means (seed 42). The Kolmogorov distance
  of the simulated Z_30 to N(0, 1), the exact D_30 where the law of the sum has a closed form (9 of the 16),
  the noise level of a distance estimated from 20,000 values, and the skewness gamma / sqrt(30) of Z_30.
* Part B (Table 7, Figure 4): the five discrete presets, exact D_N (no correction) and E_N (continuity
  correction) for N from 1 to 10,000; the leading constants of the lattice Edgeworth expansion,
  c_D = phi(0) (1/(2 sigma) + |gamma|/6) and c_E = phi(0) |gamma|/6; the smallest N with D_N < 0.01 and with
  E_N < 0.01 (searched up to N = 3,000).
"""

import math

import numpy as np
from scipy import stats

from common import save_table
from cltzoo import NAMES, ZOO, corrected_error, kolmogorov_distance, lattice_constants, sample_means
from cltzoo.style import BAND, INK, RED, SERIES, auto_axis, fit_log_axis, may_touch, save_figure, use_style, value_axis

N, M = 30, 20_000
noise = float(stats.kstwo(M).mean())   # mean distance of M exact N(0,1) draws to N(0,1)

# Part A
rowsA, zs = [], {}
for i, name in enumerate(NAMES):
    d = ZOO[name]
    p = d.preset_params()
    m = d.moments(**p)
    means = sample_means(d, p, N, M, np.random.default_rng(42))
    z = math.sqrt(N) * (means - m["mean"]) / math.sqrt(m["var"])
    zs[name] = z
    row = {"distribution": name, "skewness_Z30": m["skew"] / math.sqrt(N),
           "excess_kurtosis_Z30": m["exkurt"] / N, "D30_simulated": float(stats.kstest(z, "norm").statistic),
           "noise_level": noise}
    if d.sum_law(N, **p) is not None:
        row["D30_exact"] = kolmogorov_distance(d, p, N)[0]
    rowsA.append(row)
    print(row)
save_table(rowsA, "table6_clt_n30")

# Part B
DISCRETE = ["Bernoulli", "Binomial", "Poisson", "Geometric", "NegBinomial"]
NS = np.unique(np.round(np.logspace(0, 4, 41)).astype(int))
curves, rowsB = {}, []
for name in DISCRETE:
    d = ZOO[name]
    p = d.preset_params()
    m = d.moments(**p)
    D = np.array([kolmogorov_distance(d, p, n)[0] for n in NS])
    E = np.array([corrected_error(d, p, n) for n in NS])
    curves[name] = (D, E)
    cD, cE = lattice_constants(d, p)
    scan = np.arange(1, 3001)
    Ds = np.array([kolmogorov_distance(d, p, n)[0] for n in scan])
    Es = np.array([corrected_error(d, p, n) for n in scan])

    def first_below(x, t=0.01):
        idx = np.where(x < t)[0]
        return (int(scan[idx[0]]), bool(np.all(x[idx[0]:] < t))) if idx.size else (None, False)
    nD, stayD = first_below(Ds)
    nE, stayE = first_below(Es)
    row = {"distribution": name, "sigma": math.sqrt(m["var"]), "skewness": m["skew"], "c_D": cD, "c_E": cE,
           "sqrtN_D_at_1e4": math.sqrt(1e4) * D[-1], "sqrtN_E_at_1e4": math.sqrt(1e4) * E[-1],
           "N_E_at_1e4": 1e4 * E[-1], "D_30": float(Ds[29]), "E_30": float(Es[29]),
           "first_N_D_below_0.01": nD, "stays_below_D": stayD, "first_N_E_below_0.01": nE, "stays_below_E": stayE}
    rowsB.append(row)
    print(row)
save_table(rowsB, "table7_lattice")
save_table([{"N": int(n), **{f"{nm}_D": float(curves[nm][0][j]) for nm in DISCRETE},
             **{f"{nm}_E": float(curves[nm][1][j]) for nm in DISCRETE}} for j, n in enumerate(NS)],
           "table8_lattice_curves")

use_style()


def draw3():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(4.75, 4.6), constrained_layout=True)
    xs = np.linspace(-4, 4, 400)
    for ax, name in zip(axes.flat, NAMES):
        d = ZOO[name]
        z = zs[name]
        m = d.moments(**d.preset_params())
        if d.kind == "discrete":     # Z_30 takes values spaced 1 / sqrt(30 var) apart: align the bins with them
            step = 1 / math.sqrt(N * m["var"])
            off = -N * m["mean"] * step
            w = step * max(1, math.ceil(8 / step / 40))
            edges = np.arange(off + (math.floor((-4 - off) / step) - 0.5) * step, 4 + w, w)
        else:
            edges = np.linspace(-4, 4, 41)
        inside = z[(z >= edges[0]) & (z <= edges[-1])]
        h, _, _ = ax.hist(inside, bins=edges, weights=np.full(inside.size, 1 / (z.size * (edges[1] - edges[0]))),
                          color=BAND, edgecolor=BAND, linewidth=0.3)
        ax.plot(xs, stats.norm.pdf(xs), color=INK, lw=1.0)
        r = next(r for r in rowsA if r["distribution"] == name)
        lab = f"$D$ = {r['D30_exact']:.3f}" if "D30_exact" in r else f"$D \\approx$ {r['D30_simulated']:.3f} (sim.)"
        ax.set_title(name, fontsize=7)
        ax.text(0.98, 0.97, lab, transform=ax.transAxes, ha="right", va="top", fontsize=6)
        value_axis(ax, "x", -4, 4, 4)
        auto_axis(ax, "y", [0.0, 1.35 * max(h.max(), stats.norm.pdf(0))], intervals=(2, 3))
        ax.tick_params(labelsize=6)
    return fig


save_figure("fig3_clt_n30", draw3,
            lambda: r"Figure 3. Standardized mean $Z_{30}$ for every preset (20,000 sample means) and $\mathcal{N}(0,1)$")

def draw4():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2.5), constrained_layout=True)
    style = dict(zip(DISCRETE, SERIES))
    values = np.concatenate([np.concatenate(curves[nm]) for nm in DISCRETE])
    for k, (ax, title) in enumerate(zip(axes, ("(a) Without correction: $D_N$",
                                                 "(b) With continuity correction: $E_N$"))):
        for name in DISCRETE:
            y = curves[name][k]
            r = next(r for r in rowsB if r["distribution"] == name)
            c = r["c_D"] if k == 0 else r["c_E"]
            col, ls, mk = style[name]
            ax.plot(NS, y, color=col, ls=ls, lw=1.0, marker=mk, ms=2, label=name)
            if c > 0:
                ax.plot(NS, c / np.sqrt(NS), color=col, lw=0.6, ls=(0, (1, 1.4)))
        ref = ax.axhline(0.01, color=RED, lw=0.9, ls=(0, (1, 1.4)))
        may_touch(ref)
        fit_log_axis(ax, "x", NS, every=2)
        fit_log_axis(ax, "y", values, every=2)
        ax.set_xlabel("$N$")
        ax.set_title(title)
    axes[0].set_ylabel("Exact error")
    axes[0].legend(loc="lower left", fontsize=6.5, handlelength=2.6)
    return fig


save_figure("fig4_lattice", draw4,
            lambda: "Figure 4. Discrete members: exact distance to the normal law, with and without the correction\n"
                    "(dotted: $c/\\sqrt{N}$ from the lattice Edgeworth expansion; red: 0.01)")
