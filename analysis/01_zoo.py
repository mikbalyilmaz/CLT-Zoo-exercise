"""Analysis 1: the zoo itself, and what the Distribution Explorer of version 1 computed.

Why: every later result rests on the sixteen distributions being what they claim to be, and on knowing which
of their moments exist. Version 1 showed each distribution with a sample and a distribution function obtained
by integrating the density numerically on the plotting window; the window, not the distribution, then decides
where that function reaches 1. Some slider positions also stopped the notebook with an error.

Design:
* Table 1: exact mean, variance, skewness, excess kurtosis and moment order r* (E|X|^r finite iff r < r*) at
  every preset of version 1.
* Table 2: version 1's own code (read from the legacy notebook) at every preset: the largest error of its
  distribution function against the exact one on its plotting window; and, at every slider position of every
  parameter (the other parameters at their presets), whether its density or probability function could be
  evaluated on its window; and the same check for the exact functions of version 2.
* Table 3: the share of slider positions at which the mean or the variance does not exist.
* Figure 1: all sixteen distributions, exact density or probability function with a sample of seed 42 (the
  preset sample size of version 1, or 6,000 draws where its preset showed no sample).
"""

import math

import numpy as np

from common import save_table
from cltzoo import NAMES, ZOO
from cltzoo.legacy import load_v1
from cltzoo.panels import support_grid
from cltzoo.style import BAND, BLUE, auto_axis, free_axis, save_figure, use_style

v1 = load_v1()

# Table 1: moments at the presets
rows = []
for name in NAMES:
    d = ZOO[name]
    p = d.preset_params()
    m = d.moments(**p)
    rows.append({"distribution": name, "kind": d.kind,
                 "parameters": "; ".join(f"{s.label}={p[s.key]:g}" for s in d.params),
                 **m, "moment_order": d.moment_order(**p)})
save_table(rows, "table1_zoo_moments")


# Table 2: version 1 at its presets and over its slider ranges
def v1_grid(name, params, n, sample_on, xpad, ngrid, seed=42):
    rvs = v1["DIST"][name]["rvs"]
    sample = v1["finite_only"](rvs(np.random.default_rng(seed), size=n, **params)) if sample_on else None
    return v1["theory_grid"](name, params, ngrid=ngrid, xpad=xpad, sample=sample)


def slider_values(spec):
    k = int(round((spec.hi - spec.lo) / spec.step))
    return [round(spec.lo + i * spec.step, 10) for i in range(k + 1)]


checks = []
for name in NAMES:
    d = ZOO[name]
    pre = v1["PRESETS"][name]
    n = max(int(pre["n"]), 100)                   # version 1 raised a preset size of 0 to the slider minimum
    p = v1["enforce_param_constraints"](name, d.preset_params())
    grid = v1_grid(name, p, n, pre["sample"], pre["xpad"], pre["grid"])
    law = d.frozen(**p)
    if d.kind == "continuous":
        y = v1["DIST"][name]["pdf"](grid, **p)
        cdf = v1["approximate_cdf_from_pdf"](grid, y)
    else:
        y = v1["DIST"][name]["pmf"](grid, **p)
        cdf = np.cumsum(y) / np.sum(y)
    err = np.abs(cdf - law.cdf(grid))
    row = {"distribution": name, "v1_cdf_max_error": float(err.max()),
           "at_x": float(grid[int(err.argmax())]), "window_from": float(grid[0]), "window_to": float(grid[-1]),
           "mass_beyond_window": float(law.sf(grid[-1]))}
    # every slider position of every parameter
    fails, total = [], 0
    for spec in d.params:
        for v in slider_values(spec):
            q = dict(d.preset_params(), **{spec.key: v})
            if name == "Uniform" and q["b"] <= q["a"]:
                continue                          # version 1 changed b silently here; counted separately
            q = v1["enforce_param_constraints"](name, q)
            total += 1
            try:
                g = v1_grid(name, q, n, pre["sample"], pre["xpad"], pre["grid"])
                f = v1["DIST"][name]["pdf" if d.kind == "continuous" else "pmf"](g, **q)
                if v1["finite_only"](f).size != g.size:
                    raise ValueError("non-finite density value on the plotting window")
            except (OverflowError, ValueError) as e:
                fails.append((spec.label, v, type(e).__name__))
    # the same positions in version 2: exact functions on the plotting window, finite everywhere
    v2_fail = 0
    for spec in d.params:
        for v in slider_values(spec):
            q = dict(d.preset_params(), **{spec.key: v})
            if name == "Uniform" and q["b"] <= q["a"]:
                continue
            try:
                law = d.frozen(**q)
                x = support_grid(d, d.check(q))
                f = law.pdf(x) if d.kind == "continuous" else law.pmf(x)
                inner = (x > law.support()[0]) & (x < law.support()[1])   # a density may be infinite at an end point
                ok = np.isfinite(law.cdf(x)).all() and np.isfinite(f[inner]).all()
                v2_fail += not ok
            except Exception:  # noqa: BLE001
                v2_fail += 1
    row["v2_errors"] = v2_fail
    row["slider_positions"] = total
    row["v1_errors"] = len(fails)
    row["v1_error_type"] = ";".join(sorted({f[2] for f in fails}))
    row["failing_values"] = "; ".join(
        f"{lab} {min(v for l2, v, _ in fails if l2 == lab):g} to {max(v for l2, v, _ in fails if l2 == lab):g} "
        f"({sum(1 for l2, _, _ in fails if l2 == lab)} positions)" for lab in dict.fromkeys(f[0] for f in fails))
    if name == "Uniform":   # positions where b <= a, which version 1 replaced silently by b = a + 1e-6
        row["silently_changed_positions"] = (sum(1 for a in slider_values(d.params[0]) if a >= 1.0)
                                             + sum(1 for b in slider_values(d.params[1]) if b <= 0.0))
    checks.append(row)
    print(name, row["v1_cdf_max_error"], row["v1_errors"], fails[:2])
save_table(checks, "table2_v1_checks")

# Table 3: where the mean and the variance stop existing on the sliders of version 1
exist = []
for name, key in (("Student-t", "df"), ("Pareto", "alpha"), ("F", "d2")):
    d = ZOO[name]
    spec = next(s for s in d.params if s.key == key)
    vals = slider_values(spec)
    no_mean = [v for v in vals if math.isnan(d.moments(**{**d.preset_params(), key: v})["mean"])]
    no_var = [v for v in vals if math.isnan(d.moments(**{**d.preset_params(), key: v})["var"])]
    exist.append({"distribution": name, "parameter": spec.label, "positions": len(vals),
                  "no_finite_mean": len(no_mean), "no_finite_variance": len(no_var),
                  "mean_condition": {"df": "df > 1", "alpha": "alpha > 1", "d2": "d2 > 2"}[key],
                  "variance_condition": {"df": "df > 2", "alpha": "alpha > 2", "d2": "d2 > 4"}[key]})
save_table(exist, "table3_moment_existence")

# Figure 1
use_style()


SYMBOL = {"mu": r"$\mu$", "sigma": r"$\sigma$", "theta": r"$\theta$", "alpha": r"$\alpha$", "beta": r"$\beta$",
          "lambda": r"$\lambda$", "xm": r"$x_m$", "d1": r"$d_1$", "d2": r"$d_2$", "df": "df", "scale": "scale"}


def parameters(d, p):
    """'μ = 0, σ = 1' for the panel of distribution d at parameters p."""
    out = []
    for sp in d.params:
        base = sp.label.split("(")[0]
        out.append(f"{SYMBOL.get(base, '$' + base + '$')} = {p[sp.key]:g}")
    return ", ".join(out)


def draw():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(4.75, 4.9), constrained_layout=True)
    for ax, name in zip(axes.flat, NAMES):
        d = ZOO[name]
        p = d.preset_params()
        law = d.frozen(**p)
        pre = d.preset
        n = int(pre["n"]) if pre["sample"] else 6000
        s = d.sample(np.random.default_rng(42), n, **p)
        if d.kind == "continuous":
            lo_s, hi_s = law.support()
            lo = lo_s if math.isfinite(lo_s) else float(law.ppf(0.005))
            hi = hi_s if math.isfinite(hi_s) else float(law.ppf(0.995))
            nb = 40
            inside = s[(s >= lo) & (s <= hi)]
            # bar areas are shares of all draws; the window starts and ends at the support where it is finite
            heights, _, _ = ax.hist(inside, bins=np.linspace(lo, hi, nb + 1),
                                    weights=np.full(inside.size, nb / (s.size * (hi - lo))),
                                    color=BAND, edgecolor=BAND, linewidth=0.3)
            x = np.linspace(lo, hi, 600)
            f = law.pdf(x)
            ax.plot(x, f, color=BLUE, lw=1.2)
            top = np.concatenate([heights, f[np.isfinite(f)]])
            auto_axis(ax, "x", [lo, hi], intervals=(2, 3))
        else:
            k, c = np.unique(s, return_counts=True)
            ax.bar(k, c / s.size, width=0.8, color=BAND, edgecolor=BAND, linewidth=0.3)
            x = support_grid(d, p, xpad=0.05)
            f = law.pmf(x)
            ax.plot(x, f, color=BLUE, lw=0, marker="o", ms=1.6)
            top = np.concatenate([c / s.size, f])
            if x.size <= 3:                       # Bernoulli: two values, shown as two categories
                ax.set_xticks(x)
                ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
                free_axis(ax, "x")
            else:
                auto_axis(ax, "x", [x.min(), x.max()], intervals=(2, 3))
        auto_axis(ax, "y", [0.0, 1.3 * np.nanmax(top)], intervals=(2, 3))   # head room for the parameters
        ax.set_title(name, fontsize=7)
        ax.text(0.98, 0.97, parameters(d, p), transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.tick_params(labelsize=6)
    return fig


save_figure("fig1_zoo", draw, lambda: "Figure 1. The zoo: exact density or probability function and a sample (seed 42)")
