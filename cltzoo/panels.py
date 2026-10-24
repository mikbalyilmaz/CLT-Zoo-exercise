"""The two interactive views of the notebook, as plain functions that return a matplotlib figure.

explorer_figure : the Distribution Explorer of version 1 (PDF or PMF, a sample, the distribution function),
                  with the exact distribution function instead of a numerical integral.
clt_figure      : the three panels the README of version 1 described but the notebook did not contain:
                  a running mean (LLN), the sampling distribution of the mean with its normal approximation,
                  and the standardized mean against N(0, 1). Normal curves are drawn only when the variance
                  is finite, and the population mean only when it is finite.

Author: Muhammed Ikbal Yilmaz
"""

import math

import numpy as np

from .clt import running_mean, sample_means
from .distributions import ZOO
from .style import BAND, BLUE, INK, RED, auto_axis, fit_log_axis, use_style, value_axis


def _note(ax, text, right=False):
    ax.text(0.98 if right else 0.02, 0.97, text, transform=ax.transAxes, va="top", ha="right" if right else "left",
            fontsize=7, color=INK)


def _bins(lo, hi, step=None, target=60):
    """Histogram edges on [lo, hi]. For lattice values (multiples of `step`) the edges lie half-way between
    lattice points and the width is a whole number of steps, so that every bar holds the same number of
    possible values."""
    if step is None:
        return np.linspace(lo, hi, target + 1)
    w = step * max(1, math.ceil((hi - lo) / step / target))
    start = (math.floor(lo / step) - 0.5) * step
    return np.arange(start, hi + w, w)


def support_grid(dist, params, xpad=0.3, ngrid=900):
    """x values for drawing: exact quantiles 0.001 to 0.999 widened by xpad, clipped to the support.
    For discrete distributions: every integer between the 1e-4 and 1 - 1e-4 quantiles."""
    law = dist.frozen(**params)
    lo_s, hi_s = law.support()
    if dist.kind == "discrete":
        a, b = int(law.ppf(1e-4)), int(law.ppf(1 - 1e-4))
        a = max(a - max(1, int(xpad * (b - a))), int(lo_s) if math.isfinite(lo_s) else a)
        b = b + max(1, int(xpad * (b - a)))
        if math.isfinite(hi_s):
            b = min(b, int(hi_s))
        return np.arange(a, b + 1)
    a, b = float(law.ppf(1e-3)), float(law.ppf(1 - 1e-3))
    pad = xpad * (b - a)
    a, b = a - pad, b + pad
    if math.isfinite(lo_s):
        a = max(a, lo_s - 0.02 * (b - lo_s))
    if math.isfinite(hi_s):
        b = min(b, hi_s + 0.02 * (hi_s - a))
    x = np.linspace(a, b, ngrid)
    # include the support end points so that jumps of the density are drawn where they are
    pts = [v for v in (lo_s, hi_s) if math.isfinite(v) and a < v < b]
    return np.unique(np.concatenate([x, pts]))


def explorer_figure(name, params, n_sample=6000, seed=42, show_sample=True, overlay=True, xpad=0.3, ngrid=900):
    use_style()
    import matplotlib.pyplot as plt
    dist = ZOO[name]
    p = dist.check(params)
    law = dist.frozen(**p)
    x = support_grid(dist, p, xpad, ngrid)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)
    if dist.kind == "continuous":
        y = law.pdf(x)
        y = np.where(np.isfinite(y), y, np.nan)   # e.g. Beta with beta < 1 at x = 1: infinite, not drawn
        axes[0].plot(x, y, color=BLUE, lw=1.4)
        axes[0].set_ylabel("Density")
        axes[2].plot(x, law.cdf(x), color=BLUE, lw=1.4)
    else:
        y = law.pmf(x)
        axes[0].bar(x, y, width=0.8, color=BLUE, linewidth=0)
        axes[0].set_ylabel("Probability")
        axes[2].step(x, law.cdf(x), where="post", color=BLUE, lw=1.4)
    axes[0].set_title("Exact PDF" if dist.kind == "continuous" else "Exact PMF")
    axes[2].set_title("Exact distribution function")
    ax = axes[1]
    ax.set_title("Sample")
    if show_sample:
        rng = np.random.default_rng(seed)
        s = dist.sample(rng, int(n_sample), **p)
        if dist.kind == "continuous":
            lo, hi = x[0], x[-1]
            inside = s[(s >= lo) & (s <= hi)]
            nb = max(25, int(math.sqrt(s.size)))
            # density relative to all draws, so that the area of the bars is the share of draws in the window
            ax.hist(inside, bins=np.linspace(lo, hi, nb + 1), weights=np.full(inside.size, nb / (s.size * (hi - lo))),
                    color=BAND, edgecolor=BAND, linewidth=0.5)
            if overlay:
                ax.plot(x, y, color=BLUE, lw=1.2)
            out = s.size - inside.size
            _note(ax, f"seed {seed}, n = {s.size}" + (f"\n{out} draws outside the window" if out else ""), right=True)
        else:
            vals, counts = np.unique(s.astype(int), return_counts=True)
            ax.bar(vals, counts / s.size, width=0.8, color=BAND, edgecolor=BAND, linewidth=0.5)
            if overlay:
                ax.plot(x, y, color=BLUE, lw=0, marker="o", ms=2)
            _note(ax, f"seed {seed}, n = {s.size}", right=True)
    else:
        _note(ax, "Sample off")
    edge = [x[0] - 0.5, x[-1] + 0.5] if dist.kind == "discrete" else [x[0], x[-1]]
    exact = np.append(y[np.isfinite(y)], 0.0)
    bars = np.array([b.get_height() for b in axes[1].patches])
    for a in axes:
        a.set_xlabel("$x$" if dist.kind == "continuous" else "$k$")
        auto_axis(a, "x", edge)
    auto_axis(axes[0], "y", exact)
    auto_axis(axes[1], "y", np.concatenate([exact, bars]))
    value_axis(axes[2], "y", 0, 1, 0.25)
    fig.suptitle(f"{name}: " + ", ".join(f"{s.label} = {p[s.key]:g}" for s in dist.params), x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold")
    return fig


def clt_figure(name, params, N=30, M=5000, path_length=5000, seed=42):
    """Running mean of one path, the M sample means of size N, and their standardized version."""
    use_style()
    import matplotlib.pyplot as plt
    dist = ZOO[name]
    p = dist.check(params)
    m = dist.moments(**p)
    mu, var = m["mean"], m["var"]
    rng = np.random.default_rng(seed)
    path = running_mean(dist.sample(rng, int(path_length), **p))
    means = sample_means(dist, p, int(N), int(M), rng)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)

    ax = axes[0]
    n = np.arange(1, path.size + 1)
    ax.plot(n, path, color=BLUE, lw=1.0)
    fit_log_axis(ax, "x", [1, path.size])
    ax.set_xlabel("$n$")
    ax.set_title(r"Running mean $\bar X_n$ (LLN)")
    auto_axis(ax, "y", np.append(path, mu) if math.isfinite(mu) else path)
    if math.isfinite(mu):
        ax.axhline(mu, color=RED, lw=1.2, ls=(0, (1, 1.4)))
        _note(ax, rf"$\mu$ = {mu:.4g} (red dotted line)", right=True)
    else:
        _note(ax, "The mean is not finite:\nthe running mean has no limit", right=True)

    ax = axes[1]
    ax.set_title(rf"Sample means, $N$ = {int(N)}, $M$ = {int(M)}")
    finite_var = math.isfinite(var)
    lo, hi = np.quantile(means, [0.001, 0.999] if finite_var else [0.01, 0.99])
    lattice = 1.0 / N if dist.kind == "discrete" else None     # integer-valued X: Xbar_N lives on multiples of 1/N
    edges = _bins(lo, hi, lattice)
    inside = means[(means >= edges[0]) & (means <= edges[-1])]
    h1, _, _ = ax.hist(inside, bins=edges, weights=np.full(inside.size, 1.0 / (means.size * (edges[1] - edges[0]))),
                       color=BAND, edgecolor=BAND, linewidth=0.5)
    ax.set_xlabel(r"$\bar X_N$")
    auto_axis(ax, "x", [edges[0], edges[-1]])
    top1 = list(h1) + [0.0]
    if math.isfinite(var):
        xs = np.linspace(lo, hi, 400)
        sd = math.sqrt(var / N)
        dens = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        ax.plot(xs, dens, color=INK, lw=1.3)
        top1 += list(dens)
        _note(ax, r"$\mathcal{N}(\mu, \sigma^2/N)$ in black")
    else:
        _note(ax, "The variance is not finite:\nno normal approximation.\nWindow: 1% to 99% quantiles", right=True)
    auto_axis(ax, "y", top1)

    ax = axes[2]
    ax.set_title(r"Standardized mean $Z_N$")
    if math.isfinite(var):
        z = math.sqrt(N) * (means - mu) / math.sqrt(var)
        zstep = None if lattice is None else 1.0 / math.sqrt(N * var)       # spacing of Z_N for integer X
        if zstep is None:
            edges = _bins(-4.0, 4.0)
        else:
            off = (-N * mu) / math.sqrt(N * var)                           # Z_N = k zstep + off
            k0 = math.floor((-4.0 - off) / zstep)
            w = zstep * max(1, math.ceil(8.0 / zstep / 60))
            edges = np.arange(off + (k0 - 0.5) * zstep, 4.0 + w, w)
        inside = z[(z >= edges[0]) & (z <= edges[-1])]
        h2, _, _ = ax.hist(inside, bins=edges, weights=np.full(inside.size, 1.0 / (z.size * (edges[1] - edges[0]))),
                           color=BAND, edgecolor=BAND, linewidth=0.5)
        value_axis(ax, "x", -4, 4, 2)
        xs = np.linspace(-4, 4, 400)
        ax.plot(xs, np.exp(-xs**2 / 2) / math.sqrt(2 * math.pi), color=INK, lw=1.3)
        auto_axis(ax, "y", list(h2) + [0.0, 1 / math.sqrt(2 * math.pi)])
        g = m["skew"]
        _note(ax, r"$\mathcal{N}(0, 1)$ in black" + (f"\nskewness of $Z_N$ = {g / math.sqrt(N):.3f}"
                                                      if math.isfinite(g) else ""))
        ax.set_xlabel("$z$")
    else:
        ax.axis("off")
        _note(ax, "Not defined: the variance is not finite,\nso the mean cannot be standardized.")
    fig.suptitle(f"{name}: " + ", ".join(f"{s.label} = {p[s.key]:g}" for s in dist.params), x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold")
    return fig
