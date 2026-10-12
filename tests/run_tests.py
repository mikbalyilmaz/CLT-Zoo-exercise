"""Unit tests. Run from the repository root:  python tests/run_tests.py

Each result is compared with an independent computation: the code of version 1 itself (read from the legacy
notebook), numerical integration and summation of the densities, N-fold convolution, goodness-of-fit tests of
the samplers, and brute-force evaluation of the distances on a fine grid.
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cltzoo import NAMES, ZOO, corrected_error, kolmogorov_distance, running_mean_quantiles, sample_means  # noqa: E402
from cltzoo.clt import Phi, lattice_constants  # noqa: E402
from cltzoo.legacy import load_v1  # noqa: E402
from cltzoo.panels import clt_figure, explorer_figure, support_grid  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
failed = 0


def check(name, ok):
    global failed
    print(f"{'ok' if ok else 'FAIL':<5}{name}")
    if not ok:
        failed += 1


v1 = load_v1()


def v1_key(label):
    k = (label.replace("(shape)", "_shape").replace("(scale)", "_scale").replace("(log)", "_log")
         .replace(" ", "").replace("-", "_").replace(".", "_").replace("/", "_"))
    return "lambda_" if k == "lambda" else k


# 1. The zoo is the zoo of version 1
ok = list(v1["DIST"]) == NAMES
for name in NAMES:
    d, old = ZOO[name], v1["DIST"][name]
    ok &= d.kind == old["kind"]
    ok &= [(s.label, s.default, s.lo, s.hi, s.step) for s in d.params] == [tuple(map(lambda v: v, t)) for t in old["params"]]
    ok &= all(v1_key(s.label) == s.key for s in d.params)
    pre = v1["PRESETS"][name]
    ok &= d.preset == pre
check("16 distributions, parameter labels, defaults, slider ranges, steps and presets equal version 1", ok)

# 2. Same draws as version 1
ok = True
for name in NAMES:
    d = ZOO[name]
    p = d.preset_params()
    a = v1["DIST"][name]["rvs"](np.random.default_rng(42), size=1000, **p)
    b = d.sample(np.random.default_rng(42), 1000, **p)
    ok &= np.array_equal(a, b)
check("samplers give the same draws as version 1 for seed 42 (all 16 presets)", ok)

# 3. Exact densities equal the formulas of version 1 (which are correct)
ok = True
for name in NAMES:
    d = ZOO[name]
    p = d.preset_params()
    x = support_grid(d, p)
    if d.kind == "continuous":
        new, old = d.frozen(**p).pdf(x), v1["DIST"][name]["pdf"](x, **p)
    else:
        new, old = d.frozen(**p).pmf(x), v1["DIST"][name]["pmf"](x, **p)
    ok &= np.allclose(new, old, rtol=1e-10, atol=1e-14)
check("exact PDF/PMF equal the formulas of version 1 at every preset", ok)


# 4. Moments: closed forms against numerical integration (continuous) or summation (discrete)
def numeric_moments(d, p):
    law = d.frozen(**p)
    if d.kind == "continuous":
        lo, hi = law.support()

        def E(f):
            return integrate.quad(lambda x: f(x) * law.pdf(x), lo, hi, limit=500, epsabs=1e-13, epsrel=1e-11)[0]
    else:
        lo, hi = law.support()
        top = hi
        if not math.isfinite(top):
            top = 10
            while law.sf(top) > 1e-18:
                top *= 2
        k = np.arange(int(lo), int(top) + 1)
        w = law.pmf(k)

        def E(f):
            return float(np.sum(f(k.astype(float)) * w))
    mu = E(lambda x: x)
    var = E(lambda x: (x - mu) ** 2)
    out = [mu, var]
    r = d.moment_order(**p)
    out.append(E(lambda x: (x - mu) ** 3) / var**1.5 if r > 3 else math.nan)
    out.append(E(lambda x: (x - mu) ** 4) / var**2 - 3 if r > 4 else math.nan)
    return out


ok = True
for name in NAMES:
    d = ZOO[name]
    p = d.preset_params()
    m = list(d.moments(**p).values())
    for a, b in zip(m, numeric_moments(d, p)):
        ok &= (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-6 * max(1.0, abs(b))
check("mean, variance, skewness, excess kurtosis equal numerical integration/summation (16 presets)", ok)

# 5. Existence of moments at the boundaries of the slider ranges
ok = (math.isnan(ZOO["Student-t"].moments(df=1)["mean"]) and math.isnan(ZOO["Student-t"].moments(df=2)["var"])
      and not math.isnan(ZOO["Student-t"].moments(df=3)["var"])
      and math.isnan(ZOO["Pareto"].moments(xm=1, alpha=1.0)["mean"])
      and not math.isnan(ZOO["Pareto"].moments(xm=1, alpha=1.1)["mean"])
      and math.isnan(ZOO["Pareto"].moments(xm=1, alpha=2.0)["var"])
      and math.isnan(ZOO["F"].moments(d1=5, d2=2)["mean"]) and math.isnan(ZOO["F"].moments(d1=5, d2=4)["var"])
      and not math.isnan(ZOO["F"].moments(d1=5, d2=5)["var"]))
check("moments are reported missing exactly when they do not exist (Student-t, Pareto, F)", ok)

# 6. Samplers against the exact laws: KS (continuous) and chi-square (discrete), 200,000 draws each
ok, worst = True, 1.0
for name in NAMES:
    d = ZOO[name]
    p = d.preset_params()
    law = d.frozen(**p)
    x = d.sample(np.random.default_rng(2024), 200_000, **p)
    if d.kind == "continuous":
        pv = stats.kstest(x, law.cdf).pvalue
    else:
        k, c = np.unique(x, return_counts=True)
        exp = law.pmf(k) * x.size
        keep = exp >= 5
        obs = np.append(c[keep], c[~keep].sum() + (x.size - c.sum()))
        e = np.append(exp[keep], x.size - exp[keep].sum())
        pv = stats.chisquare(obs, e).pvalue
    worst = min(worst, pv)
    ok &= pv > 0.001
check(f"every sampler agrees with its exact law (smallest p-value {worst:.3f})", ok)

# 7. Laws of the sum against N-fold convolution (discrete) and simulation (continuous)
ok = True
for name in ("Bernoulli", "Binomial", "Poisson", "Geometric", "NegBinomial"):
    d = ZOO[name]
    p = d.preset_params()
    law = d.frozen(**p)
    k = np.arange(0, int(law.isf(1e-16)) + 1)
    f = law.pmf(k)
    conv = f.copy()
    for _ in range(4):
        conv = np.convolve(conv, f)
    kk = np.arange(conv.size)
    ok &= np.max(np.abs(conv - d.sum_law(5, **p).pmf(kk))) < 1e-12
for name in ("Normal", "Exponential", "Gamma", "Chi-square"):
    d = ZOO[name]
    p = d.preset_params()
    s = d.sample(np.random.default_rng(7), (100_000, 5), **p).sum(axis=1)
    ok &= stats.kstest(s, d.sum_law(5, **p).cdf).pvalue > 0.001
check("exact laws of S_5 equal 5-fold convolution (5 discrete) and simulated sums (4 continuous)", ok)

# 8. Exact Kolmogorov distance and corrected error against brute force on a fine grid
ok = True
for name, N in (("Bernoulli", 7), ("Poisson", 3), ("Geometric", 4), ("NegBinomial", 2), ("Binomial", 3)):
    d = ZOO[name]
    p = d.preset_params()
    m = d.moments(**p)
    law = d.sum_law(N, **p)
    loc, sc = N * m["mean"], math.sqrt(N * m["var"])
    x = np.sort(np.concatenate([np.linspace(law.ppf(1e-12) - 2, law.isf(1e-12) + 2, 400_001),
                                np.arange(law.ppf(1e-12) - 2, law.isf(1e-12) + 3),            # support points
                                np.arange(law.ppf(1e-12) - 2, law.isf(1e-12) + 3) - 1e-9]))   # just before them
    brute = np.max(np.abs(law.cdf(x) - Phi((x - loc) / sc)))
    ok &= abs(kolmogorov_distance(d, p, N)[0] - brute) < 1e-7
    k = np.arange(int(law.ppf(1e-14)) - 1, int(law.isf(1e-14)) + 2)
    ok &= abs(corrected_error(d, p, N) - np.max(np.abs(law.cdf(k) - Phi((k + 0.5 - loc) / sc)))) < 1e-12
for name, N in (("Exponential", 1), ("Gamma", 3), ("Chi-square", 2)):
    d = ZOO[name]
    p = d.preset_params()
    m = d.moments(**p)
    law = d.sum_law(N, **p)
    z = np.linspace(-12, 12, 2_400_001)
    brute = np.max(np.abs(law.cdf(N * m["mean"] + math.sqrt(N * m["var"]) * z) - Phi(z)))
    ok &= abs(kolmogorov_distance(d, p, N)[0] - brute) < 1e-8
ok &= kolmogorov_distance(ZOO["Normal"], ZOO["Normal"].preset_params(), 10)[0] < 1e-12
check("exact distances (with and without continuity correction) equal brute-force search", ok)

# 9. Lattice constants: sqrt(N) D_N and sqrt(N) E_N approach them (N = 10^4)
ok = True
for name in ("Bernoulli", "Binomial", "Poisson", "Geometric", "NegBinomial"):
    d = ZOO[name]
    p = d.preset_params()
    cu, cc = lattice_constants(d, p)
    ok &= abs(math.sqrt(1e4) * kolmogorov_distance(d, p, 10_000)[0] - cu) < 2e-3 * cu
    ok &= abs(math.sqrt(1e4) * corrected_error(d, p, 10_000) - cc) < 1e-3   # cc = 0 for the symmetric ones
check("sqrt(N) D_N and sqrt(N) E_N match the lattice Edgeworth constants at N = 10^4", ok)

# 10. Simulation helpers
d, p = ZOO["Gamma"], ZOO["Gamma"].preset_params()
a = sample_means(d, p, 30, 1000, np.random.default_rng(1), chunk=300)
b = sample_means(d, p, 30, 1000, np.random.default_rng(1), chunk=10_000_000)
direct = d.sample(np.random.default_rng(1), (1000, 30), **p).mean(axis=1)
check("sample means do not depend on the block size and equal a direct computation",
      np.allclose(a, b, rtol=0, atol=1e-13) and np.allclose(a, direct, rtol=0, atol=1e-13))
rng = np.random.default_rng(5)
x = np.concatenate([d.sample(rng, 100_000, **p), d.sample(rng, 50_000, **p)])  # the two blocks of one path
q = running_mean_quantiles(d, p, [10, 150_000], 1, np.random.default_rng(5), q=(0.5,))[0]
check("running-mean checkpoints equal a direct computation, across a block boundary",
      abs(q[0] - x[:10].mean()) < 1e-12 and abs(q[1] - x.mean()) < 1e-12)

# 11. Robustness over the slider ranges: no error, finite output (version 1 crashed for some of these)
ok = True
bad = []
for name in NAMES:
    d = ZOO[name]
    for s in d.params:
        for v in np.unique([s.lo, s.default, s.hi, s.lo + s.step, s.hi - s.step]):
            p = dict(d.preset_params(), **{s.key: v})
            if name == "Uniform" and p["b"] <= p["a"]:
                continue
            try:
                law = d.frozen(**p)
                x = support_grid(d, p)
                y = law.pdf(x) if d.kind == "continuous" else law.pmf(x)
                fin = np.isfinite(law.cdf(x)).all() and np.isfinite(y[np.isfinite(x)]).sum() > 0
                if not fin:
                    bad.append((name, s.key, v))
            except Exception as e:  # noqa: BLE001
                bad.append((name, s.key, v, type(e).__name__))
ok = not bad
check(f"exact functions finite at every slider end point (5 values per parameter){'' if ok else ': ' + str(bad)}", ok)

ok = True
for name, p in (("Binomial", {"n": 200, "p": 0.5}), ("NegBinomial", {"r": 80, "p": 0.01}),
                ("NegBinomial", {"r": 5, "p": 0.05}), ("Pareto", {"xm": 1, "alpha": 0.8}), ("Student-t", {"df": 1})):
    try:
        import matplotlib.pyplot as plt
        plt.close(explorer_figure(name, p, n_sample=2000))
        plt.close(clt_figure(name, p, N=30, M=500, path_length=1000))
    except Exception as e:  # noqa: BLE001
        print("   ", name, p, type(e).__name__, e)
        ok = False
check("both views draw the cases where version 1 failed or the moments do not exist", ok)

try:
    ZOO["Uniform"].check({"a": 2.0, "b": 1.0})
    ok = False
except ValueError:
    ok = True
check("an impossible parameter (Uniform with b <= a) raises an error instead of being changed silently", ok)

# 12. Notebook controls (ipywidgets): every distribution can be chosen in both views, each change draws once
import contextlib  # noqa: E402
import io  # noqa: E402

import cltzoo.widgets as wd  # noqa: E402

boxes, calls = [], []
wd.display = boxes.append
_e, _c = wd.explorer_figure, wd.clt_figure
wd.explorer_figure = lambda *a, **k: (calls.append(("e", a[0])), _e(*a, **k))[1]
wd.clt_figure = lambda *a, **k: (calls.append(("c", a[0])), _c(*a, **k))[1]
with contextlib.redirect_stdout(io.StringIO()) as printed:
    wd.explorer_ui()
    wd.clt_ui()
    for box in boxes:
        for nm in NAMES:
            box.children[0].children[0].value = nm
    boxes[0].children[0].children[0].value = "Uniform"
    boxes[0].children[3].children[0].value = 2.0          # a = 2 > b = 1
e_names = [c[1] for c in calls if c[0] == "e"]
c_names = [c[1] for c in calls if c[0] == "c"]
check("notebook controls: all 16 distributions in both views, one drawing per change, b <= a reported",
      sorted(set(e_names)) == sorted(NAMES) and sorted(set(c_names)) == sorted(NAMES)
      and len(c_names) == 17 and "b must be larger than a" in printed.getvalue())

print(f"\n{failed} check(s) failed.")
sys.exit(1 if failed else 0)
