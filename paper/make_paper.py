"""Writes every number and table of the project note from ../figures/*.csv, so that the text cannot drift away
from the analysis. Run from inside paper/:  python make_paper.py

Output: numbers.tex (one macro per quoted number), tab_clt.tex and tab_lattice.tex (table bodies).
"""

import csv
import math
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "figures"
OUT = Path(__file__).resolve().parent


def read(name):
    with open(FIG / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def by(rows, key="distribution"):
    return {r[key]: r for r in rows}


def dec(x, d):
    return f"{float(x):.{d}f}".replace("-", "$-$")


def sig(x, d=2):
    """d significant figures, plain notation."""
    x = float(x)
    if x == 0:
        return "0"
    p = d - 1 - int(math.floor(math.log10(abs(x))))
    return f"{x:.{max(p, 0)}f}"


v1 = by(read("table2_v1_checks"))
ex = by(read("table3_moment_existence"))
lln = {r["alpha"]: r for r in read("table4_lln_rates")}
clt = by(read("table6_clt_n30"))
lat = by(read("table7_lattice"))

m = {}
m["CdfErrPareto"] = dec(v1["Pareto"]["v1_cdf_max_error"], 3)
m["CdfErrUniform"] = dec(v1["Uniform"]["v1_cdf_max_error"], 4)
m["CdfErrExponential"] = dec(v1["Exponential"]["v1_cdf_max_error"], 4)
m["UniformWindowFrom"] = dec(v1["Uniform"]["window_from"], 1)
m["UniformWindowTo"] = dec(v1["Uniform"]["window_to"], 1)
m["UniformSilent"] = v1["Uniform"]["silently_changed_positions"]
m["UniformPositions"] = str(int(v1["Uniform"]["slider_positions"]) + int(v1["Uniform"]["silently_changed_positions"]))
m["BinomFails"] = v1["Binomial"]["v1_errors"]
m["BinomPositions"] = v1["Binomial"]["slider_positions"]
m["NegbinFails"] = v1["NegBinomial"]["v1_errors"]
m["NegbinPositions"] = v1["NegBinomial"]["slider_positions"]
m["AllPositions"] = f"{sum(int(r['slider_positions']) for r in v1.values()):,}"
m["VtwoErrors"] = str(sum(int(r["v2_errors"]) for r in v1.values()))
m["MaxCdfErrOthers"] = sig(max(float(r["v1_cdf_max_error"]) for k, r in v1.items() if k not in ("Pareto", "Uniform")), 2)
for name, key in (("Student-t", "T"), ("Pareto", "Pareto"), ("F", "F")):
    m[f"Pos{key}"] = ex[name]["positions"]
    m[f"NoMean{key}"] = ex[name]["no_finite_mean"]
    m[f"NoVar{key}"] = ex[name]["no_finite_variance"]

for a, key in (("2.5", "A"), ("1.5", "B"), ("0.8", "C")):
    r = lln[a]
    m[f"Slope{key}"] = dec(r["slope_fitted"], 3)
    m[f"Theory{key}"] = dec(r["slope_theory"], 3)
    m[f"SlopeFirst{key}"] = dec(r["slope_10_100"], 3)
    m[f"SlopeLast{key}"] = dec(r["slope_10000_100000"], 3)
    m[f"Median{key}"] = sig(r["median_at_1e5"], 3)
m["CltPredA"] = sig(lln["2.5"]["clt_prediction_at_1e5"], 3)
m["PathFinalC"] = dec(lln["0.8"]["path_final_value"], 0)

m["Noise"] = dec(clt["Normal"]["noise_level"], 4)
m["DParetoSim"] = dec(clt["Pareto"]["D30_simulated"], 3)
m["DFSim"] = dec(clt["F"]["D30_simulated"], 3)
m["DLognormalSim"] = dec(clt["Lognormal"]["D30_simulated"], 3)
m["DBernoulli"] = dec(clt["Bernoulli"]["D30_exact"], 3)
m["DExpExact"] = dec(clt["Exponential"]["D30_exact"], 3)
m["DExpSim"] = dec(clt["Exponential"]["D30_simulated"], 3)
for name, key in (("Bernoulli", "Ber"), ("Binomial", "Bin"), ("Poisson", "Poi"), ("Geometric", "Geo"),
                  ("NegBinomial", "Nb")):
    r = lat[name]
    m[f"Cd{key}"] = dec(r["c_D"], 4)
    m[f"Ce{key}"] = dec(r["c_E"], 4)
    m[f"Nd{key}"] = f"{int(r['first_N_D_below_0.01']):,}"
    m[f"Ne{key}"] = r["first_N_E_below_0.01"]
    m[f"NE{key}"] = dec(r["N_E_at_1e4"], 4)
(OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()), encoding="utf-8")

# Table: the zoo at N = 30
lines = []
for name, r in clt.items():
    g = float(r["skewness_Z30"])
    sk = "--" if math.isnan(g) else dec(g, 3)
    exact = dec(r["D30_exact"], 3) if r.get("D30_exact") else ""
    lines.append(f"{name} & {sk} & {exact} & {dec(r['D30_simulated'], 3)} \\\\")
lines.append("\\bottomrule")
(OUT / "tab_clt.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Table: lattice constants and sample sizes
lines = []
for name, r in lat.items():
    lines.append(f"{name} & {dec(r['sigma'], 3)} & {dec(r['skewness'], 3)} & {dec(r['c_D'], 4)} & "
                 f"{dec(r['sqrtN_D_at_1e4'], 4)} & {dec(r['c_E'], 4)} & {dec(r['sqrtN_E_at_1e4'], 4)} & "
                 f"{int(r['first_N_D_below_0.01']):,} & {r['first_N_E_below_0.01']} \\\\")
lines.append("\\bottomrule")
(OUT / "tab_lattice.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Wrote numbers.tex, tab_clt.tex and tab_lattice.tex.")
