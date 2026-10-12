"""The distribution zoo of version 1: sixteen distributions with their parameters, slider ranges, presets and
samplers exactly as in version 1, plus what version 2 adds: exact distribution functions, exact moments, the
order up to which moments exist, and the exact law of the sum of N draws where it has a closed form.

Parameters are given by the names version 1 used internally (k_shape, theta_scale, mu_log, sigma_log, b_scale,
lambda_ ...). The samplers make the same NumPy calls as version 1, so a seed gives the same draws.

Author: Muhammed Ikbal Yilmaz
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

INF = math.inf


@dataclass(frozen=True)
class Param:
    label: str      # slider label in version 1
    key: str        # keyword used by the formulas
    default: float
    lo: float
    hi: float
    step: float

    @property
    def integer(self):
        return self.step == 1.0


@dataclass(frozen=True)
class Distribution:
    name: str
    kind: str                       # "continuous" or "discrete"
    params: tuple
    preset: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ parameters
    def defaults(self):
        return {p.key: p.default for p in self.params}

    def preset_params(self):
        out = self.defaults()
        out.update(self.preset.get("params", {}))
        return self.check(out)

    def check(self, params):
        """Validated parameters (integers rounded as in version 1). Raises ValueError for impossible values."""
        p = dict(self.defaults(), **params)
        for spec in self.params:
            v = float(p[spec.key])
            if not math.isfinite(v):
                raise ValueError(f"{self.name}: {spec.label} must be finite")
            p[spec.key] = int(round(v)) if spec.integer else v
        _validate(self.name, p)
        return p

    # ------------------------------------------------------------------ sampling (as in version 1)
    def sample(self, rng, size, **params):
        return _SAMPLERS[self.name](rng, size, **self.check(params))

    # ------------------------------------------------------------------ exact distribution
    def frozen(self, **params):
        """scipy.stats frozen distribution with the same parametrization as the sampler."""
        return _FROZEN[self.name](**self.check(params))

    def moments(self, **params):
        """Exact mean, variance, skewness and excess kurtosis; nan where the moment does not exist."""
        p = self.check(params)
        mean, var, skew, exkurt = _MOMENTS[self.name](**p)
        r = self.moment_order(**p)
        return {"mean": mean if r > 1 else math.nan, "var": var if r > 2 else math.nan,
                "skew": skew if r > 3 else math.nan, "exkurt": exkurt if r > 4 else math.nan}

    def moment_order(self, **params):
        """r* such that E|X|^r is finite exactly for r < r* (infinity when every moment exists)."""
        p = self.check(params)
        return {"Student-t": lambda: p["df"], "Pareto": lambda: p["alpha"],
                "F": lambda: p["d2"] / 2.0}.get(self.name, lambda: INF)()

    def sum_law(self, N, **params):
        """Exact law of X_1 + ... + X_N (frozen scipy distribution), or None if it has no closed form here."""
        p = self.check(params)
        N = int(N)
        f = _SUM_LAWS.get(self.name)
        return None if f is None else f(N, **p)


def _validate(name, p):
    def need(cond, msg):
        if not cond:
            raise ValueError(f"{name}: {msg}")
    if name == "Uniform":
        need(p["b"] > p["a"], "b must be larger than a")
    for key in ("sigma", "scale", "theta_scale", "sigma_log", "b_scale", "xm", "alpha", "beta", "k_shape",
                "df", "d1", "d2", "lambda_", "r", "n"):
        if key in p:
            need(p[key] > 0, f"{key} must be positive")
    if "p" in p:
        need(0 < p["p"] < 1, "p must lie strictly between 0 and 1")


# ------------------------------------------------------------------------------------------ samplers
# Identical to the lambdas of version 1 (same NumPy generator calls, same order of arguments).
_SAMPLERS = {
    "Normal": lambda rng, size, mu, sigma: rng.normal(mu, sigma, size=size),
    "Uniform": lambda rng, size, a, b: rng.uniform(a, b, size=size),
    "Exponential": lambda rng, size, scale: rng.exponential(scale, size=size),
    "Gamma": lambda rng, size, k_shape, theta_scale: rng.gamma(k_shape, theta_scale, size=size),
    "Beta": lambda rng, size, alpha, beta: rng.beta(alpha, beta, size=size),
    "Lognormal": lambda rng, size, mu_log, sigma_log: rng.lognormal(mu_log, sigma_log, size=size),
    "Laplace": lambda rng, size, mu, b_scale: rng.laplace(mu, b_scale, size=size),
    "Student-t": lambda rng, size, df: rng.standard_t(df, size=size),
    "Pareto": lambda rng, size, xm, alpha: xm * (1.0 + rng.pareto(alpha, size=size)),
    "Chi-square": lambda rng, size, df: rng.chisquare(df, size=size),
    "F": lambda rng, size, d1, d2: rng.f(d1, d2, size=size),
    "Bernoulli": lambda rng, size, p: rng.binomial(1, p, size=size),
    "Binomial": lambda rng, size, n, p: rng.binomial(n, p, size=size),
    "Poisson": lambda rng, size, lambda_: rng.poisson(lambda_, size=size),
    "Geometric": lambda rng, size, p: rng.geometric(p, size=size),
    "NegBinomial": lambda rng, size, r, p: rng.negative_binomial(r, p, size=size),
}

# ------------------------------------------------------------------------------------------ exact laws
_FROZEN = {
    "Normal": lambda mu, sigma: stats.norm(mu, sigma),
    "Uniform": lambda a, b: stats.uniform(a, b - a),
    "Exponential": lambda scale: stats.expon(scale=scale),
    "Gamma": lambda k_shape, theta_scale: stats.gamma(k_shape, scale=theta_scale),
    "Beta": lambda alpha, beta: stats.beta(alpha, beta),
    "Lognormal": lambda mu_log, sigma_log: stats.lognorm(sigma_log, scale=math.exp(mu_log)),
    "Laplace": lambda mu, b_scale: stats.laplace(mu, b_scale),
    "Student-t": lambda df: stats.t(df),
    "Pareto": lambda xm, alpha: stats.pareto(alpha, scale=xm),
    "Chi-square": lambda df: stats.chi2(df),
    "F": lambda d1, d2: stats.f(d1, d2),
    "Bernoulli": lambda p: stats.bernoulli(p),
    "Binomial": lambda n, p: stats.binom(n, p),
    "Poisson": lambda lambda_: stats.poisson(lambda_),
    "Geometric": lambda p: stats.geom(p),                 # number of trials up to the first success: 1, 2, ...
    "NegBinomial": lambda r, p: stats.nbinom(r, p),       # number of failures before the r-th success: 0, 1, ...
}


def _beta_moments(alpha, beta):
    a, b = alpha, beta
    s = a + b
    return (a / s, a * b / (s**2 * (s + 1)), 2 * (b - a) * math.sqrt(s + 1) / ((s + 2) * math.sqrt(a * b)),
            6 * ((a - b) ** 2 * (s + 1) - a * b * (s + 2)) / (a * b * (s + 2) * (s + 3)))


def _lognormal_moments(mu_log, sigma_log):
    w = math.exp(sigma_log**2)
    return (math.exp(mu_log + sigma_log**2 / 2), (w - 1) * math.exp(2 * mu_log + sigma_log**2),
            (w + 2) * math.sqrt(w - 1), w**4 + 2 * w**3 + 3 * w**2 - 6)


def _t_moments(df):
    v = df
    return (0.0, v / (v - 2) if v > 2 else INF, 0.0, 6 / (v - 4) if v > 4 else INF)


def _pareto_moments(xm, alpha):
    a = alpha
    mean = a * xm / (a - 1) if a > 1 else INF
    var = xm**2 * a / ((a - 1) ** 2 * (a - 2)) if a > 2 else INF
    skew = 2 * (1 + a) / (a - 3) * math.sqrt((a - 2) / a) if a > 3 else math.nan
    exkurt = 6 * (a**3 + a**2 - 6 * a - 2) / (a * (a - 3) * (a - 4)) if a > 4 else math.nan
    return mean, var, skew, exkurt


def _f_moments(d1, d2):
    mean = d2 / (d2 - 2) if d2 > 2 else INF
    var = 2 * d2**2 * (d1 + d2 - 2) / (d1 * (d2 - 2) ** 2 * (d2 - 4)) if d2 > 4 else INF
    skew = ((2 * d1 + d2 - 2) * math.sqrt(8 * (d2 - 4)) / ((d2 - 6) * math.sqrt(d1 * (d1 + d2 - 2)))
            if d2 > 6 else math.nan)
    exkurt = (12 * (d1 * (5 * d2 - 22) * (d1 + d2 - 2) + (d2 - 4) * (d2 - 2) ** 2)
              / (d1 * (d2 - 6) * (d2 - 8) * (d1 + d2 - 2)) if d2 > 8 else math.nan)
    return mean, var, skew, exkurt


_MOMENTS = {
    "Normal": lambda mu, sigma: (mu, sigma**2, 0.0, 0.0),
    "Uniform": lambda a, b: ((a + b) / 2, (b - a) ** 2 / 12, 0.0, -6 / 5),
    "Exponential": lambda scale: (scale, scale**2, 2.0, 6.0),
    "Gamma": lambda k_shape, theta_scale: (k_shape * theta_scale, k_shape * theta_scale**2,
                                           2 / math.sqrt(k_shape), 6 / k_shape),
    "Beta": _beta_moments,
    "Lognormal": _lognormal_moments,
    "Laplace": lambda mu, b_scale: (mu, 2 * b_scale**2, 0.0, 3.0),
    "Student-t": _t_moments,
    "Pareto": _pareto_moments,
    "Chi-square": lambda df: (df, 2 * df, math.sqrt(8 / df), 12 / df),
    "F": _f_moments,
    "Bernoulli": lambda p: (p, p * (1 - p), (1 - 2 * p) / math.sqrt(p * (1 - p)), (1 - 6 * p * (1 - p)) / (p * (1 - p))),
    "Binomial": lambda n, p: (n * p, n * p * (1 - p), (1 - 2 * p) / math.sqrt(n * p * (1 - p)),
                              (1 - 6 * p * (1 - p)) / (n * p * (1 - p))),
    "Poisson": lambda lambda_: (lambda_, lambda_, 1 / math.sqrt(lambda_), 1 / lambda_),
    "Geometric": lambda p: (1 / p, (1 - p) / p**2, (2 - p) / math.sqrt(1 - p), 6 + p**2 / (1 - p)),
    "NegBinomial": lambda r, p: (r * (1 - p) / p, r * (1 - p) / p**2, (2 - p) / math.sqrt(r * (1 - p)),
                                 6 / r + p**2 / (r * (1 - p))),
}

# Closed-form laws of S_N = X_1 + ... + X_N (sums of independent copies)
_SUM_LAWS = {
    "Normal": lambda N, mu, sigma: stats.norm(N * mu, sigma * math.sqrt(N)),
    "Exponential": lambda N, scale: stats.gamma(N, scale=scale),
    "Gamma": lambda N, k_shape, theta_scale: stats.gamma(N * k_shape, scale=theta_scale),
    "Chi-square": lambda N, df: stats.chi2(N * df),
    "Bernoulli": lambda N, p: stats.binom(N, p),
    "Binomial": lambda N, n, p: stats.binom(N * n, p),
    "Poisson": lambda N, lambda_: stats.poisson(N * lambda_),
    "Geometric": lambda N, p: stats.nbinom(N, p, loc=N),     # N trials-to-success = N + failures
    "NegBinomial": lambda N, r, p: stats.nbinom(N * r, p),
}


# ------------------------------------------------------------------------------------------ the zoo of version 1
def _P(label, key, default, lo, hi, step):
    return Param(label, key, float(default), float(lo), float(hi), float(step))


def _preset(params, sample, n, xpad, grid, overlay):
    return {"params": params, "sample": sample, "n": n, "xpad": xpad, "grid": grid, "overlay": overlay}


ZOO = {d.name: d for d in (
    Distribution("Normal", "continuous", (_P("mu", "mu", 0.0, -5.0, 5.0, 0.1), _P("sigma", "sigma", 1.0, 0.1, 5.0, 0.1)),
                 _preset({"mu": 0.0, "sigma": 1.0}, True, 5000, 0.20, 900, True)),
    Distribution("Uniform", "continuous", (_P("a", "a", 0.0, -10.0, 5.0, 0.1), _P("b", "b", 1.0, -5.0, 10.0, 0.1)),
                 _preset({"a": 0.0, "b": 1.0}, False, 0, 0.10, 700, False)),
    Distribution("Exponential", "continuous", (_P("scale", "scale", 1.0, 0.1, 10.0, 0.1),),
                 _preset({"scale": 1.0}, True, 5000, 0.35, 900, True)),
    Distribution("Gamma", "continuous", (_P("k(shape)", "k_shape", 2.0, 0.2, 15.0, 0.1),
                                         _P("theta(scale)", "theta_scale", 1.0, 0.1, 8.0, 0.1)),
                 _preset({"k_shape": 2.0, "theta_scale": 1.0}, True, 6000, 0.30, 900, True)),
    Distribution("Beta", "continuous", (_P("alpha", "alpha", 2.0, 0.2, 15.0, 0.1), _P("beta", "beta", 5.0, 0.2, 15.0, 0.1)),
                 _preset({"alpha": 2.0, "beta": 5.0}, False, 0, 0.05, 900, False)),
    Distribution("Lognormal", "continuous", (_P("mu(log)", "mu_log", 0.0, -2.5, 2.5, 0.1),
                                             _P("sigma(log)", "sigma_log", 0.6, 0.1, 2.5, 0.1)),
                 _preset({"mu_log": 0.0, "sigma_log": 0.6}, True, 6000, 0.50, 900, True)),
    Distribution("Laplace", "continuous", (_P("mu", "mu", 0.0, -5.0, 5.0, 0.1), _P("b(scale)", "b_scale", 1.0, 0.1, 6.0, 0.1)),
                 _preset({"mu": 0.0, "b_scale": 1.0}, True, 6000, 0.25, 900, True)),
    Distribution("Student-t", "continuous", (_P("df", "df", 5.0, 1.0, 80.0, 1.0),),
                 _preset({"df": 5}, True, 8000, 0.30, 900, True)),
    Distribution("Pareto", "continuous", (_P("xm", "xm", 1.0, 0.1, 5.0, 0.1), _P("alpha", "alpha", 2.5, 0.8, 10.0, 0.1)),
                 _preset({"xm": 1.0, "alpha": 2.5}, False, 0, 0.65, 900, False)),
    Distribution("Chi-square", "continuous", (_P("df", "df", 5.0, 1.0, 80.0, 1.0),),
                 _preset({"df": 5}, True, 6000, 0.40, 900, True)),
    Distribution("F", "continuous", (_P("d1", "d1", 5.0, 1.0, 80.0, 1.0), _P("d2", "d2", 10.0, 1.0, 160.0, 1.0)),
                 _preset({"d1": 5, "d2": 10}, False, 0, 0.50, 900, False)),
    Distribution("Bernoulli", "discrete", (_P("p", "p", 0.5, 0.01, 0.99, 0.01),),
                 _preset({"p": 0.5}, True, 3000, 0.05, 400, True)),
    Distribution("Binomial", "discrete", (_P("n", "n", 10.0, 1.0, 200.0, 1.0), _P("p", "p", 0.5, 0.01, 0.99, 0.01)),
                 _preset({"n": 20, "p": 0.5}, True, 6000, 0.10, 400, True)),
    Distribution("Poisson", "discrete", (_P("lambda", "lambda_", 4.0, 0.1, 40.0, 0.1),),
                 _preset({"lambda_": 4.0}, True, 7000, 0.15, 400, True)),
    Distribution("Geometric", "discrete", (_P("p", "p", 0.3, 0.01, 0.99, 0.01),),
                 _preset({"p": 0.3}, True, 7000, 0.25, 400, True)),
    Distribution("NegBinomial", "discrete", (_P("r", "r", 5.0, 1.0, 80.0, 1.0), _P("p", "p", 0.4, 0.01, 0.99, 0.01)),
                 _preset({"r": 5, "p": 0.4}, True, 7000, 0.20, 400, True)),
)}

NAMES = list(ZOO)
