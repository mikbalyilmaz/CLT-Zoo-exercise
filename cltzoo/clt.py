"""The Law of Large Numbers and the Central Limit Theorem for the zoo: simulation tools and exact distances.

Notation: X_1, X_2, ... i.i.d. with mean mu and variance sigma^2; S_N = X_1 + ... + X_N; Xbar_N = S_N / N;
Z_N = sqrt(N) (Xbar_N - mu) / sigma = (S_N - N mu) / (sigma sqrt(N)).

Exact distances use the closed-form law of S_N (Distribution.sum_law):
* kolmogorov_distance: D_N = sup_x |P(Z_N <= x) - Phi(x)|. For integer-valued S_N the supremum is attained
  at a support point k, either at z_k = (k - N mu)/(sigma sqrt N) itself or just before it, so
  D_N = max_k max(|F(k) - Phi(z_k)|, |F(k-1) - Phi(z_k)|), computed over every k that carries probability.
  For a continuous S_N the supremum is found on a fine grid and refined by bounded scalar optimization.
* corrected_error: for integer-valued S_N, the error of the normal approximation with the continuity
  correction, E_N = max_k |F(k) - Phi((k + 1/2 - N mu)/(sigma sqrt N))|.

Author: Muhammed Ikbal Yilmaz
"""

import math

import numpy as np
from scipy import optimize, special

PHI0 = 1.0 / math.sqrt(2.0 * math.pi)   # standard normal density at 0
TAIL = 1e-15                            # probability left out at each end of a lattice support


def Phi(z):
    return special.ndtr(z)


# ------------------------------------------------------------------------------------------ simulation
def running_mean(x):
    """Xbar_n = (x_1 + ... + x_n) / n for n = 1, ..., len(x)."""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, x.size + 1)


def sample_means(dist, params, N, M, rng, chunk=200_000):
    """M independent sample means of N draws each. Draws are made row by row in blocks of whole rows, so the
    result does not depend on the block size."""
    rows = max(1, chunk // max(1, N))
    out = np.empty(M)
    done = 0
    while done < M:
        m = min(rows, M - done)
        out[done:done + m] = dist.sample(rng, (m, N), **params).mean(axis=1)
        done += m
    return out


def running_mean_quantiles(dist, params, checkpoints, M, rng, q=(0.25, 0.5, 0.75), center=0.0, absolute=False):
    """Quantiles over M independent paths of Xbar_n - center (or |Xbar_n - center|) at the given n.

    Each path is simulated in blocks up to the largest checkpoint; only the values at the checkpoints are
    kept, so memory stays small."""
    checkpoints = np.asarray(sorted(checkpoints), dtype=int)
    vals = np.empty((M, checkpoints.size))
    nmax = int(checkpoints[-1])
    block = 100_000
    for m in range(M):
        total, n_done, j = 0.0, 0, 0
        while n_done < nmax:
            k = min(block, nmax - n_done)
            x = dist.sample(rng, k, **params)
            c = total + np.cumsum(x)
            idx = checkpoints[(checkpoints > n_done) & (checkpoints <= n_done + k)]
            for n in idx:
                vals[m, j] = c[n - n_done - 1] / n
                j += 1
            total = c[-1]
            n_done += k
    d = vals - center
    if absolute:
        d = np.abs(d)
    return np.quantile(d, q, axis=0)


# ------------------------------------------------------------------------------------------ exact distances
def _lattice_range(law):
    lo = int(law.ppf(TAIL)) - 1
    hi = int(law.isf(TAIL)) + 1
    return np.arange(lo, hi + 1)


def kolmogorov_distance(dist, params, N):
    """Exact D_N for the standardized mean, from the closed-form law of S_N. Returns (D_N, z where attained)."""
    law = dist.sum_law(N, **params)
    if law is None:
        raise ValueError(f"{dist.name}: no closed-form law of the sum")
    m = dist.moments(**params)
    mu, sigma = m["mean"], math.sqrt(m["var"])
    loc, sc = N * mu, sigma * math.sqrt(N)
    if dist.kind == "discrete":
        k = _lattice_range(law)
        F = law.cdf(k)
        F_left = law.cdf(k - 1)
        z = (k - loc) / sc
        a, b = np.abs(F - Phi(z)), np.abs(F_left - Phi(z))
        i, j = int(np.argmax(a)), int(np.argmax(b))
        return (float(a[i]), float(z[i])) if a[i] >= b[j] else (float(b[j]), float(z[j]))

    def diff(z):
        return np.abs(law.cdf(loc + sc * np.asarray(z)) - Phi(z))

    zs = np.linspace(-10.0, 10.0, 20_001)
    d = diff(zs)
    best_val, best_z = float(d.max()), float(zs[int(d.argmax())])
    # refine the five largest local maxima
    peaks = np.where((d[1:-1] >= d[:-2]) & (d[1:-1] >= d[2:]))[0] + 1
    for p in peaks[np.argsort(d[peaks])[::-1][:5]]:
        res = optimize.minimize_scalar(lambda z: -diff(z), bounds=(zs[p - 1], zs[p + 1]), method="bounded",
                                       options={"xatol": 1e-12})
        if -res.fun > best_val:
            best_val, best_z = float(-res.fun), float(res.x)
    return best_val, best_z


def corrected_error(dist, params, N):
    """Integer-valued S_N: max_k |P(S_N <= k) - Phi((k + 1/2 - N mu) / (sigma sqrt N))|."""
    if dist.kind != "discrete":
        raise ValueError("the continuity correction applies to integer-valued sums")
    law = dist.sum_law(N, **params)
    m = dist.moments(**params)
    mu, sigma = m["mean"], math.sqrt(m["var"])
    k = _lattice_range(law)
    z = (k + 0.5 - N * mu) / (sigma * math.sqrt(N))
    return float(np.max(np.abs(law.cdf(k) - Phi(z))))


def lattice_constants(dist, params):
    """Leading constants c with D_N ~ c / sqrt(N) (no correction) and E_N ~ c / sqrt(N) (with correction),
    for an integer-valued X with span 1, from the first-order lattice Edgeworth expansion:
        F_N(x) - Phi(x) = [ -gamma/6 (x^2 - 1) + S(x)/sigma ] phi(x) / sqrt(N) + o(1/sqrt N),
    where S is a saw-tooth function with values in [-1/2, 1/2] that vanishes at the midpoints used by the
    continuity correction. Both terms are largest in absolute value at x = 0, which gives
        c_uncorrected = phi(0) (1/(2 sigma) + |gamma|/6),   c_corrected = phi(0) |gamma| / 6."""
    m = dist.moments(**params)
    sigma, g = math.sqrt(m["var"]), abs(m["skew"])
    return PHI0 * (0.5 / sigma + g / 6.0), PHI0 * g / 6.0
