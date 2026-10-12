# Changes

## Version 2.0.0

### New

- `cltzoo/` holds the sixteen distributions of version 1 (same names, parameters, slider ranges, presets and samplers)
  with exact distribution functions, exact moments, the order up to which moments exist, and the closed-form law of
  the sum of $N$ draws for nine of them.
- The LLN and CLT view that the README of version 1 described: running mean, sampling distribution of the mean for a
  sample size $N$ and $M$ replications, standardized mean against $\mathcal N(0,1)$; the population mean is drawn only
  when it is finite and the normal curves only when the variance is finite. For discrete distributions the histogram
  bars follow the values the mean can take.
- Exact Kolmogorov distance of the standardized mean to $\mathcal N(0,1)$, with and without the continuity correction,
  and the leading constants of the lattice Edgeworth expansion.
- Unit tests (`tests/run_tests.py`) against the code of version 1 (read from the legacy notebook), numerical
  integration and summation of the densities, $N$-fold convolution, goodness-of-fit tests of every sampler, and
  brute-force search on a fine grid; the notebook controls are tested for all sixteen distributions.
- Three analyses, four figures, CSV tables, and a project note in English with the mathematics of every step, whose
  numbers are written from the tables.
- `CITATION.cff`, `requirements.txt`, `.gitignore`, a GitHub Actions workflow that runs the tests.

### Corrections to version 1

| # | Version 1 | Version 2 |
|---|---|---|
| 1 | The introduction and the README described a running mean, the sampling distribution of the mean with $N$ and $M$ sliders, the standardized mean and normal curves "automatically disabled" for infinite mean or variance. The notebook contained none of these; its only code cell is a distribution explorer. | The LLN and CLT view is implemented, and analyses 2 and 3 study it. |
| 2 | The distribution function was the integral of the density over the plotting window, rescaled to end at 1. At the presets the error reached 0.022 (Pareto, where the density jumps) and 0.0093 (Uniform, drawn on $[-7.2, 7.2]$ for a support of $[0, 1]$). | Exact distribution functions (`scipy.stats`); plotting windows from exact quantiles and the support. |
| 3 | Binomial with $n \ge 171$ (30 of 299 slider positions) and Negative Binomial with $r \ge 45$ or $p \le 0.10$ (46 of 179) stopped with an `OverflowError`: the binomial coefficients were computed with `math.gamma`, which overflows above 171.6. | Exact probability functions from `scipy.stats`; finite at all 2,844 valid slider positions. |
| 4 | When the Uniform sliders gave $b \le a$ (92 of 302 positions), $b$ was replaced silently by $a + 10^{-6}$. | The notebook reports that $b$ must be larger than $a$. |
| 5 | The parameter box printed floating-point artefacts (`xm = 1.0000000000000002` in the saved Pareto figure). | Parameters are printed in short form. |
| 6 | The README wrote $\mu$, $\sigma^2$, $\bar X_N$ and $\mathcal N(0,1)$ inside code quotes, so GitHub showed `\mu` and `\mathcal N(0,1)` literally. | Mathematics in `$...$`. |
| 7 | References: Kolmogorov (1933) was given with the English title and the Chelsea publisher of the later translation; Feller (1968, Vol. 1) was cited for non-Gaussian limits under infinite variance, which are treated in Vol. 2. | Kolmogorov (1933), *Grundbegriffe der Wahrscheinlichkeitsrechnung*, Springer; Feller (1971, Vol. 2) and Gnedenko and Kolmogorov (1954) for stable limits. |
| 8 | Sixteen PNG files, an HTML and a PDF export were kept in the root of the repository; saved figures were written to the working directory. | Figures in `figures/`; figures saved from the notebook go to `notebook/renders/` (ignored by git). Version 1 kept as `legacy/clt_zoo_v1.ipynb`; the other files remain in the `v1.0.0` release. |

## Version 1.0.0

First version: an interactive distribution explorer for sixteen distributions (density or probability function, a
sample, a numerically integrated distribution function), with presets and a button to save the figure as PNG.
