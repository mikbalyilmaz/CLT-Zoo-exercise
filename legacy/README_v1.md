# clt-zoo
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive simulation that visualizes the **Law of Large Numbers (LLN)** and the
**Central Limit Theorem (CLT)** across a wide range of probability distributions.

The notebook renders three coordinated panels:
1. Running mean (LLN)
2. Sampling distribution of the sample mean
3. Standardized mean

---

## What it shows

### Law of Large Numbers (single path)

The running mean

~~~math
\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i
~~~

stabilizes around the population mean `\mu` as the sample size `n` increases.

---

### Central Limit Theorem (sampling distribution)

For a fixed sample size `N`, repeated sample means satisfy

~~~math
\bar X_N \approx \mathcal N\!\left(\mu,\frac{\sigma^2}{N}\right)
~~~

whenever the population variance `\sigma^2` is finite.

---

### Standardized mean

~~~math
Z_N = \frac{\sqrt{N}(\bar X_N-\mu)}{\sigma}
\;\Rightarrow\;
\mathcal N(0,1)
~~~

illustrating the classical CLT in normalized form.

---

## Key Features

### Distribution zoo

Normal, Uniform, Exponential, Gamma, Beta, Lognormal, Laplace, Student-t, Pareto,
Chi-square, F, Bernoulli, Binomial, Poisson, Geometric, Negative Binomial.

---

### Fully interactive

Change the distribution, its parameters, sample size `N`, number of Monte Carlo
replications `M`, and random seed via sliders.

---

### Three-panel visualization

1. Running mean (LLN)
2. Sampling distribution of `\bar X_N` with Normal (CLT) overlay
3. Standardized mean with `\mathcal N(0,1)` overlay

---

### Theory-aware behavior

Normal approximations and standardization are automatically disabled when the
population mean or variance is infinite (e.g. heavy-tailed distributions).

---

## Requirements

- Python ≥ 3.9
- numpy
- matplotlib
- ipywidgets

~~~bash
pip install numpy matplotlib ipywidgets
~~~
