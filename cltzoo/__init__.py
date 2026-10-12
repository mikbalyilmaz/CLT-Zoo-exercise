"""cltzoo: the distribution zoo of version 1, with exact laws, the Law of Large Numbers and the Central Limit
Theorem."""

from .clt import (corrected_error, kolmogorov_distance, lattice_constants, running_mean, running_mean_quantiles,
                  sample_means)
from .distributions import NAMES, ZOO, Distribution

__version__ = "2.0.1"
