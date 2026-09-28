"""volsmith — a small, rigorous toolkit for pricing equity options.

Modules
-------
black_scholes  closed-form European prices and Greeks
implied_vol    robust implied-volatility inversion
binomial       Cox–Ross–Rubinstein trees (European & American)
monte_carlo    variance-reduced Monte Carlo (European & Asian)
heston         Heston stochastic-volatility pricing and smile calibration
"""

from .black_scholes import Greeks, bs_greeks, bs_price
from .binomial import crr_price
from .heston import HestonParams, calibrate_heston, heston_price
from .implied_vol import implied_vol
from .monte_carlo import MCResult, mc_asian_arithmetic, mc_european

__all__ = [
    "Greeks",
    "HestonParams",
    "MCResult",
    "bs_greeks",
    "bs_price",
    "calibrate_heston",
    "crr_price",
    "heston_price",
    "implied_vol",
    "mc_asian_arithmetic",
    "mc_european",
]

__version__ = "0.1.0"
