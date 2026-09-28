"""Black–Scholes–Merton pricing with continuous dividend yield.

All functions are vectorised: any argument may be a NumPy array and the usual
broadcasting rules apply.

    d1 = [ln(S/K) + (r - q + σ²/2) T] / (σ√T),   d2 = d1 - σ√T
    C  = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    P  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _pdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _sign(kind: str) -> int:
    kind = kind.lower()
    if kind in ("call", "c"):
        return 1
    if kind in ("put", "p"):
        return -1
    raise ValueError(f"option kind must be 'call' or 'put', got {kind!r}")


def _d1_d2(S, K, T, r, sigma, q):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S, K, T, r, sigma, q=0.0, kind: str = "call"):
    """European option price under Black–Scholes–Merton.

    Parameters
    ----------
    S, K : spot and strike
    T : time to expiry in years (> 0)
    r : continuously-compounded risk-free rate
    sigma : volatility (> 0)
    q : continuous dividend yield
    kind : "call" or "put"
    """
    w = _sign(kind)
    S, K, T, r, sigma, q = np.broadcast_arrays(*map(np.asarray, (S, K, T, r, sigma, q)))
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    price = w * (S * np.exp(-q * T) * ndtr(w * d1) - K * np.exp(-r * T) * ndtr(w * d2))
    return price[()] if price.ndim == 0 else price


@dataclass(frozen=True)
class Greeks:
    """First- and second-order sensitivities of an option price.

    ``theta`` is per year (divide by 365 for a per-day figure) and ``vega`` /
    ``rho`` are per unit change (divide by 100 for per 1 %).
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float


def bs_greeks(S, K, T, r, sigma, q=0.0, kind: str = "call") -> Greeks:
    """Analytic Greeks under Black–Scholes–Merton."""
    w = _sign(kind)
    S, K, T, r, sigma, q = np.broadcast_arrays(*map(np.asarray, (S, K, T, r, sigma, q)))
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = np.sqrt(T)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    n_d1 = _pdf(d1)

    price = w * (S * dq * ndtr(w * d1) - K * dr * ndtr(w * d2))
    delta = w * dq * ndtr(w * d1)
    gamma = dq * n_d1 / (S * sigma * sqrt_t)
    vega = S * dq * n_d1 * sqrt_t
    theta = (
        -S * dq * n_d1 * sigma / (2.0 * sqrt_t)
        + w * q * S * dq * ndtr(w * d1)
        - w * r * K * dr * ndtr(w * d2)
    )
    rho = w * K * T * dr * ndtr(w * d2)
    vanna = -dq * n_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma

    def unwrap(x):
        return x[()] if x.ndim == 0 else x

    return Greeks(*(unwrap(x) for x in (price, delta, gamma, vega, theta, rho, vanna, volga)))
