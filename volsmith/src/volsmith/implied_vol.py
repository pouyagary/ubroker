"""Robust implied-volatility inversion.

Solving ``bs_price(σ) = market_price`` for σ looks trivial but fails in
practice for deep in/out-of-the-money options, where vega is tiny and plain
Newton iterations explode.  This module:

1. rejects prices that violate no-arbitrage bounds,
2. converts the quote (via put–call parity) to the *out-of-the-money* side,
   which carries all of the time value and is numerically well conditioned,
3. starts from the Corrado–Miller approximation,
4. runs Newton's method inside a shrinking bisection bracket, so each step is
   guaranteed to stay in the region where the root lives, and
5. falls back to Brent's method if Newton ever stalls.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .black_scholes import _sign, bs_greeks, bs_price

SIGMA_MIN = 1e-6
SIGMA_MAX = 10.0


class ArbitrageError(ValueError):
    """Raised when a quoted price lies outside the no-arbitrage bounds."""


def _corrado_miller(price, S, K, T, r, q):
    """Closed-form IV approximation for a call (Corrado & Miller, 1996)."""
    Sd = S * math.exp(-q * T)
    Kd = K * math.exp(-r * T)
    half_gap = 0.5 * (Sd - Kd)
    rem = price - half_gap
    disc = rem * rem - (Sd - Kd) ** 2 / math.pi
    guess = math.sqrt(2.0 * math.pi / T) / (Sd + Kd) * (rem + math.sqrt(max(disc, 0.0)))
    return guess if math.isfinite(guess) and guess > 0 else 0.2


def _implied_vol_scalar(price, S, K, T, r, q, kind, tol, max_iter):
    w = _sign(kind)
    Sd = S * math.exp(-q * T)
    Kd = K * math.exp(-r * T)

    lower = max(w * (Sd - Kd), 0.0)
    upper = Sd if w == 1 else Kd
    if not (lower - tol <= price <= upper + tol):
        raise ArbitrageError(
            f"price {price:.6g} outside no-arbitrage bounds [{lower:.6g}, {upper:.6g}]"
        )

    # Work on the OTM side: a call is OTM when K > forward.
    call_price = price if w == 1 else price + Sd - Kd
    if K >= S * math.exp((r - q) * T):
        target, side = call_price, "call"
    else:
        target, side = call_price - Sd + Kd, "put"

    if target <= 1e-14 * max(S, K):
        return 0.0  # no time value left: volatility is not identifiable

    def f(sig):
        return float(bs_price(S, K, T, r, sig, q, side)) - target

    lo, hi = SIGMA_MIN, SIGMA_MAX
    if f(hi) < 0:
        raise ArbitrageError("price implies volatility above SIGMA_MAX")

    sigma = min(max(_corrado_miller(call_price, S, K, T, r, q), lo), hi)
    for _ in range(max_iter):
        g = bs_greeks(S, K, T, r, sigma, q, side)
        diff = float(g.price) - target
        if abs(diff) < tol:
            return sigma
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        vega = float(g.vega)
        step = sigma - diff / vega if vega > 1e-12 else -1.0
        # Accept the Newton step only if it stays strictly inside the bracket.
        sigma = step if lo < step < hi else 0.5 * (lo + hi)
        if hi - lo < 1e-15:
            return sigma

    return brentq(f, SIGMA_MIN, SIGMA_MAX, xtol=1e-14, maxiter=500)


def implied_vol(price, S, K, T, r, q=0.0, kind: str = "call", tol: float = 1e-10,
                max_iter: int = 100):
    """Black–Scholes implied volatility of a European option quote.

    Accepts scalars or arrays (broadcast together).  Raises
    :class:`ArbitrageError` for quotes outside the no-arbitrage bounds.
    """
    arrays = np.broadcast_arrays(*map(np.asarray, (price, S, K, T, r, q)))
    out = np.empty(arrays[0].shape, dtype=float)
    for idx in np.ndindex(out.shape):
        args = (float(a[idx]) for a in arrays)
        out[idx] = _implied_vol_scalar(*args, kind, tol, max_iter)
    return out[()] if out.ndim == 0 else out
