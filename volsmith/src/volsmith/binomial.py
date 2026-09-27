"""Cox–Ross–Rubinstein binomial tree with Black–Scholes smoothing.

The workhorse for American options, which have no closed-form price: at every
node the holder compares the continuation value with immediate exercise.

A plain CRR tree converges slowly and in a saw-tooth pattern because the
kinked payoff falls between nodes.  We use the BBSR scheme of Broadie &
Detemple (1996):

* **BBS** — at the last time step, replace the payoff by the Black–Scholes
  price over one Δt, which smooths the kink away;
* **R** — Richardson extrapolation  2·V(n) − V(n/2)  then removes the
  leading 1/n error term.

Backward induction is vectorised over each time slice, so a 2 000-step tree
takes a few milliseconds.
"""

from __future__ import annotations

import numpy as np

from .black_scholes import _sign, bs_price


def crr_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
              kind: str = "call", american: bool = False, steps: int = 1000) -> float:
    """Price a vanilla European or American option on a smoothed CRR tree."""
    if steps < 4:
        raise ValueError("steps must be >= 4")
    fine = _bbs(S, K, T, r, sigma, q, kind, american, steps)
    coarse = _bbs(S, K, T, r, sigma, q, kind, american, steps // 2)
    return 2.0 * fine - coarse


def _bbs(S, K, T, r, sigma, q, kind, american, n):
    w = _sign(kind)
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability outside (0, 1); increase steps")

    # Nodes one step before expiry (j = number of down moves).
    spots = S * u ** (n - 1 - 2 * np.arange(n))
    values = bs_price(spots, K, dt, r, sigma, q, kind)
    if american:
        values = np.maximum(values, w * (spots - K))

    for _ in range(n - 1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            spots = spots[:-1] * d
            values = np.maximum(values, w * (spots - K))
    return float(values[0])
