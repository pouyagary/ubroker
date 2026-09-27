"""Variance-reduced Monte Carlo under geometric Brownian motion.

Two classic variance-reduction techniques are combined:

* **Antithetic variates** — every normal draw Z is paired with -Z.
* **Control variates** — a correlated payoff with a known expectation is used
  to cancel noise.  For European options the control is the discounted
  terminal stock price (E = S e^{-qT}); for arithmetic Asian options it is the
  *geometric* Asian option, whose price is known in closed form (Kemna–Vorst).

Every result carries its standard error, so the answer comes with an honest
95 % confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import _sign, bs_price


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        half = 1.959963984540054 * self.std_error
        return self.price - half, self.price + half


def _normals(rng, n_paths, n_steps, antithetic):
    if antithetic:
        z = rng.standard_normal((n_paths // 2, n_steps))
        return np.concatenate([z, -z])
    return rng.standard_normal((n_paths, n_steps))


def _controlled(y, x, x_mean, antithetic):
    """Optimal control-variate estimator  Y - β (X - E[X]).

    With antithetic sampling the draws come in dependent pairs, so the
    standard error is computed from the pair averages, which are i.i.d.
    """
    cov = np.cov(y, x)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    adj = y - beta * (x - x_mean)
    if antithetic:
        half = adj.size // 2
        adj = 0.5 * (adj[:half] + adj[half:])
    return float(adj.mean()), float(adj.std(ddof=1) / np.sqrt(adj.size))


def mc_european(S, K, T, r, sigma, q=0.0, kind="call", n_paths=200_000,
                antithetic=True, control_variate=True, seed=None) -> MCResult:
    """European option by Monte Carlo (exact one-step GBM simulation)."""
    w = _sign(kind)
    rng = np.random.default_rng(seed)
    z = _normals(rng, n_paths, 1, antithetic)[:, 0]
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    disc = np.exp(-r * T)
    y = disc * np.maximum(w * (ST - K), 0.0)
    if control_variate:
        price, se = _controlled(y, disc * ST, S * np.exp(-q * T), antithetic)
    else:
        price, se = _controlled(y, np.zeros_like(y), 0.0, antithetic)
    return MCResult(price, se, z.size)


def geometric_asian_price(S, K, T, r, sigma, q=0.0, kind="call", n_fixings=252):
    """Closed-form price of a discretely monitored geometric-average Asian option.

    The geometric average of log-normal fixings is itself log-normal, so it
    prices as a Black–Scholes option with adjusted volatility and carry.
    """
    n = n_fixings
    dt = T / n
    t = dt * np.arange(1, n + 1)
    mu = r - q - 0.5 * sigma**2
    mean_log = np.log(S) + mu * t.mean()
    var_log = sigma**2 * dt * (n + 1) * (2 * n + 1) / (6 * n)
    fwd = np.exp(mean_log + 0.5 * var_log)
    sig_eff = np.sqrt(var_log / T)
    # Price as BS with spot fwd·e^{-rT}, no dividends, volatility sig_eff.
    return float(bs_price(fwd * np.exp(-r * T), K, T, r, sig_eff, 0.0, kind))


def mc_asian_arithmetic(S, K, T, r, sigma, q=0.0, kind="call", n_fixings=252,
                        n_paths=100_000, antithetic=True, control_variate=True,
                        seed=None) -> MCResult:
    """Arithmetic-average Asian option — no closed form exists.

    With the geometric-Asian control variate the standard error typically
    falls by two orders of magnitude for the same number of paths.
    """
    w = _sign(kind)
    rng = np.random.default_rng(seed)
    dt = T / n_fixings
    z = _normals(rng, n_paths, n_fixings, antithetic)
    log_paths = np.log(S) + np.cumsum(
        (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1
    )
    disc = np.exp(-r * T)
    arith = np.exp(log_paths).mean(axis=1)
    y = disc * np.maximum(w * (arith - K), 0.0)
    if control_variate:
        geo = np.exp(log_paths.mean(axis=1))
        x = disc * np.maximum(w * (geo - K), 0.0)
        x_mean = geometric_asian_price(S, K, T, r, sigma, q, kind, n_fixings)
        price, se = _controlled(y, x, x_mean, antithetic)
    else:
        price, se = _controlled(y, np.zeros_like(y), 0.0, antithetic)
    return MCResult(price, se, z.shape[0])
