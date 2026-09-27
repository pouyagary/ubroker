"""Heston (1993) stochastic-volatility model: pricing and smile calibration.

Black–Scholes assumes one constant volatility, yet real option markets quote
a different implied volatility for every strike and maturity — the *smile*.
Heston lets the variance follow its own mean-reverting process that is
correlated with the stock:

    dS_t = (r - q) S_t dt + √v_t S_t dW₁
    dv_t = κ(θ - v_t) dt + σ √v_t dW₂,        d⟨W₁, W₂⟩ = ρ dt

Negative ρ produces the downward skew seen in equity markets and σ (the
volatility of volatility) produces the curvature.  European prices follow from
the characteristic function φ by Fourier inversion (Lewis, 2001).

We use the "little Heston trap" form of φ (Albrecher et al., 2007), which
avoids the branch-cut discontinuity of the complex logarithm, and price a
whole strike strip in one vectorised pass — fast enough to sit inside a
least-squares calibration loop.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
from scipy.optimize import least_squares

from .black_scholes import _sign, bs_greeks
from .implied_vol import implied_vol


@dataclass(frozen=True)
class HestonParams:
    v0: float     # initial variance
    kappa: float  # speed of mean reversion
    theta: float  # long-run variance
    sigma: float  # volatility of variance
    rho: float    # spot/variance correlation

    @property
    def feller_satisfied(self) -> bool:
        """2κθ > σ² keeps the variance process strictly positive."""
        return 2.0 * self.kappa * self.theta > self.sigma**2


def char_func(u, T, p: HestonParams):
    """Characteristic function of the log-forward return  x = ln(S_T / F_T).

    ``u`` may be complex.  Written in the "little Heston trap" form, which
    stays on the principal branch of the complex logarithm for all T.
    """
    v0, kappa, theta, sigma, rho = astuple(p)
    iu = 1j * u
    b = kappa - rho * sigma * iu
    d = np.sqrt(b * b + sigma**2 * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / sigma**2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (b - d) / sigma**2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(C + D * v0)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_PANEL_MAX = 8.0     # widest Gauss–Legendre panel (in u)
_U_CAP = 2.0e4       # hard ceiling on the integration range


def _lewis_integrand(u, k, T, p):
    """Re[e^{iuk} φ(u - i/2)] / (u² + ¼)  for every strike (rows) and node (cols)."""
    phi = char_func(u - 0.5j, T, p)
    return np.real(np.exp(1j * np.outer(k, u)) * phi) / (u * u + 0.25)


def _upper_limit(T, p, tol):
    """Smallest U (doubling from 16) beyond which the integrand is below ``tol``."""
    U = 16.0
    while U < _U_CAP and abs(char_func(U - 0.5j, T, p)) / (U * U) > tol:
        U *= 2.0
    return min(2.0 * U, _U_CAP)  # safety factor: |φ| need not be monotone


def _heston_call_strip(S, K, T, r, q, p: HestonParams, tol=1e-13):
    """Heston call prices for a strip of strikes sharing one maturity.

    Lewis (2001) single-integral representation:

        C = S e^{-qT} - √(S K) e^{-(r+q)T/2} / π
                        · ∫₀^∞ Re[e^{iuk} φ(u - i/2)] / (u² + ¼) du,
        k = ln(F / K)

    Evaluating φ on the line Im u = -½ keeps it bounded, and the 1/(u² + ¼)
    factor gives the integrand a fast algebraic decay on top of φ's own decay,
    so it is smooth, non-singular and easy to integrate.  We use composite
    24-point Gauss–Legendre panels whose width adapts to the oscillation
    frequency |k| and an upper limit chosen from the decay of |φ|.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = S * np.exp((r - q) * T)
    k = np.log(F / K)

    U = _upper_limit(T, p, tol)
    width = min(_PANEL_MAX, np.pi / max(float(np.max(np.abs(k))), 1e-3))
    # The 1/(u² + ¼) factor has poles at ±i/2, so resolve [0, 4] with
    # geometrically refined panels before switching to the uniform width.
    near = np.array([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    n_panels = max(int(np.ceil((U - near[-1]) / width)), 1)
    edges = np.concatenate([near, np.linspace(near[-1], U, n_panels + 1)[1:]])
    half = 0.5 * np.diff(edges)
    u = ((edges[:-1] + half)[:, None] + half[:, None] * _GL_X[None, :]).ravel()
    w = (half[:, None] * _GL_W[None, :]).ravel()

    integral = _lewis_integrand(u, k, T, p) @ w
    call = S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-0.5 * (r + q) * T) / np.pi * integral
    # Clamp to the no-arbitrage bounds to remove quadrature round-off.
    lower = np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    return np.clip(call, lower, S * np.exp(-q * T))


def heston_price(S, K, T, r, params: HestonParams, q=0.0, kind: str = "call"):
    """European option price under Heston.  ``K`` and ``T`` may be arrays."""
    w = _sign(kind)
    K, T = np.broadcast_arrays(np.asarray(K, dtype=float), np.asarray(T, dtype=float))
    out = np.empty(K.shape)
    for t in np.unique(T):
        mask = T == t
        call = _heston_call_strip(S, K[mask], t, r, q, params)
        out[mask] = call if w == 1 else call - S * np.exp(-q * t) + K[mask] * np.exp(-r * t)
    return out[()] if out.ndim == 0 else out


@dataclass(frozen=True)
class CalibrationResult:
    params: HestonParams
    rmse_iv: float   # root-mean-square implied-volatility error
    n_evals: int
    success: bool


_LOWER = np.array([1e-4, 1e-2, 1e-4, 1e-2, -0.999])
_UPPER = np.array([2.0, 20.0, 2.0, 5.0, 0.999])


def calibrate_heston(S, strikes, maturities, market_prices, r, q=0.0,
                     kind: str = "call", initial: HestonParams | None = None) -> CalibrationResult:
    """Fit Heston parameters to a set of option quotes.

    Price errors are divided by Black–Scholes vega, which to first order turns
    them into implied-volatility errors: every quote then counts equally,
    whether it is a cheap wing option or an expensive at-the-money one.
    """
    K, T, P = np.broadcast_arrays(*(np.asarray(a, dtype=float)
                                    for a in (strikes, maturities, market_prices)))
    K, T, P = K.ravel(), T.ravel(), P.ravel()

    market_iv = implied_vol(P, S, K, T, r, q, kind)
    vega = np.maximum(np.asarray(bs_greeks(S, K, T, r, market_iv, q, kind).vega), 1e-8)

    if initial is None:
        atm_var = float(np.median(market_iv) ** 2)
        initial = HestonParams(atm_var, 2.0, atm_var, 0.5, -0.5)
    x0 = np.clip(np.array(astuple(initial)), _LOWER + 1e-6, _UPPER - 1e-6)

    def residuals(x):
        model = heston_price(S, K, T, r, HestonParams(*x), q, kind)
        return (model - P) / vega

    sol = least_squares(residuals, x0, bounds=(_LOWER, _UPPER), method="trf",
                        x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
    params = HestonParams(*map(float, sol.x))
    model_iv = implied_vol(heston_price(S, K, T, r, params, q, kind), S, K, T, r, q, kind)
    rmse = float(np.sqrt(np.mean((model_iv - market_iv) ** 2)))
    return CalibrationResult(params, rmse, int(sol.nfev), bool(sol.success))
