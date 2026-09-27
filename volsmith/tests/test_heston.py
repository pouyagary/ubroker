import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from volsmith import HestonParams, bs_price, calibrate_heston, heston_price, implied_vol
from volsmith.heston import char_func

S, R, Q = 100.0, 0.03, 0.01

PARAM_SETS = [
    HestonParams(0.04, 1.5, 0.04, 0.3, -0.7),     # typical equity index
    HestonParams(0.01, 0.5, 0.09, 1.0, -0.9),     # steep skew, Feller violated
    HestonParams(0.2, 5.0, 0.1, 2.0, 0.5),        # positive correlation
    HestonParams(0.09, 0.1, 0.5, 3.0, -0.99),     # extreme vol-of-vol
]


def gil_pelaez_reference(K, T, p):
    """Independent pricer: the two-probability Heston (1993) formula with adaptive quad."""
    lk = np.log(S * np.exp((R - Q) * T) / K)

    def prob(shift):
        f = lambda u: np.real(np.exp(1j * u * lk) * char_func(u - shift, T, p) / (1j * u))
        total, a = 0.0, 1e-12
        for b in (1, 5, 20, 100, 500, 2000, 1e4, 5e4):
            total += quad(f, a, b, limit=5000, epsabs=1e-15)[0]
            a = b
        return 0.5 + total / np.pi

    return S * np.exp(-Q * T) * prob(1j) - K * np.exp(-R * T) * prob(0)


@pytest.mark.parametrize("p", PARAM_SETS)
@pytest.mark.parametrize("T", [1 / 52, 0.5, 10.0])
def test_matches_independent_quadrature(p, T):
    K = np.array([60.0, 90.0, 100.0, 110.0, 150.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = [gil_pelaez_reference(k, T, p) for k in K]
    assert np.allclose(heston_price(S, K, T, R, p, Q), np.maximum(ref, 0), atol=1e-8)


def test_reduces_to_black_scholes_without_vol_of_vol():
    # The Heston–BS gap shrinks like σ²; much smaller σ loses digits to the
    # κθ/σ² cancellation in φ, which is why calibration bounds σ ≥ 0.01.
    p = HestonParams(0.04, 1.0, 0.04, 1e-4, 0.0)
    K = np.linspace(70, 130, 7)
    assert np.allclose(heston_price(S, K, 1.0, R, p, Q), bs_price(S, K, 1.0, R, 0.2, Q),
                       atol=1e-7)


def test_put_call_parity():
    p = PARAM_SETS[0]
    K = np.linspace(70, 130, 7)
    c = heston_price(S, K, 0.7, R, p, Q, "call")
    put = heston_price(S, K, 0.7, R, p, Q, "put")
    assert np.allclose(c - put, S * np.exp(-Q * 0.7) - K * np.exp(-R * 0.7), atol=1e-12)


def test_negative_correlation_produces_skew():
    p = HestonParams(0.04, 1.5, 0.04, 0.6, -0.7)
    K = np.array([80.0, 100.0, 120.0])
    iv = implied_vol(heston_price(S, K, 0.5, R, p, Q), S, K, 0.5, R, Q)
    assert iv[0] > iv[1] > iv[2]


def test_matches_monte_carlo():
    # Full-truncation Euler scheme: a completely different route to the same price.
    p = HestonParams(0.04, 2.0, 0.05, 0.4, -0.6)
    T, n_steps, n_paths = 1.0, 200, 100_000
    rng = np.random.default_rng(11)
    dt = T / n_steps
    x = np.full(n_paths, np.log(S))
    v = np.full(n_paths, p.v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        x += (R - Q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += p.kappa * (p.theta - vp) * dt + p.sigma * np.sqrt(vp * dt) * z2
    pay = np.exp(-R * T) * np.maximum(np.exp(x) - 100.0, 0.0)
    se = pay.std() / np.sqrt(n_paths)
    assert abs(pay.mean() - heston_price(S, 100.0, T, R, p, Q)) < 4 * se + 0.02


def test_calibration_recovers_parameters():
    true = HestonParams(0.035, 2.2, 0.05, 0.6, -0.72)
    K, T = np.meshgrid(np.linspace(70, 130, 13), [0.1, 0.25, 0.5, 1.0, 2.0])
    market = heston_price(S, K, T, R, true, Q)
    res = calibrate_heston(S, K, T, market, R, Q)
    assert res.success
    assert res.rmse_iv < 1e-6
    for name in ("v0", "kappa", "theta", "sigma", "rho"):
        assert getattr(res.params, name) == pytest.approx(getattr(true, name), rel=1e-3)
