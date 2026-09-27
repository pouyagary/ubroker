import pytest

from volsmith import bs_price, crr_price, mc_asian_arithmetic, mc_european
from volsmith.monte_carlo import geometric_asian_price

ARGS = dict(S=100, K=95, T=1.0, r=0.05, sigma=0.25, q=0.02)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_crr_converges_to_black_scholes(kind):
    assert crr_price(**ARGS, kind=kind, steps=2000) == pytest.approx(
        bs_price(**ARGS, kind=kind), abs=5e-5)


def test_american_put_benchmark():
    # Classic case from Longstaff & Schwartz (2001), Table 1: S=36, K=40, r=6%,
    # σ=20%, T=1.  The high-accuracy value is 4.48667 (the paper's coarse
    # finite-difference grid reports 4.478).
    assert crr_price(36, 40, 1.0, 0.06, 0.2, kind="put", american=True,
                     steps=2000) == pytest.approx(4.48667, abs=5e-5)


def test_american_call_without_dividends_equals_european():
    # Merton: never optimal to exercise an American call early if q = 0.
    a = dict(ARGS, q=0.0)
    assert crr_price(**a, american=True) == pytest.approx(crr_price(**a), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_american_at_least_european(kind):
    assert crr_price(**ARGS, kind=kind, american=True) >= crr_price(**ARGS, kind=kind) - 1e-12


@pytest.mark.parametrize("kind", ["call", "put"])
def test_monte_carlo_within_confidence_interval(kind):
    res = mc_european(**ARGS, kind=kind, n_paths=200_000, seed=42)
    exact = bs_price(**ARGS, kind=kind)
    assert abs(res.price - exact) < 4 * res.std_error
    assert res.std_error < 0.03


def test_control_variate_reduces_variance():
    plain = mc_european(**ARGS, control_variate=False, seed=1)
    cv = mc_european(**ARGS, control_variate=True, seed=1)
    assert cv.std_error < 0.75 * plain.std_error


def test_geometric_asian_closed_form_matches_simulation():
    import numpy as np
    from volsmith.monte_carlo import _normals

    S, K, T, r, s, q, n = 100, 100, 1.0, 0.05, 0.3, 0.0, 12
    rng = np.random.default_rng(0)
    z = _normals(rng, 400_000, n, True)
    dt = T / n
    logs = np.log(S) + np.cumsum((r - q - 0.5 * s**2) * dt + s * np.sqrt(dt) * z, axis=1)
    pay = np.exp(-r * T) * np.maximum(np.exp(logs.mean(axis=1)) - K, 0)
    assert pay.mean() == pytest.approx(geometric_asian_price(S, K, T, r, s, q, "call", n),
                                       abs=0.03)


def test_asian_control_variate():
    a = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    plain = mc_asian_arithmetic(**a, n_paths=50_000, control_variate=False, seed=3)
    cv = mc_asian_arithmetic(**a, n_paths=50_000, seed=3)
    assert cv.std_error < plain.std_error / 10
    assert abs(cv.price - plain.price) < 4 * plain.std_error
    # Arithmetic ≥ geometric average (AM–GM) ⇒ arithmetic Asian call ≥ geometric one.
    assert cv.price >= geometric_asian_price(**a, n_fixings=252)
    assert cv.price < bs_price(**a)  # averaging lowers volatility
