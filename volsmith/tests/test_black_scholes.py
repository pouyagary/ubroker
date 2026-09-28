import numpy as np
import pytest

from volsmith import bs_greeks, bs_price


def test_textbook_value():
    # Hull, "Options, Futures and Other Derivatives", Example 15.6
    assert bs_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.7594, abs=1e-4)
    assert bs_price(42, 40, 0.5, 0.10, 0.20, kind="put") == pytest.approx(0.8086, abs=1e-4)


@pytest.mark.parametrize("K", [60, 90, 100, 110, 160])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(K, q):
    S, T, r, v = 100, 0.75, 0.04, 0.3
    c = bs_price(S, K, T, r, v, q, "call")
    p = bs_price(S, K, T, r, v, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-12)


def test_vectorised():
    K = np.array([80.0, 100.0, 120.0])
    prices = bs_price(100, K, 1.0, 0.02, 0.25)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)  # calls decrease in strike


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    S, K, T, r, v, q = 100, 105, 0.8, 0.03, 0.22, 0.01
    g = bs_greeks(S, K, T, r, v, q, kind)
    h = 1e-4

    def f(**kw):
        a = dict(S=S, K=K, T=T, r=r, sigma=v, q=q)
        a.update(kw)
        return bs_price(a["S"], a["K"], a["T"], a["r"], a["sigma"], a["q"], kind)

    assert g.delta == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g.gamma == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-4)
    assert g.vega == pytest.approx((f(sigma=v + h) - f(sigma=v - h)) / (2 * h), rel=1e-6)
    assert g.theta == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-6)
    assert g.rho == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-6)

    def delta(sig):
        return bs_greeks(S, K, T, r, sig, q, kind).delta

    def vega(sig):
        return bs_greeks(S, K, T, r, sig, q, kind).vega

    assert g.vanna == pytest.approx((delta(v + h) - delta(v - h)) / (2 * h), rel=1e-5)
    assert g.volga == pytest.approx((vega(v + h) - vega(v - h)) / (2 * h), rel=1e-5)


def test_bad_kind():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0, 0.2, kind="straddle")
