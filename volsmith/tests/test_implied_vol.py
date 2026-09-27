import numpy as np
import pytest

from volsmith import bs_greeks, bs_price, implied_vol
from volsmith.implied_vol import ArbitrageError


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [40, 70, 95, 100, 105, 140, 250])
@pytest.mark.parametrize("sigma", [0.03, 0.2, 0.8, 2.5])
@pytest.mark.parametrize("T", [1 / 365, 0.25, 5.0])
def test_round_trip(kind, K, sigma, T):
    S, r, q = 100, 0.04, 0.015
    price = bs_price(S, K, T, r, sigma, q, kind)
    # Skip quotes with no identifiable time value (vega below double precision).
    if bs_greeks(S, K, T, r, sigma, q, kind).vega * sigma < 1e-9:
        pytest.skip("volatility not identifiable from this price")
    iv = implied_vol(price, S, K, T, r, q, kind)
    assert bs_price(S, K, T, r, iv, q, kind) == pytest.approx(price, abs=1e-9)
    assert iv == pytest.approx(sigma, rel=1e-6)


def test_vectorised_smile():
    K = np.linspace(60, 140, 17)
    vols = 0.2 + 0.4 * (np.log(K / 100)) ** 2
    prices = bs_price(100, K, 0.5, 0.01, vols)
    assert np.allclose(implied_vol(prices, 100, K, 0.5, 0.01), vols, atol=1e-8)


@pytest.mark.parametrize("price", [-1.0, 101.0])
def test_arbitrage_rejected(price):
    with pytest.raises(ArbitrageError):
        implied_vol(price, 100, 100, 1.0, 0.0)


def test_below_intrinsic_rejected():
    with pytest.raises(ArbitrageError):
        implied_vol(15.0, 100, 80, 1.0, 0.05)  # intrinsic ≈ 23.9
