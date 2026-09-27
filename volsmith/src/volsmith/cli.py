"""Command-line interface:  ``volsmith price | iv | asian | smile | calibrate``."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .binomial import crr_price
from .black_scholes import bs_greeks, bs_price
from .heston import HestonParams, calibrate_heston, heston_price
from .implied_vol import ArbitrageError, implied_vol
from .monte_carlo import mc_asian_arithmetic, mc_european


def _market_args(p: argparse.ArgumentParser, vol: bool = True) -> None:
    p.add_argument("-S", "--spot", type=float, required=True, help="spot price")
    p.add_argument("-K", "--strike", type=float, required=True, help="strike price")
    p.add_argument("-T", "--expiry", type=float, required=True, help="time to expiry, years")
    p.add_argument("-r", "--rate", type=float, default=0.0, help="risk-free rate (0.05 = 5%%)")
    p.add_argument("-q", "--div", type=float, default=0.0, help="dividend yield")
    if vol:
        p.add_argument("-v", "--vol", type=float, required=True, help="volatility (0.2 = 20%%)")
    p.add_argument("--put", action="store_true", help="price a put instead of a call")


def _heston_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--v0", type=float, default=0.04, help="initial variance")
    p.add_argument("--kappa", type=float, default=1.5, help="mean-reversion speed")
    p.add_argument("--theta", type=float, default=0.04, help="long-run variance")
    p.add_argument("--sigma", type=float, default=0.6, help="vol of variance")
    p.add_argument("--rho", type=float, default=-0.7, help="spot/variance correlation")


def _kind(a) -> str:
    return "put" if a.put else "call"


def _row(label: str, value: str) -> None:
    print(f"  {label:<26}{value}")


def cmd_price(a) -> None:
    kind = _kind(a)
    args = (a.spot, a.strike, a.expiry, a.rate, a.vol, a.div, kind)
    g = bs_greeks(*args)
    mc = mc_european(*args, n_paths=a.paths, seed=a.seed)
    lo, hi = mc.ci95

    print(f"\n European {kind}  S={a.spot:g}  K={a.strike:g}  T={a.expiry:g}y  "
          f"r={a.rate:.2%}  q={a.div:.2%}  σ={a.vol:.2%}\n")
    _row("Black–Scholes", f"{g.price:.6f}")
    _row("Binomial (CRR, 2000)", f"{crr_price(*args, steps=2000):.6f}")
    _row(f"Monte Carlo ({mc.n_paths:,})", f"{mc.price:.6f}  95% CI [{lo:.4f}, {hi:.4f}]")
    amer = crr_price(*args, american=True, steps=2000)
    _row("American (CRR, 2000)", f"{amer:.6f}  early-exercise premium {amer - g.price:.6f}")
    print("\n Greeks")
    _row("Δ delta", f"{g.delta:+.6f}")
    _row("Γ gamma", f"{g.gamma:+.6f}")
    _row("ν vega  (per 1 vol pt)", f"{g.vega / 100:+.6f}")
    _row("Θ theta (per day)", f"{g.theta / 365:+.6f}")
    _row("ρ rho   (per 1% rate)", f"{g.rho / 100:+.6f}")
    _row("vanna", f"{g.vanna:+.6f}")
    _row("volga", f"{g.volga:+.6f}")
    print()


def cmd_iv(a) -> None:
    try:
        iv = implied_vol(a.price, a.spot, a.strike, a.expiry, a.rate, a.div, _kind(a))
    except ArbitrageError as e:
        sys.exit(f"error: {e}")
    print(f"implied volatility: {iv:.6%}")


def cmd_asian(a) -> None:
    kind = _kind(a)
    args = (a.spot, a.strike, a.expiry, a.rate, a.vol, a.div, kind, a.fixings)
    plain = mc_asian_arithmetic(*args, n_paths=a.paths, control_variate=False, seed=a.seed)
    cv = mc_asian_arithmetic(*args, n_paths=a.paths, seed=a.seed)
    print(f"\n Arithmetic Asian {kind}, {a.fixings} fixings, {cv.n_paths:,} paths\n")
    _row("antithetic only", f"{plain.price:.6f} ± {plain.std_error:.6f}")
    _row("+ geometric control", f"{cv.price:.6f} ± {cv.std_error:.6f}")
    ratio = (plain.std_error / cv.std_error) ** 2 if cv.std_error > 0 else float("inf")
    print(f"\n  variance reduction ×{ratio:,.0f}  (same accuracy with {ratio:,.0f}× fewer paths)\n")


def _smile_table(S, T, r, q, params, strikes):
    prices = heston_price(S, strikes, T, r, params, q)
    return implied_vol(prices, S, strikes, T, r, q)


def cmd_smile(a) -> None:
    params = HestonParams(a.v0, a.kappa, a.theta, a.sigma, a.rho)
    strikes = a.spot * np.linspace(0.7, 1.3, 13)
    ivs = _smile_table(a.spot, a.expiry, a.rate, a.div, params, strikes)
    print(f"\n Heston implied-volatility smile, T={a.expiry:g}y   {params}\n")
    lo, hi = ivs.min(), ivs.max()
    for k, v in zip(strikes, ivs):
        bar = "█" * int(1 + 40 * (v - lo) / max(hi - lo, 1e-12))
        print(f"  K={k:8.2f}   {v:7.2%}  {bar}")
    print(f"\n  Feller condition 2κθ > σ²: {'yes' if params.feller_satisfied else 'no'}\n")


def cmd_calibrate(a) -> None:
    """Round-trip demo: build a noisy synthetic market, then recover the model."""
    rng = np.random.default_rng(a.seed)
    true = HestonParams(a.v0, a.kappa, a.theta, a.sigma, a.rho)
    S, r, q = 100.0, 0.03, 0.01
    K, T = np.meshgrid(np.linspace(75, 125, 11), [0.1, 0.25, 0.5, 1.0, 2.0])
    iv = implied_vol(heston_price(S, K, T, r, true, q), S, K, T, r, q)
    noisy_iv = iv + rng.normal(0.0, a.noise, iv.shape)  # bid/ask noise, in vol points
    market = bs_price(S, K, T, r, noisy_iv, q)

    t0 = time.perf_counter()
    res = calibrate_heston(S, K, T, market, r, q)
    dt = time.perf_counter() - t0

    print(f"\n Calibrating Heston to {K.size} quotes (IV noise {a.noise:.2%})\n")
    print(f"  {'':8}{'true':>10}{'fitted':>10}")
    for name in ("v0", "kappa", "theta", "sigma", "rho"):
        print(f"  {name:8}{getattr(true, name):10.4f}{getattr(res.params, name):10.4f}")
    print(f"\n  IV RMSE {res.rmse_iv:.4%}   {res.n_evals} evaluations   {dt:.2f}s\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="volsmith", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("price", help="price a vanilla option with every engine + Greeks")
    _market_args(p)
    p.add_argument("--paths", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_price)

    p = sub.add_parser("iv", help="implied volatility from an option price")
    _market_args(p, vol=False)
    p.add_argument("-p", "--price", type=float, required=True, help="option premium")
    p.set_defaults(func=cmd_iv)

    p = sub.add_parser("asian", help="arithmetic Asian option via Monte Carlo")
    _market_args(p)
    p.add_argument("--fixings", type=int, default=252)
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_asian)

    p = sub.add_parser("smile", help="implied-vol smile generated by Heston")
    p.add_argument("-S", "--spot", type=float, default=100.0)
    p.add_argument("-T", "--expiry", type=float, default=0.5)
    p.add_argument("-r", "--rate", type=float, default=0.03)
    p.add_argument("-q", "--div", type=float, default=0.0)
    _heston_args(p)
    p.set_defaults(func=cmd_smile)

    p = sub.add_parser("calibrate", help="calibrate Heston to a noisy synthetic market")
    _heston_args(p)
    p.add_argument("--noise", type=float, default=0.002, help="IV noise std-dev")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_calibrate)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
