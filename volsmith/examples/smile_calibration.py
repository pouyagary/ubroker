"""Calibrate Heston to a noisy synthetic option market and plot the fit.

    python examples/smile_calibration.py            # writes docs/smile_fit.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from volsmith import HestonParams, bs_price, calibrate_heston, heston_price, implied_vol

S, R, Q = 100.0, 0.03, 0.01
TRUE = HestonParams(v0=0.03, kappa=2.0, theta=0.045, sigma=0.7, rho=-0.75)
MATURITIES = [0.1, 0.5, 2.0]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1–3

# 1. A "market": Heston implied vols plus bid/ask noise of ±0.3 vol points.
rng = np.random.default_rng(3)
K, T = np.meshgrid(np.linspace(70, 130, 15), MATURITIES)
market_iv = implied_vol(heston_price(S, K, T, R, TRUE, Q), S, K, T, R, Q)
market_iv += rng.normal(0.0, 0.003, market_iv.shape)
market_px = bs_price(S, K, T, R, market_iv, Q)

# 2. Calibrate.
res = calibrate_heston(S, K, T, market_px, R, Q)
print(res)

# 3. Plot quotes (dots) against the fitted model smile (lines).
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
fine = np.linspace(68, 132, 200)
for t, iv, color in zip(MATURITIES, market_iv, COLORS):
    model = implied_vol(heston_price(S, fine, t, R, res.params, Q), S, fine, t, R, Q)
    label = f"T = {t:g}y"
    ax.plot(fine, model * 100, color=color, lw=2, label=label)
    ax.scatter(K[0], iv * 100, s=28, color=color, edgecolor="white", linewidth=1.2, zorder=3)

ax.set_title("Heston fit to the volatility smile  ·  dots = market quotes, lines = model",
             loc="left", fontsize=11, color="#0b0b0b")
ax.set_xlabel("Strike", color="#52514e")
ax.set_ylabel("Implied volatility (%)", color="#52514e")
ax.grid(axis="y", color="#e6e5e0", lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#c3c2b7")
ax.tick_params(colors="#52514e")
ax.set_xlim(66, 134)
ax.legend(frameon=False, loc="upper right", fontsize=9)
p = res.params
fig.text(0.015, 0.015,
        f"fitted: v0={p.v0:.3f}  κ={p.kappa:.2f}  θ={p.theta:.3f}  σ={p.sigma:.2f}  "
        f"ρ={p.rho:.2f}   ·   IV RMSE {res.rmse_iv:.2%}",
        fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 1))

out = Path(__file__).resolve().parent.parent / "docs" / "smile_fit.png"
fig.savefig(out, facecolor="white")
print(f"saved {out}")
