"""
Presentation demo: causality-aware anomalies (CAROTS intuition).

Application case (process industry)
-----------------------------------
A process pump in a chemical / water plant. Operators change the speed
setpoint irregularly (batches, demand) → both signals look non-periodic.

  cause : pump speed S  [rpm]
  effect: outlet flow  F  [m^3/h]

Healthy physics (approx. linear in the operating window):
    F ≈ a + b * S + noise

Fault after the break (e.g. partially clogged filter / worn impeller /
stuck bypass valve): speed and flow each stay in their usual ranges, but
F is no longer determined by S. Univariate monitors stay quiet; only the
joint / causal residual lights up.

Time unit: minutes (N=300 → 5 hours of plant data).

Run
---
    python plot_causal_anomaly_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save only, no interactive windows
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def _ar1(n: int, phi: float, mu: float, sigma: float, rng: np.random.Generator,
         x0: float | None = None) -> np.ndarray:
    """Mean-reverting AR(1): X_t = mu + phi*(X_{t-1}-mu) + eps."""
    x = np.empty(n)
    x[0] = mu if x0 is None else x0
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(0.0, sigma)
    return x


# ---------------------------------------------------------------------------
# Synthetic plant data
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)
OUT_DIR = Path(__file__).resolve().parent / "figure" / "causal_anomaly_demo"

N = 300          # minutes
BREAK = 180      # fault starts after 3 hours
T_TIME = np.arange(N)
n_anom = N - BREAK

# Pump speed [rpm]: irregular setpoint changes, persistent but non-periodic
PHI_S, MU_S, SD_S = 0.92, 1450.0, 55.0
speed = _ar1(N, PHI_S, MU_S, SD_S, RNG)

# Healthy flow [m^3/h]: rises with speed in the operating window
# F ≈ a + b * S   (calibrated so F is roughly 20–50 m^3/h for typical rpm)
A, B = -40.0, 0.055
flow_expected = A + B * speed + RNG.normal(0.0, 0.9, size=N)
flow = flow_expected.copy()

# --- Causal break ----------------------------------------------------------
# Speed keeps the same AR(1) plant dynamics.
# Flow is replaced by an independent AR(1) matched to the pre-break marginal
# of flow — looks like normal throughput noise, but S → F is gone.
f_pre = flow[:BREAK]
mu_f = float(f_pre.mean())
phi_f = float(np.corrcoef(f_pre[:-1], f_pre[1:])[0, 1])
phi_f = float(np.clip(phi_f, 0.80, 0.97))
var_f = float(f_pre.var())
sd_f = float(np.sqrt(max(var_f * (1.0 - phi_f**2), 1e-6)))

flow_indep = _ar1(n_anom, phi_f, mu_f, sd_f, RNG, x0=float(flow[BREAK - 1]))

blend = 10
alpha = np.linspace(0.0, 1.0, blend)
f_bridge = flow[BREAK : BREAK + blend].copy()
for i, a in enumerate(alpha):
    flow[BREAK + i] = (1.0 - a) * f_bridge[i] + a * flow_indep[i]
flow[BREAK + blend :] = flow_indep[blend:]

# Counterfactual: flow if S → F still held
flow_expected[BREAK:] = A + B * speed[BREAK:] + RNG.normal(0.0, 0.9, size=n_anom)

lo = float(np.percentile(flow[:BREAK], 2))
hi = float(np.percentile(flow[:BREAK], 98))

NORMAL_SLICE = slice(0, BREAK)
ANOM_SLICE = slice(BREAK, N)
print(f"causal break at t={BREAK} min (clogged filter / impeller wear)")
print(f"corr(S,F) before={np.corrcoef(speed[:BREAK], flow[:BREAK])[0,1]:+.3f}")
print(f"corr(S,F) after ={np.corrcoef(speed[BREAK:], flow[BREAK:])[0,1]:+.3f}")
print(f"S mean/std before={speed[:BREAK].mean():.0f}/{speed[:BREAK].std():.0f}  "
      f"after={speed[BREAK:].mean():.0f}/{speed[BREAK:].std():.0f}")
print(f"F mean/std before={flow[:BREAK].mean():.1f}/{flow[:BREAK].std():.1f}  "
      f"after={flow[BREAK:].mean():.1f}/{flow[BREAK:].std():.1f}")
residual = np.abs(flow - (A + B * speed))
print(f"mean |F - E[F|S]| before={residual[:BREAK].mean():.2f}, "
      f"after={residual[BREAK:].mean():.2f}")


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved {path}")
    return path


def _shade_break(ax, color="#c0392b", alpha=0.12, label="Causal break"):
    ax.axvspan(BREAK, N - 1, color=color, alpha=alpha, label=label, zorder=0)
    ax.axvline(BREAK, color=color, ls="--", lw=1.4, alpha=0.85)


# ---------------------------------------------------------------------------
# Style A — clean dual time series, red break band
# ---------------------------------------------------------------------------

def style_a_dual_timeseries():
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor("white")

    axes[0].plot(T_TIME, speed, color="#1f4e79", lw=1.8, label="Pump speed")
    _shade_break(axes[0])
    axes[0].set_ylabel("Speed (rpm)")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].set_title("Style A — Dual time series (shared break band)", loc="left",
                      fontsize=12, pad=8)

    axes[1].plot(T_TIME, flow, color="#b85c38", lw=1.8, label="Outlet flow")
    axes[1].plot(T_TIME, flow_expected, color="#b85c38", lw=1.0, ls=":",
                 alpha=0.55, label="Expected from speed (counterfactual)")
    _shade_break(axes[1], label=None)
    axes[1].set_ylabel("Flow (m³/h)")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend(loc="upper right", frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Individually normal · causally broken  (speed → flow)",
        fontsize=13, y=1.02,
    )
    _save(fig, "A_dual_timeseries")
    return fig


# ---------------------------------------------------------------------------
# Style B — single panel, twin y-axis (compact slide)
# ---------------------------------------------------------------------------

def style_b_twin_axis():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("white")

    color_s, color_f = "#0b6e4f", "#9b2226"
    ax.plot(T_TIME, speed, color=color_s, lw=2.0, label="Pump speed")
    ax.set_ylabel("Speed (rpm)", color=color_s)
    ax.tick_params(axis="y", labelcolor=color_s)

    ax2 = ax.twinx()
    ax2.plot(T_TIME, flow, color=color_f, lw=2.0, label="Outlet flow")
    ax2.set_ylabel("Flow (m³/h)", color=color_f)
    ax2.tick_params(axis="y", labelcolor=color_f)

    _shade_break(ax, color="#9b2226", alpha=0.10)

    handles = [
        plt.Line2D([0], [0], color=color_s, lw=2, label="Pump speed"),
        plt.Line2D([0], [0], color=color_f, lw=2, label="Outlet flow"),
        Patch(facecolor="#9b2226", alpha=0.18, label="Causal break"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    ax.set_xlabel("Time (min)")
    ax.set_title("Style B — Process pump: speed → flow", loc="left")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.2)

    _save(fig, "B_twin_axis")
    return fig


# ---------------------------------------------------------------------------
# Style B2 — same as B, no break annotation
# ---------------------------------------------------------------------------

def style_b2_twin_axis_clean():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("white")

    color_s, color_f = "#0b6e4f", "#9b2226"
    ax.plot(T_TIME, speed, color=color_s, lw=2.0, label="Pump speed")
    ax.set_ylabel("Speed (rpm)", color=color_s)
    ax.tick_params(axis="y", labelcolor=color_s)

    ax2 = ax.twinx()
    ax2.plot(T_TIME, flow, color=color_f, lw=2.0, label="Outlet flow")
    ax2.set_ylabel("Flow (m³/h)", color=color_f)
    ax2.tick_params(axis="y", labelcolor=color_f)

    handles = [
        plt.Line2D([0], [0], color=color_s, lw=2, label="Pump speed"),
        plt.Line2D([0], [0], color=color_f, lw=2, label="Outlet flow"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    ax.set_xlabel("Time (min)")
    # ax.set_title("Style B2 — Twin-axis without break marker", loc="left")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.2)

    _save(fig, "B2_twin_axis_clean")
    return fig


# ---------------------------------------------------------------------------
# Style C — scatter: normal vs broken relationship
# ---------------------------------------------------------------------------

def style_c_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    coef = np.polyfit(speed[NORMAL_SLICE], flow[NORMAL_SLICE], 1)
    x_line = np.linspace(speed.min(), speed.max(), 100)
    y_line = coef[0] * x_line + coef[1]

    for ax, sl, title, color in (
        (axes[0], NORMAL_SLICE, "Before — clear speed → flow link", "#1f4e79"),
        (axes[1], ANOM_SLICE, "After — same ranges, link gone", "#c0392b"),
    ):
        ax.scatter(
            speed[sl], flow[sl],
            c=color, s=22, alpha=0.75, edgecolors="none",
        )
        ax.plot(x_line, y_line, color="#555555", ls="--", lw=1.4,
                label="Learned S → F relation")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Pump speed (rpm)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Outlet flow (m³/h)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Style C — Scatter: break only visible jointly",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "C_scatter_before_after")
    return fig


# ---------------------------------------------------------------------------
# Style D — dark slide theme
# ---------------------------------------------------------------------------

def style_d_dark():
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True,
                             gridspec_kw={"hspace": 0.1})
    bg, fg = "#12141a", "#e8eaed"
    accent = "#f4a261"
    series_s, series_f = "#4cc9f0", "#e76f51"
    fig.patch.set_facecolor(bg)

    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_color("#3a3f4b")
        ax.yaxis.label.set_color(fg)
        ax.xaxis.label.set_color(fg)
        ax.grid(True, axis="y", color="#2a2f3a", alpha=0.9)

    axes[0].plot(T_TIME, speed, color=series_s, lw=2.0)
    axes[0].set_ylabel("Speed (rpm)")
    axes[0].set_title("Style D — Dark theme (presentation-ready)",
                      loc="left", color=fg, fontsize=12)
    _shade_break(axes[0], color=accent, alpha=0.18, label="Causal break")
    axes[0].legend(loc="upper right", frameon=False, labelcolor=fg)

    axes[1].plot(T_TIME, flow, color=series_f, lw=2.0, label="Observed flow")
    axes[1].plot(T_TIME[ANOM_SLICE], flow_expected[ANOM_SLICE],
                 color=series_f, ls=":", lw=1.3, alpha=0.7,
                 label="What flow should be")
    _shade_break(axes[1], color=accent, alpha=0.18, label=None)
    axes[1].set_ylabel("Flow (m³/h)")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend(loc="upper right", frameon=False, labelcolor=fg)

    _save(fig, "D_dark_theme")
    return fig


# ---------------------------------------------------------------------------
# Style E — teaching slide: marginal OK vs causal FAIL
# ---------------------------------------------------------------------------

def style_e_teaching():
    fig = plt.figure(figsize=(11, 6.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.28)

    ax_s = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, :])

    ax_s.plot(T_TIME, speed, color="#1f4e79", lw=1.6)
    ax_s.axhline(speed[:BREAK].mean(), color="#1f4e79", ls=":", alpha=0.5)
    _shade_break(ax_s, alpha=0.10, label=None)
    ax_s.set_title("Marginal: pump speed looks unchanged", fontsize=11)
    ax_s.set_ylabel("rpm")
    ax_s.set_xlabel("Time (min)")

    ax_f.plot(T_TIME, flow, color="#b85c38", lw=1.6)
    ax_f.axhspan(lo, hi, color="#b85c38", alpha=0.08, label="Historical 2–98% band")
    _shade_break(ax_f, alpha=0.10, label=None)
    ax_f.set_title("Marginal: outlet flow looks unchanged", fontsize=11)
    ax_f.set_ylabel("m³/h")
    ax_f.set_xlabel("Time (min)")
    ax_f.legend(loc="upper right", frameon=False, fontsize=8)

    residual = np.abs(flow - (A + B * speed))
    ax_r.fill_between(T_TIME, residual, color="#6c757d", alpha=0.35, label="|F − E[F | S]|")
    ax_r.plot(T_TIME, residual, color="#212529", lw=1.4)
    _shade_break(ax_r, alpha=0.14)
    ax_r.set_title("Joint view: residual of speed → flow explodes after the break",
                   fontsize=11)
    ax_r.set_ylabel("Residual (m³/h)")
    ax_r.set_xlabel("Time (min)")
    ax_r.legend(loc="upper left", frameon=False)

    for ax in (ax_s, ax_f, ax_r):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Style E — Why causality-aware detection matters",
        fontsize=13, y=0.98,
    )
    _save(fig, "E_teaching_marginal_vs_causal")
    return fig


# ---------------------------------------------------------------------------
# Style F — minimal black & white (print / projector safe)
# ---------------------------------------------------------------------------

def style_f_bw():
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.0), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor("white")

    axes[0].plot(T_TIME, speed, color="black", lw=1.7)
    axes[0].set_ylabel("Speed (rpm)")
    axes[0].set_title("Style F — Black & white (projector / print safe)", loc="left")
    _shade_break(axes[0], color="0.55", alpha=0.25)

    axes[1].plot(T_TIME, flow, color="black", lw=1.7, label="Observed")
    axes[1].plot(T_TIME, flow_expected, color="0.45", lw=1.2, ls="--",
                 label="Expected from speed")
    _shade_break(axes[1], color="0.55", alpha=0.25, label=None)
    axes[1].set_ylabel("Flow (m³/h)")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend(loc="upper right", frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="0.85")

    _save(fig, "F_black_white")
    return fig


def main():
    builders = [
        style_a_dual_timeseries,
        style_b_twin_axis,
        style_b2_twin_axis_clean,
        style_c_scatter,
        style_d_dark,
        style_e_teaching,
        style_f_bw,
    ]
    for build in builders:
        build()
        plt.close("all")

    print(f"\nAll figures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
