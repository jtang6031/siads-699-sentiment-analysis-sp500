"""Build the report figures.

Static PNGs, because the deliverable is a written report rather than a web page — so the
interaction layer of the house data-viz method does not apply, but its form and colour rules do.

Form choices follow the data's job, not taste:
  fig 1, 2, 3, 6  emphasis   one accent series, the rest recessive grey
  fig 4           part-to-whole with a hero number
  fig 5           two-series line, the only figure needing categorical hues

Palette is the validated default. Slots 1 (blue) and 2 (orange) are the documented adjacent-safe
pair. The aqua/red pair is deliberately NOT used: the validator puts it at CVD deltaE 6.9, inside the
6-8 band that is legal only with secondary encoding, and none of these figures need two-way polarity
badly enough to spend that.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.paths import DATA_DIR, OUTPUT_DIR, REPO_ROOT  # noqa: E402

FIG_DIR = OUTPUT_DIR / "report_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- palette (validated default, light surface) ---
ACCENT = "#2a78d6"      # slot 1, blue
SECOND = "#eb6834"      # slot 2, orange
MUTED = "#b6b5ae"       # de-emphasis
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2dd"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.9,
    "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.alpha": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _finish(ax, title, subtitle=None):
    """Titles state the finding; the subtitle carries the caveat.

    Both are drawn in axes coordinates at explicit heights. set_title plus a text at y=1.02
    collides, because the title's own baseline sits at roughly y=1.0 regardless of pad.
    """
    ax.text(0, 1.135, title, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, va="bottom", ha="left")
    if subtitle:
        ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=10.5,
                color=INK_2, va="bottom", ha="left")


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path


# ---------------------------------------------------------------- market series
def market_series() -> pd.Series:
    """Equal weight of the nine sectors present for the whole window (no composition break)."""
    panel = pd.read_csv(DATA_DIR / "model_daily_panel.csv", parse_dates=["session_date"])
    wide = panel.pivot_table(index="session_date", columns="ticker", values="daily_return")
    nine = [c for c in wide.columns if c not in ("XLRE", "XLC")]
    return wide[nine].dropna(how="any").mean(axis=1)


# ---------------------------------------------------------------- fig 1
def fig_timing():
    """Emphasis bar: news lines up with today, and says nothing about tomorrow."""
    mkt = market_series()
    news = (pd.read_csv(DATA_DIR / "news_daily_df.csv", parse_dates=["session_date"])
              .set_index("session_date")["mean_event_sentiment_score"])
    joined = pd.DataFrame({"tone": news}).join(pd.DataFrame({"r": mkt}), how="inner")

    lags = list(range(-5, 6))
    corrs = []
    for k in lags:
        pair = pd.DataFrame({"tone": joined["tone"], "r": joined["r"].shift(-k)}).dropna()
        corrs.append(pair["tone"].corr(pair["r"]))

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    colors = [ACCENT if k == 0 else MUTED for k in lags]
    ax.bar(lags, corrs, color=colors, width=0.68, zorder=3)
    ax.axhline(0, color=INK_2, lw=1.0, zorder=4)

    peak = corrs[lags.index(0)]
    ax.annotate(f"same day\n{peak:+.3f}", xy=(0, peak), xytext=(0, peak + 0.035),
                ha="center", fontsize=10.5, fontweight="bold", color=INK)
    nxt = corrs[lags.index(1)]
    ax.annotate(f"next day  {nxt:+.3f}", xy=(1, nxt), xytext=(2.6, peak * 0.55),
                fontsize=10.5, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.1,
                                connectionstyle="arc3,rad=-0.25"))

    ax.set_xticks(lags)
    ax.set_xticklabels(["-5", "-4", "-3", "-2", "-1", "today", "+1", "+2", "+3", "+4", "+5"])
    ax.set_xlabel("trading day, relative to when the news was published")
    ax.set_ylabel("how strongly news lines up with market movement")
    ax.set_ylim(min(corrs) - 0.03, peak + 0.075)
    _finish(ax, "News matches the day it arrives, not the day after",
            "Daily news tone vs market movement, 2015-2025. Bars right of centre are the future.")
    return _save(fig, "fig1_timing.png")


# ---------------------------------------------------------------- fig 2
def fig_three_windows():
    """Emphasis bar: the signal exists in one window and nowhere else."""
    rows = [("Overnight\n(close to next open)", 2.95836, True),
            ("During trading hours\n(open to close)", -0.164613, False),
            ("Full day\n(close to close)", -0.0725849, False)]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [ACCENT if r[2] else MUTED for r in rows]

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.bar(range(3), vals, color=colors, width=0.55, zorder=3)
    ax.axhline(0, color=INK_2, lw=1.0, zorder=4)
    ax.axhline(2.0, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(2.46, 2.08, "threshold for a credible result", fontsize=9.5, color=INK_2,
            ha="right", va="bottom")

    for i, (lab, v, hit) in enumerate(rows):
        ax.text(i, v + (0.13 if v >= 0 else -0.30), f"{v:+.2f}", ha="center",
                fontsize=11, fontweight="bold" if hit else "normal",
                color=INK if hit else INK_2)
    # sits to the right of the bar; centred over it would be hidden behind the fill
    ax.text(0.42, 2.55, "the only window\nthat works", ha="left", va="center",
            fontsize=11, fontweight="bold", color=ACCENT)

    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("strength of prediction")
    ax.set_ylim(-0.9, 3.35)
    _finish(ax, "The signal exists overnight, and only overnight",
            "Sector prediction model, seven independent yearly tests, 2019-2025.")
    return _save(fig, "fig2_three_windows.png")


# ---------------------------------------------------------------- fig 3
def fig_collapsed():
    """Dumbbell: before -> after per finding, when tested on more data."""
    old = pd.read_csv(OUTPUT_DIR / "news_flow_magnitude_metrics.csv")
    new = pd.read_csv(OUTPUT_DIR / "grid_metrics_2015_2026.csv")
    m = old.merge(new, on="spec", suffixes=("_old", "_new"))

    pick = {"1. RavenPack news flow": "News volume predicts\nhow big the move is",
            "3. FinBERT tone": "AI sentiment predicts\nhow big the move is"}
    rows = []
    for spec, label in pick.items():
        r = m.loc[m["spec"] == spec].iloc[0]
        rows.append((label, r["auc_with_news_old"], r["auc_with_news_new"]))
    # the teammate's monthly finding, rescaled onto the same 0.5-centred axis for comparability
    rows.append(("Monthly sector model\n(deep learning)", 0.5 + 0.0508 / 2, 0.5 + 0.0041 / 2))

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ys = np.arange(len(rows))
    for y, (_, a, b) in zip(ys, rows):
        ax.plot([a, b], [y, y], color=MUTED, lw=2.4, zorder=2, solid_capstyle="round")
        ax.scatter([a], [y], s=110, color=ACCENT, zorder=4, edgecolor=SURFACE, linewidth=2)
        ax.scatter([b], [y], s=110, color=MUTED, zorder=4, edgecolor=SURFACE, linewidth=2)
        ax.text(a + 0.0009, y - 0.02, f"{a:.3f}", ha="left", va="center", fontsize=10.5,
                color=INK, fontweight="bold")
        ax.text(b - 0.0009, y - 0.02, f"{b:.3f}", ha="right", va="center", fontsize=10.5,
                color=INK_2)

    ax.axvline(0.5, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.5, -0.52, "coin flip", fontsize=9.5, color=INK_2, ha="center", va="bottom")

    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    # room for the value labels sitting outside the leftmost and rightmost dots
    lo = min(min(r[1], r[2]) for r in rows)
    hi = max(max(r[1], r[2]) for r in rows)
    ax.set_xlim(lo - 0.0055, hi + 0.0040)
    ax.set_xlabel("how often the model was right (0.500 = a coin flip)")
    ax.set_ylim(-0.75, len(rows) - 0.35)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=ACCENT, label="tested on 6 years"),
               plt.Line2D([], [], marker="o", ls="", ms=10, color=MUTED, label="tested on 11 years")]
    ax.legend(handles=handles, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.16), fontsize=10.5, labelcolor=INK_2)
    _finish(ax, "Three findings that looked real — and did not survive more data",
            "Each was convincing on the shorter sample. All three collapsed when re-tested.")
    return _save(fig, "fig3_collapsed.png")


# ---------------------------------------------------------------- fig 4
def fig_overnight_split():
    """Part-to-whole with a hero number: most of the move is untradeable."""
    panel = pd.read_csv(DATA_DIR / "model_daily_panel.csv", parse_dates=["session_date"])
    d = panel.dropna(subset=["overnight_gap", "open_to_close"])
    gap, intra = d["overnight_gap"].abs().mean(), d["open_to_close"].abs().mean()
    share = gap / (gap + intra)

    fig, ax = plt.subplots(figsize=(10.2, 2.9))
    ax.barh([0], [share], color=ACCENT, height=0.42, zorder=3)
    ax.barh([0], [1 - share], left=[share + 0.004], color=MUTED, height=0.42, zorder=3)

    ax.text(share / 2, 0, f"{share:.1%}", ha="center", va="center",
            fontsize=15, fontweight="bold", color=SURFACE)
    ax.text(share + (1 - share) / 2, 0, f"{1-share:.1%}", ha="center", va="center",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(share / 2, -0.42, "OVERNIGHT\nmarkets closed — cannot trade",
            ha="center", va="top", fontsize=10.5, color=ACCENT, fontweight="bold")
    ax.text(share + (1 - share) / 2, -0.42, "DURING TRADING HOURS\ncan trade",
            ha="center", va="top", fontsize=10.5, color=INK_2)

    ax.set_xlim(0, 1); ax.set_ylim(-1.15, 0.5)
    ax.set_yticks([]); ax.set_xticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    _finish(ax, "Two-fifths of a typical day's move happens before anyone can trade",
            "Average absolute price move, split by when it occurs. 11 sectors, 2015-2025.")
    return _save(fig, "fig4_overnight_split.png")


# ---------------------------------------------------------------- fig 5
def fig_scorer_agreement():
    """Rolling agreement, not raw levels.

    Plotting the two sentiment series together is misleading: both drift upward across the decade,
    so a whole-period correlation of 0.70 mostly measures that shared trend, and the two lines sit
    at visibly different levels before 2020 (gap +0.075 in 2015-17, closing to -0.012 by 2021-25).
    A reader sees the offset and concludes they disagree. The honest quantity is the rolling
    one-year correlation, which is level-free and shows how the agreement actually behaves.
    """
    rp = (pd.read_csv(DATA_DIR / "news_daily_df.csv", parse_dates=["session_date"])
            .set_index("session_date")["mean_event_sentiment_score"])
    fb = (pd.read_csv(DATA_DIR / "finbert_daily_df.csv", parse_dates=["session_date"])
            .set_index("session_date")["fb_mean_sentiment"])
    both = pd.DataFrame({"rp": rp, "fb": fb}).dropna()
    weekly = both.resample("W").mean()
    rolling = weekly["rp"].rolling(52).corr(weekly["fb"]).dropna()
    median = rolling.median()

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    ax.fill_between(rolling.index, 0, rolling.values, color=ACCENT, alpha=0.16, zorder=2)
    ax.plot(rolling.index, rolling.values, color=ACCENT, lw=2.0, zorder=4)
    ax.axhline(0, color=INK_2, lw=1.0, zorder=3)
    ax.axhline(median, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.text(rolling.index[2], median + 0.022, f"typical agreement {median:.2f}",
            fontsize=10.5, color=INK_2, va="bottom")

    ax.set_ylabel("agreement over the previous year\n(1.0 = identical, 0 = unrelated)")
    ax.set_ylim(-0.05, 1.0)
    _finish(ax, "The free model and the paid score agree — moderately, and not always",
            "Rolling one-year agreement between FinBERT and the commercial score. "
            "Neither can see the other.")
    return _save(fig, "fig5_scorer_agreement.png")


# ---------------------------------------------------------------- fig 6
def fig_control():
    """Emphasis: the pipeline recovers a known effect, so its nulls mean something."""
    res = pd.read_csv(OUTPUT_DIR / "grid_metrics_2015_2026.csv")
    label = {"1. RavenPack news flow": "News volume alone",
             "2. RavenPack tone": "News tone alone",
             "3. FinBERT tone": "AI sentiment alone",
             "4. Trailing volatility (control)": "Recent volatility\n(known to work)",
             "5. Flow | vol + trailing flow": "Volatility + news volume",
             "6. RavenPack news -> direction": "News, predicting direction"}
    res = res[res["spec"].isin(label)].copy()
    res["name"] = res["spec"].map(label)
    res = res.sort_values("auc_with_news")

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ys = np.arange(len(res))
    colors = [ACCENT if c else MUTED for c in res["clears_both"]]
    ax.barh(ys, res["auc_with_news"] - 0.5, left=0.5, color=colors, height=0.6, zorder=3)
    ax.errorbar(res["auc_with_news"], ys,
                xerr=[res["auc_with_news"] - res["auc_ci_low"],
                      res["auc_ci_high"] - res["auc_with_news"]],
                fmt="none", ecolor=INK_2, elinewidth=1.3, capsize=3, zorder=5)
    ax.axvline(0.5, color=INK_2, lw=1.1, zorder=4)

    for y, (_, r) in zip(ys, res.iterrows()):
        ax.text(r["auc_ci_high"] + 0.006, y, f"{r['auc_with_news']:.3f}", va="center",
                fontsize=10.5, color=INK if r["clears_both"] else INK_2,
                fontweight="bold" if r["clears_both"] else "normal")

    ax.set_yticks(ys); ax.set_yticklabels(res["name"], fontsize=10.5)
    ax.set_xlabel("how often the model was right  (0.500 = a coin flip)")
    ax.set_xlim(0.44, max(0.62, res["auc_ci_high"].max() + 0.03))
    ax.grid(axis="y", visible=False)
    # below the axes: inside the plot it lands on the bottom bar
    ax.text(0, -0.185, "blue = survived every test  ·  bars show the range of uncertainty",
            transform=ax.transAxes, fontsize=10, color=INK_2, va="top")
    _finish(ax, "The pipeline finds a real effect when one exists — so its blanks are informative",
            "Same machinery, same data, 2,206 out-of-sample days.")
    return _save(fig, "fig6_control.png")


if __name__ == "__main__":
    print("Building report figures")
    for fn in (fig_timing, fig_three_windows, fig_collapsed,
               fig_overnight_split, fig_scorer_agreement, fig_control):
        fn()
    print(f"\nAll figures in {FIG_DIR.relative_to(REPO_ROOT)}")
