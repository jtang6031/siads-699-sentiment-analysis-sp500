#!/usr/bin/env python
"""Clean the pulled data into a model-ready table, then train and evaluate.

    python run_training.py --list             # show stages and current inputs
    python run_training.py --stage clean      # raw -> model_inputs.csv
    python run_training.py --stage train      # model_inputs.csv -> metrics + figure
    python run_training.py                    # both

Run this after run_pipeline.py. The two are separate because cleaning and training iterate on
different clocks: the data pull takes hours and changes rarely, while the grid gets re-run whenever
a spec or a control changes.

The transforms live in ml/cleaning_lib.py and the harness in ml/model_lib.py, both unit-tested, so
this file is orchestration only. Notebook 11 imports the same harness -- one implementation, so a
number cannot move because the measurement quietly drifted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml import cleaning_lib as clean          # noqa: E402
from ml import model_lib as model             # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data_collection"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = REPO_ROOT / "model_outputs"

ELEVEN = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
NINE = tuple(t for t in ELEVEN if t not in clean.LATE_LAUNCH_TICKERS)


def sample_tag() -> str:
    """Read the window out of notebook 01 so this file cannot disagree with the pull."""
    nb = json.loads((DATA_DIR / "01_ravenpack_news_extraction.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    start = re.search(r'START_DATE\s*=\s*"(\d{4})', src)
    end = re.search(r'END_DATE\s*=\s*"(\d{4})', src)
    if not (start and end):
        raise SystemExit("could not read START_DATE/END_DATE from notebook 01")
    return f"{start.group(1)}_{end.group(1)}"


TAG = sample_tag()
EVENTS_CSV = RAW_DIR / f"ravenpack_core_events_{TAG}.csv"
FINBERT_CSV = RAW_DIR / "finbert_scores.csv"
MARKET_CSV = DATA_DIR / "market_daily_df.csv"
MODEL_INPUTS = DATA_DIR / f"model_inputs_{TAG}.csv"
METRICS_CSV = OUT_DIR / f"grid_metrics_{TAG}.csv"
FIGURE_PNG = OUT_DIR / f"grid_summary_{TAG}.png"


def _rel(path: Path) -> str:
    """Repo-relative path for display, falling back to the full path when outside the repo."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------
def stage_clean() -> int:
    for path in (EVENTS_CSV, MARKET_CSV, FINBERT_CSV):
        if not path.exists():
            print(f"  missing input: {_rel(path)}")
            print("  Run `python run_pipeline.py` first.")
            return 1

    print("Loading price panel")
    market = pd.read_csv(MARKET_CSV, parse_dates=["session_date"])
    wide = market.pivot_table(index="session_date", columns="ticker", values="daily_return")
    present = [t for t in ELEVEN if t in wide.columns]
    print(f"  {len(wide):,} sessions, {len(present)} tickers, "
          f"{wide.index.min().date()} .. {wide.index.max().date()}")

    # Nine sectors span the whole window; XLRE (Oct 2015) and XLC (Jun 2018) do not, so a
    # composition break would otherwise sit inside the target series.
    mkt9 = clean.build_market_series(wide, NINE).rename("mkt9")
    mkt11 = (clean.build_market_series(wide, ELEVEN).rename("mkt11")
             if all(t in wide.columns for t in ELEVEN) else None)
    print(f"  mkt9  {len(mkt9):,} sessions from {mkt9.index.min().date()}")
    if mkt11 is not None:
        print(f"  mkt11 {len(mkt11):,} sessions from {mkt11.index.min().date()}")
        overlap = pd.concat([mkt9, mkt11], axis=1).dropna()
        print(f"  overlap corr {overlap['mkt9'].corr(overlap['mkt11']):.4f} "
              f"over {len(overlap):,} sessions")

    sessions = mkt9.index

    print("Loading events and selecting the non-leaky pre-open window")
    events = pd.read_csv(
        EVENTS_CSV,
        usecols=["timestamp_utc", "event_sentiment_score", "headline", "event_text"],
    )
    events["et"] = clean.to_et(events["timestamp_utc"])
    total = len(events)
    events = events.loc[clean.pre_open_mask(events["et"])].copy()
    events["session_date"] = clean.assign_signal_session(events["et"], sessions)
    unmapped = int(events["session_date"].isna().sum())
    events = events.dropna(subset=["session_date"])
    print(f"  {total:,} events -> {len(events):,} pre-open and mapped to a session "
          f"({len(events)/total:.1%}); {unmapped:,} fell past the last session")

    rp_features = clean.build_news_features(events, "event_sentiment_score", "rp")

    print("Joining FinBERT scores")
    import hashlib

    for col in ("headline", "event_text"):
        events[col] = events[col].fillna("").astype(str).str.strip()
    events["text_id"] = [
        hashlib.sha1(f"{h}\x1f{t}".encode("utf-8")).hexdigest()[:16]
        for h, t in zip(events["headline"], events["event_text"])
    ]
    scores = pd.read_csv(FINBERT_CSV, usecols=["text_id", "sentiment"])
    if scores["text_id"].duplicated().any():
        dupes = int(scores["text_id"].duplicated().sum())
        print(f"  WARNING: finbert_scores.csv has {dupes:,} duplicate text_id rows; "
              f"keeping the first of each")
        scores = scores.drop_duplicates("text_id", keep="first")
    fb_events = events.merge(scores, on="text_id", how="inner")
    coverage = len(fb_events) / max(1, len(events))
    print(f"  FinBERT covers {len(fb_events):,} of {len(events):,} pre-open events ({coverage:.1%})")
    if coverage < 0.90:
        print("  WARNING: coverage under 90% — re-run the scoring stage before trusting fb_* specs")
    fb_features = clean.build_news_features(fb_events, "sentiment", "fb")

    print("Assembling and labelling")
    frame = pd.DataFrame({"r": mkt9}).join(rp_features, how="left").join(fb_features, how="left")
    if mkt11 is not None:
        frame = frame.join(mkt11, how="left")
    frame = clean.add_price_controls(frame)
    frame = clean.add_labels(frame)
    frame["flow_ma20"] = frame["rp_flow"].rolling(20).mean().shift(1)

    unlabelled = int(frame["y_magnitude"].isna().sum())
    if unlabelled != clean.MEDIAN_WINDOW:
        print(f"  FAILED: expected exactly {clean.MEDIAN_WINDOW} unlabelled sessions, "
              f"got {unlabelled}. The magnitude label is wrong; do not train on this.")
        return 1
    if frame.index.duplicated().any():
        print("  FAILED: duplicate session dates in the assembled frame")
        return 1

    frame.index.name = "session_date"
    frame.to_csv(MODEL_INPUTS)
    print(f"  wrote {_rel(MODEL_INPUTS)}: "
          f"{len(frame):,} sessions x {frame.shape[1]} columns")
    print(f"  magnitude base rate {frame['y_magnitude'].mean():.3f} | "
          f"direction {frame['y_direction'].mean():.3f}")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def stage_train(n_perm: int, seed: int) -> int:
    if not MODEL_INPUTS.exists():
        print(f"  missing {_rel(MODEL_INPUTS)} — run `--stage clean` first.")
        return 1

    frame = pd.read_csv(MODEL_INPUTS, parse_dates=["session_date"], index_col="session_date")
    usable = len(frame.dropna(subset=["vol20", "rp_flow", "y_magnitude"]))
    print(f"Loaded {len(frame):,} sessions ({usable:,} usable), "
          f"{frame.index.min().date()} .. {frame.index.max().date()}")
    if usable <= model.N_INIT + model.REFIT_STEP:
        print(f"  FAILED: only {usable:,} usable sessions against N_INIT={model.N_INIT}; "
              f"nothing would be left out of sample.")
        return 1
    # Gate BEFORE spending minutes on permutations: does the machinery recover a signal that is
    # present by construction? If not, nothing the grid reports afterwards means anything.
    ok, self_auc = model.harness_self_test(frame, "y_magnitude", seed=seed)
    print(f"Harness self-test (planted signal): AUC {self_auc:.4f} "
          f"(needs >= {model.SELF_TEST_MIN_AUC})")
    if not ok:
        print("  *** SELF-TEST FAILED — the walk-forward cannot recover a signal that is present")
        print("      by construction. The evaluation code is broken; the grid is not run. ***")
        return 1
    print("  passed — splitting, fitting and scoring are sound.\n")

    print(f"Grid: {len(model.GRID)} specs, {n_perm} permutation draws, seed {seed}\n")

    results = model.run_grid(frame, seed=seed, n_perm=n_perm)

    OUT_DIR.mkdir(exist_ok=True)
    results.to_csv(METRICS_CSV, index=False)
    print(f"\nWrote {_rel(METRICS_CSV)}")

    control = results.loc[results["spec"].str.startswith(model.CONTROL_SPEC_PREFIX)].iloc[0]
    print(f"\nTrailing volatility (informative, not a gate): AUC {control['auc_with_news']:.4f} "
          f"[{control['auc_ci_low']:.4f}, {control['auc_ci_high']:.4f}], p={control['p_value']:.3f}")
    if not model.control_passed(results):
        print("  Does not clear here. This spec is fragile — it clears on the eleven-sector target")
        print("  and fails on the nine-sector one, and its CI has always included 0.5. That says")
        print("  volatility persistence is weak in this sample, not that the harness is broken;")
        print("  the self-test above is what establishes that.")

    both = results.loc[results["clears_both"], "spec"].tolist()
    perm_only = results.loc[results["clears_perm"] & ~results["clears_both"], "spec"].tolist()
    print("\nClears BOTH references:")
    print("  " + ("\n  ".join(both) if both else "none"))
    if perm_only:
        print("Clears the permutation null only — suggestive, not established:")
        print("  " + "\n  ".join(perm_only))

    off = results.loc[results["null_mean"].abs() > 0.005, "spec"].tolist()
    if off:
        print("\nPermutation null does not centre at zero for "
              f"{len(off)} of {len(results)} specs; for those the bootstrap CI is the binding test.")

    _plot(results)
    return 0


def _plot(results: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = results.iloc[::-1].reset_index(drop=True)
    ypos = np.arange(len(order))
    colors = ["#1baf7a" if c else "#b9c0c8" for c in order["clears_both"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ypos, order["auc_with_news"] - 0.5, left=0.5, color=colors, height=0.6, zorder=2)
    ax.errorbar(order["auc_with_news"], ypos,
                xerr=[order["auc_with_news"] - order["auc_ci_low"],
                      order["auc_ci_high"] - order["auc_with_news"]],
                fmt="none", ecolor="#2b2b2b", elinewidth=1.4, capsize=3, zorder=5)
    for k, row in order.iterrows():
        ax.plot([row["detectable_bound"]] * 2, [k - 0.34, k + 0.34],
                color="#d1495b", lw=2.2, zorder=3)
    ax.axvline(0.5, color="#333", lw=1.1, zorder=4)
    ax.set_yticks(ypos); ax.set_yticklabels(order["spec"], fontsize=9)
    ax.set_xlabel("out-of-sample AUC")
    ax.set_title("Each spec vs both references\n"
                 "bars = 95% block-bootstrap CI · red = permutation bound · green = clears both",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(FIGURE_PNG, dpi=150, bbox_inches="tight")
    print(f"Wrote {_rel(FIGURE_PNG)}")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["clean", "train", "all"], default="all")
    parser.add_argument("--list", action="store_true", help="show stages and inputs, then exit")
    parser.add_argument("--n-perm", type=int, default=model.N_PERM,
                        help=f"permutation draws (default {model.N_PERM}; lower for a quick look)")
    parser.add_argument("--seed", type=int, default=model.SEED)
    args = parser.parse_args()

    print(f"Sample tag from notebook 01: {TAG}\n")

    if args.list:
        print("  clean   raw events + market panel -> model_inputs")
        for p in (EVENTS_CSV, MARKET_CSV, FINBERT_CSV):
            print(f"            in  {_rel(p)}  "
                  f"({'exists' if p.exists() else 'MISSING'})")
        print(f"            out {_rel(MODEL_INPUTS)}  "
              f"({'exists' if MODEL_INPUTS.exists() else 'missing'})")
        print("  train   model_inputs -> grid metrics + figure")
        print(f"            out {_rel(METRICS_CSV)}")
        print(f"            out {_rel(FIGURE_PNG)}")
        return 0

    started = time.time()
    if args.stage in ("clean", "all"):
        print("=" * 70); print("CLEAN"); print("=" * 70)
        if stage_clean() != 0:
            return 1
        print()
    if args.stage in ("train", "all"):
        print("=" * 70); print("TRAIN"); print("=" * 70)
        if stage_train(args.n_perm, args.seed) != 0:
            return 1
    print(f"\nDone in {(time.time() - started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
