"""Walk-forward evaluation with two independent references.

This is the harness verified on the 2020-2025 sample. It is a module rather than notebook cells so
that the runner and notebook 11 share one implementation: if the measurement and the sample both
change at once, movement in the numbers is unattributable.

Two references, deliberately:

  * a block permutation null (circular rotation of the news block against the returns), which
    preserves the autocorrelation of both series and destroys only their alignment; and
  * a block-bootstrap CI on the out-of-sample AUC itself.

Both are needed. The permutation nulls in this project do NOT centre at 0.5 -- rotating a trending
feature misaligns it with the volatility regime and pushes rotated AUC below chance, so beating the
permutation 95th percentile is an anti-conservative bar on its own. Where the null is off-centre,
the CI is the binding test. No in-sample AUC is ever produced here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_INIT = 500        # first training block, in sessions
REFIT_STEP = 21     # refit cadence, ~1 month
N_PERM = 250        # block permutation draws
N_BOOT = 500        # block bootstrap draws
SEED = 20260807


def walk_forward_predictions(frame: pd.DataFrame, feature_cols: list[str], y_col: str,
                             n_init: int = N_INIT, step: int = REFIT_STEP):
    """Pooled out-of-sample predictions from an expanding walk-forward, in time order."""
    preds, truth, i = [], [], n_init
    while i < len(frame):
        j = min(i + step, len(frame))
        train, test = frame.iloc[:i], frame.iloc[i:j]
        if train[y_col].nunique() > 1:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
            model.fit(train[feature_cols], train[y_col])
            preds.append(model.predict_proba(test[feature_cols])[:, 1])
            truth.append(test[y_col].to_numpy())
        i = j
    if not preds:
        raise ValueError("walk-forward produced no out-of-sample predictions; sample too short")
    return np.concatenate(preds), np.concatenate(truth)


def walk_forward_auc(frame: pd.DataFrame, feature_cols: list[str], y_col: str,
                     n_init: int = N_INIT, step: int = REFIT_STEP) -> float:
    """Pooled out-of-sample AUC. An empty feature set is a coin flip by definition."""
    if not feature_cols:
        return 0.5
    pred, truth = walk_forward_predictions(frame, feature_cols, y_col, n_init, step)
    return roc_auc_score(truth, pred)


def block_bootstrap_ci(pred: np.ndarray, truth: np.ndarray, rng: np.random.Generator,
                       block: int = REFIT_STEP, n_boot: int = N_BOOT,
                       alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for the OOS AUC, resampling contiguous blocks to respect autocorrelation."""
    n = len(truth)
    starts = np.arange(0, n, block)
    aucs = []
    for _ in range(n_boot):
        picks = rng.choice(starts, size=len(starts), replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in picks])
        t, p = truth[idx], pred[idx]
        if len(np.unique(t)) > 1:
            aucs.append(roc_auc_score(t, p))
    return float(np.percentile(aucs, 100 * alpha / 2)), float(np.percentile(aucs, 100 * (1 - alpha / 2)))


@dataclass
class SpecResult:
    spec: str
    target: str
    n_oos: int
    auc_baseline: float
    auc_with_news: float
    lift: float
    auc_ci_low: float
    auc_ci_high: float
    null_mean: float
    null_p95: float
    p_value: float
    clears_perm: bool
    beats_half: bool
    clears_both: bool
    detectable_bound: float


def evaluate(frame: pd.DataFrame, label: str, news_cols: list[str], base_cols: list[str],
             y_col: str, rng: np.random.Generator, n_perm: int = N_PERM,
             n_init: int = N_INIT, step: int = REFIT_STEP) -> SpecResult:
    """Out-of-sample lift of `news_cols` over `base_cols`, against both references."""
    needed = list(dict.fromkeys(news_cols + base_cols + [y_col]))
    data = frame.dropna(subset=needed).reset_index(drop=True)

    auc_base = walk_forward_auc(data, base_cols, y_col, n_init, step)
    pred, truth = walk_forward_predictions(data, base_cols + news_cols, y_col, n_init, step)
    auc_full = roc_auc_score(truth, pred)
    ci_low, ci_high = block_bootstrap_ci(pred, truth, rng)

    null = np.empty(n_perm)
    for k in range(n_perm):
        shift = rng.integers(30, len(data) - 30)
        rotated = data.copy()
        rotated[news_cols] = np.roll(data[news_cols].to_numpy(), shift, axis=0)
        null[k] = walk_forward_auc(rotated, base_cols + news_cols, y_col, n_init, step) - auc_base

    observed = auc_full - auc_base
    p_value = float((null >= observed).mean())
    return SpecResult(
        spec=label, target=y_col, n_oos=len(truth),
        auc_baseline=auc_base, auc_with_news=auc_full, lift=observed,
        auc_ci_low=ci_low, auc_ci_high=ci_high,
        null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
        p_value=p_value,
        clears_perm=p_value < 0.05,
        beats_half=ci_low > 0.5,
        clears_both=bool(p_value < 0.05 and ci_low > 0.5),
        detectable_bound=auc_base + float(np.percentile(null, 95)),
    )


# ---------------------------------------------------------------------------
# the pre-registered grid
# ---------------------------------------------------------------------------
RP_TONE = ["rp_tone", "rp_net", "rp_disp", "rp_absmean", "rp_tail", "rp_negshare"]
FB_TONE = ["fb_tone", "fb_net", "fb_disp", "fb_absmean", "fb_tail", "fb_negshare"]
RP_FLOW = ["rp_flow"]
VOL_BASE = ["vol20", "vol5", "abs_r1"]
DIR_BASE = ["rev1", "mom5", "vol20"]

GRID = [
    ("1. RavenPack news flow",           RP_FLOW,           [],                       "y_magnitude"),
    ("2. RavenPack tone",                RP_TONE,           [],                       "y_magnitude"),
    ("3. FinBERT tone",                  FB_TONE,           [],                       "y_magnitude"),
    ("4. Trailing volatility (control)", VOL_BASE,          [],                       "y_magnitude"),
    ("5. Flow | vol + trailing flow",    RP_FLOW,           VOL_BASE + ["flow_ma20"], "y_magnitude"),
    ("6. RavenPack news -> direction",   RP_FLOW + RP_TONE, DIR_BASE,                 "y_direction"),
]

CONTROL_SPEC_PREFIX = "4."


def run_grid(frame: pd.DataFrame, seed: int = SEED, n_perm: int = N_PERM,
             n_init: int = N_INIT, progress=print) -> pd.DataFrame:
    """Run the pre-registered grid and return one row per spec."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, news_cols, base_cols, y_col in GRID:
        result = evaluate(frame, label, news_cols, base_cols, y_col, rng,
                          n_perm=n_perm, n_init=n_init)
        rows.append(asdict(result))
        if progress:
            verdict = ("*** CLEARS BOTH ***" if result.clears_both
                       else "clears perm only" if result.clears_perm else "inside noise")
            progress(f"{result.spec:<36s} AUC={result.auc_with_news:.4f} "
                     f"[{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}]  "
                     f"p={result.p_value:.3f}  {verdict}")
    return pd.DataFrame(rows)


def control_passed(results: pd.DataFrame) -> bool:
    """Whether the trailing-volatility spec cleared its permutation null.

    NOT a gate. This was originally used as the positive control, and it is too weak for that job:
    on the 2020-2025 sample it clears on the eleven-sector target (AUC 0.5260, p=0.028) and fails
    on the nine-sector one (0.4982, p=0.352), purely because ~6% of magnitude labels differ between
    the two definitions. Its CI has always included 0.5. Treat it as an informative spec about
    volatility persistence in this data, and use `harness_self_test` for the gate.
    """
    control = results.loc[results["spec"].str.startswith(CONTROL_SPEC_PREFIX)]
    if control.empty:
        raise ValueError("no positive-control spec found in the results")
    return bool(control.iloc[0]["clears_perm"])


SELF_TEST_MIN_AUC = 0.65


def harness_self_test(frame: pd.DataFrame, y_col: str, seed: int = SEED,
                      min_auc: float = SELF_TEST_MIN_AUC,
                      n_init: int = N_INIT, step: int = REFIT_STEP) -> tuple[bool, float]:
    """Verify the machinery recovers a signal that is present by construction.

    A gate has to answer one question only: is the evaluation code working? Using a real-world
    effect for that conflates it with a second question -- is this particular effect present in
    this particular sample -- and a weak effect then fails the gate for reasons that say nothing
    about the code. So the gate plants a feature correlated with the label by construction and
    checks that the walk-forward recovers it. It fails only if splitting, fitting, or scoring is
    broken, and it is deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    data = frame.dropna(subset=[y_col]).reset_index(drop=True)
    if len(data) <= n_init + step:
        raise ValueError(f"sample too short for a self-test: {len(data)} rows vs n_init={n_init}")
    planted = data[y_col].to_numpy() + rng.normal(scale=1.0, size=len(data))
    probe = data.assign(_planted=planted)
    auc = walk_forward_auc(probe, ["_planted"], y_col, n_init, step)
    return auc >= min_auc, auc
