# Sample Extension Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the capstone sample from 2020–2025 to 2015–2026 through a six-stage resumable pipeline, so notebook 11's magnitude result and the direction null are both measured on ~2.5× the out-of-sample data.

**Architecture:** All transformation logic moves into two importable modules (`pipeline_config.py` for constants and paths, `pipeline_lib.py` for pure data transforms) so it can be unit-tested without a WRDS session. The notebooks become thin orchestration over those modules. Stages communicate only through files on disk, so any stage re-runs alone.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, pytest 9.3, `wrds` (WRDS/PostgreSQL), transformers + torch (FinBERT, CUDA).

## Global Constraints

- Requested window: `START_DATE = "2015-01-01"`. `END_DATE` is **never a literal** — stage 0 resolves it from the latest session in `crsp.dsf_v2` and records it in `resolved_config.json`.
- All new outputs are versioned with the suffix `_2015_2026`. The existing `news_daily_df.csv`, `market_daily_df.csv`, and `model_daily_panel.csv` are **never overwritten** — Christian's notebooks consume them.
- Licensed row-level WRDS data is written **only** under `data_collection/raw/`, which is gitignored. Never force-add it.
- News filters must stay identical to notebook 01: `relevance >= 90`, `event_relevance >= 90`, rank-1 non-blog institutional sources, 16:00 ET signal-date rule. The extended sample is only comparable if the filters match.
- Two market series: `mkt9` (the nine SPDRs present throughout, 2015–2026) for aggregate models; `mkt11` and the 11-ticker panel (June 2018 onward) for cross-sectional work.
- Notebook 11's harness is **unchanged**: `MEDIAN_WINDOW = 60`, `N_INIT = 500` (absolute, not rescaled), `REFIT_STEP = 21`, `N_PERM = 250`, `SEED = 20260807`, the six-spec grid, the block permutation, and the block-bootstrap CI.
- Stage 6 gate: the positive control (Spec 4, trailing volatility) **must clear**, or no other spec in the run is readable.
- `pytest.ini` at the repo root is a **prerequisite already in place**. It anchors pytest's rootdir here; without it pytest walks up to `Group/pyproject.toml`, an unrelated project's file carrying a UTF-8 BOM that fails TOML parsing, and no test in this repo can be collected. It also sets `pythonpath = .` so `from data_collection import ...` resolves in tests. Do not delete or relocate it.
- **Every notebook created or modified by this plan must open with this bootstrap cell**, before any `from data_collection import ...`. Notebooks are opened with a working directory of either the repo root or `data_collection/`, and the package import only resolves from the root:

```python
import sys
from pathlib import Path

_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data_collection").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
```

---

### Task 1: Pipeline configuration module

**Files:**
- Create: `data_collection/pipeline_config.py`
- Test: `tests/test_pipeline_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `REPO_ROOT`, `DATA_DIR`, `RAW_DIR`, `NEWS_RAW_DIR`, `OUT_DIR`, `RESOLVED_CONFIG_PATH` (all `pathlib.Path`); `SAMPLE_TAG: str`, `REQUESTED_START: str`; `MARKET_CLOSE_MIN: int`, `MARKET_OPEN_MIN: int`, `MEDIAN_WINDOW: int`, `N_INIT: int`, `REFIT_STEP: int`, `N_PERM: int`, `SEED: int`; `SECTOR_ETF_TO_ASSET: dict[str, str]`, `NINE_SECTOR_TICKERS: tuple[str, ...]`, `ELEVEN_SECTOR_TICKERS: tuple[str, ...]`; `versioned(stem: str) -> Path`; `save_resolved(d: dict) -> Path`; `load_resolved() -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_config.py
import json
import pytest
from data_collection import pipeline_config as cfg


def test_repo_root_contains_data_collection():
    assert (cfg.REPO_ROOT / "data_collection").is_dir()


def test_versioned_appends_sample_tag():
    assert cfg.versioned("market_daily").name == "market_daily_2015_2026.csv"
    assert cfg.versioned("market_daily").parent == cfg.DATA_DIR


def test_nine_tickers_exclude_late_launchers():
    assert set(cfg.NINE_SECTOR_TICKERS) == set(cfg.ELEVEN_SECTOR_TICKERS) - {"XLRE", "XLC"}
    assert len(cfg.NINE_SECTOR_TICKERS) == 9
    assert len(cfg.ELEVEN_SECTOR_TICKERS) == 11


def test_window_constants_bracket_the_trading_day():
    assert cfg.MARKET_OPEN_MIN == 9 * 60 + 30
    assert cfg.MARKET_CLOSE_MIN == 16 * 60
    assert cfg.MARKET_OPEN_MIN < cfg.MARKET_CLOSE_MIN


def test_resolved_config_round_trips(tmp_path, monkeypatch):
    target = tmp_path / "resolved_config.json"
    monkeypatch.setattr(cfg, "RESOLVED_CONFIG_PATH", target)
    payload = {"end_date": "2026-08-01", "news_years": [2015, 2016]}
    cfg.save_resolved(payload)
    assert cfg.load_resolved() == payload


def test_load_resolved_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RESOLVED_CONFIG_PATH", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="stage 0"):
        cfg.load_resolved()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.pipeline_config'`

- [ ] **Step 3: Write minimal implementation**

```python
# data_collection/pipeline_config.py
"""Single source of truth for the extended-sample pipeline.

Every stage imports its dates, paths and window constants from here. Six notebooks each carrying
a private START_DATE is how a pipeline silently desynchronises: one gets edited, five do not, and
the resulting panel is wrong in a way no assert catches.
"""
from __future__ import annotations

import json
from pathlib import Path

# --- paths -------------------------------------------------------------------
def _find_repo_root(start: Path | None = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data_collection").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the repository root (no data_collection/ found).")


REPO_ROOT = _find_repo_root(Path(__file__).parent)
DATA_DIR = REPO_ROOT / "data_collection"
RAW_DIR = DATA_DIR / "raw"
NEWS_RAW_DIR = RAW_DIR / "news"
OUT_DIR = REPO_ROOT / "model_outputs"
RESOLVED_CONFIG_PATH = DATA_DIR / "resolved_config.json"

# --- sample window -----------------------------------------------------------
SAMPLE_TAG = "2015_2026"
REQUESTED_START = "2015-01-01"

# --- news window (ET minutes past midnight) ----------------------------------
MARKET_CLOSE_MIN = 16 * 60       # 16:00 ET, end of a session
MARKET_OPEN_MIN = 9 * 60 + 30    # 09:30 ET, first tradeable price

# --- news filters (must match notebook 01 exactly) ---------------------------
RELEVANCE_MIN = 90
EVENT_RELEVANCE_MIN = 90

# --- modelling harness (unchanged from the verified notebook 11 run) ---------
MEDIAN_WINDOW = 60
N_INIT = 500
REFIT_STEP = 21
N_PERM = 250
SEED = 20260807

# --- universe ----------------------------------------------------------------
SECTOR_ETF_TO_ASSET = {
    "XLK": "Technology", "XLV": "Health_Care", "XLF": "Financials",
    "XLC": "Communication_Services", "XLY": "Consumer_Discretionary",
    "XLI": "Industrials", "XLP": "Consumer_Staples", "XLE": "Energy",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real_Estate",
}
ELEVEN_SECTOR_TICKERS = tuple(sorted(SECTOR_ETF_TO_ASSET))
# XLRE launched Oct 2015 and XLC Jun 2018, so neither spans the requested window.
LATE_LAUNCH_TICKERS = ("XLRE", "XLC")
NINE_SECTOR_TICKERS = tuple(t for t in ELEVEN_SECTOR_TICKERS if t not in LATE_LAUNCH_TICKERS)


# --- helpers -----------------------------------------------------------------
def versioned(stem: str) -> Path:
    """Path for a versioned gold output, e.g. versioned('market_daily')."""
    return DATA_DIR / f"{stem}_{SAMPLE_TAG}.csv"


def save_resolved(payload: dict) -> Path:
    RESOLVED_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESOLVED_CONFIG_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return RESOLVED_CONFIG_PATH


def load_resolved() -> dict:
    if not RESOLVED_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{RESOLVED_CONFIG_PATH} not found — run stage 0 (discovery) before any later stage."
        )
    return json.loads(RESOLVED_CONFIG_PATH.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Create the package marker so `from data_collection import ...` resolves**

```bash
touch data_collection/__init__.py
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline_config.py -v`
Expected: PASS, 6 passed

- [ ] **Step 6: Commit**

```bash
git add data_collection/pipeline_config.py data_collection/__init__.py tests/test_pipeline_config.py
git commit -m "feat: add pipeline_config as single source of truth for sample extension

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Pre-open window selection and session assignment

This is spec unit test #1. `assign_signal_session` also fixes a defect measured in the current
extract: 6.6% of events (18,585 of them Saturday-dated) carry a `signal_calendar_date` that is not a
trading session, so they vanish at the join. Rolling forward to the next real session keeps them.

**Files:**
- Create: `data_collection/pipeline_lib.py`
- Test: `tests/test_pipeline_lib.py`

**Interfaces:**
- Consumes: `pipeline_config.MARKET_CLOSE_MIN`, `MARKET_OPEN_MIN`.
- Produces: `to_et(ts_utc: pd.Series) -> pd.Series`; `minute_of_day(et: pd.Series) -> pd.Series`; `pre_open_mask(et: pd.Series) -> pd.Series[bool]`; `assign_signal_session(et: pd.Series, sessions) -> pd.Series[datetime64[ns]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_lib.py
import numpy as np
import pandas as pd
import pytest
from data_collection import pipeline_lib as lib

NY = "America/New_York"


def et(*stamps):
    return pd.Series(pd.to_datetime(list(stamps)).tz_localize(NY))


def test_pre_open_mask_selects_only_non_trading_hours():
    stamps = et(
        "2024-03-05 16:00", "2024-03-05 18:30", "2024-03-05 23:59",   # after the close
        "2024-03-05 00:00", "2024-03-05 09:29",                        # before the open
        "2024-03-05 09:30", "2024-03-05 12:00", "2024-03-05 15:59",   # intraday
    )
    assert lib.pre_open_mask(stamps).tolist() == [True, True, True, True, True, False, False, False]


def test_assign_signal_session_same_day_before_close():
    sessions = pd.to_datetime(["2024-03-05", "2024-03-06"])
    got = lib.assign_signal_session(et("2024-03-05 10:00"), sessions)
    assert got.iloc[0] == pd.Timestamp("2024-03-05")


def test_assign_signal_session_rolls_after_close():
    sessions = pd.to_datetime(["2024-03-05", "2024-03-06"])
    got = lib.assign_signal_session(et("2024-03-05 16:30"), sessions)
    assert got.iloc[0] == pd.Timestamp("2024-03-06")


def test_assign_signal_session_friday_evening_rolls_to_monday():
    sessions = pd.to_datetime(["2024-03-08", "2024-03-11"])  # Fri, Mon
    got = lib.assign_signal_session(et("2024-03-08 18:00"), sessions)
    assert got.iloc[0] == pd.Timestamp("2024-03-11")


def test_assign_signal_session_saturday_rolls_to_monday():
    """The current extract loses 6.6% of events here; they must survive."""
    sessions = pd.to_datetime(["2024-03-08", "2024-03-11"])
    got = lib.assign_signal_session(et("2024-03-09 10:00"), sessions)
    assert got.iloc[0] == pd.Timestamp("2024-03-11")


def test_assign_signal_session_skips_a_holiday():
    # 2024-03-29 Good Friday: sessions jump Thu 28 -> Mon 1 Apr
    sessions = pd.to_datetime(["2024-03-28", "2024-04-01"])
    got = lib.assign_signal_session(et("2024-03-28 17:00"), sessions)
    assert got.iloc[0] == pd.Timestamp("2024-04-01")


def test_assign_signal_session_past_the_last_session_is_nat():
    sessions = pd.to_datetime(["2024-03-05"])
    got = lib.assign_signal_session(et("2024-03-05 16:30"), sessions)
    assert pd.isna(got.iloc[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_lib.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_collection.pipeline_lib'`

- [ ] **Step 3: Write minimal implementation**

```python
# data_collection/pipeline_lib.py
"""Pure transforms for the extended-sample pipeline.

Everything here is a function of its arguments with no WRDS or filesystem dependency, so it is
unit-testable without a database session. The notebooks are thin orchestration over this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_collection.pipeline_config import (
    MARKET_CLOSE_MIN,
    MARKET_OPEN_MIN,
)

EASTERN = "America/New_York"


def to_et(ts_utc: pd.Series) -> pd.Series:
    """Parse a UTC timestamp column and convert it to US Eastern."""
    return pd.to_datetime(ts_utc, utc=True).dt.tz_convert(EASTERN)


def minute_of_day(et: pd.Series) -> pd.Series:
    return et.dt.hour * 60 + et.dt.minute


def pre_open_mask(et: pd.Series) -> pd.Series:
    """True for events arriving after the previous close and before this session's open.

    These are the only events that are not already priced at the moment the target return begins.
    """
    minutes = minute_of_day(et)
    return (minutes >= MARKET_CLOSE_MIN) | (minutes < MARKET_OPEN_MIN)


def assign_signal_session(et: pd.Series, sessions) -> pd.Series:
    """Map each timestamp to the first trading session whose close it can still affect.

    An event before 16:00 ET can affect that calendar day's close; one at or after 16:00 cannot,
    so it rolls to the next day. Either way the result is then rolled forward to the next actual
    trading session, which is what keeps weekend and holiday news instead of dropping it at the
    join. Timestamps past the final session return NaT.
    """
    session_index = pd.DatetimeIndex(sorted(pd.DatetimeIndex(sessions)))
    naive_day = et.dt.tz_localize(None).dt.normalize()
    rolls = (minute_of_day(et) >= MARKET_CLOSE_MIN).astype(int)
    candidate = naive_day + pd.to_timedelta(rolls, unit="D")

    positions = session_index.searchsorted(candidate.to_numpy(), side="left")
    result = pd.Series(pd.NaT, index=et.index, dtype="datetime64[ns]")
    in_range = positions < len(session_index)
    result.loc[in_range] = session_index[positions[in_range]]
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline_lib.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add data_collection/pipeline_lib.py tests/test_pipeline_lib.py
git commit -m "feat: add pre-open window selection and session assignment

assign_signal_session rolls weekend and holiday news forward to the next
trading session; the current extract drops 6.6% of events at that join.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: NaN-preserving label construction

This is spec unit test #2, and it guards the exact defect that failed the positive control on
2026-08-07: `(abs_r > med).astype(int)` turns the `NaN > NaN` comparison into a confident `0`, so
60 unlabelled sessions entered the first training block and `dropna` could not remove them.

**Files:**
- Modify: `data_collection/pipeline_lib.py`
- Test: `tests/test_pipeline_lib.py`

**Interfaces:**
- Consumes: `pipeline_config.MEDIAN_WINDOW`.
- Produces: `trailing_median(abs_r: pd.Series, window: int = MEDIAN_WINDOW) -> pd.Series`; `magnitude_label(abs_r: pd.Series, median: pd.Series) -> pd.Series[float64]`; `direction_label(returns: pd.Series) -> pd.Series[float64]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pipeline_lib.py

def test_trailing_median_is_strictly_past_looking():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    got = lib.trailing_median(values, window=3)
    assert got.isna().tolist() == [True, True, True, False, False]
    assert got.iloc[3] == 2.0    # median of rows 0..2, not including row 3
    assert got.iloc[4] == 3.0


def test_magnitude_label_is_nan_before_the_window_fills():
    """The bug that failed the positive control: NaN > NaN is False, and .astype(int) makes it 0."""
    abs_r = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    median = lib.trailing_median(abs_r, window=3)
    got = lib.magnitude_label(abs_r, median)

    assert got.isna().sum() == 3, "unlabelled rows must stay NaN, never 0"
    assert not (got.fillna(-1) == 0).iloc[:3].any()
    assert got.iloc[3] == 1.0    # 0.04 > median(0.01,0.02,0.03)=0.02
    assert got.iloc[4] == 1.0


def test_magnitude_label_marks_small_moves_zero_not_nan():
    abs_r = pd.Series([0.05, 0.06, 0.07, 0.001])
    median = lib.trailing_median(abs_r, window=3)
    got = lib.magnitude_label(abs_r, median)
    assert got.iloc[3] == 0.0
    assert not pd.isna(got.iloc[3])


def test_direction_label_preserves_nan():
    returns = pd.Series([0.01, -0.02, np.nan, 0.0])
    got = lib.direction_label(returns)
    assert got.iloc[0] == 1.0
    assert got.iloc[1] == 0.0
    assert pd.isna(got.iloc[2])
    assert got.iloc[3] == 0.0    # a flat session is not "up"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_lib.py -k label -v`
Expected: FAIL — `AttributeError: module 'data_collection.pipeline_lib' has no attribute 'trailing_median'`

- [ ] **Step 3: Write minimal implementation**

Add to `data_collection/pipeline_lib.py`, and extend the config import at the top to
`from data_collection.pipeline_config import MARKET_CLOSE_MIN, MARKET_OPEN_MIN, MEDIAN_WINDOW`:

```python
def trailing_median(abs_r: pd.Series, window: int = MEDIAN_WINDOW) -> pd.Series:
    """Median of the previous `window` absolute returns, excluding the current session."""
    return abs_r.rolling(window).median().shift(1)


def magnitude_label(abs_r: pd.Series, median: pd.Series) -> pd.Series:
    """1 where the move exceeds its trailing median, 0 where it does not, NaN where undefined.

    Built with np.where rather than a comparison plus .astype(int) because `NaN > NaN` is False:
    astype would silently label every session before the trailing window fills as a confident 0,
    and because the result is then never NaN, dropna could not remove those rows.
    """
    defined = median.notna() & abs_r.notna()
    return pd.Series(
        np.where(defined, abs_r > median, np.nan),
        index=abs_r.index, dtype="float64",
    )


def direction_label(returns: pd.Series) -> pd.Series:
    """1 for an up session, 0 otherwise, NaN where the return is missing."""
    return pd.Series(
        np.where(returns.notna(), returns > 0, np.nan),
        index=returns.index, dtype="float64",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline_lib.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add data_collection/pipeline_lib.py tests/test_pipeline_lib.py
git commit -m "feat: add NaN-preserving label construction

Guards the defect that failed the notebook 11 positive control: NaN > NaN
is False, so .astype(int) fabricated 60 zero labels dropna could not remove.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: News feature aggregation

**Files:**
- Modify: `data_collection/pipeline_lib.py`
- Test: `tests/test_pipeline_lib.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_news_features(events: pd.DataFrame, score_col: str, prefix: str, session_col: str = "session_date") -> pd.DataFrame` returning columns `{prefix}_flow`, `{prefix}_tone`, `{prefix}_net`, `{prefix}_disp`, `{prefix}_absmean`, `{prefix}_tail`, `{prefix}_negshare`, indexed by session.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pipeline_lib.py

def test_build_news_features_computes_expected_columns():
    events = pd.DataFrame({
        "session_date": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 2),
        "score": [0.8, -0.9, 0.0, 0.4, -0.5, 0.5],
    })
    got = lib.build_news_features(events, "score", "rp")

    assert list(got.columns) == [
        "rp_flow", "rp_tone", "rp_net", "rp_disp", "rp_absmean", "rp_tail", "rp_negshare",
    ]
    day1 = got.loc[pd.Timestamp("2024-01-02")]
    assert day1["rp_flow"] == pytest.approx(np.log1p(4))
    assert day1["rp_tone"] == pytest.approx((0.8 - 0.9 + 0.0 + 0.4) / 4)
    assert day1["rp_net"] == pytest.approx(0.3)
    assert day1["rp_absmean"] == pytest.approx((0.8 + 0.9 + 0.0 + 0.4) / 4)
    assert day1["rp_tail"] == pytest.approx(2 / 4)      # |0.8| and |-0.9| exceed 0.7
    assert day1["rp_negshare"] == pytest.approx(1 / 4)  # only -0.9 is below -0.3


def test_build_news_features_one_event_day_has_nan_dispersion():
    events = pd.DataFrame({
        "session_date": pd.to_datetime(["2024-01-02"]),
        "score": [0.5],
    })
    got = lib.build_news_features(events, "score", "rp")
    assert pd.isna(got.iloc[0]["rp_disp"])
    assert got.iloc[0]["rp_flow"] == pytest.approx(np.log1p(1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_lib.py -k news_features -v`
Expected: FAIL — `AttributeError: ... has no attribute 'build_news_features'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to data_collection/pipeline_lib.py

TAIL_THRESHOLD = 0.7
NEGATIVE_THRESHOLD = -0.3


def build_news_features(
    events: pd.DataFrame,
    score_col: str,
    prefix: str,
    session_col: str = "session_date",
) -> pd.DataFrame:
    """Session-level aggregates of a per-event sentiment score.

    `flow` is arrival volume and carries the effect measured in notebook 11; the tone-derived
    columns are kept so the RavenPack-versus-FinBERT comparison runs on identical statistics.
    """
    grouped = events.groupby(session_col)[score_col]
    absolute = events.assign(_abs=events[score_col].abs()).groupby(session_col)["_abs"]
    tail = events.assign(_t=events[score_col].abs() > TAIL_THRESHOLD).groupby(session_col)["_t"]
    negative = events.assign(_n=events[score_col] < NEGATIVE_THRESHOLD).groupby(session_col)["_n"]

    return pd.DataFrame({
        f"{prefix}_flow": np.log1p(grouped.size()),
        f"{prefix}_tone": grouped.mean(),
        f"{prefix}_net": grouped.sum(),
        f"{prefix}_disp": grouped.std(),
        f"{prefix}_absmean": absolute.mean(),
        f"{prefix}_tail": tail.mean(),
        f"{prefix}_negshare": negative.mean(),
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline_lib.py -v`
Expected: PASS, 13 passed

- [ ] **Step 5: Commit**

```bash
git add data_collection/pipeline_lib.py tests/test_pipeline_lib.py
git commit -m "feat: add session-level news feature aggregation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Market series construction (`mkt9` / `mkt11`)

**Files:**
- Modify: `data_collection/pipeline_lib.py`
- Test: `tests/test_pipeline_lib.py`

**Interfaces:**
- Consumes: `pipeline_config.NINE_SECTOR_TICKERS`, `ELEVEN_SECTOR_TICKERS`.
- Produces: `build_market_series(wide_returns: pd.DataFrame, tickers) -> pd.Series`; `first_complete_session(wide_returns: pd.DataFrame, tickers) -> pd.Timestamp`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pipeline_lib.py

def _wide():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {"XLK": [0.01, 0.02, 0.03],
         "XLF": [0.03, 0.02, 0.01],
         "XLC": [np.nan, 0.05, 0.05]},
        index=idx,
    )


def test_build_market_series_equal_weights_named_tickers_only():
    got = lib.build_market_series(_wide(), ["XLK", "XLF"])
    assert got.tolist() == pytest.approx([0.02, 0.02, 0.02])


def test_build_market_series_drops_sessions_with_a_missing_constituent():
    got = lib.build_market_series(_wide(), ["XLK", "XLF", "XLC"])
    assert len(got) == 2, "the session where XLC is absent must be dropped, not mean-imputed"
    assert got.index[0] == pd.Timestamp("2024-01-03")


def test_build_market_series_rejects_an_unknown_ticker():
    with pytest.raises(KeyError, match="XLZ"):
        lib.build_market_series(_wide(), ["XLK", "XLZ"])


def test_first_complete_session_finds_the_inception_boundary():
    assert lib.first_complete_session(_wide(), ["XLK", "XLF", "XLC"]) == pd.Timestamp("2024-01-03")
    assert lib.first_complete_session(_wide(), ["XLK", "XLF"]) == pd.Timestamp("2024-01-02")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_lib.py -k market_series -v`
Expected: FAIL — `AttributeError: ... has no attribute 'build_market_series'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to data_collection/pipeline_lib.py

def _require_tickers(wide_returns: pd.DataFrame, tickers) -> list[str]:
    wanted = list(tickers)
    missing = [t for t in wanted if t not in wide_returns.columns]
    if missing:
        raise KeyError(f"tickers absent from the return matrix: {missing}")
    return wanted


def build_market_series(wide_returns: pd.DataFrame, tickers) -> pd.Series:
    """Equal-weight return of exactly `tickers`, on sessions where all of them trade.

    Sessions missing a constituent are dropped rather than averaged over what is present, so the
    series has one constant definition throughout and carries no composition break.
    """
    wanted = _require_tickers(wide_returns, tickers)
    subset = wide_returns[wanted].dropna(how="any")
    return subset.mean(axis=1)


def first_complete_session(wide_returns: pd.DataFrame, tickers) -> pd.Timestamp:
    """Earliest session on which every one of `tickers` trades — an empirical inception date."""
    wanted = _require_tickers(wide_returns, tickers)
    complete = wide_returns[wanted].dropna(how="any")
    if complete.empty:
        raise ValueError(f"no session has all of {wanted} present")
    return complete.index.min()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline_lib.py -v`
Expected: PASS, 17 passed

- [ ] **Step 5: Commit**

```bash
git add data_collection/pipeline_lib.py tests/test_pipeline_lib.py
git commit -m "feat: add mkt9/mkt11 market series construction

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Scoring delta and per-year event split

The per-year split keys on the **UTC year of `timestamp_utc`**, not on `signal_calendar_date`.
Notebook 01's SQL filters on `rpa_date_utc`, so an event on 2019-12-31 at 18:00 ET has a signal date
of 2020-01-01 but belongs to the 2019 pull. Keying on the signal date would make the per-year files
disagree with a re-pull at every year boundary.

**Files:**
- Modify: `data_collection/pipeline_lib.py`
- Test: `tests/test_pipeline_lib.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `compute_scoring_delta(dedup: pd.DataFrame, scored_text_ids) -> pd.DataFrame`; `split_events_by_utc_year(events: pd.DataFrame, out_dir: Path, overwrite: bool = False) -> dict[int, Path]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pipeline_lib.py

def test_compute_scoring_delta_excludes_already_scored():
    dedup = pd.DataFrame({"text_id": ["a", "b", "c", "d"], "headline": list("wxyz")})
    got = lib.compute_scoring_delta(dedup, ["a", "c"])
    assert got["text_id"].tolist() == ["b", "d"]
    assert set(got["text_id"]) & {"a", "c"} == set()


def test_compute_scoring_delta_is_empty_when_all_scored():
    dedup = pd.DataFrame({"text_id": ["a", "b"], "headline": ["w", "x"]})
    assert lib.compute_scoring_delta(dedup, ["a", "b"]).empty


def test_compute_scoring_delta_rejects_duplicate_text_ids():
    dedup = pd.DataFrame({"text_id": ["a", "a"], "headline": ["w", "x"]})
    with pytest.raises(ValueError, match="duplicate text_id"):
        lib.compute_scoring_delta(dedup, [])


def test_split_events_by_utc_year_uses_utc_year_not_signal_date(tmp_path):
    events = pd.DataFrame({
        "rp_story_id": ["s1", "s2"],
        # 2019-12-31 23:00 UTC is still 2019 in UTC, though its signal date is in 2020
        "timestamp_utc": ["2019-12-31 23:00:00", "2020-01-02 12:00:00"],
        "signal_calendar_date": ["2020-01-01", "2020-01-02"],
    })
    written = lib.split_events_by_utc_year(events, tmp_path)

    assert sorted(written) == [2019, 2020]
    assert pd.read_csv(written[2019])["rp_story_id"].tolist() == ["s1"]
    assert pd.read_csv(written[2020])["rp_story_id"].tolist() == ["s2"]


def test_split_events_by_utc_year_skips_existing_without_overwrite(tmp_path):
    events = pd.DataFrame({
        "rp_story_id": ["s1"],
        "timestamp_utc": ["2019-06-01 12:00:00"],
        "signal_calendar_date": ["2019-06-01"],
    })
    lib.split_events_by_utc_year(events, tmp_path)
    target = tmp_path / "macro_2019.csv"
    target.write_text("sentinel\n", encoding="utf-8")

    lib.split_events_by_utc_year(events, tmp_path)
    assert target.read_text(encoding="utf-8") == "sentinel\n"

    lib.split_events_by_utc_year(events, tmp_path, overwrite=True)
    assert target.read_text(encoding="utf-8") != "sentinel\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_lib.py -k "delta or split" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'compute_scoring_delta'`

- [ ] **Step 3: Write minimal implementation**

Add `from pathlib import Path` to the imports at the top of `pipeline_lib.py`, then:

```python
# add to data_collection/pipeline_lib.py

def compute_scoring_delta(dedup: pd.DataFrame, scored_text_ids) -> pd.DataFrame:
    """Rows of `dedup` whose text_id has never been scored.

    The ~305k headlines already scored in raw/finbert_scores.csv are not re-scored, which is what
    keeps stage 4 to a few minutes rather than a full re-run.
    """
    if dedup["text_id"].duplicated().any():
        raise ValueError("duplicate text_id in the dedup frame; dedup before computing a delta")
    already = set(scored_text_ids)
    return dedup.loc[~dedup["text_id"].isin(already)].copy()


def split_events_by_utc_year(
    events: pd.DataFrame,
    out_dir: Path,
    overwrite: bool = False,
) -> dict[int, Path]:
    """Write one CSV per UTC year to `out_dir`, returning {year: path}.

    Keyed on the UTC year of timestamp_utc because notebook 01's SQL filters on rpa_date_utc.
    Keying on signal_calendar_date would put a 31 December evening event in the following year's
    file and make the per-year files disagree with a re-pull at every year boundary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    years = pd.to_datetime(events["timestamp_utc"], utc=True).dt.year

    written: dict[int, Path] = {}
    for year, chunk in events.groupby(years):
        target = out_dir / f"macro_{int(year)}.csv"
        written[int(year)] = target
        if target.exists() and not overwrite:
            continue
        chunk.to_csv(target, index=False)
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline_lib.py -v`
Expected: PASS, 22 passed

- [ ] **Step 5: Commit**

```bash
git add data_collection/pipeline_lib.py tests/test_pipeline_lib.py
git commit -m "feat: add scoring delta and per-UTC-year event split

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Stage 0 — discovery notebook

Requires a WRDS session. Everything before this task runs offline; everything after depends on the
`resolved_config.json` this produces.

**Files:**
- Create: `data_collection/20_stage0_discovery.ipynb`
- Test: manual — the notebook's own asserts plus the printed summary.

**Interfaces:**
- Consumes: `pipeline_config` constants; `pipeline_lib.first_complete_session`.
- Produces: `data_collection/resolved_config.json` with keys `end_date: str`, `news_years: list[int]`, `news_year_min: int`, `news_year_max: int`, `inception: dict[str, str]`, `mkt9_start: str`, `mkt11_start: str`, `resolved_at: str`.

- [ ] **Step 1: Create the notebook with a discovery cell**

Cell 1 — setup:

```python
import json
from datetime import datetime, timezone

import pandas as pd
import wrds

from data_collection import pipeline_config as cfg
from data_collection import pipeline_lib as lib

db = wrds.Connection()
print(f"Requested start: {cfg.REQUESTED_START}")
```

Cell 2 — which RavenPack years exist:

```python
rpna_tables = set(db.list_tables(library="rpna"))
requested_years = range(int(cfg.REQUESTED_START[:4]), datetime.now(timezone.utc).year + 1)

available_years = [y for y in requested_years if f"rpa_djpr_global_macro_{y}" in rpna_tables]
missing_years = [y for y in requested_years if y not in available_years]

print(f"Requested years : {list(requested_years)}")
print(f"Available       : {available_years}")
print(f"NOT available   : {missing_years or 'none'}")
assert available_years, "no rpa_djpr_global_macro_<year> tables found for the requested range"
```

Cell 3 — resolve the true end date from CRSP:

```python
end_date = db.raw_sql("SELECT MAX(dlycaldt) AS d FROM crsp.dsf_v2")["d"].iloc[0]
end_date = pd.Timestamp(end_date).date().isoformat()
print(f"Latest CRSP session: {end_date}")
```

Cell 4 — empirical ETF inception dates:

```python
ticker_sql = ", ".join(f"'{t}'" for t in cfg.ELEVEN_SECTOR_TICKERS)
returns = db.raw_sql(f"""
    WITH names AS (
        SELECT DISTINCT permno, ticker, secinfostartdt AS s,
               COALESCE(secinfoenddt, DATE '9999-12-31') AS e
        FROM crsp.stksecurityinfohist
        WHERE ticker IN ({ticker_sql})
    )
    SELECT d.dlycaldt AS session_date, n.ticker, d.dlyret AS daily_return
    FROM crsp.dsf_v2 d
    INNER JOIN names n ON d.permno = n.permno AND d.dlycaldt BETWEEN n.s AND n.e
    WHERE d.dlycaldt BETWEEN DATE '{cfg.REQUESTED_START}' AND DATE '{end_date}'
""")
wide = returns.pivot_table(index="session_date", columns="ticker", values="daily_return")
wide.index = pd.to_datetime(wide.index)

inception = {t: wide[t].first_valid_index() for t in cfg.ELEVEN_SECTOR_TICKERS}
for ticker, first in sorted(inception.items(), key=lambda kv: kv[1]):
    print(f"  {ticker:5s} first session {first.date()}")

mkt9_start = lib.first_complete_session(wide, cfg.NINE_SECTOR_TICKERS)
mkt11_start = lib.first_complete_session(wide, cfg.ELEVEN_SECTOR_TICKERS)
print(f"\nmkt9  starts {mkt9_start.date()}   mkt11 starts {mkt11_start.date()}")
```

Cell 5 — validation gate and write:

```python
assert mkt9_start <= mkt11_start, "the nine-sector series must start no later than the eleven"
assert len(cfg.NINE_SECTOR_TICKERS) == 9 and len(cfg.ELEVEN_SECTOR_TICKERS) == 11

resolved = {
    "requested_start": cfg.REQUESTED_START,
    "end_date": end_date,
    "news_years": available_years,
    "news_year_min": min(available_years),
    "news_year_max": max(available_years),
    "inception": {t: d.date().isoformat() for t, d in inception.items()},
    "mkt9_start": mkt9_start.date().isoformat(),
    "mkt11_start": mkt11_start.date().isoformat(),
    "resolved_at": datetime.now(timezone.utc).isoformat(),
}
path = cfg.save_resolved(resolved)
print(f"Wrote {path}")
print(json.dumps(resolved, indent=2))

if min(available_years) > int(cfg.REQUESTED_START[:4]):
    print(f"\nNOTE: RavenPack only reaches back to {min(available_years)}, not "
          f"{cfg.REQUESTED_START[:4]}. The pipeline runs unchanged over the shorter range; the "
          f"projected CI improvement scales down with sqrt(sample).")
```

- [ ] **Step 2: Execute the notebook**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace data_collection/20_stage0_discovery.ipynb`
Expected: exit 0; `data_collection/resolved_config.json` exists; printed inception dates show XLRE and XLC starting later than the other nine.

- [ ] **Step 3: Verify the resolved config parses**

Run: `python -c "from data_collection import pipeline_config as c; r=c.load_resolved(); print(r['news_year_min'], r['news_year_max'], r['end_date'], r['mkt9_start'], r['mkt11_start'])"`
Expected: one line of real values, no exception.

- [ ] **Step 4: Commit**

```bash
git add data_collection/20_stage0_discovery.ipynb data_collection/resolved_config.json
git commit -m "feat: add stage 0 discovery notebook resolving dates and inception

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Stage 1 — news extraction

**Files:**
- Create: `data_collection/21_stage1_news_extraction.ipynb`
- Test: manual — the notebook's own asserts.

**Interfaces:**
- Consumes: `resolved_config.json`; `pipeline_lib.split_events_by_utc_year`.
- Produces: `data_collection/raw/news/macro_<year>.csv` for every year in `news_years`.

- [ ] **Step 1: Create the notebook — split the existing extract first**

```python
import pandas as pd
import wrds

from data_collection import pipeline_config as cfg
from data_collection import pipeline_lib as lib

resolved = cfg.load_resolved()
cfg.NEWS_RAW_DIR.mkdir(parents=True, exist_ok=True)

existing = cfg.RAW_DIR / "ravenpack_core_events_2020_2025.csv"
if existing.exists():
    prior = pd.read_csv(existing)
    written = lib.split_events_by_utc_year(prior, cfg.NEWS_RAW_DIR)
    print(f"Split the existing extract into {len(written)} per-year files: {sorted(written)}")
else:
    print("No prior extract found; every year will be pulled.")

have = {int(p.stem.split("_")[1]) for p in cfg.NEWS_RAW_DIR.glob("macro_*.csv")}
need = [y for y in resolved["news_years"] if y not in have]
print(f"Already on disk: {sorted(have)}")
print(f"To pull        : {need or 'nothing'}")
```

- [ ] **Step 2: Add the pull cell, reusing notebook 01's exact filters**

```python
db = wrds.Connection()

source_attrs = db.raw_sql("""
    SELECT rp_entity_id, data_type, data_value
    FROM rpna.rpa_source_list
    WHERE data_type IN ('ENTITY_NAME', 'PUBLICATION_TYPE', 'SOURCE_RANK')
""")
sources = (source_attrs.pivot(index="rp_entity_id", columns="data_type", values="data_value")
                       .reset_index()
                       .rename(columns={"PUBLICATION_TYPE": "source_type",
                                        "SOURCE_RANK": "source_rank"}))
sources.columns.name = None
sources["source_rank"] = pd.to_numeric(sources["source_rank"], errors="coerce")
institutional = sources.loc[sources["source_rank"].eq(1)
                            & sources["source_type"].notna()
                            & sources["source_type"].ne("BLOG")]
source_id_sql = ", ".join("'" + str(v).replace("'", "''") + "'"
                          for v in institutional["rp_entity_id"].dropna().astype(str))

ET_TS = "((timestamp_utc AT TIME ZONE 'UTC') AT TIME ZONE 'America/New_York')"

for year in need:
    query = f"""
        SELECT rp_story_id, timestamp_utc,
            CASE WHEN {ET_TS}::time < TIME '16:00:00'
                 THEN {ET_TS}::date
                 ELSE ({ET_TS}::date + INTERVAL '1 day')::date END AS signal_calendar_date,
            relevance, event_relevance, rp_source_id, source_name,
            topic, "group" AS group_name, event_sentiment_score,
            headline, event_text, {year} AS source_year
        FROM rpna.rpa_djpr_global_macro_{year}
        WHERE rpa_date_utc BETWEEN DATE '{year}-01-01' AND DATE '{year}-12-31'
          AND relevance >= {cfg.RELEVANCE_MIN}
          AND event_relevance >= {cfg.EVENT_RELEVANCE_MIN}
          AND rp_source_id IN ({source_id_sql})
          AND timestamp_utc IS NOT NULL
          AND event_sentiment_score IS NOT NULL
    """
    chunk = db.raw_sql(query)
    assert len(chunk) > 0, f"{year} returned zero rows — check the table and filters"
    assert not chunk["rp_story_id"].duplicated().any(), f"{year} has duplicate rp_story_id"
    target = cfg.NEWS_RAW_DIR / f"macro_{year}.csv"
    chunk.to_csv(target, index=False)
    print(f"  {year}: {len(chunk):,} events -> {target.name}")
```

- [ ] **Step 3: Add the validation gate**

```python
files = sorted(cfg.NEWS_RAW_DIR.glob("macro_*.csv"))
frames = {int(p.stem.split("_")[1]): pd.read_csv(p) for p in files}

for year, frame in sorted(frames.items()):
    assert len(frame) > 0, f"{year} is empty"
    utc_years = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.year.unique()
    assert set(utc_years) == {year}, f"macro_{year}.csv contains UTC years {sorted(utc_years)}"

combined = pd.concat(frames.values(), ignore_index=True)
assert not combined["rp_story_id"].duplicated().any(), "duplicate rp_story_id across years"
print(f"Total events {len(combined):,} across {len(frames)} years "
      f"({min(frames)}..{max(frames)}); no duplicate story ids.")
```

- [ ] **Step 4: Execute and verify**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace data_collection/21_stage1_news_extraction.ipynb`
Expected: exit 0; one `macro_<year>.csv` per resolved year; the printed total is roughly 67k × the number of years.

- [ ] **Step 5: Commit (notebook only — `raw/` is gitignored)**

```bash
git add data_collection/21_stage1_news_extraction.ipynb
git commit -m "feat: add stage 1 resumable per-year news extraction

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Stage 2 — price extraction over the resolved window

**Files:**
- Modify: `data_collection/02_crsp_sector_etf_price_extraction.ipynb` (cell 2, the constants cell)
- Test: manual — the notebook's existing reconciliation assert.

**Interfaces:**
- Consumes: `resolved_config.json`.
- Produces: `data_collection/market_daily_2015_2026.csv`, `raw/crsp_sector_etf_daily_raw_2015_2026.csv`.

- [ ] **Step 1: Replace the hard-coded dates and output paths in cell 2**

Replace these lines:

```python
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
```

with:

```python
from data_collection import pipeline_config as cfg

_resolved = cfg.load_resolved()
START_DATE = cfg.REQUESTED_START
END_DATE = _resolved["end_date"]
```

and replace the two output-path lines:

```python
RAW_PRICES_CSV = RAW_DIR / f"crsp_sector_etf_daily_raw_{START_DATE[:4]}_{END_DATE[:4]}.csv"
MARKET_DAILY_CSV = NOTEBOOK_DIR / "market_daily_df.csv"
```

with:

```python
RAW_PRICES_CSV = RAW_DIR / f"crsp_sector_etf_daily_raw_{cfg.SAMPLE_TAG}.csv"
MARKET_DAILY_CSV = cfg.versioned("market_daily")   # never overwrites the committed gold file
```

- [ ] **Step 2: Relax the all-tickers assert, which cannot hold before June 2018**

Replace:

```python
missing_tickers = set(TARGET_TICKERS) - tickers_found
if missing_tickers:
    raise RuntimeError(f"Could not retrieve all sector ETF tickers from CRSP: missing {missing_tickers}")
```

with:

```python
missing_tickers = set(TARGET_TICKERS) - tickers_found
if missing_tickers:
    raise RuntimeError(f"Could not retrieve all sector ETF tickers from CRSP: missing {missing_tickers}")
# XLRE (Oct 2015) and XLC (Jun 2018) legitimately have no rows before their inception; the
# per-ticker coverage table below is the check that matters, not equal row counts.
coverage_start = market_daily_raw.groupby("ticker")["session_date"].min()
print(coverage_start.sort_values().to_string())
```

- [ ] **Step 3: Execute and verify**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace data_collection/02_crsp_sector_etf_price_extraction.ipynb`
Expected: exit 0; "Validation checks passed."; `open_price` missing under 2%; reconciliation assert passes; the coverage table shows XLRE and XLC starting later than the other nine.

- [ ] **Step 4: Confirm the committed gold file is untouched**

Run: `git status --short data_collection/market_daily_df.csv`
Expected: no output — the original is unchanged; only `market_daily_2015_2026.csv` is new.

- [ ] **Step 5: Commit**

```bash
git add data_collection/02_crsp_sector_etf_price_extraction.ipynb data_collection/market_daily_2015_2026.csv
git commit -m "feat: extend CRSP pull to the resolved window with versioned output

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Stages 3 and 4 — scoring prep delta and FinBERT

**Files:**
- Create: `data_collection/22_stage34_finbert_delta.ipynb`
- Test: manual — the notebook's own asserts.

**Interfaces:**
- Consumes: `raw/news/macro_<year>.csv`; existing `raw/finbert_scores.csv`; `pipeline_lib.compute_scoring_delta`.
- Produces: `raw/finbert_scores_2015_2026.csv` with columns `text_id, label, sentiment, confidence, model, scored_at`; `raw/llm_scoring_event_map_2015_2026.csv` with columns `rp_story_id, timestamp_utc, signal_calendar_date, event_sentiment_score, text_id`.

- [ ] **Step 1: Build the dedup set and the delta**

```python
import hashlib

import pandas as pd

from data_collection import pipeline_config as cfg
from data_collection import pipeline_lib as lib

events = pd.concat(
    [pd.read_csv(p) for p in sorted(cfg.NEWS_RAW_DIR.glob("macro_*.csv"))],
    ignore_index=True,
)


def text_id_of(headline: str, event_text: str) -> str:
    """Byte-identical to make_text_id in notebook 07a — verified to reproduce every existing id."""
    payload = f"{headline}\x1f{event_text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


# The .str.strip() matters: 07a strips before hashing, so omitting it would give a different digest
# for any whitespace-padded headline and make those texts look unscored.
for col in ("headline", "event_text"):
    events[col] = events[col].fillna("").astype(str).str.strip()

events["text_id"] = [text_id_of(h, t) for h, t in zip(events["headline"], events["event_text"])]

# --- GATE: the hashing must reproduce the ids already on disk, or the whole delta is wrong ---
prior_map_path = cfg.RAW_DIR / "llm_scoring_event_map.csv"
if prior_map_path.exists():
    prior_map = pd.read_csv(prior_map_path, usecols=["rp_story_id", "text_id"])
    check = events[["rp_story_id", "text_id"]].merge(
        prior_map, on="rp_story_id", how="inner", suffixes=("_new", "_old"))
    if len(check):
        agreement = (check["text_id_new"] == check["text_id_old"]).mean()
        print(f"text_id reproduction on {len(check):,} overlapping stories: {agreement:.4f}")
        assert agreement == 1.0, (
            "text_id hashing does not reproduce notebook 07a's ids; the delta would re-score "
            "everything and the join to finbert_scores.csv would silently break"
        )

dedup = (events[["text_id", "headline", "event_text"]]
         .drop_duplicates("text_id")
         .reset_index(drop=True))

prior_path = cfg.RAW_DIR / "finbert_scores.csv"
prior = pd.read_csv(prior_path) if prior_path.exists() else pd.DataFrame(columns=["text_id"])
delta = lib.compute_scoring_delta(dedup, prior["text_id"])

print(f"Events {len(events):,} | distinct texts {len(dedup):,} | "
      f"already scored {len(prior):,} | to score {len(delta):,}")
assert set(delta["text_id"]) & set(prior["text_id"]) == set()
```

- [ ] **Step 2: Score only the delta**

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Scoring {len(delta):,} texts on {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
order = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
pos_i, neg_i = order.index("positive"), order.index("negative")

rows = []
BATCH = 64
texts = delta["headline"].tolist()
ids = delta["text_id"].tolist()
with torch.no_grad():
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=64,
                        return_tensors="pt").to(device)
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        for text_id, p in zip(ids[start:start + BATCH], probs):
            rows.append({
                "text_id": text_id,
                "label": order[int(p.argmax())],
                "sentiment": float(p[pos_i] - p[neg_i]),
                "confidence": float(p.max()),
                "model": MODEL_NAME,
                "scored_at": pd.Timestamp.utcnow().isoformat(),
            })

new_scores = pd.DataFrame(rows)
print(f"Scored {len(new_scores):,} new texts")
```

- [ ] **Step 3: Concatenate, dedupe on write, and validate**

```python
all_scores = pd.concat([prior, new_scores], ignore_index=True)
before = len(all_scores)
all_scores = all_scores.drop_duplicates("text_id", keep="first").reset_index(drop=True)
if before != len(all_scores):
    print(f"Dropped {before - len(all_scores):,} duplicate text_id rows on write")

assert not all_scores["text_id"].duplicated().any()
assert set(dedup["text_id"]).issubset(set(all_scores["text_id"])), "some texts went unscored"

scores_path = cfg.RAW_DIR / f"finbert_scores_{cfg.SAMPLE_TAG}.csv"
all_scores.to_csv(scores_path, index=False)

event_map = events[["rp_story_id", "timestamp_utc", "signal_calendar_date",
                    "event_sentiment_score", "text_id"]]
map_path = cfg.RAW_DIR / f"llm_scoring_event_map_{cfg.SAMPLE_TAG}.csv"
event_map.to_csv(map_path, index=False)

print(f"Wrote {scores_path.name} ({len(all_scores):,}) and {map_path.name} ({len(event_map):,})")
```

- [ ] **Step 4: Execute and verify**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace data_collection/22_stage34_finbert_delta.ipynb`
Expected: exit 0; "to score" is far below the distinct-text count (the prior ~305k are reused); no assertion errors.

- [ ] **Step 5: Commit (notebook only)**

```bash
git add data_collection/22_stage34_finbert_delta.ipynb
git commit -m "feat: add stage 3-4 delta-only FinBERT scoring

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Stage 5 — panel assembly

**Files:**
- Create: `data_collection/23_stage5_panel_assembly.ipynb`
- Test: manual — the notebook's own asserts.

**Interfaces:**
- Consumes: `market_daily_2015_2026.csv`; `raw/news/macro_<year>.csv`; `raw/finbert_scores_2015_2026.csv`; `raw/llm_scoring_event_map_2015_2026.csv`; all of `pipeline_lib`.
- Produces: `data_collection/model_inputs_2015_2026.csv` — one row per session, indexed by `session_date`, with columns `mkt9`, `mkt11`, the seven `rp_*` features, the seven `fb_*` features, and `y_magnitude_mkt9`, `y_direction_mkt9`.

- [ ] **Step 1: Build both market series from the resolved inception dates**

```python
import numpy as np
import pandas as pd

from data_collection import pipeline_config as cfg
from data_collection import pipeline_lib as lib

resolved = cfg.load_resolved()
market = pd.read_csv(cfg.versioned("market_daily"), parse_dates=["session_date"])
wide = market.pivot_table(index="session_date", columns="ticker", values="daily_return")

mkt9 = lib.build_market_series(wide, cfg.NINE_SECTOR_TICKERS).rename("mkt9")
mkt11 = lib.build_market_series(wide, cfg.ELEVEN_SECTOR_TICKERS).rename("mkt11")

assert mkt9.index.min() <= mkt11.index.min()
assert str(mkt11.index.min().date()) >= resolved["mkt11_start"]
print(f"mkt9  {len(mkt9):,} sessions from {mkt9.index.min().date()}")
print(f"mkt11 {len(mkt11):,} sessions from {mkt11.index.min().date()}")

overlap = pd.concat([mkt9, mkt11], axis=1).dropna()
print(f"overlap corr {overlap['mkt9'].corr(overlap['mkt11']):.4f} over {len(overlap):,} sessions")
```

- [ ] **Step 2: Build the pre-open news features for both scorers**

```python
sessions = mkt9.index

events = pd.concat(
    [pd.read_csv(p) for p in sorted(cfg.NEWS_RAW_DIR.glob("macro_*.csv"))],
    ignore_index=True,
)
events["et"] = lib.to_et(events["timestamp_utc"])
events = events.loc[lib.pre_open_mask(events["et"])].copy()
events["session_date"] = lib.assign_signal_session(events["et"], sessions)
events = events.dropna(subset=["session_date"])
print(f"Pre-open events mapped to a session: {len(events):,}")

rp_features = lib.build_news_features(events, "event_sentiment_score", "rp")

scores = pd.read_csv(cfg.RAW_DIR / f"finbert_scores_{cfg.SAMPLE_TAG}.csv",
                     usecols=["text_id", "sentiment"])
fb_events = events.merge(scores, on="text_id", how="inner")
assert len(fb_events) > 0.95 * len(events), "FinBERT coverage of pre-open events fell below 95%"
fb_features = lib.build_news_features(fb_events, "sentiment", "fb")
```

- [ ] **Step 3: Assemble, label, validate, and write**

```python
panel = (pd.DataFrame({"mkt9": mkt9})
         .join(mkt11, how="left")
         .join(rp_features, how="left")
         .join(fb_features, how="left"))

abs_r = panel["mkt9"].abs()
panel["y_magnitude_mkt9"] = lib.magnitude_label(abs_r, lib.trailing_median(abs_r))
panel["y_direction_mkt9"] = lib.direction_label(panel["mkt9"])

unlabelled = int(panel["y_magnitude_mkt9"].isna().sum())
assert unlabelled == cfg.MEDIAN_WINDOW, (
    f"expected exactly {cfg.MEDIAN_WINDOW} unlabelled sessions, got {unlabelled}"
)
assert not panel.index.duplicated().any()
assert wide[list(cfg.NINE_SECTOR_TICKERS)].loc[mkt9.index].notna().all().all()

target = cfg.versioned("model_inputs")
panel.to_csv(target)
print(f"Wrote {target.name}: {len(panel):,} sessions x {panel.shape[1]} columns")
print(f"Magnitude base rate {panel['y_magnitude_mkt9'].mean():.3f} | "
      f"direction {panel['y_direction_mkt9'].mean():.3f}")
```

- [ ] **Step 4: Execute and verify**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace data_collection/23_stage5_panel_assembly.ipynb`
Expected: exit 0; session count materially above the current 1,508; magnitude base rate near 0.48; exactly 60 unlabelled sessions.

- [ ] **Step 5: Commit**

```bash
git add data_collection/23_stage5_panel_assembly.ipynb data_collection/model_inputs_2015_2026.csv
git commit -m "feat: add stage 5 panel assembly for the extended sample

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Stage 6 — re-run the model grid

The harness does not change. Only the input file and the market series it reads do.

**Files:**
- Create: `ml/24_stage6_extended_model.ipynb` (a copy of `ml/11_news_flow_magnitude_model.ipynb` with its data-loading cells replaced)
- Test: the notebook's own positive-control gate.

**Interfaces:**
- Consumes: `data_collection/model_inputs_2015_2026.csv`; `pipeline_config` harness constants.
- Produces: `model_outputs/news_flow_magnitude_metrics_2015_2026.csv`, `model_outputs/news_flow_magnitude_summary_2015_2026.png`.

- [ ] **Step 1: Copy notebook 11 and replace its data-loading cells**

```bash
cp ml/11_news_flow_magnitude_model.ipynb ml/24_stage6_extended_model.ipynb
```

Replace the three cells that load raw events, build FinBERT features, and assemble `df` with a
single cell — stage 5 has already done that work:

```python
import numpy as np
import pandas as pd

from data_collection import pipeline_config as cfg

MEDIAN_WINDOW, N_INIT = cfg.MEDIAN_WINDOW, cfg.N_INIT
REFIT_STEP, N_PERM, SEED = cfg.REFIT_STEP, cfg.N_PERM, cfg.SEED
rng = np.random.default_rng(SEED)

df = pd.read_csv(cfg.versioned("model_inputs"), parse_dates=["session_date"], index_col="session_date")
df["r"] = df["mkt9"]
df["abs_r"] = df["r"].abs()
df["rev1"] = df["r"].shift(1)
df["mom5"] = df["r"].rolling(5).sum().shift(1)
df["vol5"] = df["r"].rolling(5).std().shift(1)
df["vol20"] = df["r"].rolling(20).std().shift(1)
df["abs_r1"] = df["abs_r"].shift(1)
df["flow_ma20"] = df["rp_flow"].rolling(20).mean().shift(1)
df["y_magnitude"] = df["y_magnitude_mkt9"]
df["y_direction"] = df["y_direction_mkt9"]

RP_TONE = ["rp_tone", "rp_net", "rp_disp", "rp_absmean", "rp_tail", "rp_negshare"]
FB_TONE = ["fb_tone", "fb_net", "fb_disp", "fb_absmean", "fb_tail", "fb_negshare"]
RP_FLOW = ["rp_flow"]
VOL_BASE = ["vol20", "vol5", "abs_r1"]
DIR_BASE = ["rev1", "mom5", "vol20"]

print(f"Sessions {len(df):,} (was 1,508) | magnitude base rate {df['y_magnitude'].mean():.3f}")
```

- [ ] **Step 2: Point the two output paths at versioned filenames**

Replace `OUT_DIR / "news_flow_magnitude_metrics.csv"` with
`OUT_DIR / f"news_flow_magnitude_metrics_{cfg.SAMPLE_TAG}.csv"`, and
`OUT_DIR / "news_flow_magnitude_summary.png"` with
`OUT_DIR / f"news_flow_magnitude_summary_{cfg.SAMPLE_TAG}.png"`.

- [ ] **Step 3: Execute**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=6000 ml/24_stage6_extended_model.ipynb`
Expected: exit 0.

- [ ] **Step 4: Check the positive-control gate before reading any other number**

Run: `python -c "import pandas as pd; d=pd.read_csv('model_outputs/news_flow_magnitude_metrics_2015_2026.csv'); c=d[d.spec.str.startswith('4.')].iloc[0]; print(f\"control AUC {c.auc_with_news:.4f} p={c.p_value:.3f} clears={c.clears_perm}\")"`
Expected: `clears=True`. **If it is False, stop.** No other spec in the run is readable, and the cause must be found before the results are used.

- [ ] **Step 5: Compare against the 2020–2025 baseline**

Run: `python -c "import pandas as pd; a=pd.read_csv('model_outputs/news_flow_magnitude_metrics.csv'); b=pd.read_csv('model_outputs/news_flow_magnitude_metrics_2015_2026.csv'); m=a.merge(b,on='spec',suffixes=('_old','_new')); print(m[['spec','n_oos_old','n_oos_new','auc_with_news_old','auc_with_news_new','auc_ci_low_new','auc_ci_high_new','p_value_new']].round(4).to_string(index=False))"`
Expected: `n_oos_new` materially above 948, and CI widths narrower than the old ±0.031.

- [ ] **Step 6: Commit**

```bash
git add ml/24_stage6_extended_model.ipynb model_outputs/news_flow_magnitude_metrics_2015_2026.csv model_outputs/news_flow_magnitude_summary_2015_2026.png
git commit -m "feat: re-run the model grid on the extended 2015-2026 sample

Harness unchanged from the verified 2020-2025 run so any movement in the
numbers is attributable to sample size alone.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.** Stage 0 → Task 7. Stage 1 → Tasks 6, 8. Stage 2 → Task 9. Stages 3–4 → Task 10.
Stage 5 → Task 11. Stage 6 → Task 12. `pipeline_config.py` → Task 1. Both mandated unit tests →
Tasks 2 and 3. Validation gates → carried in the notebook of each stage plus Tasks 2–6. Versioned
outputs → Tasks 9, 10, 11, 12. Delta-only scoring → Tasks 6 and 10. The two market series →
Tasks 5, 7, 11. FinBERT double-write guard → Task 10 Step 3. Positive-control gate → Task 12 Step 4.

**Gap found and closed.** The spec's stage-1 gate says "signal dates inside the year ±1 day", which
is unenforceable once `assign_signal_session` rolls a Friday-evening event to Monday — up to three
days. Task 8 Step 3 asserts on the **UTC year of `timestamp_utc`** instead, which is the column the
SQL actually filters on and is exact.

**Naming consistency.** `versioned(stem)` is used identically in Tasks 9, 11, 12. `build_news_features`
takes `(events, score_col, prefix, session_col="session_date")` in Task 4 and is called that way in
Task 11. `magnitude_label(abs_r, median)` and `trailing_median(abs_r, window)` are defined in Task 3
and called in Task 11. `compute_scoring_delta(dedup, scored_text_ids)` is defined in Task 6 and
called in Task 10. `first_complete_session` is defined in Task 5 and called in Task 7.

**Hashing verified, not assumed.** `text_id_of` in Task 10 recomputes the sha1 rather than importing
notebook 07a. Checked against 07a's `make_text_id` and against the real data: 200,000 overlapping
stories reproduce their existing `text_id` with **1.0 agreement**. The `.str.strip()` before hashing
is required to match 07a — this sample happens to have no padded headlines, so both variants agree
today, but the 2015–2019 years are unverified and stripping is what 07a actually does. Task 10 Step 1
carries a hard assert on that agreement, so a digest mismatch fails loudly instead of silently
re-scoring 305k texts and breaking the join.
