"""Cleaning transforms: raw events + price panel -> a model-ready table.

Every function here is pure -- a function of its arguments, with no filesystem or database
dependency -- so the logic that decides which news is usable and which sessions are labelled can
be unit-tested. Two silent defects were found in this project by testing exactly these two things,
so they are not incidental helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EASTERN = "America/New_York"

MARKET_CLOSE_MIN = 16 * 60        # 16:00 ET, end of a session
MARKET_OPEN_MIN = 9 * 60 + 30     # 09:30 ET, first tradeable price
MEDIAN_WINDOW = 60                # trailing window defining the magnitude label

TAIL_THRESHOLD = 0.7
NEGATIVE_THRESHOLD = -0.3

LATE_LAUNCH_TICKERS = ("XLRE", "XLC")   # XLRE Oct 2015, XLC Jun 2018


# ---------------------------------------------------------------------------
# news window
# ---------------------------------------------------------------------------
def to_et(ts_utc: pd.Series) -> pd.Series:
    """Parse a UTC timestamp column and convert it to US Eastern."""
    return pd.to_datetime(ts_utc, utc=True).dt.tz_convert(EASTERN)


def minute_of_day(et: pd.Series) -> pd.Series:
    return et.dt.hour * 60 + et.dt.minute


def pre_open_mask(et: pd.Series) -> pd.Series:
    """True for news arriving after the previous close and before this session's open.

    This is the only news that is not already priced at the moment the target return begins.
    Measured on this dataset, tone over the full signal window correlates +0.134 (t=5.2) with the
    session it is contemporaneous with and +0.001 (t=0.0) with the next one -- the measure works,
    it was simply pointed at a return that had already happened.
    """
    minutes = minute_of_day(et)
    return (minutes >= MARKET_CLOSE_MIN) | (minutes < MARKET_OPEN_MIN)


def assign_signal_session(et: pd.Series, sessions) -> pd.Series:
    """Map each timestamp to the first trading session whose close it can still affect.

    News before 16:00 ET can affect that calendar day's close; at or after 16:00 it cannot, so it
    rolls to the next day. Either way the result rolls forward to the next actual trading session,
    which is what keeps weekend and holiday news. Joining on the raw signal_calendar_date instead
    silently drops 6.6% of events -- 18,585 of them Saturday-dated, and systematically more
    negative than the ones that survive. Timestamps past the final session return NaT.
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


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def build_news_features(
    events: pd.DataFrame,
    score_col: str,
    prefix: str,
    session_col: str = "session_date",
) -> pd.DataFrame:
    """Session-level aggregates of a per-event sentiment score.

    `flow` is arrival volume and carries the only effect that has cleared both references so far;
    the tone columns exist so RavenPack and FinBERT are compared on identical statistics.
    """
    grouped = events.groupby(session_col)[score_col]
    absolute = events.assign(_x=events[score_col].abs()).groupby(session_col)["_x"]
    tail = events.assign(_x=events[score_col].abs() > TAIL_THRESHOLD).groupby(session_col)["_x"]
    negative = events.assign(_x=events[score_col] < NEGATIVE_THRESHOLD).groupby(session_col)["_x"]

    return pd.DataFrame({
        f"{prefix}_flow": np.log1p(grouped.size()),
        f"{prefix}_tone": grouped.mean(),
        f"{prefix}_net": grouped.sum(),
        f"{prefix}_disp": grouped.std(),
        f"{prefix}_absmean": absolute.mean(),
        f"{prefix}_tail": tail.mean(),
        f"{prefix}_negshare": negative.mean(),
    })


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def trailing_median(abs_r: pd.Series, window: int = MEDIAN_WINDOW) -> pd.Series:
    """Median of the previous `window` absolute returns, excluding the current session."""
    return abs_r.rolling(window).median().shift(1)


def magnitude_label(abs_r: pd.Series, median: pd.Series) -> pd.Series:
    """1 where the move exceeds its trailing median, 0 where it does not, NaN where undefined.

    Built with np.where rather than a comparison plus .astype(int) because `NaN > NaN` is False:
    astype would label every session before the trailing window fills as a confident 0, and since
    the result would then never be NaN, dropna could not remove them. That exact defect put 60
    fabricated rows in the first training block and failed the positive control.
    """
    defined = median.notna() & abs_r.notna()
    return pd.Series(np.where(defined, abs_r > median, np.nan),
                     index=abs_r.index, dtype="float64")


def direction_label(returns: pd.Series) -> pd.Series:
    """1 for an up session, 0 otherwise, NaN where the return is missing."""
    return pd.Series(np.where(returns.notna(), returns > 0, np.nan),
                     index=returns.index, dtype="float64")


# ---------------------------------------------------------------------------
# market series
# ---------------------------------------------------------------------------
def _require_tickers(wide_returns: pd.DataFrame, tickers) -> list[str]:
    wanted = list(tickers)
    missing = [t for t in wanted if t not in wide_returns.columns]
    if missing:
        raise KeyError(f"tickers absent from the return matrix: {missing}")
    return wanted


def build_market_series(wide_returns: pd.DataFrame, tickers) -> pd.Series:
    """Equal-weight return of exactly `tickers`, on sessions where all of them trade.

    Sessions missing a constituent are dropped rather than averaged over whatever is present, so
    the series carries one constant definition and no composition break.
    """
    wanted = _require_tickers(wide_returns, tickers)
    return wide_returns[wanted].dropna(how="any").mean(axis=1)


def first_complete_session(wide_returns: pd.DataFrame, tickers) -> pd.Timestamp:
    """Earliest session on which every one of `tickers` trades -- an empirical inception date."""
    wanted = _require_tickers(wide_returns, tickers)
    complete = wide_returns[wanted].dropna(how="any")
    if complete.empty:
        raise ValueError(f"no session has all of {wanted} present")
    return complete.index.min()


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------
def add_price_controls(frame: pd.DataFrame, return_col: str = "r") -> pd.DataFrame:
    """Lagged price and volatility controls. Every column is strictly past-looking."""
    out = frame.copy()
    out["abs_r"] = out[return_col].abs()
    out["rev1"] = out[return_col].shift(1)
    out["mom5"] = out[return_col].rolling(5).sum().shift(1)
    out["vol5"] = out[return_col].rolling(5).std().shift(1)
    out["vol20"] = out[return_col].rolling(20).std().shift(1)
    out["abs_r1"] = out["abs_r"].shift(1)
    out["med60"] = trailing_median(out["abs_r"])
    return out


def add_labels(frame: pd.DataFrame, return_col: str = "r") -> pd.DataFrame:
    out = frame.copy()
    out["y_magnitude"] = magnitude_label(out["abs_r"], out["med60"])
    out["y_direction"] = direction_label(out[return_col])
    return out
