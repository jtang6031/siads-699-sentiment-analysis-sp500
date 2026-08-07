import numpy as np
import pandas as pd
import pytest

from ml import cleaning_lib as lib

NY = "America/New_York"


def et(*stamps):
    return pd.Series(pd.to_datetime(list(stamps)).tz_localize(NY))


# --------------------------------------------------------------------------
# news window
# --------------------------------------------------------------------------
def test_pre_open_mask_selects_only_non_trading_hours():
    stamps = et(
        "2024-03-05 16:00", "2024-03-05 18:30", "2024-03-05 23:59",   # after the close
        "2024-03-05 00:00", "2024-03-05 09:29",                        # before the open
        "2024-03-05 09:30", "2024-03-05 12:00", "2024-03-05 15:59",   # intraday
    )
    assert lib.pre_open_mask(stamps).tolist() == [True, True, True, True, True, False, False, False]


def test_assign_signal_session_same_day_before_close():
    sessions = pd.to_datetime(["2024-03-05", "2024-03-06"])
    assert lib.assign_signal_session(et("2024-03-05 10:00"), sessions).iloc[0] == pd.Timestamp("2024-03-05")


def test_assign_signal_session_rolls_after_close():
    sessions = pd.to_datetime(["2024-03-05", "2024-03-06"])
    assert lib.assign_signal_session(et("2024-03-05 16:30"), sessions).iloc[0] == pd.Timestamp("2024-03-06")


def test_assign_signal_session_friday_evening_rolls_to_monday():
    sessions = pd.to_datetime(["2024-03-08", "2024-03-11"])
    assert lib.assign_signal_session(et("2024-03-08 18:00"), sessions).iloc[0] == pd.Timestamp("2024-03-11")


def test_assign_signal_session_saturday_survives():
    """Joining on raw signal_calendar_date drops 6.6% of events here; they must survive."""
    sessions = pd.to_datetime(["2024-03-08", "2024-03-11"])
    assert lib.assign_signal_session(et("2024-03-09 10:00"), sessions).iloc[0] == pd.Timestamp("2024-03-11")


def test_assign_signal_session_skips_a_holiday():
    sessions = pd.to_datetime(["2024-03-28", "2024-04-01"])   # Good Friday closed
    assert lib.assign_signal_session(et("2024-03-28 17:00"), sessions).iloc[0] == pd.Timestamp("2024-04-01")


def test_assign_signal_session_past_the_last_session_is_nat():
    sessions = pd.to_datetime(["2024-03-05"])
    assert pd.isna(lib.assign_signal_session(et("2024-03-05 16:30"), sessions).iloc[0])


# --------------------------------------------------------------------------
# labels -- the defect that failed the positive control
# --------------------------------------------------------------------------
def test_trailing_median_is_strictly_past_looking():
    got = lib.trailing_median(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), window=3)
    assert got.isna().tolist() == [True, True, True, False, False]
    assert got.iloc[3] == 2.0
    assert got.iloc[4] == 3.0


def test_magnitude_label_is_nan_before_the_window_fills():
    abs_r = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    got = lib.magnitude_label(abs_r, lib.trailing_median(abs_r, window=3))
    assert got.isna().sum() == 3, "unlabelled rows must stay NaN, never 0"
    assert not (got.fillna(-1) == 0).iloc[:3].any()
    assert got.iloc[3] == 1.0


def test_magnitude_label_marks_small_moves_zero_not_nan():
    abs_r = pd.Series([0.05, 0.06, 0.07, 0.001])
    got = lib.magnitude_label(abs_r, lib.trailing_median(abs_r, window=3))
    assert got.iloc[3] == 0.0 and not pd.isna(got.iloc[3])


def test_direction_label_preserves_nan():
    got = lib.direction_label(pd.Series([0.01, -0.02, np.nan, 0.0]))
    assert got.iloc[0] == 1.0
    assert got.iloc[1] == 0.0
    assert pd.isna(got.iloc[2])
    assert got.iloc[3] == 0.0


def test_add_labels_leaves_exactly_median_window_unlabelled():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"r": rng.normal(0, 0.01, 200)})
    got = lib.add_labels(lib.add_price_controls(frame))
    assert int(got["y_magnitude"].isna().sum()) == lib.MEDIAN_WINDOW


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
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
    assert day1["rp_tone"] == pytest.approx(0.3 / 4)
    assert day1["rp_net"] == pytest.approx(0.3)
    assert day1["rp_absmean"] == pytest.approx(2.1 / 4)
    assert day1["rp_tail"] == pytest.approx(2 / 4)
    assert day1["rp_negshare"] == pytest.approx(1 / 4)


def test_build_news_features_one_event_day_has_nan_dispersion():
    events = pd.DataFrame({"session_date": pd.to_datetime(["2024-01-02"]), "score": [0.5]})
    got = lib.build_news_features(events, "score", "rp")
    assert pd.isna(got.iloc[0]["rp_disp"])


# --------------------------------------------------------------------------
# market series
# --------------------------------------------------------------------------
def _wide():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {"XLK": [0.01, 0.02, 0.03], "XLF": [0.03, 0.02, 0.01], "XLC": [np.nan, 0.05, 0.05]},
        index=idx,
    )


def test_build_market_series_equal_weights_named_tickers_only():
    assert lib.build_market_series(_wide(), ["XLK", "XLF"]).tolist() == pytest.approx([0.02, 0.02, 0.02])


def test_build_market_series_drops_sessions_with_a_missing_constituent():
    got = lib.build_market_series(_wide(), ["XLK", "XLF", "XLC"])
    assert len(got) == 2
    assert got.index[0] == pd.Timestamp("2024-01-03")


def test_build_market_series_rejects_an_unknown_ticker():
    with pytest.raises(KeyError, match="XLZ"):
        lib.build_market_series(_wide(), ["XLK", "XLZ"])


def test_first_complete_session_finds_the_inception_boundary():
    assert lib.first_complete_session(_wide(), ["XLK", "XLF", "XLC"]) == pd.Timestamp("2024-01-03")
    assert lib.first_complete_session(_wide(), ["XLK", "XLF"]) == pd.Timestamp("2024-01-02")


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------
def test_price_controls_are_strictly_past_looking():
    """A control that peeks at the current session would leak the answer."""
    frame = pd.DataFrame({"r": [0.01, -0.02, 0.03, -0.04, 0.05, 0.06]})
    got = lib.add_price_controls(frame)
    assert got["rev1"].iloc[1] == pytest.approx(0.01)
    assert pd.isna(got["rev1"].iloc[0])
    # vol20 needs 20 prior observations, so a 6-row frame yields none at all
    assert got["vol20"].isna().all()
