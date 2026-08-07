# Sample Extension Pipeline — Design

**Date:** 2026-08-07
**Author:** Dongxin (NLP/LLM lead), Team Alpha Signal, SIADS 699
**Status:** Approved for implementation planning

## Why

Notebook 11 produced the project's first result that clears both of its references:
out-of-sample AUC **0.5437, 95% CI [0.5119, 0.5737]**, predicting whether the next session's absolute
market move exceeds its trailing 60-day median, from news arrival volume in the non-leaky pre-open
window. Direction remains a null (0.4916) with a detectable-effect bound of **0.5164**.

Both numbers are limited by the same thing: `n_oos = 948`. The CI half-width is 0.031, so no model
improvement smaller than ~0.03 AUC is measurable at all, and model capacity has already been tested
and found to hurt (HistGB 0.4812, RF 0.4909, logistic 0.5002). More sample is the only lever that
moves every number in the project at once.

Extending 2020–2025 to 2015–2026 is projected to take the magnitude CI to roughly [0.524, 0.563] and
the direction bound to roughly 0.510. The tighter bound is the more valuable of the two: it upgrades
the null from "nothing above 0.5164" to "nothing above 0.510".

## Decisions taken

| Decision | Choice | Reason |
|---|---|---|
| Window | `START_DATE` 2015-01-01, `END_DATE` resolved at run time, **two market series** | See below |
| Output paths | Versioned `*_2015_2026.csv`, originals untouched | Christian's notebooks consume the existing gold files |
| Config | One `pipeline_config.py` imported by all notebooks | Six notebooks with private `START_DATE` constants desynchronise silently |
| Harness | **Unchanged** | Changing the measurement and the sample together makes movement unattributable |

### The two market series

XLRE launched October 2015 and XLC launched June 2018 (the latter when GICS split Communication
Services out of Telecom). A 2015–2026 panel is therefore unbalanced: 9 sectors, then 10, then 11.
Composition breaks inside a volatility target are a known source of spurious results.

Rather than sacrificing either sample length or cross-sectional balance, the two uses get the series
each actually needs:

- **`mkt9`** — equal weight of the nine SPDRs present throughout, 2015–2026. Carries the aggregate
  models in notebook 11.
- **`mkt11`** and the full 11-ticker panel — June 2018 onward. Carries anything cross-sectional,
  including the firm-level path in `01b_ravenpack_equity_news_extraction.ipynb`.

Measured on the existing 2020–2025 panel, the two definitions agree closely — and because the
modelling target is a binary label rather than the raw series, label agreement is the figure that
matters:

| Comparison | Value |
|---|---|
| corr(`mkt9`, `mkt11`) daily return | 0.9958 |
| corr of absolute returns | 0.9934 |
| daily std | 0.01255 vs 0.01248 |
| **magnitude label agreement** | **0.9423** |
| **direction label agreement** | **0.9635** |

So ~5.8% of magnitude labels differ between the two definitions. That is small enough that `mkt9` is
a sound proxy for the aggregate work, and large enough that the two series must not be treated as
interchangeable or silently swapped between notebooks. The report states the split explicitly and
reproduces this table.

## Stage graph

Each stage is resumable and depends only on files, not on in-memory state from a prior stage. Any
stage can be re-run alone provided its inputs exist.

```
0  discover   → data_collection/resolved_config.json
1  news       → raw/news/macro_<year>.csv          (one per year, skipped if present)
2  prices     → raw/crsp_sector_etf_daily_2015_2026.csv
3  score prep → raw/scoring_input_delta_2015_2026.csv
4  finbert    → raw/finbert_scores_2015_2026.csv
5  assemble   → market_daily_2015_2026.csv, model_daily_panel_2015_2026.csv,
                news_daily_2015_2026.csv, finbert_daily_2015_2026.csv
6  model      → notebook 11 re-run on the extended panel
```

### Stage 0 — discovery

Queries which `rpna.rpa_djpr_global_macro_<year>` tables actually exist, and the true first session
of each of the eleven ETFs from `crsp.stksecurityinfohist`. Writes `resolved_config.json` recording
the **actual** available range and inception dates.

This exists so that a subscription that only reaches back to, say, 2017 is discovered in thirty
seconds rather than nine years into a pull, and so the XLRE/XLC inception dates are established
empirically rather than assumed. Every later stage reads its dates from the resolved config, never
from a hard-coded literal.

`END_DATE` is **not** a literal in the config. Stage 0 resolves it to the latest session actually
present in `crsp.dsf_v2` at run time and records it in `resolved_config.json`, so a mid-2026 re-run
picks up new sessions without an edit, and every downstream stage shares one unambiguous end date.
The RavenPack year range is likewise clamped to the tables that exist rather than to `range(2015,
2027)`.

### Stage 1 — news extraction

Extends `01_ravenpack_news_extraction.ipynb`. Identical filters (relevance ≥ 90, event relevance
≥ 90, rank-1 non-blog institutional sources, 16:00 ET signal-date rule) so the extended sample is
comparable to the existing one. Pulls one year at a time into `raw/news/macro_<year>.csv` and skips
any year whose file already exists.

The stage **begins by splitting the existing `ravenpack_core_events_2020_2025.csv` into per-year
files**, so 2020–2025 is never re-pulled. Only 2015–2019 and 2026 hit WRDS. This is a required step,
not an optimisation: it also guarantees the already-scored headlines keep byte-identical text and
therefore identical sha1 `text_id`s, which is what makes the stage 3 delta correct.

Expected scale: ~67k events/year, so ~780k events over 11.6 years, ~195 MB raw.

### Stage 2 — price extraction

Extends `02_crsp_sector_etf_price_extraction.ipynb`, which has already been patched to select
`d.dlyopen` and derive `overnight_gap` and `open_to_close` with a reconciliation assert, **and has
already been re-run for 2020–2025**. For this stage it is therefore a date-range change only. CRSP
is a single fast pull; no per-year chunking needed. Builds both `mkt9` and `mkt11` from the resolved
inception dates.

The 2020–2025 re-run validates cleanly — `open_price` 0.00% missing, reconciliation error 4.44e-16 —
and establishes the figure the report needs: **41.8% of the average absolute session move occurs in
the overnight gap** (mean |gap| 0.591% vs mean |open→close| 0.822%). That is the share of what these
models predict which is *not* capturable by trading at the open, and it must be stated wherever a
result is described. It is the difference between a finding about predictability and a claim about
a strategy.

### Stage 3 — scoring prep (delta only)

Extends `07a_llm_scoring_input_prep.ipynb`. Dedups `(headline, event_text)` to a stable sha1
`text_id` as before, then **diffs against the existing `raw/finbert_scores.csv`** and emits only
text_ids that have never been scored. The 304,809 already-scored headlines are not re-scored.

Expected delta: ~275k new texts.

### Stage 4 — FinBERT scoring

Extends `07_finbert_sentiment_scoring.ipynb`. Scores the delta only (~3 minutes at ~1,600/s on the
RTX 5070 Ti), then concatenates with the existing scores and writes the versioned file.

### Stage 5 — panel assembly

Builds the modelling inputs on the re-aligned non-leaky pre-open window `(close d−1, 09:30 d]`:
`mkt9`, `mkt11`, the 11-ticker panel, and the daily RavenPack and FinBERT aggregates.

### Stage 6 — model

`ml/11_news_flow_magnitude_model.ipynb` re-run against the extended panel, **harness untouched**.

`N_INIT` stays at **500 sessions in absolute terms**, not rescaled proportionally. This keeps the
training-window semantics identical to the current run — the first prediction is always made after
500 sessions of history — so the whole of the additional sample lands in the out-of-sample set and
the new AUC is directly comparable to 0.5437. Everything else (the six-spec grid, the 250-draw block
permutation, the block-bootstrap CI, the positive control, `MEDIAN_WINDOW`, `REFIT_STEP`, the seed)
is unchanged.

## Validation

Every stage ends in asserts that raise rather than warn.

| Stage | Gate |
|---|---|
| 0 | every year in the requested range resolves to an existing table, or the config records the real range |
| 1 | per-year row counts non-zero; no duplicate `rp_story_id`; signal dates inside the year ±1 day |
| 2 | no duplicate `(asset, session_date)`; `open_price` missing < 2%; gap × open→close reconciles to the raw price move within 1e-9 |
| 3 | `text_id` unique; delta ∩ already-scored is empty; `sum(n_events)` equals the event row count |
| 4 | every delta text_id scored exactly once; no duplicate text_id after the concatenation |
| 5 | `mkt9` has exactly 9 constituents throughout; `mkt11` starts no earlier than XLC inception; magnitude label NaN for exactly `MEDIAN_WINDOW` sessions |
| 6 | **positive control (Spec 4, trailing volatility) must clear**, or no other spec is readable |

### Unit tests

Both bugs found on 2026-08-07 were silent, and one survived a clean exit code. The two functions
where a bug destroys the result without raising get real tests in `tests/`:

1. **Pre-open window selection** — that events at or after 16:00 ET map to the next session, events
   before 09:30 ET map to the current one, and intraday events are excluded. Table-driven across a
   normal day, a Friday→Monday roll, and a holiday.
2. **NaN-preserving label construction** — that `y` is NaN, not 0, wherever the trailing window is
   not yet full. This is the exact defect that failed the positive control: `NaN > NaN` is `False`,
   and `.astype(int)` silently turned 60 unlabelled sessions into confident zeros that `dropna`
   could not remove.

## Failure handling

- **WRDS connection drops mid-pull** — per-year files mean the run resumes at the failed year.
- **A requested year has no table** — stage 0 catches it; the resolved config records the real range
  and later stages narrow to it rather than failing.
- **FinBERT double-write** — running the notebook and a background job simultaneously previously
  produced 20k duplicate rows. Delta-only scoring plus a dedup-on-write guard makes this structurally
  impossible rather than something to remember.
- **Silent data defects** — covered by the stage gates and the two unit tests above.

## Out of scope

- Model families, hyperparameter search, ensembles, new architectures. Capacity is tested and hurts;
  the CI half-width means sub-0.03 gains are unmeasurable regardless.
- Changes to notebook 11's grid, permutation harness, or bootstrap CI.
- The firm-level equities path (`01b`), which is written but unexecuted and independent of this work.
- The `dlyopen` change itself — already written and already run for 2020–2025. This pipeline only
  widens its date range.
- Migrating Christian's notebooks onto the extended panel. Versioned outputs exist precisely so that
  switchover is a separate, deliberate team decision.

## Open dependency

Whether RavenPack reaches back to 2015 on this WRDS subscription is unconfirmed. Stage 0 resolves it.
If the real floor is later than 2015, the pipeline runs unchanged over the shorter range and the
projected CI improvements scale down accordingly.
