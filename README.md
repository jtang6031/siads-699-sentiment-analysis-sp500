# SIADS-699-sentiment-analysis-sp500 - Financial News and S&P 500 Sector Returns

**Academic research only. This project is not investment advice.**

A simple question: Can the language and amount of financial news help us anticipate what broad parts of the U.S. stock market will do next?

## Executive Summary

- **News sentiment did not improve next-session direction forecasts.** Across 11,022 later-date
  sector comparisons from 2022 through 2025, the market-only model and every news-based version
  scored near 0.50 AUC, which is chance-level ranking.
- **News tone had a small relationship with prices in the session around its arrival.** Across
  2015–2025, RavenPack and FinBERT tone had same-session correlations of 0.078 and 0.068 with an
  equal-weight nine-sector ETF composite. In the following session, the correlations were only
  0.008 and −0.006. This is consistent with news being reflected quickly, but market recaps and
  shared events are also possible explanations. Daily data cannot show a reaction within seconds or
  prove that news caused the move.
- **Lagged volatility was associated with full-session move size in an open-time test.** Volatility
  alone reached 0.5521 AUC. In a separate setup, adding current news volume to a 0.5507 baseline of
  lagged volatility plus trailing news flow raised AUC to 0.5604, a gain of +0.0097. The
  close-to-close target includes the overnight move that had already occurred by the market open,
  so this is not a fully before-the-fact forecast. The news increment is also sensitive to the
  statistical test and should not be described as settled evidence.
- **The Autoformer model is exploratory and cannot support an investment claim.** Its strongest row, M7, reports an
  overnight net Sharpe ratio of 1.3348 and an arithmetic annualized return of 0.109173. However, the
  close-to-open holding period began before the full signal was available, so that return could not
  have been earned as shown. A delayed-return check was negative for M5, M6, and M7.

Taken together, the primary analyses support three conclusions: sentiment did not improve
next-session direction forecasts, tone had a small same-session association with returns, and
lagged volatility was associated with full-session move size. The possible news-volume gain is
preliminary, while the Autoformer remains an academic analysis.

## Question we may ask

A **sector ETF** is a fund that follows one broad industry, such as technology, energy, or health
care. **News sentiment** describes whether financial news sounds positive, negative, or neutral.
We tested whether that information helped forecast:

1. whether a sector would go up or down next trading day.
2. whether a session's full close-to-close market move was relatively large or small, and
3. how sectors ranked against one another during the overnight period, from one market close to the
   next morning's open.

The models learned from earlier dates and were evaluated on later dates they had not seen during
training. This is better than randomly mixing old and new dates, but chronological testing alone does
not guarantee that every input was available before the measured return began. The timing limits are
called out beside the affected results below.

## Main Results

### Confirmed: news sentiment did not improve direction forecasts

The most direct test compares a market-only model with versions that add RavenPack news sentiment,
FinBERT sentiment, or both. RavenPack is the commercial news source used by the project. FinBERT is
a language model trained to recognize sentiment in financial writing.

**How to read AUC:** 0.50 means the model ranks up and down outcomes no better than chance. A score
closer to 1.00 would be better. “Change” compares each row with the market-only model.

| Model | Average AUC across yearly tests | AUC after combining all test rows | Change in average AUC vs. market-only |
|---|---:|---:|---:|
| Market information only | 0.4975 | 0.4885 | 0.0000 |
| Market + RavenPack sentiment | 0.4973 | 0.4936 | −0.0002 |
| Market + FinBERT sentiment | 0.4936 | 0.4839 | −0.0039 |
| Market + both sentiment sources | 0.4988 | 0.4917 | +0.0013 |
| Always predict “up” | 0.5000 | 0.5000 | +0.0025 |

None of the sentiment versions produced a meaningful improvement. A sector fund rose on 52.5% of
the tested sector-days. Because raw accuracy is affected by that imbalance, the table reports AUC
instead. A rule that always predicts “up” has AUC 0.5000 and provides no useful ranking information.
In this sample, the daily sentiment measures did not tell us which sectors would rise next.

Source: [`outputs/model_comparison_all.csv`](outputs/model_comparison_all.csv).

### Descriptive timing result: tone aligned modestly with same-session returns

A surprising news item can move prices quickly. That is different from asking whether its tone
predicts the next trading session. To examine the first idea at the resolution available here, we
compared pre-open tone with the equal-weight return of nine long-running sector ETFs in the
close-to-close session around the news and in the session that followed.

Here, “same session” means the previous close to the current close; “following session” means the
current close to the next close. News arrives inside the first window.

RavenPack and FinBERT tone had small positive same-session correlations of 0.078 and 0.068. Their
following-session correlations were 0.008 and −0.006, both near zero. The 95% uncertainty ranges
were [0.039, 0.112] and [0.027, 0.105] for the same session; both following-session ranges included
zero. Nearby trading days were resampled together in 21-session blocks. A correlation of zero would
mean no linear relationship. The pattern is consistent with information being reflected quickly,
but it is not proof of a causal market reaction or a tradeable forecast. The data are daily, and the
news window overlaps the overnight part of the same-session return, so this study cannot measure a
response within seconds. Some stories may also recap a move already underway or respond to the same
event that moved prices.

### Confirmed statistical result: lagged volatility separated large and small full-session moves

This second test classifies whether the equal-weight sector average had a large or small
close-to-close move. A “large” move means that its absolute return exceeded its own median over the
previous 60 sessions. **Volatility** simply means how much prices had been moving recently. The
price-based controls use only earlier sessions. The news features cover stories arriving after the
previous market close and before 9:30 a.m. Eastern, which places them inside the close-to-close
target interval. By the open, the overnight part of that target has already occurred. This makes the
news result a partly contemporaneous, open-time association—not a clean forecast of the next move.

The table reports later-date tests. The 95% confidence range shows bootstrap uncertainty around the
full model's AUC. The p-value compares the observed AUC with circularly shifted versions of the
tested feature block: news features for the news setups and lagged price features for the volatility
control. Smaller p-values are harder to explain with that shifted-feature comparison, but no single
number should be read as proof on its own.

| Model setup | Test sessions | AUC | 95% confidence range | p | Plain-language reading |
|---|---:|---:|---:|---:|---|
| News volume alone | 2,206 | 0.4902 | [.4552, .5294] | .172 | No clear pattern |
| RavenPack news tone alone | 2,206 | 0.4741 | [.4457, .5047] | .824 | No clear pattern |
| FinBERT tone alone | 2,206 | 0.5025 | [.4769, .5295] | .108 | No clear pattern |
| **Trailing volatility** | **2,206** | **0.5521** | **[.5155, .5834]** | **.032** | Recent price movement helped |
| Lagged volatility + trailing flow + current news volume (preliminary) | 2,206 | 0.5604 | [.5277, .5930] | .004 | Small gain in this evaluation |
| Pre-open news used for same-session close-to-close direction | 2,246 | 0.5181 | [.4987, .5418] | .044 | Range still includes 0.50 |

### Preliminary: news volume may add a small gain

The full setup improved from a 0.5507 baseline containing lagged volatility and trailing 20-session
news flow to 0.5604 after current news volume was added, a gain of +0.0097. Most of the useful
information still came from recent price movement. The saved p-value tests the full specification
against shifted versions of the current-news features; it is not a paired uncertainty range for the
+0.0097 increment, and the row-level predictions needed for that paired check were not preserved.
The careful interpretation is therefore: **current news volume may add a small amount of
information about move size, but the evidence is not yet stable enough to call it a general rule.**
Because the news and target intervals overlap, some of the gain may reflect the market's overnight
reaction to the same news rather than information about a future return.

Source: [`outputs/grid_metrics_2015_2026.csv`](outputs/grid_metrics_2015_2026.csv).

## Exploratory Timing Diagnostic — Not a Confirmed Finding

Autoformer is a deep-learning model built to learn patterns across time. It trained on earlier dates
and tested the years 2022 through 2025. The table below preserves the values produced by that run.
“Projected” models apply market-wide information across sectors, while “attributed” models assign
news to particular sectors. “Multimodal” means that several types of inputs are combined.

| Autoformer model | Mean daily rank relationship (IC) | Reported t score (not adjusted for repeated tests) | Gross Sharpe | Overnight net Sharpe | Overnight net arithmetic annual return |
|---|---:|---:|---:|---:|---:|
| M0 Market baseline | −0.00468843 | −0.149517 | −0.179366 | −1.61385 | −0.118438 |
| M1 Narrative projected | 0.0445961 | 1.84856 | 1.83103 | 0.462243 | 0.0438523 |
| M2 FinBERT projected† | −0.00689727 | −0.271448 | −0.139301 | −1.57016 | −0.111038 |
| M3 Multimodal projected† | 0.0488618 | 2.04493 | 2.0402 | 0.655388 | 0.0614283 |
| M4 Macro projected† | 0.0475541 | 1.9925 | 1.6431 | 0.252955 | 0.0350629 |
| M5 Sector attributed† | 0.0434403 | 1.79139 | 1.50441 | 0.120243 | 0.029153 |
| M6 LLM attributed† | 0.0550994 | 2.32356 | 2.04136 | 0.650439 | 0.0634247 |
| M7 Multimodal + LLM† | 0.0651768 | 2.74447 | 2.71781 | 1.3348 | 0.109173 |

Here, IC measures how closely the model's daily sector ranking matched the observed ranking; zero
means no relationship. A Sharpe ratio compares average return with how uneven those returns were.
Gross Sharpe ignores estimated trading costs; net Sharpe subtracts the notebook's assumed costs.
The final column is an arithmetic annualized average, not compound growth. It is written as a decimal
exactly as produced by the table: for example, 0.109173 means 10.9173% per year under the notebook's
assumptions.

The notebook generated random stand-in values so it could continue. Because the model feature sets
build on one another, every row from M2 through M7 uses that column. Those rows are not valid tests
of real FinBERT information.

### The delayed-outcome check changes the story

The model also shifts the measured return to the following close-to-close period. This reduces the direct
timing overlap, but it does not fully model when every daily economic input was published. All three
delayed results are negative even though the same-window overnight results are positive. The table
below preserves the notebook's reported values.

| Model | Overnight net Sharpe | Overnight net annualized mean | Delayed net Sharpe | Delayed net annualized mean |
|---|---:|---:|---:|---:|
| M5 | 0.43 | 3.3% | −0.37 | −4.8% |
| M6 | 0.87 | 6.7% | −0.14 | −1.9% |
| M7 | 1.48 | 11.2% | −0.28 | −3.5% |

The first table averages Sharpe ratios across the four yearly tests; the delayed comparison joins
the four years before calculating Sharpe. That is why, for example, M7 is 1.3348 in the matrix and
1.48 in the comparison summary.

The notebook calls this a “no look-ahead” check, but that wording is too strong. It does not model
the publication time of every same-day economic input, and its portfolio code rebalances daily while
reporting turnover of 2.00. The signal was not available at the prior close, when the measured
overnight period began. Several related model versions were also tried, so the reported t-scores
need a test that accounts for repeated comparisons and nearby trading days. The model identifies a timing
question worth retesting with valid inputs and a design that uses only information available before
the return begins; it does not show that news caused returns or that the strategy was tradeable.

Sources: [`outputs/v5_autoformer_daily_metrics.csv`](outputs/v5_autoformer_daily_metrics.csv) and
[`outputs/v5_autoformer_delayed_outcome_metrics.csv`](outputs/v5_autoformer_delayed_outcome_metrics.csv).
These aggregate files preserve the executed values without licensed text.

## What We Can Conclude

**Confirmed statistical results:**

- RavenPack and FinBERT sentiment did not improve next-session sector direction forecasts.
- In the open-time association test, lagged volatility helped distinguish large from small
  full-session moves.

**Descriptive timing result—not a causal claim:**

- More positive RavenPack and FinBERT tone had small positive relationships with the same
  close-to-close session, but almost none with the following session. Daily data cannot determine
  whether news caused the move or whether the response happened within seconds.

**Preliminary—not confirmed:**

- Current news volume may add a small amount to a baseline that already contains lagged volatility
  and trailing news flow, but the news and return intervals overlap and the size and stability of
  the improvement need confirmation.

**Not supported as a final claim:**

- that the model can earn the positive overnight returns shown in the Autoformer matrix;
- that the M2–M7 results show an effect from real FinBERT information in this execution; or
- that news is the cause of the overnight relationship.

Taken together, the evidence supports a cautious ending: sentiment aligns modestly with the session
around its arrival but does not add a reliable next-session direction forecast; lagged volatility is
associated with same-session move size; the extra news-volume gain is preliminary; the Autoformer model does not
support an executable overnight forecast.

## Data and Study Scope

- **Market window:** 2,766 trading sessions from January 2, 2015 through December 31, 2025.
- **Market data:** 29,362 daily fund observations across 11 sector ETFs.
- **News in the study window:** 880,686 RavenPack event records and 601,177 distinct mapped texts
  prepared for FinBERT matching.
- **Direction comparison:** 11,022 later-date sector outcomes from 2022 through 2025.
- **Move-size comparison:** 2,206 later-date sessions; the related direction row uses 2,246.

The 11 funds cover Technology, Health Care, Financials, Communication Services, Consumer
Discretionary, Industrials, Consumer Staples, Energy, Utilities, Materials, and Real Estate. The
Real Estate fund began in October 2015 and the Communication Services fund began in June 2018, so
nine long-running sectors are used when a full-period comparison is required.

## Important Data Limits

- RavenPack story coverage changes sharply around 2017: about 138,000 stories in 2015, 96,000 in
  2017, and 59,000 in 2018. A raw news count can therefore partly reflect a vendor coverage change.
- The pre-open rule uses clock time. It excludes weekend and holiday stories posted between 9:30
  a.m. and 4:00 p.m. even when the stock market was closed, so the timing comparison does not cover
  every story published between two sessions.
- Overnight gaps account for about 40.9% of the combined average absolute overnight and intraday
  component movement. This makes the timing window important, but it does not make a signal
  available before the gap begins.
- RavenPack and FinBERT agree moderately rather than perfectly. Their weekly correlation is 0.70
  over the full period, while rolling one-year agreement has a median of 0.50 and ranges from 0.05
  to 0.76.
- The 2015–2019 period had 12.7% annualized volatility, compared with 19.9% in 2020–2025. A result
  that appears only in the later years may not carry into calmer markets.

## How to Run This Repo

### 1. Requirements and setup

**Python 3.11 or 3.12.** The committed results were produced on 3.11 and 3.12, and the test suite is
verified on 3.12.10. Do not use Python 3.14: several pinned dependencies have no prebuilt wheel for
it yet, so `pip` falls back to compiling pandas from source and the install fails unless you have a
C++ toolchain installed.

Create and activate an isolated environment, then install the dependencies:

```bash
# Windows PowerShell  (use the py launcher to pick the interpreter)
py -3.12 -m venv venv
venv\Scripts\Activate.ps1

# macOS or Linux
python3.12 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Expect the install to take a few minutes and several GB on disk, most of it `torch`. If `pip` is not
found as a bare command, use `python -m pip` as shown above — it always resolves to the interpreter
you are running.

No credentials are needed for setup, for step 2, or for the grid rerun in step 3. WRDS access is
required only for the optional rebuild in step 4.

### 2. Verify the install

```bash
python -m pytest tests/ -q
python run_training.py --list
```

`pytest` should report **36 passing tests** in about 30 seconds, covering the shared cleaning and
evaluation code in `src/`.

`--list` prints each stage with its inputs and whether they are present. In a fresh clone it should
look like this:

```text
  clean   raw events + market panel -> model_inputs
            in  data\raw\ravenpack_core_events_2015_2026.csv  (MISSING)
            in  data\market_daily_df.csv  (exists)
            in  data\raw\finbert_scores.csv  (MISSING)
            out data\model_inputs_2015_2026.csv  (exists)
  train   model_inputs -> grid metrics + figure
            out outputs\grid_metrics_2015_2026.csv
            out outputs\grid_summary_2015_2026.png
```

The two **MISSING** rows are expected. `data/raw/` is gitignored, so the row-level licensed extracts
are not in the clone. The prepared `data/model_inputs_2015_2026.csv` is committed, and it is the only
input step 3 needs.

> Always pass an explicit `--stage` to `run_training.py`. With no arguments it defaults to
> `--stage all`, which begins with `clean` and stops on the first MISSING input above.

### 3. Reproduce the statistical grid

This is the main reproduction path. It needs no credentials and no licensed news:

```bash
python run_training.py --stage train --n-perm 250 --seed 20260807
```

Expect roughly **20–25 minutes** on a laptop CPU. The run first checks itself against a signal
planted by construction, then evaluates the six pre-registered specs:

```text
Harness self-test (planted signal): AUC 0.7519 (needs >= 0.65)
  passed — splitting, fitting and scoring are sound.

1. RavenPack news flow               AUC=0.4902   clears perm only
2. RavenPack tone                    AUC=0.4741   inside noise
3. FinBERT tone                      AUC=0.5025   inside noise
4. Trailing volatility (control)     AUC=0.5521   *** CLEARS BOTH ***
5. Flow | vol + trailing flow        AUC=0.5604   *** CLEARS BOTH ***
6. RavenPack news -> direction       AUC=0.5181   clears perm only
```

If the self-test fails, the grid is not run at all — that failure means the evaluation code is
broken, not that the data lacks a signal.

The command **overwrites** `outputs/grid_metrics_2015_2026.csv` and
`outputs/grid_summary_2015_2026.png`. To compare a rerun against the committed result, copy the two
files aside first, or restore them afterwards with
`git checkout -- outputs/grid_metrics_2015_2026.csv outputs/grid_summary_2015_2026.png`.

For a one-minute smoke test that the environment is wired correctly, lower the permutation count:

```bash
python run_training.py --stage train --n-perm 5
```

The six AUC point estimates are identical at any `--n-perm`. The confidence intervals and p-values
are **not**, because the bootstrap and permutation draws come from a single seeded random stream, so
changing the number of permutation draws shifts every later spec's draws. Only
`--n-perm 250 --seed 20260807` reproduces the committed table.

### 4. Rebuild the data from WRDS (optional)

Only needed to regenerate `data/raw/` from source. It requires a WRDS account entitled to RavenPack
and CRSP, internet access to download the FinBERT model, substantial disk space, and preferably a
CUDA-capable GPU for text scoring. Budget several hours.

Store the WRDS login in a local password file first — the notebooks run headless and cannot answer a
password prompt:

```bash
python -c "import wrds; wrds.Connection(wrds_username='YOUR_USERNAME').create_pgpass_file()"
```

Then confirm the credentials and packages resolve before committing to a long run:

```bash
python run_pipeline.py --list     # the five stages and which outputs already exist
python run_pipeline.py --check    # credentials, packages and paths only — runs nothing
```

`--check` connects to WRDS for real. A stored password that has expired fails here with
`PAM authentication failed`; recreate the file with the command above.

The five stages run in dependency order, each one a notebook executed in place:

| Stage | Notebook | Produces | Needs |
|---|---|---|---|
| `news` | `1_extract/01_ravenpack_news_extraction.ipynb` | `data/raw/ravenpack_core_events_*.csv`, `data/news_daily_df.csv` | WRDS |
| `prices` | `1_extract/02_crsp_sector_etf_price_extraction.ipynb` | `data/raw/crsp_sector_etf_daily_raw_*.csv`, `data/market_daily_df.csv` | WRDS |
| `panel` | `2_prepare/03_data_quality_visual_qa.ipynb` | `data/model_daily_panel.csv` | — |
| `scoring-input` | `2_prepare/07a_llm_scoring_input_prep.ipynb` | `data/raw/llm_scoring_input.csv`, `data/raw/llm_scoring_event_map.csv` | — |
| `finbert` | `2_prepare/07_finbert_sentiment_scoring.ipynb` | `data/raw/finbert_scores.csv` | GPU (works on CPU, slowly) |

Despite its name, `panel` is not optional — it builds `model_daily_panel.csv`. Run the full sequence:

```bash
python run_pipeline.py
python run_training.py --stage clean
python run_training.py --stage train --n-perm 250 --seed 20260807
```

### 5. Results the runners do not regenerate

Steps 3 and 4 cover the statistical grid only. The other committed results come from notebooks that
are run individually:

| Result in this README | Produced by |
|---|---|
| Direction-model table (`outputs/model_comparison_all.csv`) | [`4_model/10_combined_sentiment_model.ipynb`](notebooks/4_model/10_combined_sentiment_model.ipynb) |
| FinBERT-only comparison (`outputs/model_comparison_m0_m2.csv`) | [`4_model/09_finbert_sentiment_model.ipynb`](notebooks/4_model/09_finbert_sentiment_model.ipynb) |
| Move-size analysis (`outputs/news_flow_magnitude_metrics.csv`) | [`4_model/11_news_flow_magnitude_model.ipynb`](notebooks/4_model/11_news_flow_magnitude_model.ipynb) |
| Per-sector baselines (`outputs/sector_*.csv`) | [`4_model/07_all_sector_baseline_models.ipynb`](notebooks/4_model/07_all_sector_baseline_models.ipynb) |
| Whole M0–M7 section, `fig1`–`fig3`, all of `data/llm_files/` | [`4_model/12_llm_autoformer_models.ipynb`](notebooks/4_model/12_llm_autoformer_models.ipynb) — see below |
| `fig4_ls_portfolios.png`, `final_fig1`–`fig3` | No producer in this repo |

Notebook 11 shares `src/model_lib.py` with `run_training.py`, so its move-size numbers and the
grid's agree by construction.

#### Notebook 12 carries the entire M0–M7 story

[`notebooks/4_model/12_llm_autoformer_models.ipynb`](notebooks/4_model/12_llm_autoformer_models.ipynb)
is the single most load-bearing notebook in the repo and the least reproducible. Everything in the
[Exploratory V5 Timing Diagnostic](#exploratory-v5-timing-diagnostic--not-a-confirmed-finding)
section traces back to it, and nothing else in the repo can regenerate any of it:

- **Both V5 tables are transcribed from its printed output, not written to disk.** The eight-row
  matrix comes from its `DAILY AUTOFORMER MATRIX (2022-2025)` cell; the delayed-outcome table comes
  from its portfolio-simulation cell. `outputs/v5_autoformer_daily_metrics.csv` and
  `outputs/v5_autoformer_delayed_outcome_metrics.csv` were created by hand from those tables — the
  notebook has no `to_csv` call for either.
- **It writes `fig1_auc_m0m7.png`, `fig2_auc_lift.png` and `fig3_movesize_auc.png` to the working
  directory**, not to `outputs/report_figures/`. They were moved there manually after the run.
- **It is the sole producer of every file in `data/llm_files/`** — the per-year `macro_news_*.parquet`
  caches, the Gemini score caches, `final_engineered_m6_panel.parquet`, and `feature_sets.json`.

M7 is `M3_FEATURES + LLM_FEATURES`: the ten market features, the six narrative features, the
projected FinBERT score, and two Gemini sector-attributed features (`llm_sent_surprise`,
`llm_sent_x_attention`). Its 0.5322 AUC in `fig1` is a seed-averaged, block-bootstrapped
out-of-sample figure over the 2022–2025 walk-forward, against an **overnight** direction target.

**Rerunning it is not a `python` command.** It is a Colab notebook: it `%pip install`s its own
dependencies, `chdir`s to a `REPO_NAME` clone path, imports `google.colab`, pulls FRED and Yahoo
series over the network, and needs a `GOOGLE_API_KEY` for the Gemini scoring pass. The Gemini scores
are cached in `data/llm_files/llm_sector_scores.parquet`, so a rerun does not re-pay for scoring, but
the notebook cannot be executed by `run_pipeline.py` and is not covered by the test suite.

> **A rerun today still will not use real FinBERT data.** The notebook resolves
> `FINBERT_PATH = DATA_DIR / "finbert_daily_df.csv"` where `DATA_DIR` is `./data/llm_files`, so it
> looks for `data/llm_files/finbert_daily_df.csv`. The file is committed one directory up, at
> `data/finbert_daily_df.csv`. The `.exists()` check therefore fails and the `else` branch fabricates
> `np.random.normal(0, 0.2, ...)`, which every model from M2 up inherits through the shared feature
> sets. This is a one-line path bug, not missing data — see [Known defects](#known-defects-in-notebook-12).

## Project Guide

```text
run_pipeline.py            collects the licensed source data in notebook order
run_training.py            prepares the modeling table and reruns the statistical grid
src/                       shared cleaning and evaluation code
notebooks/1_extract/       RavenPack news and CRSP market-data collection
notebooks/2_prepare/       data checks and FinBERT text scoring
notebooks/3_explore/       exploratory analysis
notebooks/4_model/         direction, move-size, and Autoformer models
data/                      prepared tables, plus the cached news and LLM scoring inputs
data/llm_files/            extracted news cache and LLM/FinBERT scoring artifacts (see below)
outputs/                   saved metrics and figures
tests/                     checks for the shared cleaning and evaluation code
```

The exploratory root V4 and V5 notebooks are not part of the supported public reproduction path.
They must be sanitized or omitted before a public release because their saved cells contain
licensed or sensitive material. The primary move-size analysis is in
[`notebooks/4_model/11_news_flow_magnitude_model.ipynb`](notebooks/4_model/11_news_flow_magnitude_model.ipynb).
The four-year Autoformer study is in
[`notebooks/4_model/12_autoformer_extended_test.ipynb`](notebooks/4_model/12_autoformer_extended_test.ipynb).

## Data Sources

- **RavenPack News Analytics through WRDS:** licensed news events, relevance, time, and commercial
  sentiment measures.
- **CRSP through WRDS:** daily fund returns, open and close prices, and trading volume.
- **FinBERT (`ProsusAI/finbert`):** a public language model run locally to score financial text.
- **FRED and Yahoo Finance:** daily economic and market series used in the exploratory V5 notebook.

## Team

- **Jeremy Tang — Data engineering:** RavenPack and market-data collection, news-to-return matching,
  timestamp alignment, and feature preparation.
- **Christian Goelz — Machine learning:** baseline and sentiment models, Autoformer development, and
  model evaluation.
- **Dongxin Liang — NLP and interpretation:** FinBERT scoring, comparison of sentiment sources,
  evaluation design, and interpretation of results.

## Data Access and Licensing

RavenPack and CRSP data reach this project through Wharton Research Data Services (WRDS) under the
University of Michigan's institutional subscription, and are used for academic research only.
Full detail on sources, ownership, and how we connected is in [DATA_ACCESS.md](DATA_ACCESS.md).

**Why the extracted data is included here.** This repository is private. It includes the extracted
RavenPack news cache and the FinBERT/LLM scoring outputs under `data/` so that the project can be
run end to end without repeating the collection work. Re-running the WRDS extraction and the FinBERT
scoring pass takes many hours, needs an entitled WRDS account, and benefits from a CUDA-capable GPU.
Shipping those artifacts means a reviewer can run training and evaluation directly and still
reproduce the reported results.

That data is held here for this project's own use and is **not shared, published, or redistributed
anywhere else**. Credentials must never appear in code, notebook cells, saved notebook output, or
version history.

**Before making this repository public**, the licensed material under `data/llm_files/` — the
RavenPack news caches with headline text and the LLM batch inputs built from them — must be removed,
and removed from git history as well, not just from the current commit.
