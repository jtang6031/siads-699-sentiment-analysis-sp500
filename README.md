# SIADS-699-sentiment-analysis-sp500 - Financial News and S&P 500 Sector Returns

**Academic research only. This project is not investment advice.**

A simple question: Can the language and amount of financial news help us anticipate what broad parts of the U.S. stock market will do next?

## Executive Summary

- **On the next-day close-to-close task, news sentiment did not improve direction ranking.** Across
  11,022 later-date sector comparisons from 2022 through 2025, the market-only model and every
  news-based version scored near 0.50 AUC, which is chance-level ranking. This is the task where the
  news window overlaps a full session, and where sentiment adds no separable edge.
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
- **In the overnight window, the Autoformer does surface a sentiment signal, but it is an upper
  bound rather than a tradeable result.** Its strongest configuration, M7 (multimodal narrative and
  FinBERT plus LLM attribution, with macro removed), reaches an information-coefficient t-statistic
  of 2.98 and a whole-period overnight net Sharpe of 1.70, and it is the best model on every metric.
  However, the overnight features overlap the very close-to-open gap they score, so that return
  could not have been earned exactly as shown, and a delayed-return check that carries the same
  signal into the next full session was negative for M5, M6, and M7.

Taken together, the primary analyses support a layered reading. On the next-day close-to-close task
sentiment did not improve direction ranking, tone had a small same-session association with returns,
and lagged volatility was associated with full-session move size. In the overnight window the
LLM-attributed Autoformer (M7) is the strongest configuration and adds genuine cross-sectional
information, but its economic figures are an upper bound because the signal overlaps the return it
scores. The possible news-volume gain is preliminary.

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
news result a partly contemporaneous, open-time association, not a clean forecast of the next move.

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

## The Overnight Autoformer: a Genuine but Upper-Bound Signal

Autoformer is a deep-learning model built to learn patterns across time. It trained on earlier dates
and tested the years 2022 through 2025. The table below preserves the values produced by that run.
“Projected” models apply market-wide information across sectors, while “attributed” models assign
news to particular sectors. “Multimodal” means that several types of inputs are combined. Read as an
ablation, this table is a real result: it ranks the eight configurations consistently across IC,
AUC, and Sharpe, and the LLM-attributed models (M6 and M7) sit at the top. Read as a strategy, it is
an upper bound, because the overnight window overlaps the gap the model scores (see the delayed
check below).

| Autoformer model | Mean daily rank relationship (IC) | Reported t score (not adjusted for repeated tests) | Gross Sharpe | Overnight net Sharpe | Overnight net arithmetic annual return |
|---|---:|---:|---:|---:|---:|
| M0 Market baseline | −0.00468843 | −0.149517 | −0.179366 | −1.61385 | −0.118438 |
| M1 Narrative projected | 0.0445961 | 1.84856 | 1.83103 | 0.462243 | 0.0438523 |
| M2 FinBERT projected | 0.0327743 | 1.31567 | 0.927627 | −0.425341 | −0.0259675 |
| M3 Multimodal projected | 0.0582649 | 2.33307 | 2.09134 | 0.756880 | 0.0708776 |
| M4 Macro projected | 0.0551341 | 2.24605 | 1.73723 | 0.368238 | 0.0474113 |
| M5 Sector attributed | 0.0555624 | 2.28603 | 2.04172 | 0.629474 | 0.0650164 |
| M6 LLM attributed | 0.0630747 | 2.60978 | 2.46105 | 1.06320 | 0.0990593 |
| M7 Multimodal + LLM | 0.0723910 | 2.98178 | 2.92618 | 1.53119 | 0.125738 |

Here, IC measures how closely the model's daily sector ranking matched the observed ranking; zero
means no relationship. A Sharpe ratio compares average return with how uneven those returns were.
Gross Sharpe ignores estimated trading costs; net Sharpe subtracts the notebook's assumed costs.
The final column is an arithmetic annualized average, not compound growth. It is written as a decimal
exactly as produced by the table: for example, 0.125738 means 12.5738% per year under the notebook's
assumptions.

The FinBERT sentiment column is now populated from the model rather than a placeholder, so the M2
through M7 rows are valid tests of that input. FinBERT on its own (M2) is mildly informative for
ranking, with a positive IC and a t score of 1.32, but it loses money as a standalone long-short
leg. Its useful contribution shows up inside the multimodal stack, where adding it to the narrative
signal (M3) improves on the narrative-only model (M1), and where the LLM-attributed models (M6 and
M7) rank highest overall.

### The delayed-outcome check changes the story

The model also shifts the measured return to the following close-to-close period. This reduces the direct
timing overlap, but it does not fully model when every daily economic input was published. All three
delayed results are negative even though the same-window overnight results are positive. The table
below preserves the notebook's reported values.

| Model | Overnight net Sharpe | Overnight net annualized mean | Delayed net Sharpe | Delayed net annualized mean |
|---|---:|---:|---:|---:|
| M5 | 0.91 | 6.9% | −0.22 | −2.9% |
| M6 | 1.34 | 10.3% | −0.51 | −6.8% |
| M7 | 1.69 | 12.9% | −0.23 | −3.0% |

The first table averages Sharpe ratios across the four yearly tests; this delayed comparison joins
the four years before calculating Sharpe. That is why, for example, M7 is 1.53 in the matrix and
1.69 in the comparison summary. The direction of the story is unchanged from earlier runs: every
overnight leg is positive and every delayed leg is negative.

The notebook calls this a “no look-ahead” check, but that wording is too strong. It does not model
the publication time of every same-day economic input, and its portfolio code rebalances daily while
reporting turnover of 2.00. The signal was not available at the prior close, when the measured
overnight period began. Several related model versions were also tried, so the reported t-scores
need a test that accounts for repeated comparisons and nearby trading days. The model identifies a timing
question worth retesting with valid inputs and a design that uses only information available before
the return begins; it does not show that news caused returns or that the strategy was tradeable.

These aggregate files preserve the executed values without licensed text.

## What We Can Conclude

**Confirmed statistical results:**

- RavenPack and FinBERT sentiment did not improve next-session sector direction forecasts.
- In the open-time association test, lagged volatility helped distinguish large from small
  full-session moves.

**Descriptive timing result, not a causal claim:**

- More positive RavenPack and FinBERT tone had small positive relationships with the same
  close-to-close session, but almost none with the following session. Daily data cannot determine
  whether news caused the move or whether the response happened within seconds.

**Preliminary, not confirmed:**

- Current news volume may add a small amount to a baseline that already contains lagged volatility
  and trailing news flow, but the news and return intervals overlap and the size and stability of
  the improvement need confirmation.

**A genuine but qualified result:**

- In the overnight ranking, the LLM-attributed model M7 is the strongest configuration on every
  metric (IC t = 2.98, whole-period overnight net Sharpe 1.70), the M2 through M7 rows are valid tests.

**Not supported as a final claim:**

- that the model can earn the positive overnight returns shown, since the signal overlaps the gap it
  scores and the delayed check is negative; or
- that news is the cause of the overnight relationship.

Taken together, the evidence supports a cautious but constructive conclusion. On the next-day
close-to-close task sentiment does not add a reliable direction forecast; tone aligns modestly with
the session around its arrival; lagged volatility is associated with same-session move size; and the
extra news-volume gain is preliminary. In the overnight window the LLM-attributed Autoformer is the
strongest configuration and adds genuine cross-sectional information, but the overnight economic
figures are an upper bound rather than an executable forecast, and a clean ex-ante test is left for
future work.

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
found as a bare command, use `python -m pip` as shown above, since it always resolves to the
interpreter you are running.

No credentials are needed for setup, or for step 2 and the grid rerun below. WRDS access is required
only for the full rebuild described later.

### 2. Verify the install

```bash
python -m pytest tests/ -q
python run_training.py --list
```

`pytest` should report **36 passing tests**, covering the shared cleaning and evaluation code in
`src/`.

`--list` prints each stage with its inputs and whether they are present. In a fresh clone the
`clean` stage will show its inputs under `data/raw/` as **MISSING**, which is expected, because the
row-level extraction files are gitignored. The prepared `data/model_inputs_2015_2026.csv` is
committed and should show as present, which is all the grid rerun needs.

> Run `run_training.py` with an explicit `--stage`. With no arguments it defaults to `--stage all`,
> which begins with `clean` and will fail in a fresh clone for the reason above.

## Reproduce the Statistical Grid

### Rerun the grid from the prepared table

This path does not need WRDS credentials or raw licensed news:

```bash
python run_training.py --stage train --n-perm 250 --seed 20260807
```

The training command writes:

- `outputs/grid_metrics_2015_2026.csv`
- `outputs/grid_summary_2015_2026.png`

The 250-permutation run can take many minutes and overwrites both files. Start from a clean checkout
if you want to compare a rerun with the committed result. These commands do not regenerate the
direction-model table, the Autoformer matrix, or the Autoformer timing figure.

### Full statistical-grid rebuild from licensed data

A complete rebuild requires a WRDS account with access to RavenPack and CRSP, internet access for
the FinBERT model, substantial disk space, and preferably a CUDA-capable GPU for text scoring. Store
the WRDS login in a local password file before running the collection steps:

```bash
python -c "import wrds; wrds.Connection(wrds_username='YOUR_USERNAME').create_pgpass_file()"
```

```bash
python run_pipeline.py --check
python run_pipeline.py
python run_training.py --stage clean
python run_training.py --stage train --n-perm 250 --seed 20260807
```

This rebuild uses the source dates configured in the extraction notebooks. Align their end date with
December 31, 2025 before comparing with the published sample. It does not rerun the separate
direction or Autoformer notebooks.

You do not have to run this rebuild to reproduce the reported results. The curated modeling table
`data/model_inputs_2015_2026.csv` is published, so a reviewer can go straight to training and
evaluation. The row-level RavenPack and CRSP extracts are **not** published — see
[Data Access and Licensing](#data-access-and-licensing) for what is and is not included and why.

### Reproduce the Autoformer results

The daily and monthly Autoformer statistics, the eight-model ablation (M0 through M7), and Figures 1
through 4 are produced by a notebook rather than by `run_training.py`:
[`notebooks/4_model/12_llm_autoformer_models.ipynb`](notebooks/4_model/12_llm_autoformer_models.ipynb).

> **Start from the "Reloaded engineered panel from local cache" cell, not from the top.** The
> notebook's first half rebuilds the news pull and the Gemini scoring pass, and it reads the RavenPack
> news cache under `data/llm_files/cache/`, which is licensed and therefore not published. Run those
> cells and they fall through to `wrds.Connection()`. Everything from the panel-reload cell onward is
> self-contained: it loads the published `final_engineered_m6_panel.parquet` and `feature_sets.json`,
> reconstructs all eight feature sets, and produces every table and figure below.

```bash
jupyter lab notebooks/4_model/12_llm_autoformer_models.ipynb
# then run from the panel-reload cell to the end
```

Running the whole notebook end to end, including the collection half, needs a WRDS account and a
`GOOGLE_API_KEY` for the Gemini scoring pass. Reproducing the *results* needs neither. A CUDA-capable GPU is strongly
recommended, because it trains the SectorAutoformer for all eight models across three random seeds
and four walk-forward test years; this takes roughly half an hour on a single GPU and much longer on
CPU. Running it prints the daily and monthly ablation matrices, the seed-stability and per-year
tables, and the paired-lift and move-size analyses, and it writes the four figures
(`fig1_auc_m0m7.png`, `fig2_ls_portfolios.png`, `fig3_auc_lift.png`, `fig4_movesize_auc.png`) used in
this README and in the capstone report. Because the SectorAutoformer training is not fully
deterministic across GPUs, exact values can shift slightly from run to run, but the model ordering
and the headline figures are stable.

## Project Guide

```text
run_pipeline.py            collects the licensed source data in notebook order
run_training.py            prepares the modeling table and reruns the statistical grid
src/                       shared cleaning and evaluation code
notebooks/1_extract/       RavenPack news and CRSP market-data collection
notebooks/2_prepare/       data checks and FinBERT text scoring
notebooks/3_explore/       exploratory analysis
notebooks/4_model/         direction, move-size, and Autoformer models
data/                      curated tables: daily aggregates and the modeling input
data/llm_files/            model-generated LLM scoring outputs and engineered features
data/raw/                  gitignored — licensed row-level extracts, not published
outputs/                   saved metrics and figures
tests/                     checks for the shared cleaning and evaluation code
```

The primary move-size analysis is in
[`notebooks/4_model/11_news_flow_magnitude_model.ipynb`](notebooks/4_model/11_news_flow_magnitude_model.ipynb).
The four-year Autoformer study is in
[`notebooks/4_model/12_llm_autoformer_models.ipynb`](notebooks/4_model/12_llm_autoformer_models.ipynb).

## Data Sources

- **RavenPack News Analytics through WRDS:** licensed news events, relevance, time, and commercial
  sentiment measures.
- **CRSP through WRDS:** daily fund returns, open and close prices, and trading volume.
- **FinBERT (`ProsusAI/finbert`):** a public language model run locally to score financial text.
- **FRED and Yahoo Finance:** daily economic and market series used in the exploratory V5 notebook.

## Team

- **Jeremy Tang, Data engineering:** RavenPack and market-data collection, news-to-return matching,
  timestamp alignment, and feature preparation.
- **Christian Goelz, Machine learning:** baseline and sentiment models, Autoformer development, and
  model evaluation.
- **Dongxin Liang, NLP and interpretation:** FinBERT scoring, comparison of sentiment sources,
  evaluation design, and interpretation of results.

## Data Access and Licensing

RavenPack and CRSP data reach this project through Wharton Research Data Services (WRDS) under the
University of Michigan's institutional subscription, and are used for academic research only.
Full detail on sources, ownership, and how we connected is in [DATA_ACCESS.md](DATA_ACCESS.md).

**No raw WRDS data is published in this repository.** RavenPack and CRSP are licensed datasets, and
row-level extracts from them are not ours to redistribute. What you get here is the curated layer —
session-level aggregates, engineered modeling features, model outputs and figures — which is
sufficient to reproduce every reported result on the supported path.

Excluded by [.gitignore](.gitignore) and absent from any clone:

```text
data/raw/                  # RavenPack event extracts, CRSP daily extracts, FinBERT per-text scores
data/llm_files/cache/      # yearly + combined RavenPack news caches, with headline text
*_input.jsonl              # LLM batch prompts, which embed headline text
```

Published instead: `data/model_inputs_2015_2026.csv` (the modeling table), the daily aggregates in
`data/`, the model-generated scoring outputs in `data/llm_files/`, and everything in `outputs/`.

**What this costs you.** Nothing on the main path — steps 1 through 3 above need no credentials and
no licensed data. Step 4 rebuilds `data/raw/` from source and has always required your own WRDS
account with RavenPack and CRSP entitlements. Notebook 12 now also needs WRDS, because the cached
news pull it used to read is no longer published.

Credentials must never appear in code, notebook cells, saved notebook output, or version history.
Because saved notebook cells are published output in a public repository, clear any cell that would
print headline text before committing.

Two WRDS-derived identifiers do remain in published files — `rp_story_id` in the LLM scoring outputs
and `permno` in the CRSP daily panel. Neither carries licensed content, and
[DATA_ACCESS.md](DATA_ACCESS.md#residual-licensed-identifiers) records why they were kept and how to
remove them if required.
