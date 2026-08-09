# SIADS-699-sentiment-analysis-sp500 - Financial News and S&P 500 Sector Returns

**Academic research only. This project is not investment advice.**

This University of Michigan MADS capstone asks a simple question: can the language and amount of
financial news help us anticipate what broad parts of the U.S. stock market will do next?

## Executive Summary

- **News sentiment did not improve next-session direction forecasts.** Across 11,022 later-date
  sector comparisons from 2022 through 2025, the market-only model and every news-based version
  scored near 0.50 AUC, which is chance-level ranking.
- **Lagged volatility was associated with full-session move size in an open-time test.** Volatility
  alone reached 0.5521 AUC. In a separate setup, adding current news volume to a 0.5507 baseline of
  lagged volatility plus trailing news flow raised AUC to 0.5604, a gain of +0.0097. The
  close-to-close target includes the overnight move that had already occurred by the market open,
  so this is not a fully before-the-fact forecast. The news increment is also sensitive to the
  statistical test and should not be described as settled evidence.
- **V5 is exploratory and cannot support an investment claim.** Its strongest row, M7, reports an
  overnight net Sharpe ratio of 1.3348 and an arithmetic annualized return of 0.109173. However, the
  close-to-open holding period began before the full signal was available, so that return could not
  have been earned as shown. The run also replaced a missing FinBERT file with randomly generated
  stand-in values. A delayed-return check was negative for M5, M6, and M7.

Taken together, the primary analyses support two conclusions: news sentiment did not improve
direction forecasts, and lagged volatility was associated with full-session move size. The possible
news-volume gain is preliminary, while V5 remains an exploratory diagnostic.

## The Question in Everyday Language

A **sector ETF** is a fund that follows one broad industry, such as technology, energy, or health
care. **News sentiment** describes whether financial news sounds positive, negative, or neutral.
We tested whether that information helped forecast:

1. whether a sector would go up or down next;
2. whether a session's full close-to-close market move was relatively large or small; and
3. how sectors ranked against one another during the overnight period, from one market close to the
   next morning's open.

The models learned from earlier dates and were evaluated on later dates they had not seen during
training. This is better than randomly mixing old and new dates, but chronological testing alone does
not guarantee that every input was available before the measured return began. The timing limits are
called out beside the affected results below.

## Confirmed Findings from the Primary Analyses

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
information still came from recent price movement. An independent audit of the saved results used
paired resampling on that gain and included zero improvement; the gain was also concentrated in
2024 and 2025. The careful interpretation is therefore: **current news volume may add a small amount
of information about move size, but the evidence is not yet stable enough to call it a general
rule.** Because the news and target intervals overlap, some of the gain may reflect the market's
overnight reaction to the same news rather than information about a future return.

Source: [`outputs/grid_metrics_2015_2026.csv`](outputs/grid_metrics_2015_2026.csv).

## Exploratory V5 Timing Diagnostic — Not a Confirmed Finding

Autoformer is a deep-learning model built to learn patterns across time. V5 trained on earlier dates
and tested the years 2022 through 2025. The table below preserves the values produced by that run.
“Projected” models apply market-wide information across sectors, while “attributed” models assign
news to particular sectors. “Multimodal” means that several types of inputs are combined.

| V5 model | Mean daily rank relationship (IC) | Reported t score (not adjusted for repeated tests) | Gross Sharpe | Overnight net Sharpe | Overnight net arithmetic annual return |
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
exactly as produced by V5: for example, 0.109173 means 10.9173% per year under the notebook's
assumptions.

The notebook generated random stand-in values so it could continue. Because the model feature sets
build on one another, every row from M2 through M7 uses that column. Those rows are not valid tests
of real FinBERT information.

### The delayed-outcome check changes the story

V5 also shifts the measured return to the following close-to-close period. This reduces the direct
timing overlap, but it does not fully model when every daily economic input was published. All three
delayed results are negative even though the same-window overnight results are positive. The table
below preserves the notebook's reported values.

| Model | Overnight net Sharpe | Overnight net annualized mean | Delayed net Sharpe | Delayed net annualized mean |
|---|---:|---:|---:|---:|
| M5 | 0.43 | 3.3% | −0.37 | −4.8% |
| M6 | 0.87 | 6.7% | −0.14 | −1.9% |
| M7 | 1.48 | 11.2% | −0.28 | −3.5% |

The first V5 table averages Sharpe ratios across the four yearly tests; the delayed comparison joins
the four years before calculating Sharpe. That is why, for example, M7 is 1.3348 in the matrix and
1.48 in the comparison summary.

The notebook calls this a “no look-ahead” check, but that wording is too strong. It does not model
the publication time of every same-day economic input, and its portfolio code rebalances daily while
reporting turnover of 2.00. The signal was not available at the prior close, when the measured
overnight period began. Several related model versions were also tried, so the reported t-scores
need a test that accounts for repeated comparisons and nearby trading days. V5 identifies a timing
question worth retesting with valid inputs and a design that uses only information available before
the return begins; it does not show that news caused returns or that the strategy was tradeable.

Sources: [`outputs/v5_autoformer_daily_metrics.csv`](outputs/v5_autoformer_daily_metrics.csv) and
[`outputs/v5_autoformer_delayed_outcome_metrics.csv`](outputs/v5_autoformer_delayed_outcome_metrics.csv).
These aggregate files preserve the executed values without licensed text. The exploratory V5 source
notebook is excluded from the public release because its saved cells contain licensed or sensitive
material.

## What We Can Conclude

**Confirmed statistical results:**

- RavenPack and FinBERT sentiment did not improve next-session sector direction forecasts.
- In the open-time association test, lagged volatility helped distinguish large from small
  full-session moves.

**Preliminary—not confirmed:**

- Current news volume may add a small amount to a baseline that already contains lagged volatility
  and trailing news flow, but the news and return intervals overlap and the size and stability of
  the improvement need confirmation.

**Not supported as a final claim:**

- that the model can earn the positive overnight returns shown in the V5 matrix;
- that the M2–M7 results show an effect from real FinBERT information in this execution; or
- that news is the cause of the overnight relationship.

Taken together, the evidence supports a cautious ending: the primary direction result is negative,
lagged volatility is associated with same-session move size, the extra news-volume gain is
preliminary, and V5 does not support an executable overnight forecast.

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
- Overnight gaps account for about 40.9% of the combined average absolute overnight and intraday
  component movement. This makes the timing window important, but it does not make a signal
  available before the gap begins.
- RavenPack and FinBERT agree moderately rather than perfectly. Their weekly correlation is 0.70
  over the full period, while rolling one-year agreement has a median of 0.50 and ranges from 0.05
  to 0.76.
- The 2015–2019 period had 12.7% annualized volatility, compared with 19.9% in 2020–2025. A result
  that appears only in the later years may not carry into calmer markets.

## Reproduce the Statistical Grid

### Set up a local environment

Create and activate an isolated Python environment first:

```bash
python -m venv venv

# macOS or Linux
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Quick code check without licensed data

```bash
python -m pytest tests/ -q
python run_training.py --list
```

The test suite currently contains 36 tests. In a public clone, `--list` will show that the private
raw inputs for the cleaning step are missing; that is expected. The prepared
`data/model_inputs_2015_2026.csv` file should be present.

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
direction-model table, the V5 matrix, or the V5 timing figure.

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
direction or V5 notebooks. The raw RavenPack and FinBERT files are intentionally excluded from the
public repository.

## Project Guide

```text
run_pipeline.py            collects the licensed source data in notebook order
run_training.py            prepares the modeling table and reruns the statistical grid
src/                       shared cleaning, evaluation, path, and figure code
notebooks/1_extract/       RavenPack news and CRSP market-data collection
notebooks/2_prepare/       data checks and FinBERT text scoring
notebooks/3_explore/       exploratory analysis
notebooks/4_model/         direction, move-size, and Autoformer models
data/                      prepared public tables; licensed raw files stay private
outputs/                   saved metrics and figures
tests/                     checks for the shared cleaning and evaluation code
```

The exploratory V5 source notebook is excluded from the public release because its saved cells
contain licensed or sensitive material. The primary move-size analysis is in
[`notebooks/4_model/11_news_flow_magnitude_model.ipynb`](notebooks/4_model/11_news_flow_magnitude_model.ipynb).
The earlier seven-year Autoformer study is in
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

RavenPack data is licensed for academic use only. Raw event records, article text, licensed
headlines, and WRDS exports must not be published. Credentials must never appear in code, notebook
cells, saved notebook output, or version history. The public repository should contain only code,
prepared non-licensed tables, aggregated statistics, and figures that do not reproduce licensed
text.
