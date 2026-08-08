# siads-699-sentiment-analysis-sp500

**Academic research only. Not investment advice.**

University of Michigan MADS Capstone (SIADS 699) — Team Alpha Signal.

## Research Question

Does news sentiment add out-of-sample predictive power for the 11 S&P 500 sector ETFs, beyond
traditional market features?

**Headline: news sentiment does not predict next-session *direction*, and we can now say so with a
well-powered null. It does predict the *overnight* move, and only the overnight move.**

Two independent models — a transparent statistical grid and a deep-learning sequence model
(Autoformer) — were built on different features against different targets. They converge on the same
mechanism: news is absorbed within the session it arrives, so the only window where information is
not yet priced is the one when the market is shut.

## Results

Sample: **2,766 trading sessions, 2015-01-02 to 2025-12-31** · 916,896 news events · 628,905
FinBERT-scored headlines.

### Statistical grid — market aggregate

2,206 out-of-sample sessions. Each spec is tested against **two independent references**: a block
permutation null (circular rotation of the news block) *and* a block-bootstrap confidence interval on
the out-of-sample AUC. Only specs clearing both are reported as findings.
Reproduce with `python run_training.py`.

| Spec | AUC | 95% CI | p | Verdict |
|---|---|---|---|---|
| News volume alone | 0.4902 | [.4552, .5294] | .172 | — |
| News tone alone | 0.4741 | [.4457, .5047] | .824 | — |
| FinBERT tone alone | 0.5025 | [.4769, .5295] | .108 | — |
| **Trailing volatility** | **0.5521** | [.5155, .5834] | **.032** | clears both |
| **Volatility + news volume** | **0.5604** | [.5277, .5930] | **.004** | clears both |
| News → next-day direction | 0.5181 | [.4987, .5418] | .044 | CI spans 0.5 |

News volume adds a small but real increment (+0.0097, p=0.004) on top of a volatility baseline that
is itself the dominant effect. News alone predicts neither direction nor magnitude.

### Autoformer — sector cross-section

Seven independent yearly tests, 2019–2025, purged walk-forward with embargo.
Reproduce with `notebooks/4_model/12_autoformer_extended_test.ipynb`.

| Holding period | IC | t | net Sharpe | net annual |
|---|---|---|---|---|
| **Overnight (close → next open)** | **0.0739** | **2.96** | **1.15** | **+10.5%** |
| Intraday (open → close) | −0.0038 | −0.16 | −1.31 | −14.2% |
| Full session (close → close) | −0.0023 | −0.07 | −1.02 | −13.5% |
| Monthly (both logistic and Autoformer) | — | best \|t\| 0.64 | — | — |

**The predictability sits entirely in the overnight gap.** Extend the holding period to a full
trading day and it is gone; stretch to a month and it is gone.

### Why the results are credible

- **A positive control passes on the same machinery.** Trailing volatility clears both references
  through the identical pipeline, so a null elsewhere is informative rather than a broken harness.
  The training runner additionally gates on a planted-signal self-test before the grid is run.
- **Three findings were killed by their own follow-up test.** News volume (0.529 → 0.490), FinBERT
  tone (0.524 → 0.503), and the monthly Autoformer (IC 0.051 → 0.004) all looked convincing on the
  shorter sample and did not survive re-testing on 2015–2025.
- **The added years are a different regime**, not more of the same: 2015–2019 has 12.7% annualised
  volatility against 19.9% for 2020–2025. Conclusions were re-tested in calm markets, not only in
  crisis.

### Open items

- **Multiplicity is not yet corrected.** The Autoformer's best t=3.06 comes from six model variants
  (four exceed 2.4, far more than chance would give), but they are nested, so a max-statistic test is
  still owed.
- The Autoformer's IC t-statistics assume day-to-day independence; a block permutation null is owed.
- Count-based news features are not yet normalised against the 2017 coverage break (below).

## Data-quality notes

These are real properties of the data, documented rather than smoothed over.

- **RavenPack macro coverage halves across 2017** — 138k distinct stories in 2015, 96k in 2017, 59k
  in 2018, ~66k thereafter — while the number of distinct sources stays flat (7.5–9.5/day). It is a
  vendor taxonomy change, present in two independent extractions. Any *raw count* feature therefore
  carries a level break mid-sample; features read relative to a trailing baseline do not.
- **40.9% of the average absolute session move happens in the overnight gap**, before anyone can
  trade. Findings about the overnight window are statements about predictability, not tradeability.
- **XLRE listed October 2015 and XLC June 2018.** The panel is unbalanced by construction before
  those dates. Aggregate models use the nine sectors present throughout; the data-quality checks are
  inception-aware rather than requiring a balanced panel.
- **FinBERT and the commercial score agree moderately, not strongly.** The whole-period weekly
  correlation of 0.70 is inflated by a shared upward drift; rolling one-year agreement has a median
  of 0.50 and ranges 0.05–0.76, and is markedly better after 2020 than before.

## Repository layout

```
run_pipeline.py            data collection: notebooks 01 -> 07, in dependency order
run_training.py            clean -> train: raw events + prices -> model-ready table -> grid
src/
  paths.py                 canonical paths; one root detection for the whole repo
  cleaning_lib.py          pure transforms (news window, features, labels, market series)
  model_lib.py             walk-forward, block permutation, bootstrap CI, the spec grid
  report_figures.py        the report exhibits
notebooks/
  1_extract/               RavenPack news, firm-level news, CRSP prices
  2_prepare/               data-quality panel, scoring input, FinBERT scoring
  3_explore/               EDA
  4_model/                 baselines, sentiment models, magnitude grid, Autoformer
data/                      gold tables (raw/ and cache/ are gitignored, licensed)
outputs/                   metrics, figures, report_figures/
tests/                     36 unit tests over cleaning_lib and model_lib
```

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 36 tests, no credentials needed
python run_pipeline.py --list       # show the data stages and what each produces
python run_pipeline.py              # full data collection (needs WRDS; hours)
python run_training.py              # clean + train + figures
```

`run_pipeline.py` reads the sample window out of notebook 01 so it cannot drift, verifies the WRDS
login before starting (an executed notebook has no stdin for a password prompt), and fails any stage
that runs but leaves its outputs unchanged.

## Required credentials

A WRDS account with RavenPack News Analytics and CRSP access is required to regenerate raw data.

Store the WRDS credential once with `wrds.Connection(...).create_pgpass_file()`; the runner reads it
from there and exports what the notebook kernels need. Any other key belongs in `.env`, which is
gitignored. **Never commit credentials, including inside notebook cells.**

## Target universe

Eleven SPDR sector ETFs: XLK Technology · XLV Health Care · XLF Financials · XLC Communication
Services · XLY Consumer Discretionary · XLI Industrials · XLP Consumer Staples · XLE Energy ·
XLU Utilities · XLB Materials · XLRE Real Estate.

## Data sources

- **RavenPack News Analytics** via WRDS — macro news events with sentiment, relevance, and
  timestamps, 2015-01-01 onward, filtered to relevance ≥ 90 from rank-1 non-blog institutional
  sources
- **CRSP daily stock file** via WRDS — daily returns, open and close prices, and volume
- **FinBERT** (`ProsusAI/finbert`) — run locally, so licensed headline text never leaves the machine

See [notebooks/3_explore/Basic_EDA_Data_Model.md](notebooks/3_explore/Basic_EDA_Data_Model.md) for
the table and column breakdown.

## Team

- **Jeremy Tang** — Data Engineering: WRDS/RavenPack ingestion, news-returns matching, timestamp
  alignment, feature engineering, Autoformer
- **Christian Goelz** — ML: model development, baseline vs sentiment-augmented classifiers,
  evaluation metrics
- **Dongxin Liang** — NLP/LLM: FinBERT sentiment scoring, RavenPack-versus-FinBERT comparison,
  pipeline and evaluation harness, interpretation of sentiment results

## Notebooks

| Notebook | Description |
|---|---|
| `1_extract/01_ravenpack_news_extraction.ipynb` | WRDS/RavenPack extraction, 4:00 PM ET cutoff, session mapping, daily aggregation |
| `1_extract/01b_ravenpack_equity_news_extraction.ipynb` | Firm-level news mapped to sectors via CUSIP → permno → SIC. **Written, not yet executed** |
| `1_extract/02_crsp_sector_etf_price_extraction.ipynb` | CRSP prices, returns, volume, open price, and the overnight-gap decomposition |
| `2_prepare/03_data_quality_visual_qa.ipynb` | Inception-aware QA scorecard; builds `model_daily_panel.csv` (not optional despite the name) |
| `2_prepare/07a_llm_scoring_input_prep.ipynb` | Deduplicates news text to a stable `text_id` for scoring |
| `2_prepare/07_finbert_sentiment_scoring.ipynb` | Scores each distinct headline with FinBERT |
| `3_explore/Basic_EDA_Analysis.ipynb` | Initial EDA: macro sentiment by session, joined to sector returns |
| `3_explore/Equity_Sentiment_EDA2.ipynb` | Exploratory company-level extension; outside the sector-ETF MVP |
| `4_model/04_report_visual_plan.ipynb` | Report figures and presentation guardrails |
| `4_model/05_baseline_model.ipynb` | Pooled chronological baseline, market-only vs market+sentiment (**M0/M1**) |
| `4_model/06_xlk_baseline_model.ipynb` | XLK-only walk-forward baseline |
| `4_model/07_all_sector_baseline_models.ipynb` | Per-sector walk-forward across all eleven ETFs |
| `4_model/08_finbert_sentiment_summary.ipynb` | FinBERT validation and weekly exploratory analysis |
| `4_model/09_finbert_sentiment_model.ipynb` | **M2** — builds `finbert_daily_df.csv`; market vs RavenPack vs FinBERT on one row set |
| `4_model/10_combined_sentiment_model.ipynb` | **M3** — four-model scoreboard in the same harness |
| `4_model/11_news_flow_magnitude_model.ipynb` | Magnitude grid: permutation null, bootstrap CI, positive control |
| `4_model/12_autoformer_extended_test.ipynb` | Autoformer on seven folds, with the return-period decomposition |

## Licensing

WRDS RavenPack data is licensed for academic use only. Raw records, article text, and WRDS exports
must not be committed to this repository — `data/raw/` and `data/cache/` are gitignored for that
reason. Only code, aggregated summary statistics, model outputs, and visualizations are stored here.
