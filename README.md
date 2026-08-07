# siads-699-sentiment-analysis-sp500

**Academic research only. Not investment advice.**

University of Michigan MADS Capstone (SIADS 699) — Team Alpha Signal.

## Research Question

Does FinBERT-derived sentiment from financial macro news add incremental out-of-sample predictive power for the next-day direction of the 11 S&P 500 sector ETFs, beyond traditional market features and RavenPack’s existing sentiment score?

This is the final project goal. The RavenPack and CRSP data pipelines, data-quality checks, FinBERT sentiment scoring, and the full model comparison (market-only, RavenPack-enhanced, FinBERT-enhanced, and combined) are all implemented. Headline result: **daily news sentiment does not add out-of-sample predictive value at the next-session horizon** — see [Model Results](#model-results) below. The next stage moves to a higher-capacity sequence model (Autoformer) at a longer horizon.

## Environment Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

## Required Credentials

A valid WRDS account with access to RavenPack News Analytics and CRSP is required to regenerate the raw data.

Create a local `.env` file or enter credentials through the secure WRDS login prompt. Never commit credentials.

## Target Universe

Eleven SPDR sector ETFs representing the S&P 500 broken out by sector:

| Ticker | Sector |
|---|---|
| XLK | Technology |
| XLV | Health Care |
| XLF | Financials |
| XLC | Communication Services |
| XLY | Consumer Discretionary |
| XLI | Industrials |
| XLP | Consumer Staples |
| XLE | Energy |
| XLU | Utilities |
| XLB | Materials |
| XLRE | Real Estate |

## Data Sources

- **RavenPack News Analytics** via WRDS — structured macro news events with built-in sentiment scores, relevance scores, and timestamps (2020-01-01 through 2025-12-31; the derived news table may include one expected trailing roll-forward session)
- **CRSP daily stock file** via WRDS — daily returns, prices, and volume for the sector ETFs

See [Basic_EDA_Data_Model.md](Basic_EDA_Data_Model.md) for a full breakdown of the tables, columns, and data pipeline.

## Model Results

We compare four models for predicting next-session direction (`fwd_1d_positive`) across the eleven sector ETFs, using a **pooled expanding-window walk-forward** (train on strictly earlier sessions, test on each held-out year 2022–2025). Every model uses the **identical** logistic-regression specification, standardization, folds, and market features; only the sentiment-score columns change. All four are evaluated on **one shared row set** so differences are attributable solely to the sentiment source.

| Model | Features | AUC (mean folds) | Directional accuracy | AUC lift vs M0 |
|---|---|---|---|---|
| **M0** — Market only | 7 market features | 0.496 | 52.3% | — |
| **M1** — Market + RavenPack | M0 + RavenPack score, pos/neg share | 0.493 | 52.3% | −0.003 |
| **M2** — Market + FinBERT | M0 + FinBERT score, pos/neg share | 0.488 | 51.7% | −0.008 |
| **M3** — Market + RavenPack + FinBERT | M0 + both scorers' score, pos/neg share | 0.496 | 51.8% | −0.000 |
| *Always-up reference* | — | 0.500 | 52.5% | — |

**Finding — a clean, pre-registered null.** At the daily horizon, news sentiment (RavenPack, FinBERT, or both combined) does **not** improve next-session directional prediction beyond market features, and no fitted model reliably beats an always-up guess. M3 confirms the two scorers are not complementary here. Under the course rubric, a clearly-tested and explained null is a valid outcome, and it is reported as one rather than fished around.

**Why the null is credible.** The design guards that make it trustworthy are all in place: strictly temporal walk-forward splits, a lookahead-safe 4:00 PM ET news cutoff (verified in `03_data_quality_visual_qa.ipynb`), an identical row set and identical hyperparameters across models, and a naive benchmark to beat. Temporal text masking is deliberately deferred to a later masked-vs-unmasked ablation: RavenPack and FinBERT are both compared on unmasked text so the comparison stays apples-to-apples.

**Why the signal may still exist elsewhere.** A linear daily classifier is low-capacity, and daily direction is dominated by market microstructure noise; macro narrative sentiment plausibly acts at a **lower frequency** than one day. That motivates the next stage — an **Autoformer** on the LLM sentiment matrix at a longer horizon, which is where the report's monthly-macro-regime framing expects any narrative signal to live. The `outputs/model_comparison_all.csv` scoreboard is the linear-baseline bar that model must clear.

Outputs: `outputs/model_comparison_all.csv`, `all_model_scoreboard.png`, and the per-model walk-forward metrics and holdout predictions.

## Team

- **Jeremy Tang** — Data Engineering: WRDS/RavenPack ingestion, news-returns matching, timestamp alignment, feature engineering
- **Christian Goelz** — ML: model development, baseline vs. sentiment-augmented classifiers, evaluation metrics
- **Dongxin Liang** — NLP/LLM: FinBERT sentiment scoring, RavenPack-versus-FinBERT comparison, and interpretation of sentiment results

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/1_extract/01_ravenpack_news_extraction.ipynb` | WRDS/RavenPack extraction, 4:00 PM ET cutoff, session mapping, and daily sentiment aggregation |
| `notebooks/1_extract/02_crsp_sector_etf_price_extraction.ipynb` | WRDS/CRSP sector-ETF prices, returns, volume, and forward-return labels |
| `notebooks/2_prepare/03_data_quality_visual_qa.ipynb` | Schema, completeness, timestamp, forward-label, join, and visual QA scorecard |
| `notebooks/4_model/04_report_visual_plan.ipynb` | Report figures and presentation guardrails for the current derived panel |
| `notebooks/4_model/05_baseline_model.ipynb` | Simple pooled chronological baseline: market-only versus market-plus-sentiment prediction of next-session sector-ETF direction (**M0/M1**) |
| `notebooks/4_model/06_xlk_baseline_model.ipynb` | XLK-only walk-forward baseline and comparison of market-only versus market-plus-news features |
| `notebooks/4_model/07_all_sector_baseline_models.ipynb` | Per-sector walk-forward of market-only versus market-plus-RavenPack across all eleven ETFs |
| `notebooks/2_prepare/07a_llm_scoring_input_prep.ipynb` | Prepares deduplicated RavenPack text inputs and an event-to-text mapping for FinBERT scoring |
| `notebooks/2_prepare/07_finbert_sentiment_scoring.ipynb` | Scores each distinct news text with FinBERT and produces positive, negative, neutral, confidence, and continuous sentiment outputs |
| `notebooks/4_model/08_finbert_sentiment_summary.ipynb` | Validate FinBERT scores and run weekly exploratory analysis |
| `notebooks/4_model/09_finbert_sentiment_model.ipynb` | **M2** — builds the session-level FinBERT aggregate (`finbert_daily_df.csv`) and runs the pooled walk-forward comparing market-only, RavenPack, and FinBERT on one identical row set |
| `notebooks/4_model/10_combined_sentiment_model.ipynb` | **M3** — the four-model scoreboard: market, RavenPack, FinBERT, and RavenPack+FinBERT combined, in the same harness |
| `notebooks/3_explore/Basic_EDA_Analysis.ipynb` | Initial EDA: RavenPack global macro sentiment aggregated by trading session, joined to sector ETF returns (2020–2025) |
| `notebooks/3_explore/Equity_Sentiment_EDA2.ipynb` | Exploratory individual-stock extension using company-level RavenPack data; outside the current sector-ETF MVP |

## Licensing

WRDS RavenPack data is licensed for academic use only. Raw records, article text, and WRDS exports must not be committed to this repository. Only code, aggregated summary statistics, model outputs, and visualizations are stored here.
