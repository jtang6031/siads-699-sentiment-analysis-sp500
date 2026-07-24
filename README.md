# siads-699-sentiment-analysis-sp500

**Academic research only. Not investment advice.**

University of Michigan MADS Capstone (SIADS 699) — Team Alpha Signal.

## Research Question

Does FinBERT-derived sentiment from financial macro news add incremental out-of-sample predictive power for the next-day direction of the 11 S&P 500 sector ETFs, beyond traditional market features and RavenPack’s existing sentiment score?

This is the final project goal. The RavenPack and CRSP data pipelines, data-quality checks, and baseline models have been implemented. FinBERT sentiment scoring is currently being completed, followed by the final comparison of market-only, RavenPack-enhanced, and FinBERT-enhanced models.

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

## Team

- **Jeremy Tang** — Data Engineering: WRDS/RavenPack ingestion, news-returns matching, timestamp alignment, feature engineering
- **Christian Goelz** — ML: model development, baseline vs. sentiment-augmented classifiers, evaluation metrics
- **Dongxin Liang** — NLP/LLM: FinBERT sentiment scoring, RavenPack-versus-FinBERT comparison, and interpretation of sentiment results

## Notebooks

| Notebook | Description |
|---|---|
| `data_collection/01_ravenpack_news_extraction.ipynb` | WRDS/RavenPack extraction, 4:00 PM ET cutoff, session mapping, and daily sentiment aggregation |
| `data_collection/02_crsp_sector_etf_price_extraction.ipynb` | WRDS/CRSP sector-ETF prices, returns, volume, and forward-return labels |
| `data_collection/03_data_quality_visual_qa.ipynb` | Schema, completeness, timestamp, forward-label, join, and visual QA scorecard |
| `04_report_visual_plan.ipynb` | Report figures and presentation guardrails for the current derived panel |
| `05_baseline_model.ipynb` | Simple chronological baseline: market-only versus market-plus-sentiment prediction of next-session sector-ETF direction |
| `06_xlk_baseline_model.ipynb` | XLK-only walk-forward baseline and comparison of market-only versus market-plus-news features |
| `Basic_EDA_Analysis.ipynb` | Initial EDA: RavenPack global macro sentiment aggregated by trading session, joined to sector ETF returns (2020–2025) |
| `Equity_Sentiment_EDA2.ipynb` | Exploratory individual-stock extension using company-level RavenPack data; outside the current sector-ETF MVP |
| `data_collection/07a_llm_scoring_input_prep.ipynb` | Prepares deduplicated RavenPack text inputs and an event-to-text mapping for FinBERT scoring |
| `07_finbert_sentiment_scoring.ipynb` | Scores each distinct news text with FinBERT and produces positive, negative, neutral, confidence, and continuous sentiment outputs |
| `08_finbert_feature_aggregation.ipynb` | Planned: working on it 7/24/26

## Licensing

WRDS RavenPack data is licensed for academic use only. Raw records, article text, and WRDS exports must not be committed to this repository. Only code, aggregated summary statistics, model outputs, and visualizations are stored here.
