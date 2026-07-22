# siads-699-sentiment-analysis-sp500

**Academic research only. Not investment advice.**

University of Michigan MADS Capstone (SIADS 699) — Team Alpha Signal.

## Research Question

Does LLM-derived sentiment extracted from financial news meaningfully predict short-term price direction for US sector ETFs, beyond what traditional market features (recent returns, sector momentum, trading volume) already capture?

This is the final project goal. The current repository contains the RavenPack-sentiment baseline that establishes the data, timing, and modeling pipeline; the LLM scoring and LLM-versus-RavenPack comparison remain the next implementation step.

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
- **Dongxin Liang** — NLP/LLM: sentiment-analysis research framing and future LLM/RavenPack comparison design

## Notebooks

| Notebook | Description |
|---|---|
| `data_collection/01_ravenpack_news_extraction.ipynb` | WRDS/RavenPack extraction, 4:00 PM ET cutoff, session mapping, and daily sentiment aggregation |
| `data_collection/02_crsp_sector_etf_price_extraction.ipynb` | WRDS/CRSP sector-ETF prices, returns, volume, and forward-return labels |
| `data_collection/03_data_quality_visual_qa.ipynb` | Schema, completeness, timestamp, forward-label, join, and visual QA scorecard |
| `04_report_visual_plan.ipynb` | Report figures and presentation guardrails for the current derived panel |
| `06_xlk_baseline_model.ipynb` | XLK-only walk-forward baseline and comparison of market-only versus market-plus-news features |
| `Basic_EDA_Analysis.ipynb` | Initial EDA: RavenPack global macro sentiment aggregated by trading session, joined to sector ETF returns (2020–2025) |
| `Equity_Sentiment_EDA2.ipynb` | Equity-level EDA: per-company RavenPack sentiment joined to individual S&P 500 stock returns via CUSIP matching (2022–2023) |
| `05_baseline_model.ipynb` | Simple chronological baseline: market-only versus market-plus-sentiment prediction of next-session sector-ETF direction |

## Licensing

WRDS RavenPack data is licensed for academic use only. Raw records, article text, and WRDS exports must not be committed to this repository. Only code, aggregated summary statistics, model outputs, and visualizations are stored here.
