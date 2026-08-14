# Data Access Statement

**Academic research only. This project is not investment advice.**

This document states where every input to this project came from, who owns it, how our team
obtained access, and what may and may not be redistributed in this repository.

## Summary

All source data used in this project was obtained from **Wharton Research Data Services (WRDS)**
under the University of Michigan's institutional subscription. Our team did not purchase, scrape,
or otherwise collect primary data. The **only data we derived ourselves** is the FinBERT sentiment
score, which we produced by running a public pre-trained model locally over text obtained from
WRDS.

Two supplementary public series (FRED and Yahoo Finance) appear only in the exploratory V5
notebook, which is not part of the supported reproduction path.

## How we accessed the data

Each team member used their own individual WRDS account, granted through their University of
Michigan affiliation. WRDS accounts are personal and non-transferable; we did not share a single
login.

Connections are made from the extraction notebooks with the `wrds` Python client, which
authenticates to the WRDS PostgreSQL host:

```python
import wrds
db = wrds.Connection()
```

Credentials are supplied through a local `.pgpass` file created once per machine and never checked
into the repository:

```bash
python -c "import wrds; wrds.Connection(wrds_username='YOUR_USERNAME').create_pgpass_file()"
```

No WRDS username, password, or API key appears in any source file, notebook cell, saved notebook
output, or commit in this repository. `.env` and `data/raw/` are listed in [.gitignore](.gitignore).

**Reproducing the extraction steps requires the reader to supply their own WRDS account with
RavenPack and CRSP entitlements.** We cannot grant access on anyone's behalf, and access is not
included with this repository.

## Sources and ownership

### 1. RavenPack News Analytics (via WRDS, `rpna` library)

- **Owner:** RavenPack. Distributed to subscribing institutions through WRDS.
- **Access:** University of Michigan WRDS subscription, academic use only.
- **Tables used:** `rpna.rpa_djpr_global_macro_{year}` (macro news events),
  `rpna.rpa_source_list`.
- **Window pulled:** configured `2015-01-01` through `2026-12-31`; the reported study window ends
  `2025-12-31`.
- **Fields used:** event timestamps, relevance, entity identifiers, and RavenPack's structured
  sentiment scores. Headline text was used as FinBERT input but is not published.
- **Held in this repository:** the extracted news cache under `data/llm_files/` includes row-level
  event records with headline text. It is included so the project can be run without repeating the
  multi-hour WRDS extraction (see [Why the extracted data is included](#why-the-extracted-data-is-included)).
  This repository is private and the data is not shared or published anywhere else.
- **Collected in:** [notebooks/1_extract/01_ravenpack_news_extraction.ipynb](notebooks/1_extract/01_ravenpack_news_extraction.ipynb)

### 2. CRSP US Stock Database (via WRDS, `crsp` library)

- **Owner:** Center for Research in Security Prices, LLC (CRSP), an affiliate of the University of
  Chicago Booth School of Business. Distributed through WRDS.
- **Access:** University of Michigan WRDS subscription, academic use only.
- **Tables used:** `crsp.dsf_v2` (daily security file) and `crsp.stksecurityinfohist` (security
  name/identifier history, also used to map CUSIPs to SIC-based sectors).
- **Coverage:** 11 SPDR sector ETFs — XLK, XLV, XLF, XLC, XLY, XLI, XLP, XLE, XLU, XLB, XLRE —
  over 2015-01-01 through 2025-12-31 (29,362 daily fund observations).
- **Fields used:** daily return, closing price, opening price, trading volume, and `permno`.
- **Held in this repository:** the daily ETF panel is included under `data/` and
  `data/llm_files/cache/sector_prices.parquet` for the same efficiency reason. Private repository,
  not shared or published elsewhere.
- **Collected in:** [notebooks/1_extract/02_crsp_sector_etf_price_extraction.ipynb](notebooks/1_extract/02_crsp_sector_etf_price_extraction.ipynb)

### 3. FinBERT sentiment scores — derived by this project

- **Model:** [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert), a publicly released
  BERT model fine-tuned for financial sentiment, distributed on Hugging Face under the **Apache
  License 2.0**. We did not train or fine-tune it; we ran the released weights unmodified.
- **What we derived:** positive / negative / neutral probabilities for each distinct headline in
  the RavenPack extract, reduced to a per-headline sentiment score and then aggregated to a daily
  series. This is the only data product this project created rather than obtained.
- **Held in this repository:** the daily aggregate is in `data/finbert_daily_df.csv`, and the LLM
  scoring artifacts are under `data/llm_files/`. These are included so the scoring pass — which needs
  a GPU and several hours — does not have to be repeated. The scores derive from licensed RavenPack
  text, so they are held under the same terms as the source: private repository, not shared or
  published elsewhere.
- **Produced in:** [notebooks/2_prepare/07a_llm_scoring_input_prep.ipynb](notebooks/2_prepare/07a_llm_scoring_input_prep.ipynb),
  [notebooks/2_prepare/07_finbert_sentiment_scoring.ipynb](notebooks/2_prepare/07_finbert_sentiment_scoring.ipynb)

### 4. Supplementary public series (exploratory V5 notebook only)

- **FRED** (Federal Reserve Bank of St. Louis), accessed via `pandas-datareader`. FRED series are
  freely redistributable, though individual series may carry the originating provider's terms.
- **Yahoo Finance**, accessed via the third-party `yfinance` package, used as a price fallback.
  Yahoo Finance data is subject to Yahoo's terms of service and is **not** redistributed here.

These appear only in `SIADS_699_Capstone_Features_LLM_vfinal.ipynb` and
[notebooks/4_model/12_autoformer_extended_test.ipynb](notebooks/4_model/12_autoformer_extended_test.ipynb),
which the README identifies as exploratory and outside the supported public reproduction path.

## Why the extracted data is included

This repository is **private**. It deliberately includes the extracted source data and the scoring
artifacts so the project can be reproduced without repeating the collection work.

Rebuilding those inputs from scratch requires an entitled WRDS account, one query per year against
RavenPack, and a FinBERT scoring pass over roughly 600,000 distinct texts that is many hours of
work and benefits from a CUDA-capable GPU. Including the extracts means a reviewer can clone the
repository, install the requirements, and run training and evaluation directly — the collection
stages become optional rather than mandatory.

This data is held for this project's use only. It is **not shared, published, or redistributed
anywhere outside this private repository.**

## What is in this repository

Session-level aggregates in [data/](data/):

| File | Rows | Grain | Content |
|---|---:|---|---|
| `news_daily_df.csv` | 2,890 | one row per session | Daily aggregate counts and mean RavenPack sentiment |
| `finbert_daily_df.csv` | 2,766 | one row per session | Daily aggregate FinBERT sentiment shares |
| `model_inputs_2015_2026.csv` | 2,766 | one row per session | Modeling features and targets |
| `market_daily_df.csv` | 29,362 | one row per ETF per session | CRSP daily price, return, volume, `permno` |
| `model_daily_panel.csv` | 29,362 | one row per ETF per session | The above joined to daily news aggregates |

Extracted source data and scoring artifacts in [data/llm_files/](data/llm_files/):

| File | Size | Content |
|---|---:|---|
| `cache/macro_news_{2015..2025}.parquet` | ~83 MB | RavenPack event records **including headline text** |
| `cache/macro_news_all.parquet` | 82 MB | The eleven yearly files combined (1,199,890 rows) |
| `cache/sector_prices.parquet` | 1.3 MB | CRSP daily OHLCV for the eleven sector ETFs |
| `llm_batch_input.jsonl` | 28 MB | Gemini batch prompts, containing RavenPack headlines |
| `llm_batch_output.jsonl` | 32 MB | Model-generated sector attribution scores |
| `llm_monthly_input.jsonl` | 6.5 MB | Monthly digest prompts, containing RavenPack headlines |
| `llm_monthly_output.jsonl` | 1.0 MB | Model-generated monthly themes |
| `llm_sector_scores.parquet`, `llm_processed_ids.parquet`, `llm_monthly_themes.parquet`, `llm_sample_scores.parquet` | ~5 MB | Scoring outputs and resume state |
| `final_engineered_m6_panel.parquet` | 12 MB | Engineered modeling panel for the V5 study |
| `feature_sets.json` | small | Feature-set definitions for the V5 models |

The files marked as containing headline text hold licensed RavenPack material at row grain. They
are present for the efficiency reason above and are covered by the same restriction: private
repository, no external sharing.

Outputs in [outputs/](outputs/) are model metrics and figures. They contain no licensed text.

> **Before making this repository public.** The licensed material under `data/llm_files/` — the
> news caches with headline text and the LLM batch inputs built from them — must be removed from
> the working tree **and from git history**, since a plain deletion leaves the data recoverable in
> earlier commits. `market_daily_df.csv` and `model_daily_panel.csv` also carry CRSP fields at their
> original row grain and should be reviewed at the same time. The reproduction path described in the
> README depends only on `model_inputs_2015_2026.csv`.

## Obligations we observe

- WRDS data is used for academic research only, consistent with the terms attached to the
  University of Michigan subscription.
- The extracted RavenPack and CRSP records held in this private repository are used solely for this
  project. They are not shared, published, or redistributed outside it, and no licensed headline
  text is reproduced in the report or in any figure.
- No credentials appear in code, notebook output, or version history.
- The exploratory V4 and V5 root notebooks contain licensed or sensitive material in their saved
  cells and must be sanitized or removed before any public release, as noted in the README.
- FinBERT is used under Apache 2.0; the model is attributed above and its weights are not
  redistributed here (they download from Hugging Face at runtime).

## Authoritative terms

Readers should consult the source terms directly rather than relying on this summary:

- WRDS terms of use — <https://wrds-www.wharton.upenn.edu/>
- CRSP — <https://www.crsp.org/>
- RavenPack — <https://www.ravenpack.com/>
- FinBERT model card and license — <https://huggingface.co/ProsusAI/finbert>
- FRED terms of use — <https://fred.stlouisfed.org/legal/>

## Contact

Questions about data handling in this repository can be directed to the project team listed in the
[README](README.md#team). Questions about entitlement to the underlying WRDS datasets should go to
the University of Michigan library's WRDS support or to WRDS directly.
