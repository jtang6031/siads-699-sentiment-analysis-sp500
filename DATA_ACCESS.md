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

## What this repository publishes

**This repository is public, and it does not contain raw WRDS extracts.** No RavenPack article text,
and no row-level RavenPack event records, are published here. What is published is the *curated*
layer: session-level aggregates, engineered modeling features, model outputs, and figures.

The row-level licensed material is excluded by [.gitignore](.gitignore):

```text
data/raw/                  # RavenPack event extracts, CRSP daily extracts, FinBERT per-text scores
data/llm_files/cache/      # the yearly and combined RavenPack news caches, with headline text
*_input.jsonl              # LLM batch prompts, which embed headline text
```

Anyone reproducing the collection stages must pull that data themselves from WRDS under their own
entitlement. See [What is in this repository](#what-is-in-this-repository) for the file-by-file
split, and [Residual licensed identifiers](#residual-licensed-identifiers) for two categories that
remain and are a deliberate judgment call.

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
  sentiment scores. Headline text was used as FinBERT and LLM input.
- **Published here:** **nothing at row grain.** No headline text, no per-event records, no RavenPack
  relevance or event-sentiment values at story level. Only daily aggregates computed from them —
  counts, shares, and mean sentiment per session — appear in `data/news_daily_df.csv` and
  `data/model_inputs_2015_2026.csv`. The row-level extract lives in gitignored `data/raw/` and
  `data/llm_files/cache/` on team machines only.
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
- **Published here:** the daily ETF panel is published in `data/market_daily_df.csv` and
  `data/model_daily_panel.csv`, including the `permno` column — see
  [Residual licensed identifiers](#residual-licensed-identifiers). These cover eleven exchange-traded
  funds whose daily prices are also available from public sources; the CRSP-specific contribution is
  the cleaning and the `permno` mapping. The raw extract under `data/raw/` is gitignored.
- **Collected in:** [notebooks/1_extract/02_crsp_sector_etf_price_extraction.ipynb](notebooks/1_extract/02_crsp_sector_etf_price_extraction.ipynb)

### 3. FinBERT sentiment scores — derived by this project

- **Model:** [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert), a publicly released
  BERT model fine-tuned for financial sentiment, distributed on Hugging Face under the **Apache
  License 2.0**. We did not train or fine-tune it; we ran the released weights unmodified.
- **What we derived:** positive / negative / neutral probabilities for each distinct headline in
  the RavenPack extract, reduced to a per-headline sentiment score and then aggregated to a daily
  series. This is the only data product this project created rather than obtained.
- **Published here:** the daily aggregate in `data/finbert_daily_df.csv`, and the model-generated
  scoring outputs under `data/llm_files/`. These are scores, not text: no headline they were computed
  from is reproduced. The per-text scores keyed to individual stories stay in gitignored
  `data/raw/finbert_scores.csv`.
- **Produced in:** [notebooks/2_prepare/07a_llm_scoring_input_prep.ipynb](notebooks/2_prepare/07a_llm_scoring_input_prep.ipynb),
  [notebooks/2_prepare/07_finbert_sentiment_scoring.ipynb](notebooks/2_prepare/07_finbert_sentiment_scoring.ipynb)

### 4. Supplementary public series (exploratory V5 notebook only)

- **FRED** (Federal Reserve Bank of St. Louis), accessed via `pandas-datareader`. FRED series are
  freely redistributable, though individual series may carry the originating provider's terms.
- **Yahoo Finance**, accessed via the third-party `yfinance` package, used as a price fallback.
  Yahoo Finance data is subject to Yahoo's terms of service and is **not** redistributed here.

These appear only in
[notebooks/4_model/12_llm_autoformer_models.ipynb](notebooks/4_model/12_llm_autoformer_models.ipynb),
which the README identifies as exploratory and outside the supported reproduction path.

## What reproduction is still possible

Excluding the row-level extracts costs less reproducibility than it might appear, because the
curated layer is what the supported path actually consumes:

| Path | Needs WRDS? | Why |
|---|---|---|
| `run_training.py --stage train` | **No** | Reads only `data/model_inputs_2015_2026.csv`, which is published |
| `run_training.py --stage clean` | Yes | Reads `data/raw/`, which was already gitignored before this repository went public |
| `run_pipeline.py` (all five stages) | Yes | These *are* the collection stages; rebuilding `data/raw/` is their purpose |
| Notebooks 05–11 | Mixed | The daily-aggregate notebooks run from published `data/*.csv`; those reading `data/raw/` do not |
| Notebook 12, collection half | Yes | Reads the RavenPack news cache under `data/llm_files/cache/`, which is not published, so it falls through to its WRDS branch |
| Notebook 12, modelling half | **No** | Resumes from the published `final_engineered_m6_panel.parquet` and `feature_sets.json`, and produces every M0–M7 table and figure from there |

In other words, the headline reproduction path in the README — install, test, rerun the statistical
grid — works from a clean public clone with no credentials at all, and so does the entire Autoformer
ablation provided it is started from its panel-reload cell rather than from the top. What requires
WRDS is rebuilding the inputs, which required WRDS before as well.

Rebuilding from scratch requires an entitled WRDS account, one query per year against RavenPack, and
a FinBERT scoring pass over roughly 600,000 distinct texts that takes many hours and benefits from a
CUDA-capable GPU.

## Residual licensed identifiers

Two categories of WRDS-derived identifier remain in published files. Both are deliberate, and both
are recorded here rather than left implicit:

1. **`rp_story_id` in the LLM scoring outputs.** `llm_sector_scores.parquet` (171,146 rows),
   `llm_processed_ids.parquet` (151,524 rows) and `llm_sample_scores.parquet` (72 rows) are keyed by
   RavenPack story ID. The *scores* are model-generated by this project; the ID is an opaque 32-hex
   key carrying no text, no sentiment value and no metadata, and it is inert without a RavenPack
   subscription. It is retained because dropping it would make the sector-attribution scores
   impossible to rejoin and would silently break notebook 12's merge.
2. **`permno` in the CRSP daily panel.** `market_daily_df.csv` and `model_daily_panel.csv` carry
   CRSP's permanent security identifier alongside daily price, return and volume for eleven ETFs.

Neither reproduces licensed *content*. If the reviewing instructor or WRDS would prefer these
removed, the story IDs can be replaced with a salted hash applied consistently on both sides of the
join, and `permno` can be dropped without affecting any model — no feature in
`model_inputs_2015_2026.csv` derives from it.

## What is in this repository

Session-level aggregates in [data/](data/):

| File | Rows | Grain | Content |
|---|---:|---|---|
| `news_daily_df.csv` | 2,890 | one row per session | Daily aggregate counts and mean RavenPack sentiment |
| `finbert_daily_df.csv` | 2,766 | one row per session | Daily aggregate FinBERT sentiment shares |
| `model_inputs_2015_2026.csv` | 2,766 | one row per session | Modeling features and targets |
| `market_daily_df.csv` | 29,362 | one row per ETF per session | CRSP daily price, return, volume, `permno` |
| `model_daily_panel.csv` | 29,362 | one row per ETF per session | The above joined to daily news aggregates |

Scoring artifacts and engineered features in [data/llm_files/](data/llm_files/):

| File | Size | Content |
|---|---:|---|
| `llm_batch_output.jsonl` | 32 MB | Model-generated sector attribution scores, keyed by story ID |
| `llm_monthly_output.jsonl` | 1.0 MB | Model-generated monthly themes |
| `llm_sector_scores.parquet`, `llm_processed_ids.parquet`, `llm_sample_scores.parquet` | ~10 MB | Scoring outputs and resume state, keyed by `rp_story_id` |
| `llm_monthly_themes.parquet` | 0.1 MB | Monthly theme labels, model-generated |
| `final_engineered_m6_panel.parquet` | 12 MB | Engineered modeling panel for the V5 study |
| `feature_sets.json` | small | Feature-set definitions for the V5 models |

Outputs in [outputs/](outputs/) are model metrics and figures. They contain no licensed text.

**Excluded by [.gitignore](.gitignore), not published:**

| Path | Content |
|---|---|
| `data/raw/` | RavenPack event extracts with headline text, CRSP daily extracts, FinBERT per-text scores |
| `data/llm_files/cache/macro_news_*.parquet` | 1,199,890 RavenPack event records with headline text, relevance, event sentiment and event taxonomy |
| `data/llm_files/*_input.jsonl` | LLM batch prompts, which embed headline text verbatim |

`data/llm_files/cache/sector_prices.parquet` (CRSP daily OHLCV) is excluded by the same `cache/`
rule. It carries no licensed identifier, but it is regenerated by the notebook that used it, so it
was not worth a narrower exception.

## Obligations we observe

- WRDS data is used for academic research only, consistent with the terms attached to the
  University of Michigan subscription.
- Row-level RavenPack and CRSP extracts are **not redistributed**. They remain on team machines
  under gitignored paths, and no licensed headline text appears in this repository, in the report,
  or in any figure.
- No credentials appear in code, notebook output, or version history.
- Notebook saved cells are treated as published output: any cell that would print licensed headline
  text must be cleared before commit, not merely scrolled past.
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
