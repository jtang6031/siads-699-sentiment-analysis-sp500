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
- **Redistribution:** **Not permitted.** Raw event records, article text, and licensed headlines
  are excluded from this repository and written only to the gitignored `data/raw/` directory.
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
- **Redistribution:** **Not permitted.** The raw CRSP pull is written only to the gitignored
  `data/raw/` directory.
- **Collected in:** [notebooks/1_extract/02_crsp_sector_etf_price_extraction.ipynb](notebooks/1_extract/02_crsp_sector_etf_price_extraction.ipynb)

### 3. FinBERT sentiment scores — derived by this project

- **Model:** [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert), a publicly released
  BERT model fine-tuned for financial sentiment, distributed on Hugging Face under the **Apache
  License 2.0**. We did not train or fine-tune it; we ran the released weights unmodified.
- **What we derived:** positive / negative / neutral probabilities for each distinct headline in
  the RavenPack extract, reduced to a per-headline sentiment score and then aggregated to a daily
  series. This is the only data product this project created rather than obtained.
- **Important:** the scores are derived *from licensed RavenPack text*. The per-headline score file
  (`data/raw/finbert_scores.csv`) is row-aligned to licensed records and is therefore **not
  published**. Only the daily aggregate is committed.
- **Produced in:** [notebooks/2_prepare/07a_llm_scoring_input_prep.ipynb](notebooks/2_prepare/07a_llm_scoring_input_prep.ipynb),
  [notebooks/2_prepare/07_finbert_sentiment_scoring.ipynb](notebooks/2_prepare/07_finbert_sentiment_scoring.ipynb)

### 4. Supplementary public series (exploratory V5 notebook only)

- **FRED** (Federal Reserve Bank of St. Louis), accessed via `pandas-datareader`. FRED series are
  freely redistributable, though individual series may carry the originating provider's terms.
- **Yahoo Finance**, accessed via the third-party `yfinance` package, used as a price fallback.
  Yahoo Finance data is subject to Yahoo's terms of service and is **not** redistributed here.

These appear only in `Corrected_SIADS_699_Capstone_Features_v5.ipynb` and
[notebooks/4_model/12_autoformer_extended_test.ipynb](notebooks/4_model/12_autoformer_extended_test.ipynb),
which the README identifies as exploratory and outside the supported public reproduction path.

## What is committed to this repository

Licensed row-level source records are **not** committed. The files in [data/](data/) are:

| File | Rows | Grain | Content |
|---|---:|---|---|
| `news_daily_df.csv` | 2,890 | one row per session | Daily aggregate counts and mean RavenPack sentiment |
| `finbert_daily_df.csv` | 2,766 | one row per session | Daily aggregate FinBERT sentiment shares |
| `model_inputs_2015_2026.csv` | 2,766 | one row per session | Modeling features and targets |
| `market_daily_df.csv` | 29,362 | one row per ETF per session | CRSP daily price, return, volume, `permno` |
| `model_daily_panel.csv` | 29,362 | one row per ETF per session | The above joined to daily news aggregates |

The three session-level files contain only aggregated statistics — no story identifiers, no
headline text, and no per-record RavenPack fields. They do not permit reconstruction of the
licensed source records.

> **Note on the two ETF-level files.** `market_daily_df.csv` and `model_daily_panel.csv` carry
> CRSP daily price, return, volume, and `permno` values at their original row grain rather than as
> aggregates. Anyone preparing this repository for distribution beyond the course should confirm
> with the University of Michigan WRDS representative or CRSP that publishing these specific fields
> is acceptable, and remove or further aggregate them if it is not. The reproduction path described
> in the README depends only on `model_inputs_2015_2026.csv`.

Outputs in [outputs/](outputs/) are model metrics and figures. They contain no licensed text.

## Obligations we observe

- WRDS data is used for academic research only, consistent with the terms attached to the
  University of Michigan subscription.
- No RavenPack or CRSP record-level data is redistributed in this repository or in the report.
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
