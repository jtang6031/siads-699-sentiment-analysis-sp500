# Basic_EDA_Analysis — Data Model & Column Reference

> Academic research only. Not investment advice.

This document breaks down every table, intermediate dataframe, and column that `Basic_EDA_Analysis.ipynb` touches, in pipeline order.

---

## 1. Source tables (WRDS / PostgreSQL)

### `rpna.rpa_source_list`
A lookup table of every news publisher known to RavenPack. Each source has multiple attribute rows (long format).

| Column | Type | Description |
|---|---|---|
| `rp_entity_id` | string | RavenPack's internal ID for the source entity |
| `data_type` | string | Attribute name: `ENTITY_NAME`, `PUBLICATION_TYPE`, or `SOURCE_RANK` |
| `data_value` | string | Value for that attribute |

The notebook pivots this into wide format and keeps only **rank-1, non-blog** sources (372 out of 25,988 total). These are tier-1 wires and newspapers — Reuters, Dow Jones Newswires, Financial Times, Bloomberg, etc.

---

### `rpna.rpa_djpr_global_macro_{year}` (one table per year, 2020–2025)
The core RavenPack dataset. Each row is a single news **event** extracted from one article, associated with a macro-level entity (not a specific company). Tables are partitioned by year.

| Column | Type | Description |
|---|---|---|
| `rpa_date_utc` | date | Publication date in UTC — used for year-range filtering |
| `rpa_time_utc` | time | Publication time in UTC |
| `timestamp_utc` | timestamp | Full UTC timestamp — converted to Eastern Time for the 4 PM cutoff |
| `rp_story_id` | string | Unique article identifier. Multiple event rows share the same story ID when one article mentions multiple macro events |
| `rp_entity_id` | string | The macro entity this event refers to (country, central bank, commodity, etc.) |
| `rp_source_id` | string | The publishing source — joined to `rpa_source_list` to apply the quality filter |
| `source_name` | string | Human-readable source name (e.g., "Dow Jones Newswires") |
| `relevance` | float | How relevant the article is to the entity, 0–100. Notebook requires ≥ 90 |
| `event_relevance` | float | How relevant the specific event type is to the entity, 0–100. Notebook requires ≥ 90 |
| `event_sentiment_score` | float | Per-event sentiment on approximately **−1 to +1** scale. Confirmed empirically: observed min −0.96, max +0.95, mean +0.11 |
| `topic` | string | Broad topic: `business`, `economy`, `society`, `politics`, `environment` |
| `group` | string | Finer sub-topic: `products-services`, `commodity-prices`, `foreign-exchange`, `credit`, `interest-rates`, etc. |
| `css` | float | Composite Sentiment Score — an alternative aggregated score also present on the row |
| `headline` | string | Article headline (not used in aggregation; used only in spot-check queries) |

**Row volume after filters (relevance ≥ 90, rank-1 sources):** roughly 18–22 million events per year.

---

### `crsp.stksecurityinfohist` (or `crsp.stocknames` as fallback)
The CRSP security name/ticker history table. Used to identify which `permno` corresponds to each ETF ticker at a given point in time.

| Column | Type | Description |
|---|---|---|
| `permno` | int | CRSP permanent number — the stable, never-reused identifier for a security |
| `ticker` | string | Exchange ticker (e.g., `SPY`, `QQQ`) |
| `secinfostartdt` | date | Date this ticker–permno mapping became active |
| `secinfoenddt` | date | Date this mapping expired (NULL = still active) |

---

### `crsp.dsf_v2` (or `crsp.dsf` as fallback)
CRSP daily stock file — one row per security per trading session.

| Column | Type | Description |
|---|---|---|
| `dlycaldt` | date | Trading session date |
| `permno` | int | CRSP permanent number |
| `dlyret` | float | Daily total return (decimal, e.g. 0.012 = +1.2%) |
| `dlyprc` | float | Closing price (may be negative for bid-ask midpoint estimates; notebook takes `ABS()`) |
| `dlyvol` | float | Daily trading volume in shares |

---

## 2. Intermediate dataframes (Python)

### `sources_df` / `institutional_sources_df`
Produced by pivoting `rpa_source_list`. One row per source entity.

| Column | Description |
|---|---|
| `rp_entity_id` | Source entity ID |
| `source_name` | Publisher name |
| `source_type` | `NEWS`, `WIRE`, `BLOG`, etc. |
| `source_rank` | Numeric rank (1 = top tier) |
| `is_rank1_non_blog` | Boolean filter flag — `True` for the 372 kept sources |

`valid_source_ids` — the list of `rp_entity_id` values used in every subsequent SQL `WHERE rp_source_id IN (...)` filter.

---

### `market_daily_df`
One row per **(asset, trading session)**. 4 assets × 1,508 sessions = 6,032 rows.

| Column | Description |
|---|---|
| `session_date` | CRSP trading session date |
| `ticker` | ETF ticker (`SPY`, `QQQ`, `DIA`, `SOXX`) |
| `permno` | CRSP permanent number |
| `daily_return` | That session's total return |
| `price` | Closing price |
| `volume` | Trading volume |
| `asset` | Human label (`SPX_proxy`, `NASDAQ_100_proxy`, `Dow_Jones_proxy`, `SOX_proxy`) |
| `fwd_1d_return` | Compounded return over the **next 1 trading session** |
| `fwd_1d_positive` | 1 if `fwd_1d_return > 0`, else 0 (NaN for last session) |
| `fwd_5d_return` | Compounded return over the **next 5 trading sessions** |
| `fwd_5d_positive` | 1 if `fwd_5d_return > 0`, else 0 (NaN for last 5 sessions) |

---

### `news_signal_calendar_df`
Intermediate step: one row per **calendar date** (not yet aligned to trading sessions). 2,191 rows covering 2020–2025 including weekends and holidays.

| Column | Description |
|---|---|
| `signal_calendar_date` | Calendar date assigned to the news batch. Articles before 4 PM ET → same calendar date. Articles after 4 PM ET → next calendar date |
| `event_record_count` | Total filtered events for this calendar date |
| `unique_story_count` | Distinct articles (`rp_story_id`) for this date |
| `mean_event_sentiment_score` | Average `event_sentiment_score` across all events |
| `positive_event_count` | Events with score > +0.05 |
| `negative_event_count` | Events with score < −0.05 |
| `neutral_event_count` | Events with score between −0.05 and +0.05 |
| `unique_source_count` | Distinct publishing sources active this date |

---

### `news_daily_df`
Calendar dates collapsed onto trading sessions. When a weekend or holiday date maps to a Monday, its event counts are merged with any Sunday/Saturday counts that also map there. 1,508 rows — one per trading session.

Same columns as `news_signal_calendar_df`, plus:

| Column | Description |
|---|---|
| `session_date` | Actual CRSP trading session (≥ signal calendar date) |
| `positive_event_share` | `positive_event_count / event_record_count` |
| `negative_event_share` | `negative_event_count / event_record_count` |
| `neutral_event_share` | `neutral_event_count / event_record_count` (the three shares sum to 1.0) |
| `sentiment_bucket` | `"positive"` if mean score > +0.05, `"negative"` if < −0.05, else `"neutral"` |

---

## 3. Final panel: `eda_panel_df`

**Shape:** 6,032 rows × 26 columns  
**Grain:** one row per **(asset, trading session)**

Produced by a left join of `market_daily_df` onto `news_daily_df` on `session_date`. The macro news features are **shared across all four assets on a given day** — the same sentiment score appears in all four rows for 2020-01-02, for example.

### Complete column list

| Column | Source | Description |
|---|---|---|
| `session_date` | CRSP | Trading session date |
| `ticker` | CRSP | ETF ticker |
| `permno` | CRSP | CRSP permanent number |
| `daily_return` | CRSP | That session's return |
| `price` | CRSP | Closing price |
| `volume` | CRSP | Trading volume |
| `asset` | Derived | Human-readable ETF label |
| `fwd_1d_return` | Derived | Next-session compounded return |
| `fwd_1d_positive` | Derived | Binary: did next session go up? |
| `fwd_5d_return` | Derived | Next-5-session compounded return |
| `fwd_5d_positive` | Derived | Binary: did next 5 sessions go up? |
| `event_record_count` | RavenPack | Total news events for this session (0 if no news) |
| `unique_story_count` | RavenPack | Distinct articles for this session |
| `mean_event_sentiment_score` | RavenPack | Volume-weighted average sentiment (−1 to +1) |
| `positive_event_count` | RavenPack | Events with score > +0.05 |
| `negative_event_count` | RavenPack | Events with score < −0.05 |
| `neutral_event_count` | RavenPack | Events with score between −0.05 and +0.05 |
| `unique_source_count` | RavenPack | Distinct publishers active this session |
| `positive_event_share` | Derived | Fraction of events that were positive |
| `negative_event_share` | Derived | Fraction of events that were negative |
| `neutral_event_share` | Derived | Fraction of events that were neutral |
| `sentiment_bucket` | Derived | `"positive"` / `"negative"` / `"neutral"` label for the day |
| `has_macro_news` | Derived | Boolean: was `event_record_count > 0`? (always True in this dataset) |
| `year` | Derived | Calendar year of `session_date` |
| `month` | Derived | First day of the month containing `session_date` |

---

## 4. Key design decisions

**After-hours cutoff:** Any article published after 4:00 PM Eastern Time is credited to the *next* calendar day, then mapped to the next trading session. This prevents lookahead bias — a 5 PM article cannot influence returns for the session that already closed.

**Source quality filter:** Only 372 rank-1, non-blog sources are included. This eliminates noise from low-quality aggregators and ensures the sentiment signal comes from authoritative financial journalism.

**Relevance thresholds:** Both `relevance ≥ 90` and `event_relevance ≥ 90` are required. This keeps only articles where the macro event and entity are highly central to the story — not passing mentions.

**Sentiment scale:** Confirmed empirically to be approximately −1 to +1. Thresholds of ±0.05 define the neutral band, which keeps ~25–35% of events in each direction bucket.

**Forward returns:** Computed as compounded products (not simple sums), so `fwd_5d_return = (1+r₁)(1+r₂)(1+r₃)(1+r₄)(1+r₅) − 1`. Validated by asserting `fwd_1d_return[t] == daily_return[t+1]` for all non-terminal rows.
