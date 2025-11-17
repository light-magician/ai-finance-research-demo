# Data Dictionary

## Overview

This document describes the canonical data tables generated during Phase 3 (Data Engineering). All data is standardized to UTC timestamps and uses canonical symbol naming (BTC, ETH, etc.).

---

## 1. OHLCV Panel (ohlcv_panel)

Historical Open, High, Low, Close, Volume data for spot and perpetual markets.

### Schema

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `timestamp` | datetime (UTC) | Time of candle close | All sources |
| `symbol` | string | Canonical symbol (BTC, ETH, SOL) | Normalized from exchange |
| `exchange` | string | Exchange name (binance, bybit, okx) | Source identifier |
| `market` | string | Market type (spot, perp_usdt, perp_coin) | Exchange specific |
| `open` | float | Opening price (USDT) | Native OHLCV |
| `high` | float | Highest price in period | Native OHLCV |
| `low` | float | Lowest price in period | Native OHLCV |
| `close` | float | Closing price (USDT) | Native OHLCV |
| `volume` | float | Volume in base asset (BTC, ETH, etc.) | Native OHLCV |
| `quote_volume` | float | Volume in quote asset (USDT) | Native OHLCV |
| `trades` | int | Number of trades in period (if available) | Exchange data |
| `taker_buy_volume` | float | Taker buy volume (if available) | Exchange data |
| `interval` | string | Candle interval (1m, 5m, 1h, 1d) | Aggregation level |

### Indexing

- Primary: (timestamp, symbol, exchange, market)
- Secondary: (exchange, symbol), (market), timestamp range

### Frequency

- 1-minute to daily intervals depending on analysis requirements
- Resample as needed from raw tick data or provided bars

### Data Quality Notes

- Binance Vision: High quality, checksummed
- CryptoDataDownload: Good quality, user-validated
- Tardis.dev: Excellent quality, tick-reconstructable

---

## 2. Funding Rates Panel (funding_panel)

Historical perpetual futures funding rates.

### Schema

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `timestamp` | datetime (UTC) | Time of funding accrual | Exchange settlement time |
| `symbol` | string | Canonical symbol (BTC, ETH) | Normalized from exchange |
| `exchange` | string | Exchange name (binance, bybit, okx, deribit) | Source identifier |
| `funding_rate` | float | Funding rate as decimal (e.g., 0.00123) | Native funding data |
| `funding_rate_pct` | float | Funding rate as percentage (e.g., 0.123%) | Converted from rate |
| `mark_price` | float | Mark price at funding time (USDT) | Exchange data |
| `index_price` | float | Index price at funding time (USDT) | Exchange data |
| `basis_bps` | float | Basis in basis points | (mark - index) / index * 10000 |
| `funding_interval_hours` | int | Hours between funding accruals | 1 (hourly) or 8 (every 8h) |

### Indexing

- Primary: (timestamp, symbol, exchange)
- Secondary: (exchange, symbol), timestamp range

### Frequency

- Typically 8 hours (Binance, Bybit, OKX) or hourly (some exchanges)
- Aligned to exchange settlement times (0:00, 8:00, 16:00 UTC for Binance)

### Data Quality Notes

- Tardis.dev, CoinAPI: Complete historical (2019+)
- CryptoDataDownload, Coinalyze: Free sources available
- GitHub tools: Open-source but may have gaps

---

## 3. Open Interest Panel (oi_panel)

Aggregated open interest across perpetual contracts.

### Schema

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `timestamp` | datetime (UTC) | Snapshot time | Exchange data |
| `symbol` | string | Canonical symbol (BTC, ETH) | Normalized |
| `exchange` | string | Exchange name | Source identifier |
| `open_interest` | float | Total OI in USD equivalent | Native OI |
| `open_interest_contracts` | float | OI in contract counts (if available) | Exchange data |
| `long_position_count` | int | Number of open long positions (if available) | Exchange data |
| `short_position_count` | int | Number of open short positions (if available) | Exchange data |
| `long_short_ratio` | float | Ratio of long to short OI | long_oi / short_oi |
| `top_trader_long_ratio` | float | Top trader long ratio (if available) | CoinGlass API |
| `top_trader_short_ratio` | float | Top trader short ratio (if available) | CoinGlass API |

### Indexing

- Primary: (timestamp, symbol, exchange)
- Secondary: (exchange, symbol), timestamp range

### Frequency

- Typically daily or hourly, depending on source
- CoinGlass: Real-time, resampled to 1-hour
- Official APIs: Various frequencies

### Data Quality Notes

- CoinGlass: Specialized, 1-second updates
- Tardis.dev: Tick-level reconstructed
- CoinAPI: Historical metrics available

---

## 4. Liquidations Panel (liquidations_panel)

Aggregated liquidation events across exchanges.

### Schema

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `timestamp` | datetime (UTC) | Liquidation time | Exchange events |
| `symbol` | string | Canonical symbol (BTC, ETH) | Normalized |
| `exchange` | string | Exchange name | Source identifier |
| `side` | string | Liquidation side (BUY, SELL) | Event data |
| `liquidation_price` | float | Price at which liquidation occurred (USDT) | Event data |
| `quantity` | float | Quantity liquidated (in base asset) | Event data |
| `volume_usd` | float | Liquidation volume in USD | quantity * price |
| `liquidation_type` | string | Type (ADL, bankruptcy, force close) | Exchange specific |
| `taker_order_id` | string | ID of order that took liquidation | Exchange data |
| `count` | int | Number of liquidations in aggregation window | Aggregate count |

### Indexing

- Primary: (timestamp, symbol, exchange, side)
- Secondary: (exchange, symbol), timestamp range

### Frequency & Aggregation

- Raw: Tick-level (individual liquidations)
- Processed: 1-hour, 4-hour, 1-day aggregates

### Data Quality Notes

- CoinGlass: Specialized liquidation aggregator (highest quality)
- Tardis.dev: Tick-level data reconstructable
- CryptoDataDownload: Aggregated liquidations (5-minute intervals)
- Free sources: Limited history and exchange coverage

---

## 5. Long/Short Ratio Panel (longsshort_panel)

Aggregated long/short positioning from liquidation or settlement data.

### Schema

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `timestamp` | datetime (UTC) | Snapshot time | Data provider |
| `symbol` | string | Canonical symbol (BTC, ETH) | Normalized |
| `exchange` | string | Exchange name | Source identifier |
| `long_positions` | float | Number or volume of long positions | Source data |
| `short_positions` | float | Number or volume of short positions | Source data |
| `long_short_ratio` | float | Long / short ratio | Computed |
| `long_ratio_pct` | float | % of total that are longs | Long / (long+short) |
| `short_ratio_pct` | float | % of total that are shorts | Short / (long+short) |
| `dominant_side` | string | Dominant side (long, short, balanced) | Inferred |

### Indexing

- Primary: (timestamp, symbol, exchange)
- Secondary: (exchange, symbol), timestamp range

### Frequency

- 1-hour or daily aggregates
- Source-dependent (not always available)

### Data Quality Notes

- CoinGlass, Coinalyze: Free API access
- Not all exchanges publish positioning data
- May be inferred from liquidation counts

---

## 6. Processed Features Panel (features_panel)

Engineered features for modeling.

### Schema

**Price-based features:**
| Column | Type | Description |
|--------|------|-------------|
| `return_1h` | float | 1-hour simple return (%) |
| `return_5h` | float | 5-hour simple return (%) |
| `return_24h` | float | 24-hour simple return (%) |
| `log_return_1h` | float | 1-hour log return (%) |
| `forward_return_1h` | float | Next hour return (% target for modeling) |
| `forward_return_5h` | float | Next 5-hour return (% target) |
| `forward_return_24h` | float | Next 24-hour return (% target) |

**Funding-based features:**
| Column | Type | Description |
|--------|------|-------------|
| `funding_rate` | float | Current funding rate (decimal) |
| `funding_change_1h` | float | Change in funding rate over 1h |
| `funding_change_8h` | float | Change in funding rate over 8h |
| `funding_zscore` | float | Z-score of current funding (rolling 30-period) |
| `funding_extreme` | int | Binary flag: |zscore| > 2 |

**Open Interest features:**
| Column | Type | Description |
|--------|------|-------------|
| `open_interest` | float | Current open interest (USD) |
| `oi_change_1h` | float | 1-hour OI percentage change |
| `oi_change_8h` | float | 8-hour OI percentage change |
| `oi_zscore` | float | Z-score of current OI (rolling 30-period) |

### Indexing & Frequency

- Typically 1-hour or daily, depending on analysis
- Aligned to first available data point after feature completion

---

## 7. Model Datasets

### Training Sets

**Dataset:** `model_btc_funding_1h.csv`
- Timespan: 2020-01-01 to 2023-12-31 (train), 2024-01-01 onwards (test)
- Frequency: 1-hour bars
- Symbols: BTC
- Exchanges: Binance primary, Bybit/OKX validation
- Target: `forward_return_1h` (next 1-hour return)
- Features: Funding, OI, returns, z-scores

**Dataset:** `model_eth_funding_4h.csv`
- Timespan: 2021-01-01 to 2023-12-31 (train), 2024-01-01 onwards (test)
- Frequency: 4-hour bars
- Symbols: ETH
- Exchanges: Multi-exchange average
- Target: `forward_return_4h`

---

## Data Cleaning Standards

### Timestamp Handling

- **Input:** Varies by source (Unix milliseconds, ISO 8601, etc.)
- **Standard:** UTC timezone, ISO 8601 format, second precision
- **Checks:** No gaps > 1 period, monotonically increasing

### Symbol Normalization

- **Input:** BTCUSDT, BTC-USDT, BTC-PERPETUAL, etc.
- **Standard:** Canonical (BTC, ETH, SOL, etc.)
- **Mapping:** Exchange-specific to canonical via SymbolNormalizer

### Numeric Precision

- **Prices:** Minimum 2 decimal places (typically 0.01 USDT)
- **Rates:** 8 decimal places (e.g., 0.00012345)
- **Percentages:** 4 decimal places
- **Volumes:** 2 decimal places minimum

### Missing Data Handling

- **Forward fill:** For OI, funding rates (when appropriate)
- **Drop:** For returns or analysis targets
- **Impute:** For z-scores (use 0 for missing values)

### Duplicate Handling

- **Exact duplicates:** Keep first occurrence
- **Near-duplicates:** Check timestamp alignment

---

## Data Validation Checklist

Before each analysis phase:

- [ ] **Completeness:** No unexpected gaps in timestamp series
- [ ] **Consistency:** Cross-verify between sources where possible
- [ ] **Outliers:** Flag and document extreme values
- [ ] **Schema:** All expected columns present with correct types
- [ ] **Timezone:** All timestamps UTC
- [ ] **Symbols:** All symbols in canonical form
- [ ] **Chronology:** Timestamps monotonically increasing per (symbol, exchange, market)
- [ ] **Duplicates:** No exact duplicates on (timestamp, symbol, exchange)

---

## Storage & Format

### Raw Data

- **Format:** CSV (preferred), Parquet (for large files), JSON (API responses)
- **Location:** `/data/raw/{data_type}/{exchange}/{symbol}/`
- **Naming:** `{symbol}_{exchange}_{interval}_{start_date}_{end_date}.csv`

### Processed Data

- **Format:** Parquet (columnar, efficient)
- **Location:** `/data/processed/`
- **Naming:** `{data_type}_{symbol}_{frequency}.parquet`
- **Metadata:** Stored as Parquet schema + metadata dict

### Metadata

Each processed file includes metadata:

```json
{
  "symbol": "BTC",
  "data_type": "ohlcv",
  "frequency": "1h",
  "exchange": "binance",
  "start_date": "2020-01-01",
  "end_date": "2024-11-17",
  "rows": 43000,
  "null_counts": { "volume": 0, "trades": 120 },
  "generated_at": "2024-11-17T14:30:00Z",
  "pipeline_version": "1.0"
}
```

---

## Updates & Maintenance

This data dictionary should be updated:

- After each data acquisition run (new date ranges)
- When new features are added
- When data sources change or are deprecated
- Monthly (check for schema changes)

**Last Updated:** 2024-11-17
**Next Review:** 2024-12-17
