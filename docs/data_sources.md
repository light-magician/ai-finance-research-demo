# Crypto Derivatives Data Sources

## Overview
This document catalogs verified data sources for crypto derivatives microstructure research, including historical funding rates, open interest, liquidations, and OHLCV data for BTC/ETH perpetual futures.

**Last Updated:** 2025-11-17

---

## Summary Table

| Source | Data Types | Coverage | Granularity | Free/Paid | Rate Limits | Quality |
|--------|-----------|----------|-------------|-----------|-------------|---------|
| Binance Vision | OHLCV, Trades | 2020-present | 1s-1mo bars | Free | None (bulk) | High |
| Tardis.dev | Full L2/L3, Funding, OI, Liquidations | 2019-03-30+ | Tick-level | Paid (free samples) | 30M/day (Pro) | Excellent |
| CoinGlass | Liquidations, Funding, OI, Long/Short | 2021-present | 1s updates | Paid (30-300/min) | 30-300 req/min | High |
| CryptoDataDownload | OHLCV, Funding, Trades | 2017-present | 1m-1d bars | Free | None | Good |
| Coinalyze | Funding, OI, Liquidations, Long/Short | Limited (1500-2000 bars intraday) | 1m-1d | Free | 40 req/min | Good |
| Kaggle (ramkrijal) | Funding Rates | Unknown | 8h intervals | Free | N/A | Unknown |
| CoinAPI | OHLCV, Funding, OI, Metrics | 2021-present (4 years) | Per exchange | Paid ($79-599/mo) | 1K-100K/day | High |
| Amberdata | Full Derivatives Suite | Exchange-dependent | Tick to daily | Enterprise | Custom | Excellent |
| GitHub (fundingrate) | Funding Rates | Real-time + historical | 8h intervals | Free (open source) | API-dependent | Good |
| Deribit API | Options, Futures, Liquidations | Since inception | Tick-level | Free (API key) | Per endpoint | High |

---

## Detailed Source Profiles

### 1. Binance Vision (Official)

**URL:** https://data.binance.vision/
**GitHub:** https://github.com/binance/binance-public-data

#### Data Available
- **OHLCV (Klines):** All spot and futures pairs
- **Trades:** Aggregate trades and individual trades
- **Futures:** Both USD-M and COIN-M perpetuals
- **Note:** Does NOT include funding rates, open interest, or liquidations

#### Access Method
- Direct bulk download via HTTP (curl/wget)
- Python/Shell helper scripts in GitHub repo
- S3-style browser interface

#### Coverage
- **Start Date:** 2020 (varies by symbol)
- **Symbols:** All BTC/ETH pairs (BTCUSDT, ETHUSDT, etc.)
- **Exchanges:** Binance Spot, USD-M Futures, COIN-M Futures

#### Granularity
All kline intervals: 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1mo

#### File Format
- CSV files inside ZIP archives
- Monthly and daily archives
- Each file includes CHECKSUM for verification

#### Rate Limits
None (direct bulk download)

#### Licensing
Free, public data from Binance

#### Caveats
- Timestamp changed to microseconds for spot data from Jan 1, 2025 onwards
- Monthly archives released first Monday of each month
- Archives may be revised if inconsistencies discovered

#### Data Quality
**High** - Official exchange data, well-documented, verified with checksums

---

### 2. Tardis.dev

**URL:** https://tardis.dev/
**Documentation:** https://docs.tardis.dev/

#### Data Available
- **Full order book:** Tick-level L2/L3 snapshots + incremental updates
- **Trades & Liquidations:** All trade events
- **Funding Rates:** Historical and real-time
- **Open Interest:** Time series data
- **Options:** Full chains with Greeks
- **Derivatives:** Index prices, mark prices

#### Access Method
- HTTP API (REST)
- WebSocket API (real-time)
- Bulk CSV downloads (gzipped)
- Python and Node.js client libraries

#### Coverage
- **Start Date:** 2019-03-30 for most exchanges
- **Exchanges:** BitMEX, Deribit, Binance, Bybit, OKX, Kraken, Bitfinex, Coinbase, Huobi, Gate.io, FTX (until 2022-11-13), and 30+ others
- **Instruments:** 100,000+ spot, perpetual, futures, and options
- **Completeness:** >99.9% average historical data completeness

#### Granularity
Tick-level (millisecond timestamps), reconstructable to any timeframe

#### File Format
- NDJSON (newline-delimited JSON)
- Gzipped CSV files
- Exchange-native and normalized formats

#### Rate Limits
- **Free:** First day of each month available without API key
- **Professional:** 30M requests/day, 60 concurrent requests
- **Business:** Unlimited (fair usage)

#### Pricing
- Free tier: Limited (first day/month samples)
- Paid tiers: Professional and Business (pricing on website)

#### Licensing
Commercial use requires subscription

#### Caveats
- Minute-by-minute API responses (1 minute of data per call)
- FTX data ends 2022-11-13
- Requires API key for most historical data

#### Data Quality
**Excellent** - Industry standard for institutional research, high completeness, rigorous validation

---

### 3. CoinGlass API

**URL:** https://www.coinglass.com/
**Documentation:** https://docs.coinglass.com/

#### Data Available
- **Liquidations:** Historical liquidation events (pair and aggregated)
- **Liquidation Heatmaps:** 3 variations for pairs and coins
- **Liquidation Maps & Max Pain:** Visual models
- **Funding Rates:** Historical and real-time
- **Open Interest:** Per exchange and aggregated
- **Long/Short Ratios:** Position data
- **Order Data:** Last 7 days with 1-second updates

#### Access Method
- REST API (requires API key)
- WebSocket (for liquidation orders)

#### Coverage
- **Exchanges:** Binance, Bybit, OKX, Deribit, BitMEX, and others
- **Historical Depth:** Not explicitly stated (likely 2021+)
- **Symbols:** BTC, ETH, and major altcoins

#### Granularity
- Liquidations: 1-second updates
- Historical: Varies by endpoint

#### File Format
JSON via REST API

#### Rate Limits
- **Hobbyist:** 30 requests/min
- **Startup:** 80 requests/min
- **Standard:** 300 requests/min

#### Pricing
- Hobbyist: TBD (check website)
- Startup: ~$79+/mo (estimated)
- Standard: Higher tier

#### Licensing
Requires account registration and API key (some endpoints may be free)

#### Caveats
- Some endpoints require API key
- Liquidation orders only available for last 7 days
- Rate limits enforced per tier

#### Data Quality
**High** - Specialized liquidation data aggregator, real-time updates, comprehensive exchange coverage

---

### 4. CryptoDataDownload

**URL:** https://www.cryptodatadownload.com/data/

#### Data Available
- **OHLCV:** Spot and futures (multiple timeframes)
- **Funding Rates:** Daily rates in 8-hour intervals (UM and CM futures)
- **Trades:** Transaction-level histories
- **Liquidations:** BTCUSDT/ETHUSDT (UM futures) in 5-minute intervals

#### Access Method
- Direct CSV download (no registration for most data)
- Organized by exchange/symbol/timeframe
- Some granularities may require free account

#### Coverage
- **Start Date:** 2017+ (varies by exchange)
- **Exchanges:** 25+ including Binance, Deribit, Kraken, Coinbase, Bitfinex, OKEx, Gemini, Poloniex, HitBTC
- **Note:** Bybit not explicitly listed
- **Symbols:** BTC, ETH, major altcoins

#### Granularity
1-minute to daily bars, 5-minute liquidation summaries

#### File Format
CSV (standardized format)

#### Rate Limits
None (direct download)

#### Pricing
Free for non-commercial use

#### Licensing
Creative Commons Attribution-NonCommercial-ShareAlike 4.0
Commercial use requires paid upgrade

#### Caveats
- May require free account for some data
- Coverage varies by exchange
- Non-commercial license by default

#### Data Quality
**Good** - Community-trusted since 2017, standardized formats, but user should verify data integrity

---

### 5. Coinalyze API

**URL:** https://coinalyze.net/
**API Docs:** https://api.coinalyze.net/v1/doc/

#### Data Available
- **Funding Rates:** Current, predicted, and historical
- **Open Interest:** Historical time series
- **Liquidations:** Historical events
- **Long/Short Ratios:** Historical position data
- **OHLCV:** Historical candlestick data

#### Access Method
REST API (requires API key)

#### Coverage
- **Historical Retention:**
  - Intraday (1m-12h): 1500-2000 datapoints (rolling deletion)
  - Daily: Permanent retention
- **Exchanges:** Major derivatives exchanges
- **Symbols:** BTC, ETH, and altcoins

#### Granularity
1-minute to daily intervals

#### File Format
JSON (returns timestamp, open, high, low, close)

#### Rate Limits
40 API calls per minute per API key

#### Pricing
Free (with citation requirement)

#### Licensing
Free API with request to cite Coinalyze as data source

#### Caveats
- Limited historical depth for intraday data (1500-2000 bars)
- Old intraday data deleted daily
- Must cite Coinalyze when using data publicly

#### Data Quality
**Good** - Free and accessible, but limited historical retention for high-frequency analysis

---

### 6. Kaggle: Crypto Perpetuals Funding Rates (ramkrijal)

**URL:** https://www.kaggle.com/datasets/ramkrijal/perpetuals-funding-rates

#### Data Available
Funding rates for 12 cryptocurrency perpetual contracts

#### Coverage
- **Symbols:** BTC, ETH, LTC, BNB, BUSD, ETC, GALA, ICP, JASMY, MANA (USD/USDT denominations)
- **Date Range:** Not specified (dataset updated March 16, 2023)
- **Exchanges:** Not specified

#### Granularity
8-hour funding intervals (typical perpetual funding schedule)

#### File Format
12 separate Excel files (.xlsx), ~129.6 KB compressed

#### Access Method
Direct download from Kaggle (requires Kaggle account)

#### Rate Limits
None (static dataset)

#### Pricing
Free

#### Licensing
Kaggle dataset license (check specific dataset page)

#### Caveats
- Limited documentation on data sources
- No information on exchange coverage
- Last updated March 2023 (may be stale)
- Unknown data quality/validation

#### Data Quality
**Unknown** - User-contributed dataset without extensive documentation; requires validation

---

### 7. CoinAPI

**URL:** https://www.coinapi.io/
**Pricing:** https://www.coinapi.io/products/market-data-api/pricing

#### Data Available
- **OHLCV:** Historical candlestick data
- **Funding Rates:** Up to 4 years of history
- **Open Interest:** DERIVATIVES_OPEN_INTEREST metric
- **Liquidations:** Liquidation metrics and clusters
- **Full market data:** 370+ exchanges

#### Access Method
- REST API (Metrics API)
- Flat Files (bulk CSV/Parquet downloads)
- WebSocket, FIX protocols

#### Coverage
- **Start Date:** Early 2021 for most exchanges (4+ years)
- **Exchanges:** Binance, Bybit, OKX, Deribit, Bitfinex, Kraken, and 370+ total
- **Symbols:** BTC, ETH, all major cryptocurrencies

#### Granularity
Per exchange (typically 1-second to daily)

#### File Format
JSON, CSV, Parquet

#### Rate Limits
- **Free Credits:** $25 test credits (no expiration)
- **Startup ($79/mo):** 1,000 requests/day
- **Streamer ($249/mo):** 10,000 requests/day
- **Pro ($599/mo):** 100,000 requests/day
- **Autoscaling:** Dynamic infrastructure (no hard throttling)

#### Pricing
$79-$599/mo + Enterprise custom

#### Licensing
Commercial use allowed with paid plans

#### Caveats
- Free credits limited to testing
- Historical depth limited to ~4 years
- Per-protocol rate limits (REST, WebSocket, FIX separate)

#### Data Quality
**High** - Professional-grade infrastructure, 370+ exchanges, institutional clients

---

### 8. Amberdata

**URL:** https://www.amberdata.io/
**Derivatives Docs:** https://www.amberdata.io/ad-derivatives

#### Data Available
- **Complete derivatives suite:** Futures, perpetuals, options
- **Liquidations:** Historical and real-time via REST/WebSocket
- **Open Interest:** Latest and time-series data
- **Funding Rates:** Historical rates and term structures
- **Greeks:** Options analytics
- **Long/Short Ratios:** Position data
- **Order Books:** Snapshots and updates
- **Trades:** Full trade history

#### Access Method
- REST API
- WebSocket (real-time)
- AWS S3 (bulk historical downloads)
- CSV exports
- Python modules

#### Coverage
- **Exchanges:** Binance, Bybit, OKX, Deribit, BitMEX, and others
- **Historical Depth:** Varies by exchange
- **Data Normalization:** Consistently normalized across venues

#### Granularity
Tick-level to daily aggregates

#### File Format
JSON, CSV, Parquet (via S3)

#### Rate Limits
Custom (enterprise pricing)

#### Pricing
Enterprise/institutional pricing (contact sales)

#### Licensing
Commercial use with subscription

#### Caveats
- Enterprise-focused (no public free tier)
- Pricing not publicly listed
- Requires direct contact for access

#### Data Quality
**Excellent** - Institutional-grade, deep normalization, used by hedge funds and research institutions

---

### 9. GitHub: fundingrate (jironghuang)

**URL:** https://github.com/jironghuang/fundingrate
**PyPI:** https://pypi.org/project/fundingrate/

#### Data Available
Funding rates for perpetual derivatives

#### Coverage
- **Exchanges:** dYdX, Binance
- **Symbols:** BTC-USD, ETH-USD
- **Historical:** Queryable and storable

#### Access Method
Python library (pip install fundingrate)

#### Granularity
8-hour funding intervals

#### File Format
SQLite3 database storage

#### Rate Limits
Dependent on underlying exchange API limits

#### Pricing
Free (open source)

#### Licensing
Open source (check repository for specific license)

#### Caveats
- Limited exchange support (dYdX, Binance)
- Requires Python programming
- Depends on exchange API availability

#### Data Quality
**Good** - Open source, community-maintained, suitable for research prototyping

---

### 10. GitHub: historical-funding-rates-fetcher (supervik)

**URL:** https://github.com/supervik/historical-funding-rates-fetcher

#### Data Available
Historical funding rates from multiple exchanges

#### Coverage
- **Exchanges:** Binance, Bybit, dYdX, Gate, HTX, KuCoin, MEXC
- **Symbols:** Major perpetual contracts

#### Access Method
Python script (download and run)

#### Granularity
8-hour funding intervals

#### File Format
CSV exports

#### Rate Limits
Dependent on exchange APIs

#### Pricing
Free (open source)

#### Licensing
Open source

#### Caveats
- Requires manual execution
- Coverage depends on exchange API access
- May need updates for API changes

#### Data Quality
**Good** - Useful for multi-exchange funding rate collection

---

### 11. Deribit API (Official)

**URL:** https://docs.deribit.com/
**Data via Deribit Insights:** https://insights.deribit.com/

#### Data Available
- **Options:** Full option chains
- **Futures/Perpetuals:** All derivatives contracts
- **Trades:** Complete trade history since inception
- **Liquidations:** Available for perpetuals
- **Greeks:** Options analytics

#### Access Method
- REST API: `https://history.deribit.com/api/v2/public/get_last_trades_by_currency`
- WebSocket API (preferred)

#### Coverage
- **Symbols:** BTC, ETH, SOL, USDC derivatives
- **Historical:** All trades since platform launch
- **Free access:** Historical data via API

#### Granularity
Tick-level (millisecond timestamps)

#### File Format
JSON (API responses)

#### Rate Limits
Per endpoint (check API docs)

#### Pricing
Free (requires API authentication)

#### Licensing
Free for research (commercial terms TBD)

#### Caveats
- Primarily options-focused (limited perpetuals)
- WebSocket preferred over REST for performance
- Bulk CSV export requires third-party tools (Tardis, Amberdata)

#### Data Quality
**High** - Official exchange data, comprehensive options market coverage

---

### 12. Bybit Historical Data (Official)

**URL:** https://www.bybit.com/derivatives/en/history-data
**API Docs:** https://bybit-exchange.github.io/docs/

#### Data Available
- User account exports (trades, orders, P&L)
- Public market data via API

#### Access Method
- Account data export (authenticated)
- REST API for public market data
- Third-party bulk downloaders (GitHub: bybit-bulk-downloader)

#### Coverage
- **Asset Types:** Spot, USDT Perpetual, Inverse Perpetual, Inverse Futures
- **Symbols:** BTC, ETH, and altcoins

#### Granularity
Varies by data type

#### File Format
CSV (account exports), JSON (API)

#### Rate Limits
API rate limits apply (check docs)

#### Pricing
Free (requires Bybit account)

#### Licensing
Free for personal use

#### Caveats
- Official bulk historical data download unclear
- May require API scraping (risk of blocks)
- Third-party tools recommended for bulk downloads

#### Data Quality
**High** - Official exchange data, but bulk access less streamlined than Binance

---

### 13. OKX Historical Data (Official)

**URL:** https://www.okx.com/historical-data
**API Docs:** https://www.okx.com/docs-v5/en/

#### Data Available
- **OHLCV:** Historical price data
- **Open Interest:** Via API
- **Liquidations:** Private liquidation orders via API v5
- **Trades:** Public trade history

#### Access Method
- REST API v5
- Public data endpoints

#### Coverage
- **Symbols:** BTC, ETH, and all listed pairs
- **Historical:** Per API endpoint

#### Granularity
Varies by endpoint

#### File Format
JSON (API responses)

#### Rate Limits
Per endpoint (check API v5 docs)

#### Pricing
Free (API key required)

#### Licensing
Free for personal use

#### Caveats
- API v5 required for Unified Accounts
- Liquidation data limited to private orders
- Bulk CSV downloads not officially documented

#### Data Quality
**High** - Official exchange, comprehensive derivatives coverage

---

### 14. FTX Historical Archive (via Tardis)

**URL:** https://tardis.dev/ftx
**Documentation:** https://docs.tardis.dev/historical-data-details/ftx

#### Data Available
- **Complete FTX market data:** All instruments
- **Perpetuals, Futures, Spot**
- **Funding Rates**
- **Order Books, Trades**

#### Coverage
- **Date Range:** 2019-08-01 to 2022-11-13 (exchange shutdown)
- **Symbols:** BTC, ETH, and all FTX instruments

#### Access Method
- Tardis.dev API/CSV downloads

#### Granularity
Tick-level

#### File Format
NDJSON, CSV (via Tardis)

#### Pricing
Free for first day of each month; paid for full access

#### Licensing
Via Tardis subscription

#### Caveats
- Exchange defunct (historical data only)
- No new data after November 2022

#### Data Quality
**High** - Complete archive of defunct exchange, useful for historical research

---

## Recommended Sources by Use Case

### Best for OHLCV Data
1. **Binance Vision** - Free, comprehensive, bulk downloads
2. **Tardis.dev** - Tick-level precision, multi-exchange
3. **CryptoDataDownload** - Free, multi-exchange, standardized CSVs

### Best for Funding Rates
1. **Tardis.dev** - Complete historical data, multi-exchange
2. **CoinAPI** - 4 years of history, 370+ exchanges
3. **GitHub (fundingrate, historical-funding-rates-fetcher)** - Free, open source

### Best for Liquidation Data
1. **CoinGlass** - Specialized liquidation aggregator, 1-second updates
2. **Tardis.dev** - Tick-level liquidations, multi-exchange
3. **Amberdata** - Institutional-grade liquidation analytics

### Best for Open Interest
1. **CoinGlass** - Aggregated OI across exchanges
2. **CoinAPI** - Historical OI metrics
3. **Coinalyze** - Free API with OI history
4. **Amberdata** - Institutional-grade OI analytics

### Best for Multi-Year Research (2019+)
1. **Tardis.dev** - Since 2019-03-30, tick-level
2. **Binance Vision** - Since 2020, free bulk downloads
3. **CryptoDataDownload** - Since 2017, free

### Best Free Sources
1. **Binance Vision** - OHLCV and trades
2. **CryptoDataDownload** - OHLCV, funding, liquidations
3. **Coinalyze** - Funding, OI, liquidations (API)
4. **GitHub (fundingrate)** - Funding rates via Python

### Best for Tick-Level Data
1. **Tardis.dev** - Industry standard
2. **Amberdata** - Institutional-grade
3. **Deribit API** - Official exchange tick data

---

## Data Quality Assessment

### Tier 1 (Institutional/Research Grade)
- **Tardis.dev** - 99.9%+ completeness, rigorous validation
- **Amberdata** - Institutional clients, deep normalization
- **Binance Vision** - Official exchange data, verified

### Tier 2 (High Quality, Production-Ready)
- **CoinGlass** - Specialized liquidation data
- **CoinAPI** - Professional infrastructure, 370+ exchanges
- **Deribit API** - Official exchange data
- **OKX/Bybit APIs** - Official exchange data

### Tier 3 (Good Quality, Research-Suitable)
- **CryptoDataDownload** - Community-trusted, requires validation
- **Coinalyze** - Free API, limited retention
- **GitHub repositories** - Open source, variable maintenance

### Tier 4 (Unknown/Requires Validation)
- **Kaggle datasets** - User-contributed, limited documentation

---

## Authentication & Setup Requirements

### No Authentication Required
- Binance Vision (bulk downloads)
- CryptoDataDownload (most data)
- Tardis.dev (first day of month samples)

### Free Account/API Key Required
- CoinGlass (free tier uncertain)
- Coinalyze (free API key)
- Deribit API (free registration)
- Bybit API (free account)
- OKX API (free account)

### Paid Subscription Required
- Tardis.dev (full historical access)
- CoinAPI ($79-599/mo)
- CoinGlass (paid tiers)
- Amberdata (enterprise pricing)

---

## Known Gaps & Limitations

### Liquidation Data
- **Challenge:** Liquidations are the hardest data type to obtain comprehensively
- **Best Sources:** CoinGlass, Tardis.dev, Amberdata
- **Free Options:** CryptoDataDownload (limited pairs), Coinalyze (limited retention)

### Long/Short Positioning
- **Challenge:** Not all exchanges expose position data
- **Sources:** CoinGlass, Coinalyze, exchange-specific APIs

### Historical Depth (Pre-2019)
- **Challenge:** Limited tick-level data before 2019
- **Best Options:** Binance Vision (2020+), CryptoDataDownload (2017+)

### Options Data
- **Challenge:** Limited open-source options data
- **Best Sources:** Deribit API, Tardis.dev, Amberdata

---

## Recommended Data Stack

### For Academic Research (Free/Low Budget)
1. **OHLCV:** Binance Vision + CryptoDataDownload
2. **Funding Rates:** GitHub (fundingrate) + Coinalyze API
3. **Liquidations:** CryptoDataDownload (limited) + Coinalyze
4. **Open Interest:** Coinalyze API

### For Professional Research (Mid Budget)
1. **All Data Types:** Tardis.dev (Professional tier)
2. **Liquidations:** CoinGlass API (Startup tier)
3. **Supplementary:** CoinAPI (Startup tier)

### For Institutional Research (High Budget)
1. **Primary:** Amberdata (full derivatives suite)
2. **Backup:** Tardis.dev (Business tier)
3. **Specialized:** CoinGlass (liquidations), CoinAPI (multi-exchange)

---

## Data Integration Notes

### File Format Preferences
- **CSV:** Best for initial exploration, spreadsheets, R/Python pandas
- **Parquet:** Best for large datasets, columnar efficiency
- **JSON/NDJSON:** Best for API integration, nested data
- **SQLite:** Best for local querying, intermediate storage

### Time Zone Considerations
- Most exchanges use UTC timestamps
- Binance Vision: Microseconds from Jan 2025+ (spot)
- Tardis: Millisecond precision
- Verify timezone in documentation

### Data Validation
- Always verify checksums (Binance Vision)
- Cross-reference data across sources
- Check for gaps in time series
- Validate funding rate calculations (should match exchange UI)

---

## Maintenance & Updates

**This document should be updated when:**
- New exchanges launch with public data APIs
- Existing sources change pricing/rate limits
- New aggregators or data providers emerge
- Academic datasets are published

**Contribution Guide:**
- Verify all URLs and access methods before adding
- Test API rate limits and document findings
- Include data quality assessment based on actual usage
- Note any regional restrictions or compliance requirements

---

## Additional Resources

### Documentation & Tutorials
- Tardis.dev Docs: https://docs.tardis.dev/
- CoinGlass API Docs: https://docs.coinglass.com/
- Binance API Docs: https://developers.binance.com/
- CCXT Library: https://github.com/ccxt/ccxt (multi-exchange API wrapper)

### Data Analysis Tools
- CCXT: Multi-exchange API library (Python/JS/PHP)
- Pandas: DataFrame analysis (Python)
- QuantLib: Derivatives pricing (C++/Python)
- PyAlgoTrade: Backtesting framework

### Academic Papers on Data Sources
- Search IEEE Xplore, arXiv for "cryptocurrency market microstructure data"
- Check university repositories for shared datasets

---

## Contact & Support

For questions about specific data sources:
- **Tardis.dev:** support@tardis.dev
- **CoinGlass:** Check website for support
- **CoinAPI:** support@coinapi.io
- **Amberdata:** Institutional sales team

For this document:
- Open an issue in the project repository
- Contribute via pull request with verified sources

---

**Document Version:** 1.0
**Last Verified:** 2025-11-17
**Next Review:** 2025-12-17 (monthly review recommended)
