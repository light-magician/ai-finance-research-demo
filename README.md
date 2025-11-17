# Crypto Derivatives Microstructure Research

A comprehensive, production-grade research toolkit for discovering and validating statistically significant relationships between funding rates, open interest, liquidations, and short-horizon price moves in BTC and ETH perpetual futures.

## Research Objective

This project systematically investigates:

1. **Funding Rate Predictability:** Do funding rate levels/changes predict future returns on BTC/ETH perps?
2. **Open Interest Dynamics:** How do OI and long/short positioning interact with funding to predict returns?
3. **Liquidation Events:** Are extreme liquidation events predictive of short-term reversals or momentum?
4. **Exchange-Specific Effects:** Which microstructure relationships are universal vs. exchange-dependent?
5. **Transaction Cost Robustness:** Do identified effects survive realistic trading costs?

## Key Features

- **Multi-source data acquisition:** 14+ verified data providers (free, professional, and institutional tiers)
- **Robust data cleaning:** Timestamp normalization, symbol standardization, duplicate handling
- **Comprehensive feature engineering:** Returns, funding rates, open interest, liquidation aggregations
- **Hypothesis testing framework:** Time-series model evaluation with proper train/test splits
- **Robustness validation:** Out-of-sample testing, multiple-hypothesis corrections, transaction cost modeling
- **Production-grade codebase:** Type hints, logging, unit tests, modular architecture

## Quick Start

See [SETUP.md](SETUP.md) for detailed installation instructions.

### Basic Usage

```python
from crypto_perp_research.loaders import DataLoader
from crypto_perp_research.cleaners import DataCleaner
from crypto_perp_research.features import FeatureEngineer

# Load data
loader = DataLoader()
ohlcv_df = loader.binance.download_ohlcv("BTCUSDT", interval="1d")

# Clean data
cleaned = DataCleaner.clean_ohlcv(ohlcv_df)

# Engineer features
engineer = FeatureEngineer()
featured = engineer.create_research_features(cleaned, symbol="BTC", has_funding=True)
```

## Project Structure

```
├── data/                          # Raw and processed data
│   ├── raw/                       # Downloaded from various sources
│   └── processed/                 # Cleaned, normalized, engineered
├── src/crypto_perp_research/      # Core research library
│   ├── loaders.py                 # Data acquisition
│   ├── cleaners.py                # Data normalization
│   ├── features.py                # Feature engineering
│   └── utils.py                   # Utilities
├── analysis/                      # Research notebooks and scripts
├── tests/                         # Unit tests and validation
├── docs/                          # Comprehensive documentation
│   ├── research_plan_v1.md        # Research objectives & phases
│   ├── data_sources.md            # 14 verified data providers
│   ├── data_dictionary.md         # Schema (generated Phase 3)
│   ├── api_reference.md           # API docs (generated Phase 3)
│   └── validation_notes.md        # Robustness report (Phase 6)
└── reports/
    └── crypto_perp_funding_research.md  # Final report
```

## Data Sources

This project uses **14 verified, publicly available data sources** including:

**Free Tier:**
- Binance Vision (OHLCV, 2020+)
- CryptoDataDownload (funding, liquidations, 2017+)
- Coinalyze API (funding, OI, liquidations)
- GitHub open-source tools

**Professional Tier:**
- Tardis.dev (tick-level, 2019+, $300-500/mo)
- CoinAPI (370+ exchanges, $79-599/mo)
- CoinGlass (liquidation specialist, $79+/mo)

**Institutional Tier:**
- Amberdata (full derivatives suite)

See [docs/data_sources.md](docs/data_sources.md) for complete details on coverage, rate limits, and data quality.

## Research Phases

The project follows a structured 7-phase approach:

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Research plan & hypotheses | ✓ Complete |
| 1 | Data source documentation | ✓ Complete |
| 2 | Data acquisition scripts | ✓ In Progress |
| 3 | Data cleaning & feature engineering | Pending |
| 4 | Exploratory data analysis | Pending |
| 5 | Hypothesis testing & modeling | Pending |
| 6 | Robustness validation | Pending |
| 7 | Final research report | Pending |

See [docs/research_plan_v1.md](docs/research_plan_v1.md) for full details.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Dependencies

- **Core:** Python 3.8+, pandas, numpy
- **Analysis:** scipy, statsmodels, scikit-learn
- **Visualization:** matplotlib, seaborn
- **Data:** requests (for API calls)
- **Testing:** pytest

See [SETUP.md](SETUP.md) for installation.

## Research Hypotheses

1. **H1:** Positive funding rates predict negative next-period perp returns (mean reversion)
2. **H2:** High OI + negative funding predict liquidations and reversals
3. **H3:** Long/short imbalance extremes (|z-score| > 2) predict directional moves
4. **H4:** Exchange-specific idiosyncrasies exist in funding/OI relationships
5. **H5:** Spot-perp basis anomalies + high OI changes predict intraday vol spikes

## Key Design Principles

- **Scientific Rigor:** Proper train/test splits, time-series validation, multiple-hypothesis corrections
- **Reproducibility:** All analysis can be re-run from source code and data
- **Production-Ready:** Type hints, logging, unit tests, error handling
- **Modularity:** Reusable components for data acquisition, cleaning, feature engineering
- **Transparency:** All assumptions, data sources, and limitations documented

## Documentation

- [SETUP.md](SETUP.md) - Installation and quick start
- [docs/research_plan_v1.md](docs/research_plan_v1.md) - Research objectives, hypotheses, and timeline
- [docs/data_sources.md](docs/data_sources.md) - Complete catalog of 14 data providers with comparison table
- [src/crypto_perp_research/](src/crypto_perp_research/) - Inline code documentation

## Contributing

Contributions and improvements welcome! Please:

1. Create a new branch for your work
2. Add tests for new functionality
3. Update documentation
4. Submit a pull request

## License

See LICENSE file (if present).

## Contact & Support

For questions or issues:
- Check [SETUP.md](SETUP.md) for troubleshooting
- Review [docs/data_sources.md](docs/data_sources.md) for data provider support
- Open an issue in the repository
