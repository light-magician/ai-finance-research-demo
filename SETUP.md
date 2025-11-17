# Setup Instructions for Crypto Derivatives Research

## Environment Requirements

- Python 3.8+
- pip or conda for package management

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/light-magician/ai-finance-research-demo.git
cd ai-finance-research-demo
```

### 2. Install Python Dependencies

```bash
pip install pandas numpy seaborn matplotlib statsmodels scikit-learn requests jupyter pytest
```

Or with conda:

```bash
conda install -c conda-forge pandas numpy seaborn matplotlib statsmodels scikit-learn requests jupyter pytest
```

### 3. Install the Research Package

```bash
pip install -e .
```

Or manually add `src/` to your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Project Structure

```
ai-finance-research-demo/
├── data/
│   ├── raw/              # Downloaded raw data files
│   │   ├── ohlcv/
│   │   ├── funding/
│   │   ├── liquidations/
│   │   └── oi/
│   └── processed/        # Cleaned and processed data
├── src/
│   └── crypto_perp_research/
│       ├── loaders.py    # Data acquisition from various sources
│       ├── cleaners.py   # Data cleaning and normalization
│       ├── features.py   # Feature engineering
│       └── utils.py      # Utility functions
├── analysis/             # Research notebooks and scripts
│   ├── figures/
│   └── results/
├── tests/
│   └── test_data_pipeline.py  # Unit tests
├── docs/
│   ├── research_plan_v1.md
│   ├── data_sources.md
│   ├── data_dictionary.md
│   ├── api_reference.md
│   └── validation_notes.md
└── reports/
    └── crypto_perp_funding_research.md  # Final report
```

## Quick Start

### 1. Download Data

```python
from crypto_perp_research.loaders import DataLoader

loader = DataLoader(output_base_dir="./data/raw")
loader.load_all_sources(
    symbols=["BTCUSDT", "ETHUSDT"],
    exchanges=["Binance"],
    data_types=["ohlcv", "funding"]
)
```

### 2. Clean Data

```python
from crypto_perp_research.cleaners import DataCleaner
import pandas as pd

ohlcv_df = pd.read_csv("./data/raw/ohlcv/BTCUSDT_1d.csv")
cleaned_df = DataCleaner.clean_ohlcv(ohlcv_df)
```

### 3. Engineer Features

```python
from crypto_perp_research.features import FeatureEngineer

engineer = FeatureEngineer()
featured_df = engineer.create_research_features(
    cleaned_df,
    symbol="BTC",
    has_funding=True,
    has_oi=True,
    price_col="close"
)
```

### 4. Run Tests

```bash
pytest tests/test_data_pipeline.py -v
```

## Running the Full Pipeline

See `/analysis/run_full_pipeline.py` for a complete end-to-end example.

## Data Sources Documentation

All data sources are documented in `/docs/data_sources.md` with:
- URLs and access methods
- Rate limits and pricing
- Coverage dates and symbols
- Data quality assessments
- Authentication requirements

## Key Documentation Files

- **research_plan_v1.md**: Overall research objectives, hypotheses, and phases
- **data_sources.md**: Comprehensive catalog of 14 verified data providers
- **data_dictionary.md**: Schema of all processed data tables (to be generated in Phase 3)
- **api_reference.md**: Full API documentation for the research toolkit (to be generated in Phase 3)
- **validation_notes.md**: Robustness checks and critical assessment (to be generated in Phase 6)

## Troubleshooting

### ImportError: No module named 'pandas'

Install missing dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```

### API Rate Limits

Most free data sources have rate limits:
- Coinalyze: 40 requests/min
- CoinGlass: 30-300 requests/min (tier-dependent)
- Exchange APIs: Varies by exchange

Adjust batch sizes and add delays between requests as needed.

### Data Missing or Incomplete

See `/docs/data_sources.md` for known limitations:
- Some sources only have recent history (2021+)
- Some require paid subscriptions for full history
- Liquidation data is particularly hard to find comprehensively

Recommended: Use multiple sources and cross-validate.

## Next Steps

1. Start with Phase 2 data acquisition in `/analysis`
2. Run Phase 3 data cleaning pipeline
3. Explore Phase 4 EDA notebooks
4. Run Phase 5 hypothesis tests and modeling
5. Review Phase 6 robustness checks
6. Read Phase 7 final report

See `docs/research_plan_v1.md` for detailed timeline and milestones.
