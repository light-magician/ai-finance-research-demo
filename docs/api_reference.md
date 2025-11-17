# API Reference: Crypto Derivatives Research Library

Complete API documentation for `crypto_perp_research` package.

---

## Table of Contents

1. [Loaders](#loaders) - Data acquisition
2. [Cleaners](#cleaners) - Data normalization
3. [Features](#features) - Feature engineering
4. [Utils](#utils) - Utilities

---

## Loaders

### `BinanceVisionLoader`

Access historical OHLCV data from Binance Vision (free, 2020+).

#### `download_ohlcv(symbol, interval, start_date, end_date, output_dir)`

Download OHLCV data for a symbol.

**Parameters:**
- `symbol` (str): Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
- `interval` (str): Kline interval ('1m', '5m', '1h', '1d', etc.)
- `start_date` (str, optional): Start date 'YYYY-MM-DD'
- `end_date` (str, optional): End date 'YYYY-MM-DD'
- `output_dir` (str): Output directory for CSV files

**Returns:** `pd.DataFrame` with columns: timestamp, open, high, low, close, volume, symbol

**Example:**
```python
from crypto_perp_research.loaders import BinanceVisionLoader

loader = BinanceVisionLoader()
df = loader.download_ohlcv(
    symbol="BTCUSDT",
    interval="1d",
    start_date="2023-01-01",
    end_date="2024-01-01"
)
```

---

### `CryptoDataDownloadLoader`

Access multi-exchange data from CryptoDataDownload (free, 2017+).

#### `download_funding_rates(symbol, output_dir)`

Download funding rates.

**Parameters:**
- `symbol` (str): Perpetual symbol (e.g., 'BTCUSDT', 'ETHUSDT')
- `output_dir` (str): Output directory

**Returns:** `pd.DataFrame` with columns: timestamp, symbol, funding_rate

**Example:**
```python
from crypto_perp_research.loaders import CryptoDataDownloadLoader

loader = CryptoDataDownloadLoader()
df = loader.download_funding_rates("BTCUSDT")
```

---

### `CoinalyzeAPILoader`

Access free API for funding, OI, liquidations (40 req/min limit).

#### `get_funding_rates(symbol, exchange, interval, limit, api_key)`

Fetch funding rate data.

**Parameters:**
- `symbol` (str): Crypto symbol ('BTC', 'ETH')
- `exchange` (str): Exchange name ('Binance', 'Bybit', 'OKX')
- `interval` (str): Interval ('1m', '5m', '1h', '1d')
- `limit` (int): Number of candles to fetch (max 500)
- `api_key` (str, optional): Coinalyze API key

**Returns:** `pd.DataFrame` with funding data

**Example:**
```python
from crypto_perp_research.loaders import CoinalyzeAPILoader

loader = CoinalyzeAPILoader()
df = loader.get_funding_rates(symbol="BTC", exchange="Binance", interval="1d", limit=100)
```

#### `get_open_interest(symbol, exchange, interval, limit, api_key)`

Fetch open interest data.

**Parameters:** Same as `get_funding_rates()`

**Returns:** `pd.DataFrame` with open interest data

---

### `DeribitAPILoader`

Access Deribit official API for options, futures, liquidations.

#### `get_last_trades(instrument, count, before)`

Fetch recent trades.

**Parameters:**
- `instrument` (str): Deribit instrument ('BTC-PERPETUAL', 'ETH-PERPETUAL')
- `count` (int): Number of trades (max 1000)
- `before` (int, optional): Trade ID for pagination

**Returns:** `pd.DataFrame` with trade data

**Example:**
```python
from crypto_perp_research.loaders import DeribitAPILoader

loader = DeribitAPILoader()
df = loader.get_last_trades("BTC-PERPETUAL", count=100)
```

---

### `DataLoader`

Unified orchestrator for all data sources.

#### `load_all_sources(symbols, exchanges, data_types)`

Load data from all available sources.

**Parameters:**
- `symbols` (list): Symbols to download (e.g., ['BTCUSDT', 'ETHUSDT'])
- `exchanges` (list): Exchanges (e.g., ['Binance'])
- `data_types` (list): Data types to load (['ohlcv', 'funding', 'open_interest'])

**Returns:** Dict mapping keys to DataFrames

**Example:**
```python
from crypto_perp_research.loaders import DataLoader

loader = DataLoader()
results = loader.load_all_sources(
    symbols=["BTCUSDT", "ETHUSDT"],
    exchanges=["Binance"],
    data_types=["ohlcv", "funding"]
)
```

---

## Cleaners

### `SymbolNormalizer`

Convert exchange-specific symbols to canonical form.

#### `normalize(symbol, exchange)`

Convert symbol to canonical form.

**Parameters:**
- `symbol` (str): Exchange-specific symbol
- `exchange` (str): Exchange name ('binance', 'okx', 'deribit')

**Returns:** Canonical symbol (str)

**Example:**
```python
from crypto_perp_research.cleaners import SymbolNormalizer

canonical = SymbolNormalizer.normalize("BTC-USDT", "okx")
# Returns: "BTC"

canonical = SymbolNormalizer.normalize("BTCUSDT", "binance")
# Returns: "BTC"
```

---

### `TimestampNormalizer`

Normalize timestamps to UTC and standardize precision.

#### `to_utc(df, timestamp_col)`

Convert timestamps to UTC timezone.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `timestamp_col` (str): Name of timestamp column

**Returns:** DataFrame with UTC timestamps

**Example:**
```python
from crypto_perp_research.cleaners import TimestampNormalizer

df_utc = TimestampNormalizer.to_utc(df, "timestamp")
```

#### `standardize_precision(df, timestamp_col, target_precision)`

Standardize timestamp precision.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `timestamp_col` (str): Name of timestamp column
- `target_precision` (str): 'second', 'millisecond', 'microsecond'

**Returns:** DataFrame with standardized precision

---

### `DataCleaner`

Main data cleaning orchestrator.

#### `clean_ohlcv(df)`

Clean and standardize OHLCV data.

**Parameters:**
- `df` (pd.DataFrame): Raw OHLCV DataFrame

**Returns:** Cleaned DataFrame

**Example:**
```python
from crypto_perp_research.cleaners import DataCleaner

cleaned = DataCleaner.clean_ohlcv(raw_ohlcv_df)
```

#### `clean_funding_rates(df)`

Clean funding rate data.

**Parameters:**
- `df` (pd.DataFrame): Raw funding DataFrame

**Returns:** Cleaned DataFrame

#### `clean_open_interest(df)`

Clean open interest data.

**Parameters:**
- `df` (pd.DataFrame): Raw OI DataFrame

**Returns:** Cleaned DataFrame

#### `clean_liquidations(df)`

Clean liquidation data.

**Parameters:**
- `df` (pd.DataFrame): Raw liquidations DataFrame

**Returns:** Cleaned DataFrame

#### `validate_data_quality(df, context)`

Validate data quality and return report.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `context` (str): Context name (e.g., 'OHLCV', 'funding')

**Returns:** Dict with validation metrics

**Example:**
```python
report = DataCleaner.validate_data_quality(df, context="OHLCV")
print(f"Total rows: {report['total_rows']}")
print(f"Null counts: {report['null_counts']}")
```

---

## Features

### `ReturnsFeatures`

Compute return-based features.

#### `compute_returns(df, price_col, periods)`

Compute simple returns over periods.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `price_col` (str): Price column name
- `periods` (list): List of periods (e.g., [1, 5, 24])

**Returns:** DataFrame with return columns

**Example:**
```python
from crypto_perp_research.features import ReturnsFeatures

df = ReturnsFeatures.compute_returns(df, "close", periods=[1, 5, 24])
# Adds: return_1h, return_5h, return_24h
```

#### `compute_log_returns(df, price_col, periods)`

Compute log returns.

#### `compute_forward_returns(df, price_col, horizons)`

Compute forward-looking returns (targets for modeling).

**Example:**
```python
df = ReturnsFeatures.compute_forward_returns(df, "close", horizons=[1, 5, 24])
# Adds: forward_return_1h, forward_return_5h, forward_return_24h
```

---

### `FundingFeatures`

Compute funding rate features.

#### `compute_funding_changes(df, funding_col, periods)`

Compute funding rate changes.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `funding_col` (str): Funding rate column
- `periods` (list): Periods (e.g., [1, 8, 24])

**Returns:** DataFrame with funding change columns

#### `compute_funding_z_scores(df, funding_col, window)`

Compute funding rate z-scores (standardized).

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `funding_col` (str): Funding column
- `window` (int): Rolling window (e.g., 30)

**Returns:** DataFrame with `funding_zscore` column

#### `compute_funding_extremes(df, funding_col, zscore_threshold)`

Identify extreme funding events.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `funding_col` (str): Funding column
- `zscore_threshold` (float): Threshold for extremes (e.g., 2.0)

**Returns:** DataFrame with `funding_extreme` binary column

---

### `OpenInterestFeatures`

Compute open interest features.

#### `compute_oi_changes(df, oi_col, periods)`

Compute OI percentage changes.

#### `compute_oi_z_scores(df, oi_col, window)`

Compute OI z-scores.

---

### `FeatureEngineer`

Main feature engineering orchestrator.

#### `create_research_features(df, symbol, has_funding, has_oi, has_liquidations, ...)`

Create comprehensive feature set.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `symbol` (str): Symbol name (for logging)
- `has_funding` (bool): Include funding features
- `has_oi` (bool): Include OI features
- `has_liquidations` (bool): Include liquidation features
- `price_col` (str): Price column name
- `funding_col` (str): Funding column name
- `oi_col` (str): OI column name

**Returns:** DataFrame with engineered features

**Example:**
```python
from crypto_perp_research.features import FeatureEngineer

engineer = FeatureEngineer()
featured = engineer.create_research_features(
    df,
    symbol="BTC",
    has_funding=True,
    has_oi=True,
    price_col="close",
    funding_col="funding_rate",
    oi_col="open_interest"
)
```

#### `prepare_modeling_data(df, target_col, feature_cols, dropna)`

Prepare data for modeling.

**Parameters:**
- `df` (pd.DataFrame): Feature-engineered DataFrame
- `target_col` (str): Target column name
- `feature_cols` (list, optional): Feature columns (None = auto-select)
- `dropna` (bool): Drop rows with NaN targets

**Returns:** Tuple (X, y) of feature and target DataFrames

**Example:**
```python
X, y = engineer.prepare_modeling_data(
    featured,
    target_col="forward_return_1h",
    dropna=True
)
```

---

## Utils

### `ConfigManager`

Manage research configuration.

#### `__init__(config_path)`

Initialize manager.

**Parameters:**
- `config_path` (str, optional): Path to JSON config file

#### `load_config(path)`

Load configuration from JSON.

#### `save_config(path)`

Save configuration to JSON.

#### `get(key, default)`

Get configuration value.

#### `set(key, value)`

Set configuration value.

**Example:**
```python
from crypto_perp_research.utils import ConfigManager

config = ConfigManager()
config.set("symbols", ["BTC", "ETH"])
config.save_config("config.json")
```

---

### `DataPersistence`

Save and load data in various formats.

#### `save_dataframe(df, path, format)`

Save DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to save
- `path` (str): Output file path
- `format` (str): 'csv', 'parquet', or 'feather'

**Example:**
```python
from crypto_perp_research.utils import DataPersistence

DataPersistence.save_dataframe(df, "data.parquet", format="parquet")
```

#### `load_dataframe(path, format)`

Load DataFrame.

**Parameters:**
- `path` (str): Input file path
- `format` (str): 'csv', 'parquet', or 'feather'

**Returns:** Loaded DataFrame

---

### `ExperimentLogger`

Log experimental results.

#### `log_experiment(experiment_name, results)`

Log results to JSON.

**Parameters:**
- `experiment_name` (str): Experiment name
- `results` (dict): Results dictionary

**Returns:** Path to log file

**Example:**
```python
from crypto_perp_research.utils import ExperimentLogger

logger = ExperimentLogger()
logger.log_experiment("funding_model_v1", {
    "accuracy": 0.52,
    "p_value": 0.03,
    "symbol": "BTC"
})
```

---

### `ensure_directories(base_path)`

Ensure all required directories exist.

**Parameters:**
- `base_path` (str): Base directory path

**Returns:** Dict mapping directory names to paths

**Example:**
```python
from crypto_perp_research.utils import ensure_directories

dirs = ensure_directories("./data")
# Creates: ./data/raw, ./data/processed, ./data/raw/ohlcv, etc.
```

---

## Common Workflows

### Workflow 1: Load and Clean OHLCV Data

```python
from crypto_perp_research.loaders import BinanceVisionLoader
from crypto_perp_research.cleaners import DataCleaner

# Load
loader = BinanceVisionLoader()
raw_df = loader.download_ohlcv("BTCUSDT", interval="1d")

# Clean
cleaned_df = DataCleaner.clean_ohlcv(raw_df)

# Validate
report = DataCleaner.validate_data_quality(cleaned_df, context="OHLCV")
print(f"Rows: {report['total_rows']}, Nulls: {report['null_counts']}")
```

### Workflow 2: Load, Clean, and Feature-Engineer

```python
from crypto_perp_research.loaders import DataLoader
from crypto_perp_research.cleaners import DataCleaner
from crypto_perp_research.features import FeatureEngineer

# Load
loader = DataLoader()
results = loader.load_all_sources(
    symbols=["BTCUSDT"],
    data_types=["ohlcv", "funding", "open_interest"]
)

# Merge
df = results["BTCUSDT_ohlcv"]
df = df.merge(results["BTCUSDT_funding"], on=["timestamp"], how="left")
df = df.merge(results["BTCUSDT_oi"], on=["timestamp"], how="left")

# Clean
df_clean = DataCleaner.clean_ohlcv(df)

# Engineer features
engineer = FeatureEngineer()
featured = engineer.create_research_features(
    df_clean,
    symbol="BTC",
    has_funding=True,
    has_oi=True
)

# Prepare for modeling
X, y = engineer.prepare_modeling_data(featured, target_col="forward_return_1h")
```

### Workflow 3: Multi-Exchange Analysis

```python
from crypto_perp_research.cleaners import SymbolNormalizer, DataCleaner
import pandas as pd

# Load from different exchanges
binance_df = pd.read_csv("binance_ohlcv.csv")
okx_df = pd.read_csv("okx_ohlcv.csv")

# Normalize symbols
binance_df["symbol"] = binance_df["symbol"].apply(
    lambda x: SymbolNormalizer.normalize(x, "binance")
)
okx_df["symbol"] = okx_df["symbol"].apply(
    lambda x: SymbolNormalizer.normalize(x, "okx")
)

# Combine
combined = pd.concat([binance_df, okx_df], ignore_index=True)

# Clean
cleaned = DataCleaner.clean_ohlcv(combined)
```

---

## Error Handling

All functions include error handling with logging. Common exceptions:

- `ImportError`: Missing pandas/numpy
- `requests.RequestException`: API call failure
- `ValueError`: Invalid parameters
- `FileNotFoundError`: Data file not found

Example:

```python
try:
    df = loader.download_ohlcv("BTCUSDT")
except requests.RequestException as e:
    logger.error(f"Failed to download data: {e}")
except ValueError as e:
    logger.error(f"Invalid parameters: {e}")
```

---

## Performance Notes

- **Loaders:** Can download months/years of data; may take minutes
- **Cleaners:** Fast (< 1s for typical datasets)
- **Features:** Fast vectorized pandas operations
- **Memory:** Parquet format recommended for large datasets

---

## Version

- **Library Version:** 0.1.0
- **Last Updated:** 2024-11-17
- **Python:** 3.8+
- **Dependencies:** pandas, numpy, requests

---

## Further Documentation

- [Data Dictionary](data_dictionary.md) - Schema of processed tables
- [Research Plan](research_plan_v1.md) - Research objectives
- [Data Sources](data_sources.md) - Complete data provider guide
