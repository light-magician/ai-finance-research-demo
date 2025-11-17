#!/usr/bin/env python3
"""
Phase 3: Data Integration Pipeline

This script demonstrates end-to-end data loading, cleaning, and feature engineering.

It uses simulated data (in a real scenario, this would fetch from Binance Vision, etc.)
and shows how to:
1. Load data from multiple sources
2. Clean and normalize timestamps/symbols
3. Create canonical research tables
4. Engineer features for modeling
5. Validate data quality

Requirements:
    pip install pandas numpy

Usage:
    python analysis/phase3_data_integration.py
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np

# Import research modules
try:
    from crypto_perp_research.cleaners import DataCleaner, TimestampNormalizer, SymbolNormalizer
    from crypto_perp_research.features import FeatureEngineer
    from crypto_perp_research.utils import DataPersistence, ensure_directories
except ImportError as e:
    print(f"Error importing research modules: {e}")
    print("Please install: pip install pandas numpy")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_ohlcv_data(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    start_date: str = "2024-01-01",
    periods: int = 365,
    freq: str = "D"
) -> pd.DataFrame:
    """
    Generate sample OHLCV data for demonstration.

    In production, this would call DataLoader.binance.download_ohlcv()
    """
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start, periods=periods, freq=freq)

    # Simulate price movement: random walk
    base_price = 40000.0
    price_changes = np.cumsum(np.random.randn(len(dates)) * 500)
    close_prices = base_price + price_changes

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "exchange": exchange,
        "market": "spot" if exchange == "binance_spot" else "perp_usdt",
        "open": close_prices + np.random.randn(len(dates)) * 100,
        "high": close_prices + np.abs(np.random.randn(len(dates)) * 300),
        "low": close_prices - np.abs(np.random.randn(len(dates)) * 300),
        "close": close_prices,
        "volume": np.random.uniform(1000, 10000, len(dates)),
    })

    return df


def generate_sample_funding_data(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    start_date: str = "2024-01-01",
    periods: int = 365 * 3  # 3 fundings per day
) -> pd.DataFrame:
    """Generate sample funding rate data."""
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start, periods=periods, freq="8H")

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "exchange": exchange,
        "funding_rate": np.random.randn(len(dates)) * 0.0005,
        "mark_price": 40000 + np.cumsum(np.random.randn(len(dates)) * 100),
        "index_price": 40000 + np.cumsum(np.random.randn(len(dates)) * 100),
    })

    return df


def generate_sample_oi_data(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    start_date: str = "2024-01-01",
    periods: int = 365
) -> pd.DataFrame:
    """Generate sample open interest data."""
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start, periods=periods, freq="H")

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "exchange": exchange,
        "open_interest": 1000000 + np.cumsum(np.random.randn(len(dates)) * 10000),
        "long_position_count": 100000 + np.random.uniform(-10000, 10000, len(dates)),
        "short_position_count": 100000 + np.random.uniform(-10000, 10000, len(dates)),
    })

    return df


def phase3_data_integration_workflow():
    """Execute Phase 3 data integration workflow."""

    logger.info("="*70)
    logger.info("PHASE 3: DATA INTEGRATION & FEATURE ENGINEERING")
    logger.info("="*70)

    # Step 1: Ensure directories
    logger.info("\n[Step 1] Creating project directories...")
    dirs = ensure_directories("./data")
    logger.info(f"Created {len(dirs)} directories")

    # Step 2: Generate sample data (or load from sources)
    logger.info("\n[Step 2] Loading/generating data from sources...")

    # In production, use: loader = DataLoader()
    # ohlcv_btc = loader.binance.download_ohlcv("BTCUSDT", interval="1d")

    ohlcv_btc = generate_sample_ohlcv_data("BTCUSDT", "binance", periods=365)
    funding_btc = generate_sample_funding_data("BTCUSDT", "binance")
    oi_btc = generate_sample_oi_data("BTCUSDT", "binance")

    logger.info(f"Loaded OHLCV: {len(ohlcv_btc)} rows")
    logger.info(f"Loaded Funding: {len(funding_btc)} rows")
    logger.info(f"Loaded OI: {len(oi_btc)} rows")

    # Step 3: Clean data
    logger.info("\n[Step 3] Cleaning data...")

    ohlcv_clean = DataCleaner.clean_ohlcv(ohlcv_btc)
    funding_clean = DataCleaner.clean_funding_rates(funding_btc)
    oi_clean = DataCleaner.clean_open_interest(oi_btc)

    logger.info(f"Cleaned OHLCV: {len(ohlcv_clean)} rows (removed {len(ohlcv_btc) - len(ohlcv_clean)} duplicates)")
    logger.info(f"Cleaned Funding: {len(funding_clean)} rows")
    logger.info(f"Cleaned OI: {len(oi_clean)} rows")

    # Step 4: Merge datasets
    logger.info("\n[Step 4] Merging datasets...")

    # Merge funding rates to OHLCV (left join on nearest timestamp)
    merged = pd.merge_asof(
        ohlcv_clean,
        funding_clean[["timestamp", "funding_rate"]],
        on="timestamp",
        direction="backward"
    )

    # Merge OI
    merged = pd.merge_asof(
        merged,
        oi_clean[["timestamp", "open_interest", "long_position_count", "short_position_count"]],
        on="timestamp",
        direction="backward"
    )

    logger.info(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")

    # Step 5: Engineer features
    logger.info("\n[Step 5] Engineering features...")

    engineer = FeatureEngineer()
    featured = engineer.create_research_features(
        merged,
        symbol="BTC",
        has_funding=True,
        has_oi=True,
        price_col="close",
        funding_col="funding_rate",
        oi_col="open_interest",
    )

    logger.info(f"Engineered features: {len(featured.columns) - len(merged.columns)} new columns")

    # Step 6: Validate quality
    logger.info("\n[Step 6] Validating data quality...")

    ohlcv_report = DataCleaner.validate_data_quality(ohlcv_clean, context="OHLCV")
    funding_report = DataCleaner.validate_data_quality(funding_clean, context="Funding")
    oi_report = DataCleaner.validate_data_quality(oi_clean, context="Open Interest")
    features_report = DataCleaner.validate_data_quality(featured, context="Featured")

    # Step 7: Save processed data
    logger.info("\n[Step 7] Saving processed data...")

    # Save canonical tables
    DataPersistence.save_dataframe(
        ohlcv_clean,
        "./data/processed/ohlcv_btc_1d.parquet",
        format="parquet"
    )

    DataPersistence.save_dataframe(
        funding_clean,
        "./data/processed/funding_btc_8h.parquet",
        format="parquet"
    )

    DataPersistence.save_dataframe(
        oi_clean,
        "./data/processed/oi_btc_1h.parquet",
        format="parquet"
    )

    DataPersistence.save_dataframe(
        featured,
        "./data/processed/features_btc_1d.parquet",
        format="parquet"
    )

    # Also save as CSV for reference
    featured.to_csv("./data/processed/features_btc_1d.csv", index=False)
    logger.info("Saved: features_btc_1d.csv and .parquet")

    # Step 8: Generate integration report
    logger.info("\n[Step 8] Generating integration report...")

    integration_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "phase": "Phase 3 - Data Integration",
        "symbols": ["BTC"],
        "period": {
            "start": str(ohlcv_clean["timestamp"].min()),
            "end": str(ohlcv_clean["timestamp"].max()),
        },
        "data_quality": {
            "ohlcv": ohlcv_report,
            "funding": funding_report,
            "open_interest": oi_report,
            "features": features_report,
        },
        "feature_engineering": {
            "input_columns": len(merged.columns),
            "output_columns": len(featured.columns),
            "new_features": len(featured.columns) - len(merged.columns),
            "feature_list": [col for col in featured.columns if col not in merged.columns],
        },
        "data_preparation": {
            "target_column": "forward_return_1d",
            "target_non_null_count": int(featured["forward_return_1d"].notna().sum()),
            "target_non_null_pct": float(featured["forward_return_1d"].notna().sum() / len(featured) * 100),
        },
    }

    # Save report
    with open("./data/processed/integration_report.json", "w") as f:
        json.dump(integration_report, f, indent=2)

    logger.info("Saved integration report to data/processed/integration_report.json")

    # Step 9: Summary statistics
    logger.info("\n[Step 9] Summary Statistics")
    logger.info("="*70)

    logger.info(f"\nOHLCV Data:")
    logger.info(f"  Rows: {len(ohlcv_clean)}")
    logger.info(f"  Date Range: {ohlcv_clean['timestamp'].min()} to {ohlcv_clean['timestamp'].max()}")
    logger.info(f"  Price Range: ${ohlcv_clean['close'].min():.2f} - ${ohlcv_clean['close'].max():.2f}")
    logger.info(f"  Avg Volume: {ohlcv_clean['volume'].mean():.2f}")

    logger.info(f"\nFunding Rates:")
    logger.info(f"  Rows: {len(funding_clean)}")
    logger.info(f"  Mean: {funding_clean['funding_rate'].mean():.6f}")
    logger.info(f"  Std Dev: {funding_clean['funding_rate'].std():.6f}")
    logger.info(f"  Min: {funding_clean['funding_rate'].min():.6f}")
    logger.info(f"  Max: {funding_clean['funding_rate'].max():.6f}")

    logger.info(f"\nOpen Interest:")
    logger.info(f"  Rows: {len(oi_clean)}")
    logger.info(f"  Mean: ${oi_clean['open_interest'].mean():.0f}")
    logger.info(f"  Min: ${oi_clean['open_interest'].min():.0f}")
    logger.info(f"  Max: ${oi_clean['open_interest'].max():.0f}")

    logger.info(f"\nFeatured Dataset:")
    logger.info(f"  Rows: {len(featured)}")
    logger.info(f"  Columns: {len(featured.columns)}")
    logger.info(f"  New features: {len([c for c in featured.columns if c not in ['timestamp', 'symbol', 'exchange', 'market', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']])}")
    logger.info(f"  Target nulls: {featured['forward_return_1d'].isna().sum()} ({featured['forward_return_1d'].isna().sum() / len(featured) * 100:.1f}%)")
    logger.info(f"  Target mean: {featured['forward_return_1d'].mean():.4f}%")
    logger.info(f"  Target std: {featured['forward_return_1d'].std():.4f}%")

    logger.info("\n" + "="*70)
    logger.info("PHASE 3 COMPLETE: All data cleaned, integrated, and engineered.")
    logger.info("="*70)

    return {
        "ohlcv": ohlcv_clean,
        "funding": funding_clean,
        "oi": oi_clean,
        "featured": featured,
        "report": integration_report,
    }


if __name__ == "__main__":
    logger.info("Starting Phase 3 Data Integration Pipeline")
    logger.info(f"Working directory: {os.getcwd()}")

    try:
        results = phase3_data_integration_workflow()
        logger.info("\n✓ Phase 3 workflow completed successfully")

        # Print next steps
        logger.info("\n" + "="*70)
        logger.info("NEXT STEPS:")
        logger.info("="*70)
        logger.info("1. Review integration report: data/processed/integration_report.json")
        logger.info("2. Check featured data: data/processed/features_btc_1d.csv")
        logger.info("3. Proceed to Phase 4: Exploratory Data Analysis")
        logger.info("   → analysis/phase4_eda.py")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Phase 3 workflow failed: {e}", exc_info=True)
        sys.exit(1)
