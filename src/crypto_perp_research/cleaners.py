"""
Data cleaning and normalization functions for crypto derivatives research.

This module provides utilities to:
- Normalize timestamps to UTC
- Standardize symbol naming across exchanges
- Handle missing data and duplicates
- Align data across multiple exchanges
- Validate data integrity
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class SymbolNormalizer:
    """
    Standardizes symbol names across different exchanges.

    Binance: BTCUSDT, ETHUSDT
    Bybit: BTCUSDT, ETHUSDT
    OKX: BTC-USDT, ETH-USDT
    Deribit: BTC-PERPETUAL, ETH-PERPETUAL
    """

    CANONICAL_SYMBOLS = {
        "BTC": "BTC",
        "ETH": "ETH",
        "SOL": "SOL",
    }

    # Mapping from exchange-specific to canonical
    EXCHANGE_MAPPINGS = {
        "binance": {
            "BTCUSDT": "BTC",
            "ETHUSDT": "ETH",
            "SOLUSDT": "SOL",
        },
        "bybit": {
            "BTCUSDT": "BTC",
            "ETHUSDT": "ETH",
            "SOLUSDT": "SOL",
        },
        "okx": {
            "BTC-USDT": "BTC",
            "ETH-USDT": "ETH",
            "SOL-USDT": "SOL",
        },
        "deribit": {
            "BTC-PERPETUAL": "BTC",
            "ETH-PERPETUAL": "ETH",
            "BTC_USDT": "BTC",
        },
    }

    @staticmethod
    def normalize(symbol: str, exchange: str = "binance") -> str:
        """
        Convert exchange-specific symbol to canonical form.

        Args:
            symbol: Original symbol (e.g., 'BTC-USDT', 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance', 'okx', 'deribit')

        Returns:
            Canonical symbol (e.g., 'BTC')
        """
        exchange = exchange.lower()

        if exchange in SymbolNormalizer.EXCHANGE_MAPPINGS:
            mapping = SymbolNormalizer.EXCHANGE_MAPPINGS[exchange]
            return mapping.get(symbol, symbol)

        # Fallback: extract base symbol
        return symbol.replace("-", "").replace("_", "").rstrip("USDT").rstrip("PERP").rstrip("USD")


class TimestampNormalizer:
    """
    Normalizes timestamps to UTC and standardizes precision.
    """

    @staticmethod
    def to_utc(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Convert timestamps to UTC timezone.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with UTC timestamps
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found")
            return df

        df = df.copy()

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            except Exception as e:
                logger.error(f"Failed to convert {timestamp_col} to datetime: {e}")
                return df

        # Localize to UTC if naive
        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
        else:
            # Convert to UTC if already timezone-aware
            df[timestamp_col] = df[timestamp_col].dt.tz_convert("UTC")

        return df

    @staticmethod
    def standardize_precision(
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        target_precision: str = "second"
    ) -> pd.DataFrame:
        """
        Standardize timestamp precision across sources.

        Args:
            df: DataFrame with timestamps
            timestamp_col: Name of timestamp column
            target_precision: Target precision ('second', 'millisecond', 'microsecond')

        Returns:
            DataFrame with standardized precision
        """
        df = df.copy()

        if timestamp_col not in df.columns:
            return df

        if target_precision == "second":
            df[timestamp_col] = df[timestamp_col].dt.floor("S")
        elif target_precision == "millisecond":
            df[timestamp_col] = df[timestamp_col].dt.floor("ms")
        elif target_precision == "microsecond":
            df[timestamp_col] = df[timestamp_col].dt.floor("us")

        return df


class DataCleaner:
    """
    Main data cleaner orchestrating all cleaning operations.
    """

    @staticmethod
    def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize OHLCV data.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Normalize timestamp
        if "timestamp" in df.columns:
            df = TimestampNormalizer.to_utc(df, "timestamp")
            df = TimestampNormalizer.standardize_precision(df, "timestamp", "second")
        else:
            logger.warning("No timestamp column found")

        # Remove duplicates (keep first occurrence)
        df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="first")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Convert price/volume columns to numeric
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        volume_cols = ["volume", "quote_asset_volume"]
        for col in volume_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for gaps
        if len(df) > 1 and "timestamp" in df.columns:
            gaps = df["timestamp"].diff()
            max_gap = gaps.max()
            logger.info(f"Max timestamp gap: {max_gap}")

            if pd.notna(max_gap) and max_gap > pd.Timedelta(days=1):
                logger.warning(f"Large gap detected: {max_gap}")

        return df

    @staticmethod
    def clean_funding_rates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize funding rate data.

        Args:
            df: Raw funding rate DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Normalize timestamp
        if "timestamp" in df.columns:
            df = TimestampNormalizer.to_utc(df, "timestamp")
        else:
            logger.warning("No timestamp column found")

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp", "symbol", "exchange"], keep="first")

        # Convert funding rate to numeric
        funding_cols = ["funding_rate", "funding", "rate"]
        for col in funding_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.rename(columns={col: "funding_rate"})
                break

        # Sort
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    @staticmethod
    def clean_open_interest(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize open interest data.

        Args:
            df: Raw open interest DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Normalize timestamp
        if "timestamp" in df.columns:
            df = TimestampNormalizer.to_utc(df, "timestamp")
        else:
            logger.warning("No timestamp column found")

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp", "symbol", "exchange"], keep="first")

        # Convert OI to numeric
        oi_cols = ["open_interest", "oi", "openInterest"]
        for col in oi_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.rename(columns={col: "open_interest"})
                break

        # Sort
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    @staticmethod
    def clean_liquidations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize liquidation data.

        Args:
            df: Raw liquidation DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Normalize timestamp
        if "timestamp" in df.columns:
            df = TimestampNormalizer.to_utc(df, "timestamp")
        else:
            logger.warning("No timestamp in liquidations")

        # Remove duplicates
        if "exchange" in df.columns:
            df = df.drop_duplicates(subset=["timestamp", "symbol", "exchange"], keep="first")
        else:
            df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="first")

        # Convert liquidation volume/price to numeric
        for col in ["volume", "price", "liq_price", "liquidation_price"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    @staticmethod
    def validate_data_quality(df: pd.DataFrame, context: str = "data") -> Dict[str, any]:
        """
        Validate data quality and return a report.

        Args:
            df: DataFrame to validate
            context: Data context (e.g., 'OHLCV', 'funding')

        Returns:
            Dictionary with validation metrics
        """
        report = {
            "context": context,
            "total_rows": len(df),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_count": df.duplicated().sum(),
            "date_range": None,
            "columns": df.columns.tolist(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }

        if "timestamp" in df.columns and len(df) > 0:
            report["date_range"] = {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
            }

        logger.info(f"Validation report for {context}:")
        logger.info(f"  Total rows: {report['total_rows']}")
        logger.info(f"  Null counts: {report['null_counts']}")
        logger.info(f"  Duplicates: {report['duplicate_count']}")
        logger.info(f"  Memory: {report['memory_usage']:.2f} MB")

        return report


if __name__ == "__main__":
    # Example: Test cleaners
    import sys

    logging.basicConfig(level=logging.INFO)

    # Create sample data
    sample_ohlcv = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
        "symbol": ["BTC"] * 5,
        "open": [40000, 40100, 40200, 40300, 40400],
        "high": [40500, 40600, 40700, 40800, 40900],
        "low": [39900, 40000, 40100, 40200, 40300],
        "close": [40200, 40300, 40400, 40500, 40600],
        "volume": [1000, 1100, 1200, 1300, 1400],
    })

    cleaned = DataCleaner.clean_ohlcv(sample_ohlcv)
    report = DataCleaner.validate_data_quality(cleaned, context="Sample OHLCV")
    print(report)
