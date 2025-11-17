"""
Feature engineering for crypto derivatives microstructure research.

This module provides functions to create research features:
- Returns and momentum
- Funding rate changes and levels
- Open interest changes
- Liquidation aggregations
- Statistical features (z-scores, rolling volatility)
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ReturnsFeatures:
    """
    Computes various return metrics.
    """

    @staticmethod
    def compute_returns(df: pd.DataFrame, price_col: str = "close", periods: List[int] = [1, 5, 24]) -> pd.DataFrame:
        """
        Compute simple returns over various periods.

        Args:
            df: DataFrame with price data
            price_col: Name of price column
            periods: List of periods (in rows) to compute returns for

        Returns:
            DataFrame with added return columns
        """
        df = df.copy()

        for period in periods:
            col_name = f"return_{period}h"
            df[col_name] = df[price_col].pct_change(periods=period) * 100  # In percentages

        return df

    @staticmethod
    def compute_log_returns(df: pd.DataFrame, price_col: str = "close", periods: List[int] = [1, 5, 24]) -> pd.DataFrame:
        """
        Compute log returns.

        Args:
            df: DataFrame with price data
            price_col: Name of price column
            periods: List of periods to compute log returns for

        Returns:
            DataFrame with added log return columns
        """
        df = df.copy()

        for period in periods:
            col_name = f"log_return_{period}h"
            df[col_name] = np.log(df[price_col] / df[price_col].shift(period)) * 100

        return df

    @staticmethod
    def compute_forward_returns(df: pd.DataFrame, price_col: str = "close", horizons: List[int] = [1, 5, 24]) -> pd.DataFrame:
        """
        Compute forward-looking returns (for model targets).

        Args:
            df: DataFrame with price data
            price_col: Name of price column
            horizons: List of forward periods

        Returns:
            DataFrame with forward return columns
        """
        df = df.copy()

        for horizon in horizons:
            col_name = f"forward_return_{horizon}h"
            df[col_name] = df[price_col].shift(-horizon).pct_change(periods=horizon) * 100

        return df


class FundingFeatures:
    """
    Computes features based on funding rates.
    """

    @staticmethod
    def compute_funding_changes(df: pd.DataFrame, funding_col: str = "funding_rate", periods: List[int] = [1, 8, 24]) -> pd.DataFrame:
        """
        Compute funding rate changes.

        Args:
            df: DataFrame with funding rates
            funding_col: Name of funding rate column
            periods: List of periods for computing changes

        Returns:
            DataFrame with funding change columns
        """
        df = df.copy()

        for period in periods:
            col_name = f"funding_change_{period}h"
            df[col_name] = df[funding_col].diff(periods=period)

        return df

    @staticmethod
    def compute_funding_z_scores(df: pd.DataFrame, funding_col: str = "funding_rate", window: int = 30) -> pd.DataFrame:
        """
        Compute z-scores of funding rates (standardized values).

        Args:
            df: DataFrame with funding rates
            funding_col: Name of funding rate column
            window: Rolling window for z-score calculation

        Returns:
            DataFrame with z-score column
        """
        df = df.copy()

        mean = df[funding_col].rolling(window=window).mean()
        std = df[funding_col].rolling(window=window).std()

        df["funding_zscore"] = (df[funding_col] - mean) / (std + 1e-8)

        return df

    @staticmethod
    def compute_funding_extremes(df: pd.DataFrame, funding_col: str = "funding_rate", zscore_threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify extreme funding rate events.

        Args:
            df: DataFrame with funding rates
            funding_col: Name of funding rate column
            zscore_threshold: Z-score threshold for extreme events

        Returns:
            DataFrame with extreme event flag
        """
        df = df.copy()

        df = FundingFeatures.compute_funding_z_scores(df, funding_col)
        df["funding_extreme"] = (df["funding_zscore"].abs() > zscore_threshold).astype(int)

        return df


class OpenInterestFeatures:
    """
    Computes features based on open interest.
    """

    @staticmethod
    def compute_oi_changes(df: pd.DataFrame, oi_col: str = "open_interest", periods: List[int] = [1, 8, 24]) -> pd.DataFrame:
        """
        Compute open interest changes.

        Args:
            df: DataFrame with open interest
            oi_col: Name of OI column
            periods: List of periods for computing changes

        Returns:
            DataFrame with OI change columns
        """
        df = df.copy()

        for period in periods:
            col_name = f"oi_change_{period}h"
            df[col_name] = df[oi_col].pct_change(periods=period) * 100

        return df

    @staticmethod
    def compute_oi_z_scores(df: pd.DataFrame, oi_col: str = "open_interest", window: int = 30) -> pd.DataFrame:
        """
        Compute z-scores of open interest.

        Args:
            df: DataFrame with open interest
            oi_col: Name of OI column
            window: Rolling window

        Returns:
            DataFrame with OI z-score
        """
        df = df.copy()

        mean = df[oi_col].rolling(window=window).mean()
        std = df[oi_col].rolling(window=window).std()

        df["oi_zscore"] = (df[oi_col] - mean) / (std + 1e-8)

        return df


class LiquidationFeatures:
    """
    Computes features based on liquidation events.
    """

    @staticmethod
    def aggregate_liquidations(
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        symbol_col: str = "symbol",
        volume_col: str = "volume",
        side_col: str = "side",
        window_periods: List[int] = [1, 4, 24]
    ) -> pd.DataFrame:
        """
        Aggregate liquidation volumes over time windows.

        Args:
            df: DataFrame with liquidation events
            timestamp_col: Timestamp column name
            symbol_col: Symbol column name
            volume_col: Volume column name
            side_col: Side column (buy/sell) name
            window_periods: Time windows (in hours)

        Returns:
            Aggregated liquidation DataFrame
        """
        df = df.copy()

        # Resample to hourly
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col).sort_index()

        aggregated = pd.DataFrame()

        for period in window_periods:
            # Aggregate by symbol and side
            for symbol in df[symbol_col].unique() if symbol_col in df.columns else [None]:
                subset = df if symbol is None else df[df[symbol_col] == symbol]

                # Buy liquidations
                buy_liq = subset[subset[side_col] == "BUY"][volume_col].rolling(f"{period}h").sum() if side_col in df.columns else subset[volume_col].rolling(f"{period}h").sum()
                # Sell liquidations
                sell_liq = subset[subset[side_col] == "SELL"][volume_col].rolling(f"{period}h").sum() if side_col in df.columns else subset[volume_col].rolling(f"{period}h").sum()

                aggregated[f"buy_liq_{period}h"] = buy_liq
                aggregated[f"sell_liq_{period}h"] = sell_liq

        return aggregated


class FeatureEngineer:
    """
    Main feature engineering orchestrator.
    """

    @staticmethod
    def create_research_features(
        df: pd.DataFrame,
        symbol: str,
        has_funding: bool = False,
        has_oi: bool = False,
        has_liquidations: bool = False,
        price_col: str = "close",
        funding_col: str = "funding_rate",
        oi_col: str = "open_interest",
    ) -> pd.DataFrame:
        """
        Create a comprehensive feature set for research.

        Args:
            df: Input DataFrame
            symbol: Symbol (for logging)
            has_funding: Whether funding rate data is available
            has_oi: Whether open interest data is available
            has_liquidations: Whether liquidation data is available
            price_col: Price column name
            funding_col: Funding rate column name
            oi_col: Open interest column name

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        logger.info(f"Engineering features for {symbol}")

        # Returns
        if price_col in df.columns:
            df = ReturnsFeatures.compute_returns(df, price_col, periods=[1, 5, 24])
            df = ReturnsFeatures.compute_forward_returns(df, price_col, horizons=[1, 5, 24])
            df = ReturnsFeatures.compute_log_returns(df, price_col, periods=[1, 5, 24])

        # Funding
        if has_funding and funding_col in df.columns:
            df = FundingFeatures.compute_funding_changes(df, funding_col)
            df = FundingFeatures.compute_funding_z_scores(df, funding_col)
            df = FundingFeatures.compute_funding_extremes(df, funding_col)

        # Open Interest
        if has_oi and oi_col in df.columns:
            df = OpenInterestFeatures.compute_oi_changes(df, oi_col)
            df = OpenInterestFeatures.compute_oi_z_scores(df, oi_col)

        # Remove rows with NaN in target (forward returns) for modeling
        logger.info(f"Rows after feature engineering: {len(df)}")

        return df

    @staticmethod
    def prepare_modeling_data(
        df: pd.DataFrame,
        target_col: str = "forward_return_1h",
        feature_cols: Optional[List[str]] = None,
        dropna: bool = True
    ) -> tuple:
        """
        Prepare data for modeling.

        Args:
            df: Feature-engineered DataFrame
            target_col: Name of target column
            feature_cols: List of feature columns (None = all except target)
            dropna: Whether to drop rows with NaN

        Returns:
            Tuple of (X, y) DataFrames
        """
        if dropna:
            df = df.dropna(subset=[target_col])

        y = df[target_col]

        if feature_cols is None:
            # Exclude target and timestamp columns
            exclude_cols = {target_col, "timestamp", "symbol", "exchange", "date", "open", "high", "low", "close", "volume"}
            feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]

        logger.info(f"Prepared modeling data: X shape {X.shape}, y shape {y.shape}")

        return X, y


if __name__ == "__main__":
    # Example: Test feature engineering
    import sys

    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    sample_df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["BTC"] * 100,
        "close": 40000 + np.cumsum(np.random.randn(100) * 100),
        "funding_rate": np.random.randn(100) * 0.001,
        "open_interest": 100000 + np.cumsum(np.random.randn(100) * 1000),
    })

    engineer = FeatureEngineer()
    features_df = engineer.create_research_features(
        sample_df,
        symbol="BTC",
        has_funding=True,
        has_oi=True,
    )

    print(f"Engineered {features_df.shape[1]} columns")
    print(features_df.head())

    # Prepare for modeling
    X, y = engineer.prepare_modeling_data(features_df, dropna=True)
    print(f"X: {X.shape}, y: {y.shape}")
