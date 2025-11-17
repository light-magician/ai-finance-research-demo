"""
Unit tests for the crypto derivatives research data pipeline.

These tests validate:
- Data loading and acquisition
- Data cleaning and normalization
- Feature engineering
- Data integrity and quality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from crypto_perp_research import cleaners, features, utils, loaders


class TestTimestampNormalizer(unittest.TestCase):
    """Test timestamp normalization."""

    def setUp(self):
        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1, 2, 3, 4, 5]
        })

    def test_to_utc_naive_timestamps(self):
        """Test converting naive timestamps to UTC."""
        result = cleaners.TimestampNormalizer.to_utc(self.df, "timestamp")
        self.assertIsNotNone(result["timestamp"].dt.tz)
        self.assertEqual(str(result["timestamp"].dt.tz), "UTC")

    def test_to_utc_already_utc(self):
        """Test handling already UTC timestamps."""
        df = self.df.copy()
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        result = cleaners.TimestampNormalizer.to_utc(df, "timestamp")
        self.assertEqual(str(result["timestamp"].dt.tz), "UTC")

    def test_standardize_precision(self):
        """Test standardizing timestamp precision."""
        df = self.df.copy()
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        result = cleaners.TimestampNormalizer.standardize_precision(df, "timestamp", "second")
        self.assertEqual(result["timestamp"].dtype, df["timestamp"].dtype)


class TestSymbolNormalizer(unittest.TestCase):
    """Test symbol normalization."""

    def test_normalize_binance_symbols(self):
        """Test normalizing Binance symbols."""
        self.assertEqual(cleaners.SymbolNormalizer.normalize("BTCUSDT", "binance"), "BTC")
        self.assertEqual(cleaners.SymbolNormalizer.normalize("ETHUSDT", "binance"), "ETH")

    def test_normalize_okx_symbols(self):
        """Test normalizing OKX symbols."""
        self.assertEqual(cleaners.SymbolNormalizer.normalize("BTC-USDT", "okx"), "BTC")
        self.assertEqual(cleaners.SymbolNormalizer.normalize("ETH-USDT", "okx"), "ETH")

    def test_normalize_deribit_symbols(self):
        """Test normalizing Deribit symbols."""
        self.assertEqual(cleaners.SymbolNormalizer.normalize("BTC-PERPETUAL", "deribit"), "BTC")


class TestDataCleaner(unittest.TestCase):
    """Test data cleaning functions."""

    def setUp(self):
        self.ohlcv_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="H"),
            "symbol": ["BTC"] * 10,
            "open": [40000 + i*100 for i in range(10)],
            "high": [40100 + i*100 for i in range(10)],
            "low": [39900 + i*100 for i in range(10)],
            "close": [40050 + i*100 for i in range(10)],
            "volume": [1000 + i*100 for i in range(10)],
        })

    def test_clean_ohlcv(self):
        """Test OHLCV cleaning."""
        result = cleaners.DataCleaner.clean_ohlcv(self.ohlcv_df)
        self.assertFalse(result.empty)
        self.assertGreater(len(result), 0)
        self.assertTrue(result["timestamp"].is_monotonic_increasing)

    def test_clean_ohlcv_removes_duplicates(self):
        """Test that duplicate removal works."""
        df = self.ohlcv_df.copy()
        df = pd.concat([df, df.iloc[0:2]], ignore_index=True)
        result = cleaners.DataCleaner.clean_ohlcv(df)
        self.assertLess(len(result), len(df))

    def test_validate_data_quality(self):
        """Test data quality validation."""
        report = cleaners.DataCleaner.validate_data_quality(self.ohlcv_df, context="test")
        self.assertIn("total_rows", report)
        self.assertIn("null_counts", report)
        self.assertEqual(report["total_rows"], len(self.ohlcv_df))


class TestReturnsFeatures(unittest.TestCase):
    """Test returns feature engineering."""

    def setUp(self):
        self.df = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105] * 10,
        })

    def test_compute_returns(self):
        """Test computing simple returns."""
        result = features.ReturnsFeatures.compute_returns(self.df, "close", periods=[1, 5])
        self.assertIn("return_1h", result.columns)
        self.assertIn("return_5h", result.columns)
        self.assertIsNotNone(result["return_1h"].iloc[1])

    def test_compute_log_returns(self):
        """Test computing log returns."""
        result = features.ReturnsFeatures.compute_log_returns(self.df, "close", periods=[1])
        self.assertIn("log_return_1h", result.columns)

    def test_compute_forward_returns(self):
        """Test computing forward returns."""
        result = features.ReturnsFeatures.compute_forward_returns(self.df, "close", horizons=[1, 5])
        self.assertIn("forward_return_1h", result.columns)
        self.assertIn("forward_return_5h", result.columns)


class TestFundingFeatures(unittest.TestCase):
    """Test funding rate features."""

    def setUp(self):
        self.df = pd.DataFrame({
            "funding_rate": np.random.randn(100) * 0.001,
        })

    def test_compute_funding_changes(self):
        """Test computing funding rate changes."""
        result = features.FundingFeatures.compute_funding_changes(self.df, "funding_rate", periods=[1, 8])
        self.assertIn("funding_change_1h", result.columns)
        self.assertIn("funding_change_8h", result.columns)

    def test_compute_funding_z_scores(self):
        """Test computing funding rate z-scores."""
        result = features.FundingFeatures.compute_funding_z_scores(self.df, "funding_rate", window=30)
        self.assertIn("funding_zscore", result.columns)
        self.assertLess(result["funding_zscore"].std(), 2.0)  # Should be roughly normal


class TestFeatureEngineer(unittest.TestCase):
    """Test the main feature engineering orchestrator."""

    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="H")
        self.df = pd.DataFrame({
            "timestamp": dates,
            "symbol": ["BTC"] * 100,
            "close": 40000 + np.cumsum(np.random.randn(100) * 100),
            "funding_rate": np.random.randn(100) * 0.001,
            "open_interest": 100000 + np.cumsum(np.random.randn(100) * 1000),
        })

    def test_create_research_features(self):
        """Test comprehensive feature engineering."""
        result = features.FeatureEngineer.create_research_features(
            self.df,
            symbol="BTC",
            has_funding=True,
            has_oi=True,
            price_col="close",
            funding_col="funding_rate",
            oi_col="open_interest",
        )

        # Check that new columns were added
        self.assertGreater(len(result.columns), len(self.df.columns))
        self.assertIn("return_1h", result.columns)
        self.assertIn("funding_change_1h", result.columns)
        self.assertIn("oi_change_1h", result.columns)

    def test_prepare_modeling_data(self):
        """Test preparing data for modeling."""
        # Create features first
        result = features.FeatureEngineer.create_research_features(
            self.df,
            symbol="BTC",
            has_funding=True,
            has_oi=False,
            price_col="close",
        )

        # Prepare for modeling
        X, y = features.FeatureEngineer.prepare_modeling_data(
            result,
            target_col="forward_return_1h",
            dropna=True
        )

        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertFalse(y.isnull().any())


class TestDataPersistence(unittest.TestCase):
    """Test data persistence functions."""

    def setUp(self):
        self.df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })
        self.temp_dir = "/tmp/test_crypto_research"
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_and_load_csv(self):
        """Test saving and loading CSV files."""
        path = os.path.join(self.temp_dir, "test.csv")
        utils.DataPersistence.save_dataframe(self.df, path, format="csv")
        loaded = utils.DataPersistence.load_dataframe(path, format="csv")
        pd.testing.assert_frame_equal(self.df, loaded)


class TestEnsureDirectories(unittest.TestCase):
    """Test directory creation utility."""

    def setUp(self):
        self.temp_dir = "/tmp/test_crypto_dirs"

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_ensure_directories_creates_all(self):
        """Test that all required directories are created."""
        dirs = utils.ensure_directories(self.temp_dir)
        for dir_path in dirs.values():
            self.assertTrue(os.path.exists(dir_path))


def run_basic_sanity_checks():
    """Run basic sanity checks without pytest."""

    print("\n" + "="*60)
    print("RUNNING SANITY CHECKS")
    print("="*60)

    # Test 1: Timestamp normalization
    print("\n[Test 1] Timestamp Normalization")
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5),
        "value": [1, 2, 3, 4, 5]
    })
    result = cleaners.TimestampNormalizer.to_utc(df, "timestamp")
    assert result["timestamp"].dt.tz is not None, "UTC timezone not applied"
    print("✓ Timestamp normalization works")

    # Test 2: Symbol normalization
    print("\n[Test 2] Symbol Normalization")
    assert cleaners.SymbolNormalizer.normalize("BTCUSDT", "binance") == "BTC"
    assert cleaners.SymbolNormalizer.normalize("BTC-USDT", "okx") == "BTC"
    print("✓ Symbol normalization works")

    # Test 3: OHLCV cleaning
    print("\n[Test 3] OHLCV Data Cleaning")
    ohlcv = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="H"),
        "symbol": ["BTC"] * 10,
        "open": np.arange(40000, 40010),
        "high": np.arange(40100, 40110),
        "low": np.arange(39900, 39910),
        "close": np.arange(40050, 40060),
        "volume": np.arange(1000, 1010),
    })
    cleaned = cleaners.DataCleaner.clean_ohlcv(ohlcv)
    assert len(cleaned) == len(ohlcv), "Data length changed unexpectedly"
    assert cleaned["timestamp"].is_monotonic_increasing, "Timestamp not sorted"
    print("✓ OHLCV cleaning works")

    # Test 4: Returns computation
    print("\n[Test 4] Returns Feature Engineering")
    df = pd.DataFrame({"close": np.arange(100, 150)})
    returns = features.ReturnsFeatures.compute_returns(df, "close", periods=[1, 5])
    assert "return_1h" in returns.columns, "Return columns not created"
    print("✓ Returns feature engineering works")

    # Test 5: Funding features
    print("\n[Test 5] Funding Rate Features")
    df = pd.DataFrame({"funding_rate": np.random.randn(100) * 0.001})
    funding_df = features.FundingFeatures.compute_funding_changes(df, "funding_rate", periods=[1, 8])
    assert "funding_change_1h" in funding_df.columns, "Funding features not created"
    print("✓ Funding features work")

    # Test 6: Data quality validation
    print("\n[Test 6] Data Quality Validation")
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10),
        "value": np.arange(10)
    })
    report = cleaners.DataCleaner.validate_data_quality(df)
    assert report["total_rows"] == 10, "Quality report incorrect"
    print("✓ Data quality validation works")

    print("\n" + "="*60)
    print("ALL SANITY CHECKS PASSED ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run sanity checks first (don't require pytest)
    try:
        run_basic_sanity_checks()
    except Exception as e:
        print(f"Sanity check failed: {e}")
        import traceback
        traceback.print_exc()

    # Then run unit tests with unittest
    unittest.main(argv=[''], verbosity=2, exit=False)
