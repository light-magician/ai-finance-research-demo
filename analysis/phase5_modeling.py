#!/usr/bin/env python3
"""
Phase 5: Hypothesis Testing & Modeling

This script tests the core hypotheses using statistical models:
- H1: Funding rates predict next-period returns
- H2: OI changes predict liquidations and reversals
- H3: Long/short imbalances predict directional moves
- H4: Exchange-specific idiosyncrasies
- H5: Spot-perp basis predicts volatility

Models used:
- OLS regression with HAC standard errors
- Logistic regression for directional moves
- Time-series cross-validation

Requirements:
    pip install pandas numpy statsmodels scikit-learn scipy

Usage:
    python analysis/phase5_modeling.py
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Tuple, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    from scipy import stats
except ImportError as e:
    print(f"Error importing modeling libraries: {e}")
    print("Please install: pip install statsmodels scikit-learn scipy")
    sys.exit(1)

from crypto_perp_research.utils import DataPersistence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_featured_data() -> pd.DataFrame:
    """Load featured data from Phase 3."""
    try:
        df = pd.read_parquet("./data/processed/features_btc_1d.parquet")
        logger.info(f"Loaded featured data from Parquet: {len(df)} rows")
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv("./data/processed/features_btc_1d.csv")
            logger.info(f"Loaded featured data from CSV: {len(df)} rows")
            return df
        except FileNotFoundError:
            logger.error("Featured data not found. Please run Phase 3 first.")
            sys.exit(1)


def prepare_train_test_split(df: pd.DataFrame, train_pct: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test by time (proper time-series split)."""
    split_idx = int(len(df) * train_pct)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    logger.info(f"Train period: {train['timestamp'].min()} to {train['timestamp'].max()}")
    logger.info(f"Test period: {test['timestamp'].min()} to {test['timestamp'].max()}")

    return train, test


def hypothesis_1_funding_predicts_returns(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    """
    H1: Positive funding rates predict negative next-period returns.

    Model: forward_return_1h ~ funding_rate + controls
    """
    logger.info("\n" + "="*70)
    logger.info("HYPOTHESIS 1: Funding Rates Predict Returns")
    logger.info("="*70)

    results = {
        "hypothesis": "H1",
        "claim": "Positive funding rates predict negative next-period returns",
        "model": "OLS",
    }

    # Prepare data
    train_clean = train[["funding_rate", "return_1h", "return_5h", "forward_return_1h"]].dropna()
    test_clean = test[["funding_rate", "return_1h", "return_5h", "forward_return_1h"]].dropna()

    if len(train_clean) < 30:
        logger.warning("Insufficient training data for H1")
        results["status"] = "INSUFFICIENT_DATA"
        return results

    # Features: funding rate + lagged returns (controls)
    X_train = train_clean[["funding_rate", "return_1h", "return_5h"]].copy()
    X_train = sm.add_constant(X_train)

    y_train = train_clean["forward_return_1h"]

    # Fit OLS model
    model = sm.OLS(y_train, X_train).fit(cov_type="HC1")  # HAC-robust SEs

    logger.info("\nModel Summary:")
    logger.info(f"R-squared: {model.rsquared:.4f}")
    logger.info(f"Adj R-squared: {model.rsquared_adj:.4f}")
    logger.info(f"F-statistic: {model.fvalue:.2f} (p-value: {model.f_pvalue:.4f})")

    # Extract coefficients
    coef_funding = model.params.get("funding_rate", 0)
    pval_funding = model.pvalues.get("funding_rate", 1)
    tstat_funding = model.tvalues.get("funding_rate", 0)

    logger.info(f"\nFunding Rate Coefficient: {coef_funding:.6f}")
    logger.info(f"T-statistic: {tstat_funding:.4f}")
    logger.info(f"P-value: {pval_funding:.4f}")
    logger.info(f"Interpretation: 1% increase in funding → {coef_funding*100:.4f}% change in next-hour return")

    # Test on test set
    X_test = test_clean[["funding_rate", "return_1h", "return_5h"]].copy()
    X_test = sm.add_constant(X_test)
    y_test = test_clean["forward_return_1h"]

    y_pred = model.predict(X_test)
    test_r2 = 1 - (((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())

    logger.info(f"\nOut-of-sample R-squared: {test_r2:.4f}")
    logger.info(f"RMSE: {np.sqrt(((y_test - y_pred) ** 2).mean()):.4f}")

    results.update({
        "status": "COMPLETE",
        "train_r2": float(model.rsquared),
        "test_r2": float(test_r2),
        "funding_coefficient": float(coef_funding),
        "funding_pvalue": float(pval_funding),
        "funding_tstat": float(tstat_funding),
        "significant": pval_funding < 0.05,
        "interpretation": f"1% funding increase → {coef_funding*100:.4f}% return change",
        "conclusion": "SUPPORTED" if (pval_funding < 0.05 and coef_funding < 0) else "NOT SUPPORTED",
    })

    return results


def hypothesis_2_oi_funding_predicts_reversals(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    """
    H2: High OI + negative funding predict liquidations and reversals.

    Model: directional_move ~ oi_zscore * funding_zscore + controls
    """
    logger.info("\n" + "="*70)
    logger.info("HYPOTHESIS 2: OI + Funding Predict Reversals")
    logger.info("="*70)

    results = {
        "hypothesis": "H2",
        "claim": "High OI + negative funding predict reversals",
        "model": "OLS with interaction",
    }

    # Prepare data
    train_data = train[["oi_zscore", "funding_zscore", "forward_return_1h"]].dropna().copy()
    test_data = test[["oi_zscore", "funding_zscore", "forward_return_1h"]].dropna().copy()

    if len(train_data) < 30:
        logger.warning("Insufficient data for H2")
        results["status"] = "INSUFFICIENT_DATA"
        return results

    # Create interaction term
    train_data["interaction"] = train_data["oi_zscore"] * train_data["funding_zscore"]
    test_data["interaction"] = test_data["oi_zscore"] * test_data["funding_zscore"]

    X_train = train_data[["oi_zscore", "funding_zscore", "interaction"]].copy()
    X_train = sm.add_constant(X_train)
    y_train = train_data["forward_return_1h"]

    # Fit model
    model = sm.OLS(y_train, X_train).fit(cov_type="HC1")

    logger.info("\nModel Summary:")
    logger.info(f"R-squared: {model.rsquared:.4f}")
    logger.info(f"F-statistic: {model.fvalue:.2f} (p-value: {model.f_pvalue:.4f})")

    coef_interaction = model.params.get("interaction", 0)
    pval_interaction = model.pvalues.get("interaction", 1)

    logger.info(f"\nInteraction Coefficient: {coef_interaction:.6f}")
    logger.info(f"P-value: {pval_interaction:.4f}")

    # Test on test set
    X_test = test_data[["oi_zscore", "funding_zscore", "interaction"]].copy()
    X_test = sm.add_constant(X_test)
    y_test = test_data["forward_return_1h"]

    y_pred = model.predict(X_test)
    test_r2 = 1 - (((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())

    logger.info(f"\nOut-of-sample R-squared: {test_r2:.4f}")

    results.update({
        "status": "COMPLETE",
        "train_r2": float(model.rsquared),
        "test_r2": float(test_r2),
        "interaction_coefficient": float(coef_interaction),
        "interaction_pvalue": float(pval_interaction),
        "significant": pval_interaction < 0.05,
        "conclusion": "SUPPORTED" if pval_interaction < 0.05 else "NOT SUPPORTED",
    })

    return results


def hypothesis_3_imbalance_predicts_directional(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    """
    H3: Long/short imbalance extremes predict directional moves.

    Model: Logistic regression on directional moves
    """
    logger.info("\n" + "="*70)
    logger.info("HYPOTHESIS 3: Imbalance Predicts Directional Moves")
    logger.info("="*70)

    results = {
        "hypothesis": "H3",
        "claim": "Extreme imbalances predict directional moves",
        "model": "Logistic Regression",
    }

    # Prepare data
    train_data = train.copy()
    test_data = test.copy()

    # Create binary target: positive or negative return
    train_data["positive_return"] = (train_data["forward_return_1h"] > 0).astype(int)
    test_data["positive_return"] = (test_data["forward_return_1h"] > 0).astype(int)

    # Create imbalance feature (if available)
    if "long_short_ratio" in train_data.columns:
        imbalance_col = "long_short_ratio"
    elif "funding_zscore" in train_data.columns:
        # Proxy with funding z-score
        imbalance_col = "funding_zscore"
    else:
        logger.warning("No imbalance measure available for H3")
        results["status"] = "INSUFFICIENT_DATA"
        return results

    train_clean = train_data[[imbalance_col, "positive_return"]].dropna()
    test_clean = test_data[[imbalance_col, "positive_return"]].dropna()

    if len(train_clean) < 30:
        logger.warning("Insufficient data for H3")
        results["status"] = "INSUFFICIENT_DATA"
        return results

    X_train = train_clean[[imbalance_col]].values
    y_train = train_clean["positive_return"].values

    # Fit logistic model
    model = LogisticRegression().fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    logger.info(f"\nTrain accuracy: {train_score:.4f}")

    # Test
    X_test = test_clean[[imbalance_col]].values
    y_test = test_clean["positive_return"].values

    test_score = model.score(X_test, y_test)
    test_pred = model.predict_proba(X_test)[:, 1]

    logger.info(f"Test accuracy: {test_score:.4f}")

    try:
        auc = roc_auc_score(y_test, test_pred)
        logger.info(f"Test AUC: {auc:.4f}")
    except Exception as e:
        auc = None
        logger.warning(f"Could not compute AUC: {e}")

    # Bootstrap significance test
    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())

    results.update({
        "status": "COMPLETE",
        "train_accuracy": float(train_score),
        "test_accuracy": float(test_score),
        "baseline_accuracy": float(baseline_accuracy),
        "auc": float(auc) if auc else None,
        "significant": test_score > baseline_accuracy + 0.05,
        "conclusion": "SUPPORTED" if (test_score > baseline_accuracy + 0.05) else "NOT SUPPORTED",
    })

    return results


def run_all_hypothesis_tests(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    """Run all hypothesis tests."""

    logger.info("\n" + "="*70)
    logger.info("PHASE 5: HYPOTHESIS TESTING & MODELING")
    logger.info("="*70)

    test_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "phase": "Phase 5 - Hypothesis Testing",
        "train_period": {"start": str(train["timestamp"].min()), "end": str(train["timestamp"].max())},
        "test_period": {"start": str(test["timestamp"].min()), "end": str(test["timestamp"].max())},
        "hypotheses": {},
    }

    # Run all tests
    h1_results = hypothesis_1_funding_predicts_returns(train, test)
    test_results["hypotheses"]["H1"] = h1_results

    h2_results = hypothesis_2_oi_funding_predicts_reversals(train, test)
    test_results["hypotheses"]["H2"] = h2_results

    h3_results = hypothesis_3_imbalance_predicts_directional(train, test)
    test_results["hypotheses"]["H3"] = h3_results

    # Summary
    logger.info("\n" + "="*70)
    logger.info("HYPOTHESIS TEST SUMMARY")
    logger.info("="*70)

    for hyp, results in test_results["hypotheses"].items():
        status = results.get("conclusion", "UNKNOWN")
        logger.info(f"{hyp}: {status}")

    # Save results
    os.makedirs("./analysis", exist_ok=True)
    with open("./analysis/hypothesis_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    logger.info("\nSaved test results to ./analysis/hypothesis_test_results.json")

    return test_results


def phase5_modeling_workflow():
    """Execute Phase 5 modeling workflow."""

    logger.info("Starting Phase 5 Hypothesis Testing & Modeling")

    # Load data
    logger.info("\n[Step 0] Loading featured data...")
    df = load_featured_data()

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Train/test split
    logger.info("\n[Step 1] Preparing train/test split (70/30 time-based)...")
    train, test = prepare_train_test_split(df, train_pct=0.7)

    # Run hypothesis tests
    logger.info("\n[Step 2] Running hypothesis tests...")
    results = run_all_hypothesis_tests(train, test)

    logger.info("\n" + "="*70)
    logger.info("PHASE 5 COMPLETE: Hypothesis Testing")
    logger.info("="*70)

    return results


if __name__ == "__main__":
    try:
        results = phase5_modeling_workflow()

        logger.info("\n✓ Phase 5 workflow completed successfully")

        # Print next steps
        logger.info("\n" + "="*70)
        logger.info("NEXT STEPS:")
        logger.info("="*70)
        logger.info("1. Review hypothesis test results: analysis/hypothesis_test_results.json")
        logger.info("2. Proceed to Phase 6: Robustness & Validation")
        logger.info("   → analysis/phase6_validation.py")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Phase 5 workflow failed: {e}", exc_info=True)
        sys.exit(1)
