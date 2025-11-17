#!/usr/bin/env python3
"""
Phase 4: Exploratory Data Analysis (EDA)

This script performs comprehensive exploratory analysis of the featured crypto derivatives data:
- Summary statistics and distributions
- Time-series visualizations
- Correlation analysis and heatmaps
- Market regime identification (bull/bear/sideways)
- Volatility patterns
- Key insights and observations

Requirements:
    pip install pandas numpy matplotlib seaborn

Usage:
    python analysis/phase4_eda.py
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Error importing visualization libraries: {e}")
    print("Please install: pip install matplotlib seaborn")
    sys.exit(1)

from crypto_perp_research.utils import DataPersistence, ExperimentLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_featured_data() -> pd.DataFrame:
    """Load featured data from Phase 3."""
    try:
        # Try parquet first
        df = pd.read_parquet("./data/processed/features_btc_1d.parquet")
        logger.info(f"Loaded featured data from Parquet: {len(df)} rows")
        return df
    except FileNotFoundError:
        try:
            # Fallback to CSV
            df = pd.read_csv("./data/processed/features_btc_1d.csv")
            logger.info(f"Loaded featured data from CSV: {len(df)} rows")
            return df
        except FileNotFoundError:
            logger.error("Featured data not found. Please run Phase 3 first: python analysis/phase3_data_integration.py")
            sys.exit(1)


def summary_statistics(df: pd.DataFrame) -> dict:
    """Generate summary statistics."""
    logger.info("\n[Step 1] Computing summary statistics...")

    stats = {
        "total_rows": len(df),
        "date_range": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "symbols": df["symbol"].unique().tolist() if "symbol" in df.columns else ["unknown"],
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
    }

    # Price statistics
    if "close" in df.columns:
        stats["price"] = {
            "mean": float(df["close"].mean()),
            "std": float(df["close"].std()),
            "min": float(df["close"].min()),
            "max": float(df["close"].max()),
            "pct_change": float((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100),
        }

    # Return statistics
    for col in df.columns:
        if "return" in col.lower():
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "skew": float(df[col].skew()),
                "null_count": int(df[col].isnull().sum()),
            }

    # Funding statistics
    if "funding_rate" in df.columns:
        stats["funding_rate"] = {
            "mean": float(df["funding_rate"].mean()),
            "std": float(df["funding_rate"].std()),
            "min": float(df["funding_rate"].min()),
            "max": float(df["funding_rate"].max()),
            "positive_pct": float((df["funding_rate"] > 0).sum() / len(df) * 100),
        }

    # OI statistics
    if "open_interest" in df.columns:
        stats["open_interest"] = {
            "mean": float(df["open_interest"].mean()),
            "std": float(df["open_interest"].std()),
            "min": float(df["open_interest"].min()),
            "max": float(df["open_interest"].max()),
        }

    logger.info(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    logger.info(f"Total rows: {stats['total_rows']}")

    return stats


def plot_price_series(df: pd.DataFrame, output_dir: str = "./analysis/figures"):
    """Plot price time series."""
    logger.info("\n[Step 2] Creating price time-series plot...")

    os.makedirs(output_dir, exist_ok=True)

    if "close" not in df.columns:
        logger.warning("No 'close' column found")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Price
    ax1.plot(df["timestamp"], df["close"], linewidth=1.5, color="steelblue")
    ax1.set_title("BTC Close Price Over Time", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, alpha=0.3)

    # Daily returns
    if "return_1h" in df.columns:
        returns_24h = df["close"].pct_change(24) * 100
        colors = ["green" if r > 0 else "red" for r in returns_24h]
        ax2.bar(df["timestamp"], returns_24h, color=colors, alpha=0.6, width=0.5)
        ax2.set_title("24-Hour Returns", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Return (%)")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "01_price_series.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_returns_distribution(df: pd.DataFrame, output_dir: str = "./analysis/figures"):
    """Plot returns distributions."""
    logger.info("\n[Step 3] Creating returns distribution plots...")

    os.makedirs(output_dir, exist_ok=True)

    # Find return columns
    return_cols = [col for col in df.columns if "forward_return" in col]

    if not return_cols:
        logger.warning("No forward_return columns found")
        return

    fig, axes = plt.subplots(1, len(return_cols), figsize=(14, 5))
    if len(return_cols) == 1:
        axes = [axes]

    for idx, col in enumerate(return_cols):
        data = df[col].dropna()
        axes[idx].hist(data, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        axes[idx].set_title(f"{col}\nμ={data.mean():.3f}%, σ={data.std():.3f}%", fontsize=11)
        axes[idx].set_xlabel("Return (%)")
        axes[idx].set_ylabel("Frequency")
        axes[idx].axvline(data.mean(), color="red", linestyle="--", linewidth=2, label="Mean")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "02_returns_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_funding_analysis(df: pd.DataFrame, output_dir: str = "./analysis/figures"):
    """Plot funding rate analysis."""
    logger.info("\n[Step 4] Creating funding rate analysis plots...")

    os.makedirs(output_dir, exist_ok=True)

    if "funding_rate" not in df.columns:
        logger.warning("No funding_rate column found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series
    axes[0, 0].plot(df["timestamp"], df["funding_rate"] * 100, linewidth=1, color="orange")
    axes[0, 0].set_title("Funding Rate Over Time", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Funding Rate (%)")
    axes[0, 0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[0, 0].grid(True, alpha=0.3)

    # Distribution
    axes[0, 1].hist(df["funding_rate"] * 100, bins=50, alpha=0.7, color="orange", edgecolor="black")
    axes[0, 1].set_title("Funding Rate Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Funding Rate (%)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(df["funding_rate"].mean() * 100, color="red", linestyle="--", linewidth=2)
    axes[0, 1].grid(True, alpha=0.3)

    # Z-scores
    if "funding_zscore" in df.columns:
        axes[1, 0].plot(df["timestamp"], df["funding_zscore"], linewidth=1, color="purple")
        axes[1, 0].set_title("Funding Rate Z-Score", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Z-Score")
        axes[1, 0].axhline(y=2, color="red", linestyle="--", linewidth=1, alpha=0.5, label="±2σ")
        axes[1, 0].axhline(y=-2, color="red", linestyle="--", linewidth=1, alpha=0.5)
        axes[1, 0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Funding vs returns
    if "forward_return_1h" in df.columns:
        valid_data = df[["funding_rate", "forward_return_1h"]].dropna()
        axes[1, 1].scatter(valid_data["funding_rate"] * 100, valid_data["forward_return_1h"], alpha=0.3, s=10)
        axes[1, 1].set_title("Funding Rate vs Forward Returns", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Funding Rate (%)")
        axes[1, 1].set_ylabel("Forward Return (%)")
        axes[1, 1].grid(True, alpha=0.3)

        # Add trend line
        if len(valid_data) > 10:
            z = np.polyfit(valid_data["funding_rate"], valid_data["forward_return_1h"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(valid_data["funding_rate"].min(), valid_data["funding_rate"].max(), 100)
            axes[1, 1].plot(x_trend * 100, p(x_trend), "r--", linewidth=2, alpha=0.8, label="Trend")
            axes[1, 1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "03_funding_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_oi_analysis(df: pd.DataFrame, output_dir: str = "./analysis/figures"):
    """Plot open interest analysis."""
    logger.info("\n[Step 5] Creating open interest analysis plots...")

    os.makedirs(output_dir, exist_ok=True)

    if "open_interest" not in df.columns:
        logger.warning("No open_interest column found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series
    axes[0, 0].plot(df["timestamp"], df["open_interest"] / 1e6, linewidth=1, color="green")
    axes[0, 0].set_title("Open Interest Over Time", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Open Interest ($M)")
    axes[0, 0].grid(True, alpha=0.3)

    # Distribution
    axes[0, 1].hist(df["open_interest"] / 1e6, bins=50, alpha=0.7, color="green", edgecolor="black")
    axes[0, 1].set_title("Open Interest Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Open Interest ($M)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Z-scores
    if "oi_zscore" in df.columns:
        axes[1, 0].plot(df["timestamp"], df["oi_zscore"], linewidth=1, color="darkgreen")
        axes[1, 0].set_title("Open Interest Z-Score", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Z-Score")
        axes[1, 0].axhline(y=2, color="red", linestyle="--", linewidth=1, alpha=0.5)
        axes[1, 0].axhline(y=-2, color="red", linestyle="--", linewidth=1, alpha=0.5)
        axes[1, 0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[1, 0].grid(True, alpha=0.3)

    # OI vs returns
    if "forward_return_1h" in df.columns:
        valid_data = df[["open_interest", "forward_return_1h"]].dropna()
        axes[1, 1].scatter(valid_data["open_interest"] / 1e6, valid_data["forward_return_1h"], alpha=0.3, s=10)
        axes[1, 1].set_title("Open Interest vs Forward Returns", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Open Interest ($M)")
        axes[1, 1].set_ylabel("Forward Return (%)")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "04_oi_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "./analysis/figures"):
    """Plot correlation heatmap."""
    logger.info("\n[Step 6] Creating correlation heatmap...")

    os.makedirs(output_dir, exist_ok=True)

    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to key columns
    key_cols = [col for col in numeric_cols if any(x in col for x in ["return", "funding", "oi", "close"])]

    if len(key_cols) < 2:
        logger.warning("Not enough key columns for correlation analysis")
        return

    corr = df[key_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "05_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def identify_regimes(df: pd.DataFrame) -> dict:
    """Identify market regimes based on volatility and price."""
    logger.info("\n[Step 7] Identifying market regimes...")

    regimes = {
        "total_periods": len(df),
        "date_range": {"start": str(df["timestamp"].min()), "end": str(df["timestamp"].max())},
    }

    if "close" not in df.columns:
        return regimes

    # Compute rolling volatility
    rolling_vol = df["close"].pct_change().rolling(window=20).std() * np.sqrt(252) * 100

    vol_mean = rolling_vol.mean()
    vol_std = rolling_vol.std()

    # Classify regimes
    high_vol = (rolling_vol > vol_mean + vol_std).sum()
    low_vol = (rolling_vol < vol_mean - vol_std).sum()
    normal_vol = len(df) - high_vol - low_vol

    regimes["volatility"] = {
        "mean": float(vol_mean),
        "std": float(vol_std),
        "high_vol_periods": int(high_vol),
        "normal_vol_periods": int(normal_vol),
        "low_vol_periods": int(low_vol),
    }

    # Bull/bear based on price trend
    price_trend = df["close"].pct_change(60).mean() * 100  # 60-day trend
    if price_trend > 1:
        trend = "bullish"
    elif price_trend < -1:
        trend = "bearish"
    else:
        trend = "sideways"

    regimes["price_trend"] = {
        "60_day_return": float(price_trend),
        "classification": trend,
    }

    logger.info(f"Volatility regime: mean={vol_mean:.2f}%, std={vol_std:.2f}%")
    logger.info(f"High vol periods: {high_vol}, Normal: {normal_vol}, Low: {low_vol}")
    logger.info(f"Price trend (60d): {price_trend:.2f}% ({trend})")

    return regimes


def compute_key_insights(df: pd.DataFrame, stats: dict) -> list:
    """Generate key insights from the data."""
    logger.info("\n[Step 8] Generating key insights...")

    insights = []

    # Insight 1: Price movement
    if "close" in df.columns:
        pct_change = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        insights.append(
            f"Price Movement: BTC price changed {pct_change:.1f}% over the analysis period "
            f"(${df['close'].min():.0f} - ${df['close'].max():.0f})"
        )

    # Insight 2: Funding bias
    if "funding_rate" in df.columns:
        positive_pct = (df["funding_rate"] > 0).sum() / len(df) * 100
        avg_funding = df["funding_rate"].mean() * 100
        insights.append(
            f"Funding Rate Bias: {positive_pct:.1f}% of periods had positive funding (avg: {avg_funding:.3f}%). "
            f"This suggests {'net long bias' if positive_pct > 50 else 'net short bias'} among traders."
        )

    # Insight 3: Return distribution
    if "forward_return_1h" in df.columns:
        returns = df["forward_return_1h"].dropna()
        positive_returns = (returns > 0).sum() / len(returns) * 100
        avg_ret = returns.mean()
        insights.append(
            f"Return Characteristics: {positive_returns:.1f}% of forward returns were positive "
            f"(avg: {avg_ret:.4f}%, std: {returns.std():.4f}%). "
            f"Return distribution shows {'right skew' if returns.skew() > 0 else 'left skew'} "
            f"(skew: {returns.skew():.2f})"
        )

    # Insight 4: Volatility
    if "close" in df.columns:
        daily_vol = df["close"].pct_change().std() * np.sqrt(365) * 100
        insights.append(f"Annualized Volatility: {daily_vol:.1f}% (daily vol: {df['close'].pct_change().std() * 100:.2f}%)")

    # Insight 5: OI trend
    if "open_interest" in df.columns:
        oi_trend = (df["open_interest"].iloc[-1] / df["open_interest"].iloc[0] - 1) * 100
        insights.append(f"Open Interest Trend: {oi_trend:+.1f}% change over period")

    # Insight 6: Correlation insight
    if "funding_rate" in df.columns and "forward_return_1h" in df.columns:
        valid_data = df[["funding_rate", "forward_return_1h"]].dropna()
        if len(valid_data) > 10:
            corr = valid_data["funding_rate"].corr(valid_data["forward_return_1h"])
            insights.append(
                f"Funding-Return Correlation: {corr:.3f}. "
                f"This suggests funding rates and next-period returns are "
                f"{'strongly' if abs(corr) > 0.3 else 'weakly' if abs(corr) > 0.1 else 'negligibly'} correlated."
            )

    for i, insight in enumerate(insights, 1):
        logger.info(f"Insight {i}: {insight}")

    return insights


def phase4_eda_workflow():
    """Execute Phase 4 EDA workflow."""

    logger.info("="*70)
    logger.info("PHASE 4: EXPLORATORY DATA ANALYSIS")
    logger.info("="*70)

    # Load data
    logger.info("\n[Step 0] Loading featured data from Phase 3...")
    df = load_featured_data()
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Ensure timestamp is datetime
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Summary statistics
    stats = summary_statistics(df)

    # Visualizations
    plot_price_series(df)
    plot_returns_distribution(df)
    plot_funding_analysis(df)
    plot_oi_analysis(df)
    plot_correlation_heatmap(df)

    # Regime identification
    regimes = identify_regimes(df)

    # Key insights
    insights = compute_key_insights(df, stats)

    # Compile EDA report
    eda_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "phase": "Phase 4 - Exploratory Data Analysis",
        "summary_statistics": stats,
        "market_regimes": regimes,
        "key_insights": insights,
        "visualizations": [
            "01_price_series.png",
            "02_returns_distribution.png",
            "03_funding_analysis.png",
            "04_oi_analysis.png",
            "05_correlation_heatmap.png",
        ],
    }

    # Save report
    os.makedirs("./analysis", exist_ok=True)
    with open("./analysis/eda_notes.json", "w") as f:
        json.dump(eda_report, f, indent=2, default=str)

    logger.info("\nSaved EDA report to ./analysis/eda_notes.json")

    # Save insights to markdown
    with open("./analysis/eda_notes.md", "w") as f:
        f.write("# Exploratory Data Analysis (Phase 4)\n\n")
        f.write(f"**Generated:** {datetime.utcnow().isoformat()}\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Periods:** {stats['total_rows']}\n")
        f.write(f"- **Date Range:** {stats['date_range']['start']} to {stats['date_range']['end']}\n")

        if "price" in stats:
            f.write(f"\n### Price\n")
            f.write(f"- **Mean:** ${stats['price']['mean']:.2f}\n")
            f.write(f"- **Std Dev:** ${stats['price']['std']:.2f}\n")
            f.write(f"- **Range:** ${stats['price']['min']:.2f} - ${stats['price']['max']:.2f}\n")
            f.write(f"- **Total Change:** {stats['price']['pct_change']:+.2f}%\n")

        f.write("\n## Market Regimes\n\n")
        if "volatility" in regimes:
            vol = regimes["volatility"]
            f.write(f"- **Annualized Volatility:** {vol['mean']:.1f}% (±{vol['std']:.1f}%)\n")
            f.write(f"- **High Vol Periods:** {vol['high_vol_periods']} ({vol['high_vol_periods']/regimes['total_periods']*100:.1f}%)\n")
            f.write(f"- **Normal Vol Periods:** {vol['normal_vol_periods']} ({vol['normal_vol_periods']/regimes['total_periods']*100:.1f}%)\n")
            f.write(f"- **Low Vol Periods:** {vol['low_vol_periods']} ({vol['low_vol_periods']/regimes['total_periods']*100:.1f}%)\n")

        if "price_trend" in regimes:
            trend = regimes["price_trend"]
            f.write(f"\n- **60-Day Price Trend:** {trend['60_day_return']:+.2f}% ({trend['classification']})\n")

        f.write("\n## Key Insights\n\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n\n")

        f.write("\n## Visualizations Generated\n\n")
        f.write("- `01_price_series.png` - Price time-series and daily returns\n")
        f.write("- `02_returns_distribution.png` - Forward return distributions\n")
        f.write("- `03_funding_analysis.png` - Funding rate analysis and correlation with returns\n")
        f.write("- `04_oi_analysis.png` - Open interest analysis and trends\n")
        f.write("- `05_correlation_heatmap.png` - Feature correlation matrix\n")

        f.write("\n## Next Steps (Phase 5)\n\n")
        f.write("- Test hypothesis H1: Funding rates predict next-period returns\n")
        f.write("- Build OLS and logistic regression models\n")
        f.write("- Evaluate statistical significance\n")
        f.write("- Perform out-of-sample validation\n")

    logger.info("Saved EDA notes to ./analysis/eda_notes.md")

    logger.info("\n" + "="*70)
    logger.info("PHASE 4 COMPLETE: Exploratory Data Analysis")
    logger.info("="*70)

    return eda_report


if __name__ == "__main__":
    logger.info("Starting Phase 4 Exploratory Data Analysis")

    try:
        report = phase4_eda_workflow()

        logger.info("\n✓ Phase 4 workflow completed successfully")

        # Print next steps
        logger.info("\n" + "="*70)
        logger.info("NEXT STEPS:")
        logger.info("="*70)
        logger.info("1. Review EDA findings: analysis/eda_notes.md")
        logger.info("2. Examine visualizations: analysis/figures/*.png")
        logger.info("3. Proceed to Phase 5: Hypothesis Testing & Modeling")
        logger.info("   → analysis/phase5_modeling.py")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Phase 4 workflow failed: {e}", exc_info=True)
        sys.exit(1)
