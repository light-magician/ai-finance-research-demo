"""
Crypto Derivatives Market Microstructure Research Library

A comprehensive toolkit for acquiring, processing, and analyzing:
- Historical funding rates
- Open interest and positioning data
- Liquidation events
- Spot and perpetual OHLCV data

Across major exchanges (Binance, Bybit, OKX, Deribit, etc.)
"""

__version__ = "0.1.0"
__author__ = "Quant Research Team"

from . import loaders
from . import cleaners
from . import features
from . import utils

__all__ = ["loaders", "cleaners", "features", "utils"]
