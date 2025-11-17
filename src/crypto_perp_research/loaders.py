"""
Data loaders for crypto derivatives microstructure research.

This module provides functions to load and download data from various public sources:
- Binance Vision (OHLCV data)
- CryptoDataDownload (funding rates, liquidations, OHLCV)
- CoinGlass API (liquidations, open interest)
- Coinalyze API (funding rates, open interest)
- Official exchange APIs (Deribit, OKX, Bybit)
"""

import os
import io
import json
import csv
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

# Note: These imports assume the required packages are available
# If not installed, users should run: pip install requests pandas numpy
try:
    import requests
    import pandas as pd
    import numpy as np
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}. Please install with: pip install requests pandas numpy")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceVisionLoader:
    """
    Loader for Binance Vision data.

    Free source for historical OHLCV data from Binance.
    Coverage: 2020-present
    Access: Direct HTTP downloads from data.binance.vision
    """

    BASE_URL = "https://data.binance.vision/data"

    @staticmethod
    def download_ohlcv(
        symbol: str,
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: str = "./data/raw/binance_ohlcv"
    ) -> pd.DataFrame:
        """
        Download OHLCV data for a symbol from Binance Vision.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Kline interval ('1m', '5m', '1h', '1d', etc.)
            start_date: Start date (YYYY-MM-DD format), None = earliest available
            end_date: End date (YYYY-MM-DD format), None = today
            output_dir: Directory to store downloaded CSV files

        Returns:
            pandas DataFrame with columns: Open, High, Low, Close, Volume, etc.

        Raises:
            requests.RequestException: If download fails
            ValueError: If date range is invalid
        """
        os.makedirs(output_dir, exist_ok=True)

        # Parse dates
        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = datetime(2020, 1, 1)  # Binance Vision starts 2020

        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = datetime.now()

        logger.info(f"Downloading {symbol} {interval} from {start.date()} to {end.date()}")

        # Build URLs for monthly archives
        dfs = []
        current = start

        while current <= end:
            year_month = current.strftime("%Y-%m")
            url = f"{BinanceVisionLoader.BASE_URL}/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year_month}.zip"

            logger.debug(f"Attempting to download: {url}")

            try:
                # Download and extract CSV from ZIP
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    # Extract CSV from ZIP (Binance uses CSV inside ZIP)
                    import zipfile
                    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
                    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]

                    if csv_files:
                        with zip_file.open(csv_files[0]) as csv_file:
                            df = pd.read_csv(csv_file, names=[
                                'timestamp', 'open', 'high', 'low', 'close',
                                'volume', 'close_time', 'quote_asset_volume',
                                'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
                            ])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df['symbol'] = symbol
                            dfs.append(df)
                            logger.info(f"Downloaded {len(df)} rows from {year_month}")
                else:
                    logger.warning(f"Status {response.status_code} for {year_month}")

            except Exception as e:
                logger.warning(f"Failed to download {year_month}: {e}")

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        if not dfs:
            logger.warning(f"No data downloaded for {symbol}")
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['timestamp', 'symbol'])
        result = result.sort_values('timestamp').reset_index(drop=True)

        # Save to CSV
        output_path = os.path.join(output_dir, f"{symbol}_{interval}.csv")
        result.to_csv(output_path, index=False)
        logger.info(f"Saved {len(result)} rows to {output_path}")

        return result


class CryptoDataDownloadLoader:
    """
    Loader for CryptoDataDownload data.

    Free source for OHLCV, funding rates, and liquidations.
    Coverage: 2017-present
    Access: Direct CSV downloads from cryptodatadownload.com
    """

    BASE_URL = "https://www.cryptodatadownload.com/data"

    @staticmethod
    def download_funding_rates(
        symbol: str = "BTCUSDT",
        output_dir: str = "./data/raw/cryptodatadownload_funding"
    ) -> pd.DataFrame:
        """
        Download funding rates from CryptoDataDownload.

        Args:
            symbol: Perpetual symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            output_dir: Directory to store downloaded files

        Returns:
            pandas DataFrame with funding rate data
        """
        os.makedirs(output_dir, exist_ok=True)

        # CryptoDataDownload funding rates CSV URL pattern
        url = f"{CryptoDataDownloadLoader.BASE_URL}/binance_um/daily/funding_rate/{symbol}_binance_um_funding_rate_daily.csv"

        logger.info(f"Downloading funding rates for {symbol} from CryptoDataDownload")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))
            df['timestamp'] = pd.to_datetime(df.get('date', df.get('Date', df.iloc[:, 0])))
            df['symbol'] = symbol

            output_path = os.path.join(output_dir, f"{symbol}_funding_rates.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} rows to {output_path}")

            return df

        except Exception as e:
            logger.error(f"Failed to download funding rates for {symbol}: {e}")
            return pd.DataFrame()


class CoinalyzeAPILoader:
    """
    Loader for Coinalyze API data (funding rates, open interest, liquidations).

    Free API source with rate limits.
    Coverage: Limited intraday history (1500-2000 bars), permanent daily history
    Access: REST API (requires free API key)
    """

    BASE_URL = "https://api.coinalyze.net/v1"

    @staticmethod
    def get_funding_rates(
        symbol: str = "BTC",
        exchange: str = "Binance",
        interval: str = "1m",
        limit: int = 500,
        api_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch funding rate data from Coinalyze API.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            exchange: Exchange name (e.g., 'Binance', 'Bybit', 'OKX')
            interval: Candle interval ('1m', '5m', '1h', '1d')
            limit: Number of candles to fetch (max varies by interval)
            api_key: Coinalyze API key (optional for free tier)

        Returns:
            pandas DataFrame with funding rate data
        """

        endpoint = f"{CoinalyzeAPILoader.BASE_URL}/pair/{exchange}/{symbol}USDT/funding_rate/{interval}"

        params = {
            'limit': limit,
            'format': 'json'
        }

        if api_key:
            params['key'] = api_key

        logger.info(f"Fetching funding rates for {symbol} from {exchange} ({interval})")

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'result' in data:
                df = pd.DataFrame(data['result'])
                df['timestamp'] = pd.to_datetime(df.get('time', df.get('timestamp')))
                df['symbol'] = symbol
                df['exchange'] = exchange

                logger.info(f"Fetched {len(df)} funding rate records")
                return df
            else:
                logger.warning(f"Unexpected API response structure: {data}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch funding rates from Coinalyze: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_open_interest(
        symbol: str = "BTC",
        exchange: str = "Binance",
        interval: str = "1d",
        limit: int = 500,
        api_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch open interest data from Coinalyze API.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            exchange: Exchange name (e.g., 'Binance', 'Bybit', 'OKX')
            interval: Candle interval ('1m', '5m', '1h', '1d')
            limit: Number of candles to fetch
            api_key: Coinalyze API key

        Returns:
            pandas DataFrame with open interest data
        """

        endpoint = f"{CoinalyzeAPILoader.BASE_URL}/pair/{exchange}/{symbol}USDT/open_interest/{interval}"

        params = {
            'limit': limit,
            'format': 'json'
        }

        if api_key:
            params['key'] = api_key

        logger.info(f"Fetching open interest for {symbol} from {exchange} ({interval})")

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'result' in data:
                df = pd.DataFrame(data['result'])
                df['timestamp'] = pd.to_datetime(df.get('time', df.get('timestamp')))
                df['symbol'] = symbol
                df['exchange'] = exchange

                logger.info(f"Fetched {len(df)} open interest records")
                return df
            else:
                logger.warning(f"Unexpected API response: {data}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch open interest from Coinalyze: {e}")
            return pd.DataFrame()


class DeribitAPILoader:
    """
    Loader for Deribit API data (options, futures, liquidations).

    Free API for research (requires API key for higher rate limits).
    Coverage: Since Deribit inception (tick-level)
    Access: REST/WebSocket API
    """

    BASE_URL = "https://history.deribit.com/api/v2/public"

    @staticmethod
    def get_last_trades(
        instrument: str,
        count: int = 100,
        before: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch recent trades for a Deribit instrument.

        Args:
            instrument: Deribit instrument name (e.g., 'BTC-PERPETUAL', 'ETH-PERPETUAL')
            count: Number of trades to fetch (max 1000)
            before: Trade ID to start from (for pagination)

        Returns:
            pandas DataFrame with trade data
        """

        endpoint = f"{DeribitAPILoader.BASE_URL}/get_last_trades_by_instrument"

        params = {
            'instrument_name': instrument,
            'count': min(count, 1000),
            'sorting': 'desc'
        }

        if before:
            params['before'] = before

        logger.info(f"Fetching trades for {instrument}")

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('result'):
                trades = data['result'].get('trades', [])
                df = pd.DataFrame(trades)
                df['timestamp'] = pd.to_datetime(df.get('timestamp', 0), unit='ms')

                logger.info(f"Fetched {len(df)} trades")
                return df
            else:
                logger.warning("No trade data in response")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return pd.DataFrame()


class DataLoader:
    """
    Unified data loader that orchestrates downloads from multiple sources.
    """

    def __init__(self, output_base_dir: str = "./data/raw"):
        self.output_base_dir = output_base_dir
        self.binance = BinanceVisionLoader()
        self.cryptodatadownload = CryptoDataDownloadLoader()
        self.coinalyze = CoinalyzeAPILoader()
        self.deribit = DeribitAPILoader()

    def load_all_sources(
        self,
        symbols: List[str] = ["BTCUSDT", "ETHUSDT"],
        exchanges: List[str] = ["Binance"],
        data_types: List[str] = ["ohlcv", "funding", "open_interest"]
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from all available sources.

        Args:
            symbols: List of symbols to download
            exchanges: List of exchanges to include
            data_types: Types of data to download

        Returns:
            Dictionary mapping (exchange, symbol, data_type) to DataFrames
        """

        results = {}

        for symbol in symbols:
            for exchange in exchanges:
                key = f"{exchange}_{symbol}"

                if "ohlcv" in data_types:
                    logger.info(f"Loading OHLCV for {key}")
                    try:
                        results[f"{key}_ohlcv"] = self.binance.download_ohlcv(
                            symbol=symbol,
                            interval="1d",
                            output_dir=os.path.join(self.output_base_dir, "ohlcv")
                        )
                    except Exception as e:
                        logger.error(f"Failed to load OHLCV for {key}: {e}")

                if "funding" in data_types:
                    logger.info(f"Loading funding rates for {key}")
                    try:
                        results[f"{key}_funding"] = self.cryptodatadownload.download_funding_rates(
                            symbol=symbol,
                            output_dir=os.path.join(self.output_base_dir, "funding")
                        )
                    except Exception as e:
                        logger.error(f"Failed to load funding for {key}: {e}")

        return results


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()

    # Download sample data
    logger.info("Starting data acquisition...")
    results = loader.load_all_sources(
        symbols=["BTCUSDT", "ETHUSDT"],
        exchanges=["Binance"],
        data_types=["ohlcv", "funding"]
    )

    for key, df in results.items():
        if not df.empty:
            logger.info(f"{key}: {len(df)} rows")
