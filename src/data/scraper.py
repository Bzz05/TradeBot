import asyncio
import aiohttp
import pandas as pd
import sys
import yaml
import logging
import json
from datetime import datetime, timedelta
import os

class BinanceDataScraper:
    def __init__(self, config_path):
        # init scraper
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.api_url = self.config.get("api_url", "https://testnet.binancefuture.com")
        self.session = aiohttp.ClientSession()
        logging.info("BinanceDataScraper initialized.")

    async def fetch_historical_klines(self, symbol, interval, start_time, end_time=None):
        all_data = []
        current_time = start_time
        interval_map = {
            "5m": 5 * 60, "15m": 15*60, "1h": 60*60,
        }
        interval_seconds = interval_map.get(interval, 60*60)  # default 1h
        limit = 1000

        while current_time < end_time:
            end_batch_time = current_time + timedelta(seconds=interval_seconds * limit)
            if end_batch_time > end_time:
                end_batch_time = end_time

            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": int(current_time.timestamp() * 1000),
                "endTime": int(end_batch_time.timestamp() * 1000),
                "limit": limit
            }

            try:
                async with self.session.get(f"{self.api_url}/fapi/v1/klines", params=params) as response:
                    data = await response.json()
                if not data:
                    break
                all_data.extend(data)
                last_ts = data[-1][0] / 1000
                current_time = datetime.fromtimestamp(last_ts + interval_seconds)
                await asyncio.sleep(0.2)  # rate limit
            except Exception as e:
                logging.error(f"Error fetching klines: {e}")
                break

        df = pd.DataFrame(all_data, columns=[
            'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteAssetVolume', 'NumberofTrades',
            'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
        ])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='ms', utc = True)
        df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.drop_duplicates(subset=['Timestamp'], keep='first', inplace=True)
        df.sort_values('Timestamp', inplace=True)
        logging.info(f"Fetched {len(df)} klines for {symbol}.")
        return df

    def save_data_to_disk(self, data, data_type, symbol, timeframe):
        # timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        filename = f"{symbol}_{timeframe}.csv"
        directory = os.path.join("data", data_type)
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        try:
            if isinstance(data, pd.DataFrame):
                # Convert datetime columns to string format without milliseconds.
                for col in data.select_dtypes(include=['datetime64[ns]']).columns:
                    data[col] = data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                data.to_csv(filepath, index=False)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            logging.info(f"Data saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving data to disk: {e}")
            raise

    async def close(self):
        await self.session.close()

async def main():
    config_path = "config/config.yaml"
    scraper = BinanceDataScraper(config_path)
    
    symbol = "BTCUSDT" # set this urself
    intervals = ["5m", "15m", "1h"] # this too
    start_time = datetime(2019, 9, 1)
    end_time = datetime(2025, 7, 1)  
    
    for interval in intervals:
        klines = await scraper.fetch_historical_klines(symbol, interval, start_time, end_time)
        
        # save the df
        output_directory = "data/historical_raw/BTCUSDT"
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, f"klines_{interval}.csv")
        
        # remove miliseconds, close time have microseconds and it kinda messes up reading in excel bcz of formatting
        klines.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Data saved to {output_path}")
    
    await scraper.close()

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())