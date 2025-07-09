import os
import time
import logging
import json
import yaml
import signal
import pandas as pd
import threading
import websocket
from kafka import KafkaProducer
from kafka.errors import KafkaError
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("BinanceProducerKafka")
shutdown_flag = threading.Event()
CONFIG_PATH = "config/config.yaml"

def signal_handler(signum, frame):
    logger.warning(f"Shutdown signal {signum} received. Producer stopping...")
    shutdown_flag.set()

class BinanceDataProducer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
        
        kafka_cfg = self.config['kafka']
        producer_cfg = self.config['producer']
        data_paths_cfg = self.config['data_paths']
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=kafka_cfg['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=5, linger_ms=20, acks='all')
            logger.info(f"KafkaProducer connected to {kafka_cfg['bootstrap_servers']}")
        except KafkaError as e: 
            logger.error(f"Kafka Producer init failed: {e}", exc_info=True); raise
        
        self.topic_prefix = kafka_cfg["topic_market_data_prefix"]
        self.symbols = producer_cfg["symbols_to_fetch"]
        self.timeframes = self.config["data_processing"]["feature_engineer_timeframes"]
        self.raw_data_dir = os.path.join(self.config.get("project_root", "."), data_paths_cfg["raw_data_directory"])
        self.save_batch_size = producer_cfg.get("save_batch_size", 50)
        self.kline_save_buffers = {f"{s}_{tf}": [] for s in self.symbols for tf in self.timeframes}
        for s in self.symbols: os.makedirs(os.path.join(self.raw_data_dir, s), exist_ok=True)

    def _save_buffer_to_disk(self, buffer_key):
        if not self.kline_save_buffers.get(buffer_key): return
        symbol, timeframe = buffer_key.split('_')
        filename = f"klines_{timeframe}.csv"
        filepath = os.path.join(self.raw_data_dir, symbol, filename)
        new_data_df = pd.DataFrame(self.kline_save_buffers[buffer_key])
        try:
            header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
            new_data_df.to_csv(filepath, mode='a', header=header, index=False)
            logger.info(f"SAVED BATCH: Appended {len(self.kline_save_buffers[buffer_key])} new klines to {filepath}")
            self.kline_save_buffers[buffer_key].clear()
        except Exception as e: logger.error(f"Error saving data to {filepath}: {e}")

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
            if 'stream' in msg and 'data' in msg and (data := msg['data']):
                if data.get('e') == 'kline' and data.get('k', {}).get('x'):
                    kline = data['k']; symbol, timeframe = kline['s'], kline['i']
                    kline_open_time = datetime.fromtimestamp(int(kline['t'])/1000, tz=timezone.utc)
                    logger.info(
                        f"LIVE DATA: {symbol} {timeframe} | "
                        f"Open: {kline['o']} High: {kline['h']} Low: {kline['l']} Close: {kline['c']} Volume: {kline['v']}"
                    )
                    payload = {"timestamp": int(kline['t']), "open": kline['o'], "high": kline['h'], "low": kline['l'], "close": kline['c'], "volume": kline['v'], "symbol": symbol, "timeframe": timeframe}
                    self.producer.send(f"{self.topic_prefix}{timeframe}-{symbol}", value=payload)
                    if self.config['producer']['save_to_disk']:
                        csv_row = {"Timestamp": pd.to_datetime(payload['timestamp'], unit='ms', utc=True), "Open": float(payload['open']), "High": float(payload['high']), "Low": float(payload['low']), "Close": float(payload['close']), "Volume": float(payload['volume'])}
                        buffer_key = f"{symbol}_{timeframe}"
                        if buffer_key in self.kline_save_buffers:
                            self.kline_save_buffers[buffer_key].append(csv_row)
                            if len(self.kline_save_buffers[buffer_key]) >= self.save_batch_size:
                                self._save_buffer_to_disk(buffer_key)
        except Exception as e: logger.error(f"Error in on_message: {e}", exc_info=True)

    def on_error(self, ws, error): logger.error(f"WebSocket Error: {error}")
    def on_close(self, ws, close_status_code, close_msg): logger.warning(f"WebSocket closed: {close_status_code} {close_msg}")
    def on_open(self, ws): logger.info(f"WebSocket connected to {ws.url}")

    def run(self):
        ws_base_url = "wss://fstream.binance.com/stream?streams="
        streams = [f"{s.lower()}@kline_{tf}" for s in self.symbols for tf in self.timeframes]
        combined_stream_url = ws_base_url + '/'.join(streams)
        
        logger.info(f"Connecting to WebSocket: {combined_stream_url}")
        ws_app = websocket.WebSocketApp(combined_stream_url, on_open=self.on_open, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close)
        
        wst = threading.Thread(target=ws_app.run_forever); wst.daemon = True; wst.start()
        try: shutdown_flag.wait()
        finally:
            logger.info("Shutting down producer..."); ws_app.close()
            for buffer_key in self.kline_save_buffers:
                if self.kline_save_buffers[buffer_key]: self._save_buffer_to_disk(buffer_key)
            if self.producer: self.producer.flush(timeout=10); self.producer.close()
            logger.info("Producer shut down gracefully.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    producer = BinanceDataProducer(config_path=CONFIG_PATH)
    producer.run()