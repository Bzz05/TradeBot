import json
from websocket import create_connection

def stream_ohlcv(pair: str, interval: str):
    print(f"Connection : {pair} {interval}")
    stream = f"{pair.lower()}@kline_{interval}"
    endpoint = f"wss://stream.binance.com:9443/ws/{stream}"
    ws = create_connection(endpoint)
    try:
        while True:
            message = ws.recv()
            data = json.loads(message)
            k = data["k"]
            ohlcv = {
                "open_time":    k["t"],
                "open":         float(k["o"]),
                "high":         float(k["h"]),
                "low":          float(k["l"]),
                "close":        float(k["c"]),
                "volume":       float(k["v"]),
                "close_time":   k["T"],        
                "is_final":     k["x"],       
            }

            yield ohlcv

    except KeyboardInterrupt:
        print("Stop Connection")
    finally:
        ws.close()



# how to use
# stream_ohlcv("pair_name","interval/candle duration")
if __name__ == "__main__":
    for bar in stream_ohlcv("ETHUSDT", "5m"):
        print(bar)