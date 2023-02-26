
#pip install websocket-client[test]

#pip install rel

import websocket , json
import rel
import pandas as pd
import sqlalchemy
from binance.client import Client
import time
from datetime import datetime
from dateutil.tz import tzutc, tzlocal

"""
response example : 
{
  "e": "kline",     // Event type
  "E": 123456789,   // Event time
  "s": "BNBBTC",    // Symbol
  "k": {
    "t": 123400000, // Kline start time
    "T": 123460000, // Kline close time
    "s": "BNBBTC",  // Symbol
    "i": "1m",      // Interval
    "f": 100,       // First trade ID
    "L": 200,       // Last trade ID
    "o": "0.0010",  // Open price
    "c": "0.0020",  // Close price
    "h": "0.0025",  // High price
    "l": "0.0015",  // Low price
    "v": "1000",    // Base asset volume ***
    "n": 100,       // Number of trades --> à ajouter à la liste
    "x": false,     // Is this kline closed?
    "q": "1.0000",  // Quote asset volume
    "V": "500",     // Taker buy base asset volume
    "Q": "0.500",   // Taker buy quote asset volume
    "B": "123456"   // Ignore
  }
}
"""

engine = sqlalchemy.create_engine('sqlite:///BinanceBDDStream.db')


def on_message(ws, message):
    json_message = json.loads(message)
    df = pd.DataFrame([json_message["k"]])
    df =df[df['x'].astype(str) == 'True']
    df =df.loc[:,['T','s','o','c','h','l','v','n']]
    df.columns = ["time","symbol","open","close","high","low","volume","trade"]
    df.time =pd.to_datetime(df["time"], unit='ms', utc = True).map(lambda x: x.tz_convert('Europe/Paris'))
    df.close = df.close.astype(float)
    df.high = df.high.astype(float)
    df.low = df.low.astype(float)
    df.volume = df.volume.astype(float)
    df.trade = df.trade.astype(float)
    df.to_sql('Crypto_streamdata', engine, if_exists='append', index=False)
def on_close(ws):
    print("Connection closed")


def binance_kline_stream_data(symbols = [], interval="1m"): 
    for symbol in symbols :
        socket = f'wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}'
        ws = websocket.WebSocketApp(socket, on_message=on_message,on_close=on_close)
        ws.run_forever(dispatcher = rel, reconnect=60)
    rel.signal(2, rel.abort)  # Keyboard Interrupt  
    rel.dispatch()


# définition des paires à récupérer 
exchange_info = Client().get_exchange_info() 
dic = exchange_info['symbols']
symbols_list = []
for element in range(len(dic)):
    symbols_list.append(dic[element]['symbol'])
pairusdtlist = [symbols_list[i].lower() for i in range(len(symbols_list)) if symbols_list[i].endswith('USDT')][:25]


binance_kline_stream_data(symbols=pairusdtlist, interval="1s")











