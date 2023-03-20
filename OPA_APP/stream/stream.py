
#pip install websocket-client[test]

#pip install rel

import websocket , json 
import rel
import pandas as pd
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

import os
import psycopg2
# Récupérer le mot de passe de la base de données PostgreSQL
conn_live=psycopg2.connect(host="postgres_stream",
                                 database="streamdb",
                                user="usr",
                                password="pwd")
cursor_live = conn_live.cursor()
# création de la table "streamdb" avec les colonnes nécessaires
create_table_query = """
CREATE TABLE IF NOT EXISTS streamtable (
        time TIMESTAMP NOT NULL,
        symbol varchar(10),
        open NUMERIC(30, 4) NOT NULL,
        close NUMERIC(30, 4) NOT NULL,
        high NUMERIC(30, 4) NOT NULL,
        low NUMERIC(30, 4) NOT NULL,
        volume NUMERIC(30, 4) NOT NULL,
        trade INTEGER NOT NULL
);
"""
cursor_live.execute(create_table_query)

def on_message(ws, message):
    json_message = json.loads(message)
    df = pd.DataFrame([json_message["k"]])
    df =df[df['x'].astype(str) == 'True']
    df =df.loc[:,['T','s','o','c','h','l','v','n']]
    df.columns = ["time","symbol","open","close","high","low","volume","trade"]
    df.time =pd.to_datetime(df["time"], unit='ms', utc = True).map(lambda x: x.tz_convert('Europe/Paris'))
    df.close = df.close.astype(float).round(3)
    df.high = df.high.astype(float).round(3)
    df.low = df.low.astype(float).round(3)
    df.volume = df.volume.astype(float).round(3)
    #df.trade = df.trade.astype(float)
    #df.symbol=df.symbol.lower()
    for i, row in df.iterrows():
        insert_query = f"""
        INSERT INTO streamtable (time, symbol, open,close, high, low,  volume, trade) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor_live.execute(insert_query, (row['time'], row['symbol'],row['open'],row['close'], row['high'], row['low'],  row['volume'], row['trade']))
    conn_live.commit()
def on_close(ws):
    print("Connection closed")


def binance_kline_stream_data(symbols = [], interval="1s"): 
    for symbol in symbols :
        socket = f'wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}'
        ws = websocket.WebSocketApp(socket, on_message=on_message,on_close=on_close)
        ws.run_forever(dispatcher = rel, reconnect=60)
    rel.signal(2, rel.abort)  # Keyboard Interrupt  
    rel.dispatch()

pairusdtlist=['btcusdt', 'ethusdt', 'bnbusdt', 'bccusdt', 'neousdt',
            'ltcusdt', 'qtumusdt', 'adausdt', 'xrpusdt', 'eosusdt',
             'tusdusdt', 'iotausdt', 'xlmusdt', 'ontusdt', 'trxusdt']

binance_kline_stream_data(symbols=pairusdtlist, interval="1s")
