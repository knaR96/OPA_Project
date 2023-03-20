#!/usr/bin/env python
# coding: utf-8

from binance.client import Client
import pandas as pd
import os
import time
import datetime
from datetime import datetime, timedelta
import os
import io
import psycopg2
# Récupérer le mot de passe de la base de données PostgreSQL
conn_hist=psycopg2.connect(host="postgres_hist",
                                 database="historicdb",
                                user="usr",
                                password="pwd")
cursor_hist = conn_hist.cursor()

pairusdtlist = ['btcusdt', 'ethusdt', 'bnbusdt', 'bccusdt', 'neousdt',
                'ltcusdt', 'qtumusdt', 'adausdt', 'xrpusdt', 'eosusdt',
                'tusdusdt', 'iotausdt', 'xlmusdt', 'ontusdt', 'trxusdt']

create_table_query = """
CREATE TABLE IF NOT EXISTS symbols (
    id SERIAL PRIMARY KEY,
    name VARCHAR(10) NOT NULL
);
"""
cursor_hist.execute(create_table_query)

for pair in pairusdtlist:
    insert_query = f"INSERT INTO symbols (name) VALUES ('{pair}')"
    cursor_hist.execute(insert_query)

#conn_hist.commit()
start = "01 january 2017"
end = (datetime.now() - timedelta(days=1)).strftime('%d %B %Y')
#def get_historical_data(pair, interval, lookback, end):
for pair in pairusdtlist:    # Get historical data from API
    pair_hist=pair.upper()
    Hist_data = Client().get_historical_klines(pair_hist, Client.KLINE_INTERVAL_1DAY, start, end)
    df = pd.DataFrame(Hist_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote-av', 'trades', 'tb_base-av', 'tb_quote_av', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']]
    df.timestamp =pd.to_datetime(df["timestamp"], unit='ms', utc = True).map(lambda x: x.tz_convert('Europe/Paris'))
    df.open = df.open.astype(float).round(5)
    df.close = df.close.astype(float).round(5)
    df.high = df.high.astype(float).round(5)
    df.low = df.low.astype(float).round(5)
    df.volume = df.volume.astype(float).round(5)
    df.trades = df.trades.astype(float).round(5)

    # Create table for the pair if it does not exist
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {pair} (
        id SERIAL PRIMARY KEY,
        pair_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        open NUMERIC(30, 8) NOT NULL,
        high NUMERIC(30, 8) NOT NULL,
        low NUMERIC(30, 8) NOT NULL,
        close NUMERIC(30, 8) NOT NULL,
        volume NUMERIC(30, 8) NOT NULL,
        trades INTEGER NOT NULL,
        FOREIGN KEY (pair_id) REFERENCES symbols(id)
    );
    """
    cursor_hist.execute(create_table_query)

    # Get symbol ID from symbols table
    select_query = f"SELECT id FROM symbols WHERE name = '{pair}'"
    cursor_hist.execute(select_query)
    result = cursor_hist.fetchone()
    if result:
        pair_id = result[0]
    else:
        print(f"The pair {pair} was not found in the symbols table")
    df['pair_id'] =pair_id
    
    # Prepare the data as a list of tuples

    # Insert data into pair table
    for i, row in df.iterrows():
        insert_query = f"""
        INSERT INTO {pair} (pair_id, timestamp, open, high, low, close, volume, trades) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor_hist.execute(insert_query, (row['pair_id'], row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume'], row['trades']))

conn_hist.commit()   
conn_hist.close()




 
