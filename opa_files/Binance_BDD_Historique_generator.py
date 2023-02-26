#!/usr/bin/env python
# coding: utf-8

from binance.client import Client
import pandas as pd
import os
import ccxt
import time
import datetime
import ta   # permet de definir tous les indicateurs qui existent 
import sqlalchemy
from datetime import datetime, timedelta
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import Table, Column, Integer, String, Float, Date, ForeignKey, MetaData, create_engine, text, inspect, exists, func
from Paires_ML_file_main import pairusdtlist


engine = create_engine('sqlite:///BinanceBDDHistorique.db')
conn = engine.connect()
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

class Symbol(Base):
    __tablename__ = 'Symbols'

    id = Column(Integer,primary_key=True)
    name = Column(String(10))
Base.metadata.create_all(engine)

session = Session()
for pair in pairusdtlist:
    existing_pair = session.query(Symbol).filter_by(name=pair).first()
    if not existing_pair:
        if pair in pairusdtlist :
            session.add(Symbol(name=pair))
        else :
            print("La paire de cryptomonaie demandée n'existe pas")
            break
session.commit()

def get_historical_data(pair, interval,lookback, end):
    Hist_data = Client().get_historical_klines(pair, interval, lookback, end)
    frame = pd.DataFrame(Hist_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote-av', 'trades', 'tb_base-av', 'tb_quote_av', 'ignore'])
    frame = frame[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']]
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], unit='ms')
    frame['date'] = frame['timestamp'].dt.date
    frame['time'] = frame['timestamp'].dt.time        
    return frame

class_registry = {}
for pair in pairusdtlist:
    pair_symbol = session.query(Symbol).filter_by(name=pair).first()
    if sqlalchemy.inspect(engine).has_table(pair):
        stmt = text("SELECT Timestamp FROM " +pair+ " ORDER BY id DESC LIMIT 1;")
        result = conn.execute(stmt)
        t = result.fetchall()
        t = str(t).strip('[]').strip('()')[1:-2]
        start = (datetime.strptime(t, '%Y-%m-%d') + timedelta(days=1)).strftime('%d %B %Y')
    else:
        start = "01 january 2017"
    end = (datetime.now() - timedelta(days=1)).strftime('%d %B %Y')
    data = get_historical_data(pair, Client.KLINE_INTERVAL_1DAY, start, end)
    # Créer une classe pour chaque crypto avec un nom spécifique à la paire
    class_name = pair
    CryptoData = type(class_name, (Base,), {
        '__tablename__': pair,
        'id': Column(Integer, primary_key=True),
        'timestamp': Column(Date),
        'open': Column(Float),
        'high': Column(Float),
        'low': Column(Float),
        'close': Column(Float),
        'volume': Column(Float),
        'trades': Column(Float),
        'pair_id': Column(Integer, ForeignKey('Symbols.id'))
    })
    class_registry[pair] = CryptoData
    Base.metadata.create_all(engine)
    session = Session()
    for i, row in data.iterrows():
        session.add(CryptoData(timestamp=row['timestamp'], open=row['open'], high=row['high'], low=row['low'], close=row['close'], volume=row['volume'], trades=row['trades'], pair_id=pair_symbol.id))
    session.commit()
