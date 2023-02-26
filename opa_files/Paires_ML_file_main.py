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
import sys
sys.path.append('./opa_files') # à changer en cas de modification
from opa_files.forecasting_file import import_df_paire_for_forecast , model_training_for_forecast
from joblib import dump

exchange_info = Client().get_exchange_info()
dic = exchange_info['symbols']
symbols_list = []
for element in range(len(dic)):
    symbols_list.append(dic[element]['symbol'])
pairusdtlist = [symbols_list[i] for i in range(len(symbols_list)) if symbols_list[i].endswith('USDT')]
pairusdtlist = pairusdtlist[:15]

def main():
    dic_df_test = {} # utilisée pour la validation
    for pair in pairusdtlist:
        df_train , df_test, df = import_df_paire_for_forecast(pair)
        dic_df_test[f'{pair}'] = df_test
        model_pair_train= model_training_for_forecast(df_train)
        model = model_training_for_forecast(df)
        joblib_file_train= f'C:/Users/kevin/anaconda3/envs/prophet39/opa_files/pkl_files/model_train/{pair}.pkl'
        joblib_file= f'C:/Users/kevin/anaconda3/envs/prophet39/opa_files/pkl_files/model/{pair}.pkl'
        dump(model_pair_train,joblib_file_train)
        dump(model,joblib_file)

if __name__ == '__main__':
    main()