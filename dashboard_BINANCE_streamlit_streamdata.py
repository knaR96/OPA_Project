#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
from Kline_graph_plot import candelistick_plot
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'
from plotly.subplots import make_subplots
import sqlite3, sqlalchemy
import time
import datetime
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
with open(r"C:\Users\kevin\1_FORMATION_DE\OPA_BINANCE_streamlit\style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #101f61;'>DASHBOARD CRYPTOMONAIE BINANCE</h1>", unsafe_allow_html=True)

#Initialisation pour récupérer les valeurs de choix de paires
conn = sqlite3.connect('/Users/kevin/1_FORMATION_DE/project-env/BinanceBDDStream.db')
c = conn.cursor()
c.execute('SELECT * FROM Crypto_streamdata')
df = pd.DataFrame(c.fetchall(), columns= ["time","symbol","open","close","high","low","volume","trade"])   
paires = list(df.symbol.unique())
df_dic ={} 
for paire in paires: 
    df_paire = df[df["symbol"] == paire]
    df_dic[paire] = df_paire

# Sidebar options
Paire = st.sidebar.selectbox('Choix de la paire', options= list(df_dic.keys()) )
time_laps = st.sidebar.selectbox("Choisir la plage d'affichage", options= ["1m","3m","5m","15m","1h","2h","5h"])
dic_timelaps = {"1m":60,"3m":180,"5m":300,"15m":900,"1h":3600,"2h":7200, "5h":18000}

variation = st.sidebar.selectbox('Choisir la plage de variation des cours', options= ["10s","30s","1m","3m","5m","10m","1h"])
dic_variation = {"10s":10,"30s":30,"1m":60,"3m":180,"5m":300,"10m":600, "1h":3600}

ma1 = st.sidebar.number_input(
    'Moving Average 1',
    value = 250,
    min_value = 1,
    max_value = 900,
    step = 1,    
)

ma2 = st.sidebar.number_input(
    'Moving Average 2',
    value = 500,
    min_value = 1,
    max_value = 900,
    step = 1,    
)
placeholder =st.empty()

def main():
    
    while True :
        #deuxième appel de la fonction pour récupérer les données stream
        conn = sqlite3.connect('/Users/kevin/1_FORMATION_DE/project-env/BinanceBDDStream.db')
        c = conn.cursor()
        c.execute('SELECT * FROM Crypto_streamdata')
        df = pd.DataFrame(c.fetchall(), columns= ["time","symbol","open","close","high","low","volume","trade"])   
        paires = list(df.symbol.unique())
        for paire in paires: 
            df_paire = df[df["symbol"] == paire]
            df_dic[paire] = df_paire
            
        with placeholder.container():
            
            # connexion à la bade de données relationnelle et production du df pour le plot
            today =datetime.date.today().strftime('%A %d %B %Y')
            current_time = datetime.datetime.now()
            lastvariation = str(round((list(df_dic[Paire]["close"])[-1] - list(df_dic[Paire]["close"])[- dic_variation[variation]])*100/(list(df_dic[Paire]["close"])[-1]),2)) +"%"
            max = df_dic[Paire]["high"][-dic_timelaps[time_laps]:].max()
            col1, col2, col3 = st.columns(3)
            col1.metric(label = str(today), value = str(current_time.time().strftime('%X')), delta = " ")
            col2.metric(label = "Prix maximal sur la période choisie" , value = max , delta = " ")
            col3.metric(label = Paire , value = list(df_dic[Paire]["close"])[-1] , delta = lastvariation)
            
            # affichage du dashboard
            st.plotly_chart(
                candelistick_plot(df_dic[Paire], ma1, ma2, Paire, timelaps = dic_timelaps[time_laps]),
                use_container_width = True,)

            time.sleep(1)


if __name__ == '__main__':
  main()                           
            