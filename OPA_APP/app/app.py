#!/usr/bin/env python
# coding: utf-8
#pip install streamlit 
#pip install PILdate

import streamlit as st
import pandas as pd
from Plot_Kline_graph_file import candelistick_plot , typical_price_plot
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'
from plotly.subplots import make_subplots
import time
import datetime
import pytz
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from prophet.plot import plot_plotly , plot_components_plotly
import plotly.express as px
pio.renderers.default='browser'
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu as om
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PIL import Image
from prophet import Prophet          
from joblib import load
from streamlit_lottie import st_lottie
import requests

#########################
st.set_page_config(layout="wide",initial_sidebar_state='expanded')

padding_top = 0

st.markdown(f"""
    <style>
        .reportview-container .main .block-container{{
            padding-top: {padding_top}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)

with open(r"style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 1rem;
                    padding-left: 1rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True) 

hide_st_style = """
        <style>
        #MainMenu {visibility:hidden;}
        footer{visibility:hidden;}
        header {visibility:hidden;}
        </style>
        """

st.markdown("""
  <style>
    .css-znku1x.e16nr0p33 {
      margin-top: -1px;
    }
  </style>
""", unsafe_allow_html=True)
st.markdown(hide_st_style, unsafe_allow_html=True) 




# Créer une instance de connexion à la base de données stream_db
#from sqlalchemy import create_engine

# Créer une instance de connexion à la base de données historic_db
import os
import psycopg2
# Récupérer le mot de passe de la base de données PostgreSQL
#password = os.getenv("POSTGRES_PASSWORD")

#Initialisation pour récupérer les valeurs de choix de paires

pairusdtlist=['btcusdt', 'ethusdt', 'bnbusdt','neousdt',
               'ltcusdt', 'qtumusdt', 'adausdt', 'xrpusdt', 'eosusdt',
                'tusdusdt', 'iotausdt', 'xlmusdt', 'ontusdt', 'trxusdt']
paires=pairusdtlist


#models_train = {}
#models = {}
#for paire in paires: 
    #paire_up = paire.upper()
    #models_train[paire] = load(f'pkl_files/model_train/{paire_up}.pkl')
    #models[paire] = load(f'pkl_files/model/{paire_up}.pkl')

def model_training_for_forecast(df_model_train:pd.DataFrame): 
    prophet_model = Prophet(changepoint_prior_scale = 0.1, seasonality_prior_scale = 0.01) #changepoint_prior_scale = 0.1, seasonality_prior_scale = 0.01 best parameters for BTC
    model_fiting = prophet_model.fit(df_model_train)
    return prophet_model

def price_forecast(model,forecasting_period : int): # forecasting_period in days
    future = model.make_future_dataframe(periods=forecasting_period , freq='D')
    forecast = model.predict(df=future)
    return forecast

def import_df_paire_for_forecast(paire_name:str):
    con_hist=psycopg2.connect(
                            host="hist-db-container",
                            database="historicdb",
                            user="usr",
                            password="pwd")
    cursor_hist = con_hist.cursor()
    select_query = f"SELECT * FROM {paire_name} ;"
    cursor_hist.execute(select_query)
    result_hist=cursor_hist.fetchall()
    df = pd.DataFrame(result_hist, columns= ["id","pairs_id","time","open","high","low","close","volume","trade"])   
    df = df.set_index("id")
    df.time = pd.to_datetime(df["time"])
    date = df["time"]
    df['typical_price'] = df.apply(lambda x : (x["open"]+ x["close"]+ x["high"])/3, axis=1)
    df = df[["time","typical_price"]]
    df = df.set_index('time')
    df = df.reset_index().rename(columns={'time':'ds', 'typical_price':'y'}) # L'appellation des colonnes telle que définie à cette ligne est necessaire pour le lancement de fbprophet
    df_model = df
    df_model['y'] = np.log(df_model['y'].astype(float))
    df_model_train = df_model[:-120]
    df_model_test = df_model[-120:]
    return (df_model_train ,df_model_test, df_model)

conn_stream=psycopg2.connect(host="stream-db-container",
                                 database="streamdb",
                                user="usr",
                                password="pwd")
cursor_stream = conn_stream.cursor()
cursor_stream.execute("select * from streamtable;")
result_stream=cursor_stream.fetchall()
df_strema = pd.DataFrame(result_stream, columns=["time", "symbol", "open", "close", "high", "low", "volume", "trade"])
df_dic = {}
for paire in paires:
        paire_sl=paire.upper()
        df_paire = df_strema[df_strema["symbol"] == paire_sl]
        df_dic[paire_sl] = df_paire

# For lottie
def load_lottieurl(url : str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def main():
    
    with st.sidebar.container():
        image = Image.open(r"Crypto_image.jpg")
     
        st.image(image, use_column_width=True)
    
    st.sidebar.markdown(
            f'''
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {1}rem;
                }}
                .reportview-container .main .block-container {{
                    padding-top: {1}rem;
                }}

                .sidebar .sidebar-content {{
                    background: url("https://www.cafedelabourse.com/wp-content/uploads/2022/09/crypto-monnaie-bitcoin-investir.jpg")
                }}
                               
                
            </style>
            ''',unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #101f61;'>CryptoBot Binance</h1>", unsafe_allow_html=True)
    menu = om("Menu",["Page d'accueil","Données stream","Données historiques"], icons=["arrow-down-square-fill","bar-chart-line",'archive-fill'],menu_icon="cast",default_index = 0, orientation ="horizontal",
              styles={
            "container": {"padding": "0!important", "background-color": "#e3e3e3"},
            "icon": {"color": "white", "font-size": "20px"}, 
            "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#000000"},
            "nav-link-selected": {"background-color": "#000000"},
    }
              )
    if  menu == "Page d'accueil":
        image = Image.open(r"Crypto_image_2.jpg")
        row_spacer1, row, row_spacer2 = st.columns((1.2, 7, 1))
        lottie_crypto= load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_gonpfxdh.json")
        with row:
            st_lottie(lottie_crypto,speed =1,
                      reverse = False,
                      loop = True,
                      quality = "low",
                      height=750,
                      width=1100,
                      key=None,)
        st.sidebar.markdown(" ")
        st.sidebar.markdown("---")
        st.sidebar.markdown("<h2 style='text-align: left; color: #fafbfc;'> MEMBRES DU GROUPES : </h2>", unsafe_allow_html=True)
        st.sidebar.markdown("<h3 style='text-align: left; color: #fafbfc ;'> - Fatma AIDI  </h3>", unsafe_allow_html=True)
        st.sidebar.markdown("<h3 style='text-align: left; color: #fafbfc ;'> - Hugo MENAGE </h3>", unsafe_allow_html=True)
        st.sidebar.markdown("<h3 style='text-align: left; color: #fafbfc ;'> - Kevin ROGER </h3>", unsafe_allow_html=True)
        st.sidebar.markdown("<h3 style='text-align: left; color: #fafbfc;'> - Melissa TELLIER </h3>", unsafe_allow_html=True)
        st.sidebar.markdown("<h2 style='text-align: left; color: #fafbfc;'> MENTOR PROJET : </h2>", unsafe_allow_html=True)
        st.sidebar.markdown("<h3 style='text-align: left; color: #fafbfc;'> Dimitri C. </h3>", unsafe_allow_html=True)
        st.sidebar.markdown("---")

            
    if  menu == "Données stream":
        #df_dic = get_stream_data(paires)
        # Sidebar options

        st.sidebar.markdown("<h1 style='text-align: left; color: #fafbfc;'> PARAMETRES </h1>", unsafe_allow_html=True)
        Paire = st.sidebar.selectbox('Choix de la paire', options= list(df_dic.keys()))
        Graph_type = st.sidebar.selectbox('Choix le type de graphique', options= ["K-line","Typical price"])
        dic_graph = {"K-line": candelistick_plot,"Typical price":typical_price_plot}
        time_laps = st.sidebar.selectbox("Choisir la plage d'affichage", options= ["1m","3m","5m","15m","1h","2h","5h"])
        dic_timelaps = {"1m":60,"3m":180,"5m":300,"15m":900,"1h":3600,"2h":7200, "5h":18000}
        variation = st.sidebar.selectbox('Choisir la plage de variation des cours', options= ["10s","30s","1m","3m","5m","10m","1h"])
        dic_variation = {"10s":10,"30s":30,"1m":60,"3m":180,"5m":300,"10m":600, "1h":3600}

        ma1 = st.sidebar.number_input(
            'Moyenne mobile 1',
            value = 250,
            min_value = 1,
            max_value = 900,
            step = 1,    
        )

        ma2 = st.sidebar.number_input(
            'Moyenne mobile 2',
            value = 500,
            min_value = 1,
            max_value = 900,
            step = 1,    
        )
        placeholder =st.empty()

        
        while True :
            #deuxième appel de la fonction pour récupérer les données stream
            conn_stream=psycopg2.connect(host="stream-db-container",
                                 database="streamdb",
                                user="usr",
                                password="pwd")
            cursor_stream = conn_stream.cursor()
            cursor_stream.execute("select * from streamtable;")
            result_stream=cursor_stream.fetchall()
            df_strema = pd.DataFrame(result_stream, columns=["time", "symbol", "open", "close", "high", "low", "volume", "trade"])
            df_dc = {}
            #paires = list(df_strema.symbol.unique())
            for paire in paires:
                paire_sl=paire.upper()
                df_paire = df_strema[df_strema["symbol"] == paire_sl]# le probleme les symbole majuscule
                df_dc[paire] = df_paire
            
            
            with placeholder.container():
                
                # connexion à la bade de données relationnelle et production du df pour le plot
                today =datetime.date.today().strftime('%A %d %B %Y')
                current_time = datetime.datetime.now(pytz.timezone('Europe/Brussels'))
                lastvariation = str(round((list(df_dc[Paire.lower()]["close"])[-1] - list(df_dc[Paire.lower()]["close"])[- dic_variation[variation]])*100/(list(df_dc[Paire.lower()]["close"])[-1]),2)) +"%"
                max = df_dc[Paire.lower()]["high"][-dic_timelaps[time_laps]:].max()
                max_str = str(round(max, 2))
                col1, col2, col3 = st.columns(3)
                col1.metric(label = str(today), value = str(current_time.time().strftime('%X')), delta = " ")
                col2.metric(label = "Prix maximal sur la période choisie" , value = max_str+' $', delta = " ")
                lis_str = str(round(list(df_dc[Paire.lower()]["close"])[-1], 2))
                col3.metric(label = Paire , value = lis_str + " $", delta = lastvariation)
                
                # affichage du dashboard
                st.plotly_chart(
                    dic_graph[Graph_type](df_dc[Paire.lower()], ma1, ma2, Paire, timelaps = dic_timelaps[time_laps]),
                    use_container_width = True,)

                time.sleep(1)
    
    elif  menu == "Données historiques":
        st.markdown("<h4 style='text-align: left; color: #101f61;'> DONNEES HISTORIQUES </h4>", unsafe_allow_html=True)
        
        st.sidebar.markdown("<h3 style='text-align: left; color: #fafbfc;'> DONNES HISTORIQUES </h3>", unsafe_allow_html=True)
        pairusdtlistupper = [paire.upper() for paire in paires]
        Graph_type = st.sidebar.selectbox('Choix la paire', options= pairusdtlistupper)
        st.markdown(str(Graph_type).upper()) 
        df_train , df_test, df = import_df_paire_for_forecast(Graph_type)
        historicaldata = (np.exp(df['y']))
        date_historique = df.ds
        df_historique = pd.DataFrame(list(zip(date_historique,historicaldata)), columns=["date","historicaldata"])
        df_historique["date"]=pd.to_datetime(df_historique["date"]).dt.date
        # fonction pour affichage calendrier
        last_day = df_historique["date"][len(df_historique)-1]#initialisation
        initialday = datetime.date.today() + datetime.timedelta(days=-61) #initialisation
        start_date = st.sidebar.date_input("Date de début",initialday)
        end_date = st.sidebar.date_input('Date de fin', last_day)
        reset = st.sidebar.button('Réinitialiser les dates')
        if reset :
            start_date = initialday 
            end_date = last_day
        
        if end_date <= start_date :
            st.sidebar.warning("End date doit être posterieure à Start date !", icon="⚠️")
        elif df_historique[df_historique["date"]==start_date].index.values[0] < 0 :
            st.sidebar.write(f'Start date doit être postérieur ou égale à {df.iloc[0].date} !')
            st.sidebar.warning(f'Start date doit être postérieur ou égale à {df.iloc[0].date} !', icon="⚠️")
            start_date = initialday 
            end_date = last_day
        else :
            df_historique = df_historique.iloc[df_historique[df_historique["date"]==start_date].index.values[0]: df_historique[df_historique["date"]==end_date].index.values[0]+1]

#add cryptocurrency max min and mean Price
        max_currency = df_historique["historicaldata"].max()
        min_currency = df_historique["historicaldata"].min()
        mean_currency = df_historique["historicaldata"].mean()
        
        col1, col2, col3= st.columns(3)
        col1.metric(label = 'PRIX MAXIMAL', value = str(round(max_currency,2))+' $', delta = "")
        col2.metric(label = 'PRIX MINIMAL', value = str(round(min_currency,2))+' $', delta = "")
        col3.metric(label = 'PRIX MOYEN', value = str(round(mean_currency,2))+' $', delta = "")


        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
                go.Line(x=df_historique["date"], y= df_historique["historicaldata"], name="historical data"),
                secondary_y=False,
            )

        # Add figure title
        fig.update_layout(
                    title_text="",
                    legend_title_font_size=2,
                )

                # Set x-axis title
        fig.update_xaxes(title_text="Date")

                # Set y-axes titles
        fig.update_yaxes(title_text=" ", secondary_y=False)
        fig.update_yaxes(title_text=" ", secondary_y=False)
                
        fig['layout']['xaxis']['title'] = 'DATE'
        fig['layout']['yaxis']['title'] = 'PRIX ($)'
        
        fig.update_xaxes(
        rangeslider_visible = True,
    )
            
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
     
#########################
            
    ### Phase de validation et metrics    
        st.sidebar.markdown("<h3 style='text-align: left; color: #fafbfc;'> PERFORMANCE DU MODELE </h3>", unsafe_allow_html=True) 
        validation_period = st.sidebar.slider("Nombre de jours pour la validation",1, 120, 60)
        error_limit = st.sidebar.slider("Erreur maximale admissible (%)",0,10,5)
        
        df_train , df_test, df = import_df_paire_for_forecast(Graph_type)
        model_validation_forecast = model_training_for_forecast(df_model_train=df_train)
        validation_forcast = price_forecast(model_validation_forecast,120)
        validation_price = np.exp(validation_forcast['yhat'])
        validation_date = validation_forcast["ds"]
        validation_df_forecast = pd.DataFrame(list(zip(validation_date,validation_price)), columns=["date","Price_forecast"])
        test_price = (np.exp(df_test['y']))
        test_date = df_test["ds"]
        df_test= pd.DataFrame(list(zip(test_date,test_price)), columns=["Date","Real_price"])
        df_metrics = pd.concat([validation_df_forecast[-120:].reset_index(drop= True),df_test.loc[:].reset_index(drop= True)], axis=1,)
        #df_metrics =df_metrics.drop(columns='Date')
        df_metrics = df_metrics[:validation_period-1]
        df_metrics["real_price_+x%"] = df_metrics["Real_price"]*(1+error_limit/100)
        df_metrics["real_price_-x%"] = df_metrics["Real_price"]*(1-error_limit/100)
        RMSE = np.sqrt(mean_squared_error(df_metrics["Real_price"].astype(float),df_metrics["Price_forecast"].astype(float)))
        MAE = mean_absolute_error(df_metrics["Real_price"].astype(float),df_metrics["Price_forecast"].astype(float))
        Mean_Realprice = df_metrics["Real_price"].astype(float).mean()
        Pourcentage_RMSE = RMSE*100/Mean_Realprice
        Pourcentage_MAE = MAE*100/Mean_Realprice
        st.markdown("-----------------------------")
        st.markdown("<h4 style='text-align: left; color: #101f61;'> EVALUATION DU MODELE </h4>", unsafe_allow_html=True)
        st.markdown(str(Graph_type).upper()) 
        col1 , col2, col3 = st.columns(3)
        col1.metric("MAE", value =round(MAE,2), delta="", delta_color="inverse")
        col2.metric("RMSE", value =round(RMSE,2), delta="", delta_color="inverse")
        col3.metric("PRIX MOYEN - DONNEES HISTORIQUES", value = str(round(Mean_Realprice,2))+" $" , delta = "")

# Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
                go.Line(x=df_metrics["date"][:], y= df_metrics["Price_forecast"][:], name="PREVISON ($)"),
                secondary_y=False,
            )

        fig.add_trace(
            go.Scatter(x=df_metrics["Date"][:], y= df_metrics["Real_price"][:], name="PRIX REELS ($)" ),
                secondary_y=False
        )
        
        fig.add_trace(
            go.Line(x=df_metrics["date"][:], y= df_metrics["real_price_+x%"][:], name=f'Error +{error_limit}%',mode='markers+lines', marker= dict(symbol= 'cross') ,line=dict(color='#F7230C')),
            secondary_y=False
            )
        
        fig.add_trace(
            go.Line(x=df_metrics["date"][:], y= df_metrics["real_price_-x%"][:], name=f'Error -{error_limit}%',mode='markers+lines', marker= dict(symbol= 'cross') , line=dict(color='#F7230C')),
            secondary_y=False
            )

        #Add figure title
        #fig.update_layout(
        #          title_text=f"PAIRE : {Graph_type}",
        #       legend_title_font_size=2,
        #   )

        # Set x-axis title
        fig.update_xaxes(title_text="Date")

                # Set y-axes titles
        fig.update_yaxes(title_text=" ", secondary_y=False)
        fig.update_yaxes(title_text=" ", secondary_y=False)
                
        fig['layout']['xaxis']['title'] = 'DATE'
        fig['layout']['yaxis']['title'] = 'PRIX ($)'
        
        fig.update_xaxes(
        rangeslider_visible = True,
    )
            
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
     

### Valeurs futurs / futur forcast

        st.sidebar.markdown("<h3 style='text-align: left; color: #fafbfc;'> PREVISION DES FUTURS PRIX </h3>", unsafe_allow_html=True) 
        forecast_period = st.sidebar.slider("Période à prédire (Jours)",1, 20, 10)
        model_forecast = model_training_for_forecast(df_model_train=df)
        forcast = price_forecast(model_forecast,forecast_period)
        futur_price = np.exp(forcast['yhat'])
        forcast_date = forcast["ds"]
        df_forecast = pd.DataFrame(list(zip(forcast_date,futur_price)), columns=["date","Futurs_price"])
        st.markdown("-----------------------------")
        st.markdown("<h4 style='text-align: left; color: #101f61;'> PREVISION DES FUTURS PRIX </h4>", unsafe_allow_html=True)
        st.markdown(str(Graph_type).upper()) 
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])


        df_historique = pd.DataFrame(list(zip(date_historique,historicaldata)), columns=["date","historicaldata"])
        df_historique["date"]=pd.to_datetime(df_historique["date"]).dt.date
        # Add traces
        visibility = 30
        fig.add_trace(
                go.Line(x=df_historique["date"][-visibility:], y= df_historique["historicaldata"][-visibility:], name="PRIX REELS ($)"),
                secondary_y=False,
            )
        fig.add_trace(
                go.Scatter(x=df_forecast["date"][-forecast_period-visibility:], y= df_forecast["Futurs_price"][-forecast_period-visibility:], name="PREVISON ($)"),
                secondary_y=False
                )

        # Add figure title
        fig.update_layout(
                    title_text="",
                    legend_title_font_size=2,
                )

                # Set x-axis title
        fig.update_xaxes(title_text="Date")

                # Set y-axes titles
        fig.update_yaxes(title_text=" ", secondary_y=False)
        fig.update_yaxes(title_text=" ", secondary_y=False)
                
        fig['layout']['xaxis']['title'] = 'DATE'
        fig['layout']['yaxis']['title'] = 'PRIX ($)'
        
        fig.update_xaxes(
        rangeslider_visible = True,
    )
            
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
  
        


if __name__ == '__main__':
  main()                           
            