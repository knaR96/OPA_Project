#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from Plot_Kline_graph_file import candelistick_plot , typical_price_plot
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'
from plotly.subplots import make_subplots
import sqlite3, sqlalchemy
import time
import datetime
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
            
from joblib import load
import sys
# setting path
sys.path.append('../opa_files') # à changer en cas de modification
from opa_files.Paires_ML_file_main import pairusdtlist 
from opa_files.forecasting_file import price_forecast, import_df_paire_for_forecast

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

with open(r"C:\Users\kevin\anaconda3\envs\prophet39\opa_files\streamlit_files\style.css") as f:
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



#Initialisation pour récupérer les valeurs de choix de paires
conn = sqlite3.connect('/Users/kevin/anaconda3/envs/prophet39/opa_files/streamlit_files/BinanceBDDStream.db')
c = conn.cursor()
c.execute('SELECT * FROM Crypto_streamdata')
df = pd.DataFrame(c.fetchall(), columns= ["time","symbol","open","close","high","low","volume","trade"])   
paires = list(df.symbol.unique())
df_dic ={} 
for paire in paires: 
    df_paire = df[df["symbol"] == paire]
    df_dic[paire] = df_paire

models_train = {}
models = {}
for paire in pairusdtlist: 
    models_train[paire] = load(f'C:/Users/kevin/anaconda3/envs/prophet39/opa_files/pkl_files/model_train/{paire}.pkl')
    models[paire] = load(f'C:/Users/kevin/anaconda3/envs/prophet39/opa_files/pkl_files/model/{paire}.pkl')

image = Image.open(r"C:\Users\kevin\anaconda3\envs\prophet39\opa_files\Crypto_image.jpeg")

def main():

    
    st.sidebar.markdown(
            f'''
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {1}rem;
                }}
                .reportview-container .main .block-container {{
                    padding-top: {1}rem;
                }}
            </style>
            ''',unsafe_allow_html=True)

    menu = om("Menu",["Front page","Stream data","Historical data"], icons=["arrow-down-square-fill","bar-chart-line",'archive-fill'],menu_icon="cast",default_index = 0, orientation ="horizontal",
              styles={
            "container": {"padding": "0!important", "background-color": "#e3e3e3"},
            "icon": {"color": "white", "font-size": "20px"}, 
            "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#000000"},
            "nav-link-selected": {"background-color": "#000000"},
    }
              )
    if  menu == "Front page":
        st.markdown("<h1 style='text-align: center; color: #101f61;'>DASHBOARD CRYPTOMONAIE BINANCE</h1>", unsafe_allow_html=True)
        row_spacer1, row, row_spacer2 = st.columns((0.8, 7, 1))
        with row:
            st.image(image, use_column_width='auto')
            
    if  menu == "Stream data":
        # Sidebar options
        st.sidebar.markdown("<h1 style='text-align: left; color: #fafbfc;'> PARAMETRES </h1>", unsafe_allow_html=True)
        Paire = st.sidebar.selectbox('Choix de la paire', options= list(df_dic.keys()) )
        Graph_type = st.sidebar.selectbox('Choix le type de graphique', options= ["K-line","Typical price"])
        dic_graph = {"K-line": candelistick_plot,"Typical price":typical_price_plot}
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

        
        while True :
            #deuxième appel de la fonction pour récupérer les données stream
            conn = sqlite3.connect('/Users/kevin/anaconda3/envs/prophet39/opa_files/streamlit_files/BinanceBDDStream.db')
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
                    dic_graph[Graph_type](df_dic[Paire], ma1, ma2, Paire, timelaps = dic_timelaps[time_laps]),
                    use_container_width = True,)

                time.sleep(1)
    
    elif  menu == "Historical data":
        st.markdown("<h4 style='text-align: left; color: #101f61;'> DONNEES HISTORIQUES </h4>", unsafe_allow_html=True) 
        st.sidebar.markdown("<h1 style='text-align: left; color: #fafbfc;'> DONNES HISTORIQUES </h1>", unsafe_allow_html=True)
        Graph_type = st.sidebar.selectbox('Choix la paire', options= pairusdtlist)
        df_train , df_test, df = import_df_paire_for_forecast(Graph_type)
        historicaldata = (np.exp(df['y']))
        date_historique = df.ds
        df_historique = pd.DataFrame(list(zip(date_historique,historicaldata)), columns=["date","historicaldata"])
        df_historique["date"]=pd.to_datetime(df_historique["date"]).dt.date
        # fonction pour affichage calendrier
        last_day = df_historique["date"][len(df_historique)-1]#initialisation
        initialday = datetime.date.today() + datetime.timedelta(days=-61) #initialisation
        start_date = st.sidebar.date_input("Start date",initialday)
        end_date = st.sidebar.date_input('End date', last_day)
        reset = st.sidebar.button('Reset dates')
        if reset :
            start_date = initialday 
            end_date = last_day
        
        if end_date <= start_date :
            st.sidebar.warning("End date doit être posterieure à Start date !", icon="⚠️")
        elif df_historique[df_historique["date"]==start_date].index.values[0] < 0 :
            st.sidebar.warning(f'Start date doit être postérieur ou égale à {df.iloc[0].date} !', icon="⚠️")
        else :
            df_historique = df_historique.iloc[df_historique[df_historique["date"]==start_date].index.values[0]: df_historique[df_historique["date"]==end_date].index.values[0]+1]
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
                go.Line(x=df_historique["date"], y= df_historique["historicaldata"], name="historical data"),
                secondary_y=False,
            )

        # Add figure title
        fig.update_layout(
                    title_text=f"DONNEES HISTORIQUE - PAIRE: {Graph_type}",
                    legend_title_font_size=2,
                )

                # Set x-axis title
        fig.update_xaxes(title_text="Date")

                # Set y-axes titles
        fig.update_yaxes(title_text=" ", secondary_y=False)
        fig.update_yaxes(title_text=" ", secondary_y=False)
                
        fig['layout']['xaxis']['title'] = 'DATE'
        fig['layout']['yaxis']['title'] = 'PRICE'
        
        fig.update_xaxes(
        rangeslider_visible = True,
    )
            
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
     
#########################
            
    ### Phase de validation et metrics    
        st.sidebar.markdown("<h1 style='text-align: left; color: #fafbfc;'> PERFORMANCE DU MODELE </h1>", unsafe_allow_html=True) 
        validation_period = st.sidebar.slider("Nombre de jours pour la validation",1, 60, 20)
        error_limit = st.sidebar.slider("Erreur maximale admissible (%)",0,10,5)
        validation_forcast = price_forecast(models_train[Graph_type],validation_period)
        validation_price = np.exp(validation_forcast['yhat'])
        validation_date = validation_forcast["ds"]
        validation_df_forecast = pd.DataFrame(list(zip(validation_date,validation_price)), columns=["date","Price_forecast"])
        test_price = (np.exp(df_test['y']))
        test_date = df_test["ds"]
        df_test= pd.DataFrame(list(zip(test_date,test_price)), columns=["Date","Real_price"])
        df_metrics = pd.concat([validation_df_forecast[-validation_period:].reset_index(drop= True),df_test.loc[:validation_period-1].reset_index(drop= True)], axis=1,)
        df_metrics =df_metrics.drop(columns='Date')
        df_metrics["real_price_+x%"] = df_metrics["Real_price"]*(1+error_limit/100)
        df_metrics["real_price_-x%"] = df_metrics["Real_price"]*(1-error_limit/100)
        RMSE = np.sqrt(mean_squared_error(df_metrics["Real_price"].astype(float),df_metrics["Price_forecast"].astype(float)))
        MAE = mean_absolute_error(df_metrics["Real_price"].astype(float),df_metrics["Price_forecast"].astype(float))
        Mean_Realprice = df_metrics["Real_price"].astype(float).mean()
        Pourcentage_RMSE = RMSE*100/Mean_Realprice
        Pourcentage_MAE = MAE*100/Mean_Realprice
        st.markdown("-----------------------------")
        st.markdown("<h4 style='text-align: left; color: #101f61;'> EVALUATION DU MODELE </h4>", unsafe_allow_html=True)
        st.markdown(f'PAIRE : {Graph_type}',unsafe_allow_html=True)  
        col1 , col2, col3 = st.columns(3)
        col1.metric("MAE", value =MAE, delta="", delta_color="inverse")
        col2.metric("RMSE", value =RMSE, delta="", delta_color="inverse")
        col3.metric("REAL PRICE MEAN VALUE", value = round(Mean_Realprice,2) , delta = "")
  
# Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
                go.Line(x=validation_df_forecast["date"][-validation_period:], y= validation_df_forecast["Price_forecast"][-validation_period:], name="Price_forecast"),
                secondary_y=False,
            )

        fig.add_trace(
            go.Scatter(x=df_test["Date"][:validation_period-1], y= df_test["Real_price"][:validation_period-1], name="Real_price" ),
                secondary_y=False
        )
        
        fig.add_trace(
            go.Line(x=df_metrics["date"][:validation_period-1], y= df_metrics["real_price_+x%"][:validation_period-1], name=f'Error +{error_limit}%',mode='markers+lines', marker= dict(symbol= 'cross') ,line=dict(color='#F7230C')),
            secondary_y=False
            )
        
        fig.add_trace(
            go.Line(x=df_metrics["date"][:validation_period-1], y= df_metrics["real_price_-x%"][:validation_period-1], name=f'Error -{error_limit}%',mode='markers+lines', marker= dict(symbol= 'cross') , line=dict(color='#F7230C')),
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
        fig['layout']['yaxis']['title'] = 'PRICE'
        
        fig.update_xaxes(
        rangeslider_visible = True,
    )
            
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
     

### Valeurs futurs / futur forcast

        st.sidebar.markdown("<h1 style='text-align: left; color: #fafbfc;'> PREVISION DES FUTURS PRIX </h1>", unsafe_allow_html=True) 
        forecast_period = st.sidebar.slider("Période à prédire",1, 20, 10)
        forcast = price_forecast(models[Graph_type],forecast_period)
        futur_price = np.exp(forcast['yhat'])
        forcast_date = forcast["ds"]
        df_forecast = pd.DataFrame(list(zip(forcast_date,futur_price)), columns=["date","Futurs_price"])
        st.markdown("-----------------------------")
        st.markdown("<h4 style='text-align: left; color: #101f61;'> PREVISION DES FUTURS PRIX </h4>", unsafe_allow_html=True) 
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        visibility = 30
        fig.add_trace(
                go.Line(x=df_historique["date"][-visibility:], y= df_historique["historicaldata"][-visibility:], name="historical data"),
                secondary_y=False,
            )
        fig.add_trace(
                go.Scatter(x=df_forecast["date"][-forecast_period-visibility:], y= df_forecast["Futurs_price"][-forecast_period-visibility:], name="Forecast"),
                secondary_y=False
                )

        # Add figure title
        fig.update_layout(
                    title_text=f"{Graph_type} Forcast Price",
                    legend_title_font_size=2,
                )

                # Set x-axis title
        fig.update_xaxes(title_text="Date")

                # Set y-axes titles
        fig.update_yaxes(title_text=" ", secondary_y=False)
        fig.update_yaxes(title_text=" ", secondary_y=False)
                
        fig['layout']['xaxis']['title'] = 'DATE'
        fig['layout']['yaxis']['title'] = 'PRICE'
        
        fig.update_xaxes(
        rangeslider_visible = True,
    )
            
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
  
        


if __name__ == '__main__':
  main()                           
            