#pip install -U scikit-learn
#pip install prophet
#pip install plotly
#pip install seaborn
#pip install --upgrade nbformat
#pip install sqlalchemy
#pip install statsmodels

import warnings
import pandas as pd
import sqlalchemy , sqlite3
 
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

import seaborn as sns
sns.set()
sns.set(rc={'figure.figsize':(14.7,10.27)})
 
from plotly import tools
import plotly.graph_objs as go
import gc
 
from datetime import datetime
 
# prophet model 
from prophet import Prophet
# prophet preformance

from IPython.display import set_matplotlib_formats
from prophet.plot import plot_plotly
import plotly.offline as py

import numpy as np
import sys


def import_df_paire_for_forecast(paire_name:str):
    conn = sqlite3.connect('/Users/kevin/anaconda3/envs/prophet39/opa_files/BinanceBDDHistorique.db')
    c = conn.cursor()
    c.execute('SELECT * FROM '+ paire_name +' ')
    df = pd.DataFrame(c.fetchall(), columns= ["id","time","open","high","low","close","volume","trade","pairs_id"])   
    df = df.set_index("id")
    df.time = pd.to_datetime(df["time"])
    date = df["time"]
    df['typical_price'] = df.apply(lambda x : (x["open"]+ x["close"]+ x["high"])/3, axis=1)
    df = df[["time","typical_price"]]
    df = df.set_index('time')
    df = df.reset_index().rename(columns={'time':'ds', 'typical_price':'y'}) # L'appellation des colonnes telle que définie à cette ligne est necessaire pour le lancement de fbprophet
    df_model = df
    df_model['y'] = np.log(df_model['y'])
    df_model_train = df_model[:-60]
    df_model_test = df_model[-60:]
    return (df_model_train ,df_model_test, df_model)

def model_training_for_forecast(df_model_train:pd.DataFrame): 
    prophet_model = Prophet(changepoint_prior_scale = 0.1, seasonality_prior_scale = 0.01) #changepoint_prior_scale = 0.1, seasonality_prior_scale = 0.01 best parameters for BTC
    model_fiting = prophet_model.fit(df_model_train)
    return prophet_model
    
def price_forecast(model,forecasting_period : int): # forecasting_period in days
    future = model.make_future_dataframe(periods=forecasting_period , freq='D')
    forecast = model.predict(df=future)
    return (forecast)






