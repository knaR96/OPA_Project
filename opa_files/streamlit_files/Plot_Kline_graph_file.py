#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='browser'
from plotly.subplots import make_subplots
import sqlite3, sqlalchemy

#fonction de génération du graph 
def candelistick_plot(
    df : pd.DataFrame,
    ma1 : int,
    ma2 : int,
    ticker: str,
    timelaps : int
): 

# Création du dashboard faisant apparaître le candlestick , deux moyennes mobiles et l'affichage des volumes
    plot_data = df.iloc[-timelaps:].copy()
    plot_data[f'{ma1}_ma'] = plot_data["close"].rolling(ma1).mean()
    plot_data[f'{ma2}_ma'] = plot_data["close"].rolling(ma2).mean()
    
    
    fig = make_subplots(
        rows = 2,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing = 0.1,
        subplot_titles = (f'{ticker} Price ', 'Volume'),
        row_width = [0.3, 0.7]
    )

    fig.add_trace(
        go.Candlestick(
                x = plot_data["time"],
                open = plot_data['open'], 
                high = plot_data['high'],
                low = plot_data['low'],
                close = plot_data['close'],
                name = 'Candlestick chart'
        ),
        row = 1,
        col = 1,
    )   
    
    fig.add_trace(
        go.Line(x = plot_data['time'], y = plot_data[f'{ma1}_ma'], name = f'{ma1} SMA'),
        row = 1,
        col = 1,
    )
    
    fig.add_trace(
        go.Line(x = plot_data['time'], y = plot_data[f'{ma2}_ma'], name = f'{ma2} SMA'),
        row = 1,
        col = 1,
    )
    
    fig.add_trace(
        go.Bar(x = plot_data['time'], y = plot_data['volume'], name = 'Volume'),
        row = 2,
        col = 1,   
    )
    
    

    
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'Volume'
    
    fig.update_xaxes(
        rangeslider_visible = False,
    )
    
    return fig

def typical_price_plot(
    df : pd.DataFrame,
    ma1 : int,
    ma2 : int,
    ticker: str,
    timelaps : int
): 

# Création du dashboard faisant apparaître le candlestick , deux moyennes mobiles et l'affichage des volumes
    plot_data = df.iloc[-timelaps:].copy()
    plot_data[f'{ma1}_ma'] = plot_data["close"].rolling(ma1).mean()
    plot_data[f'{ma2}_ma'] = plot_data["close"].rolling(ma2).mean()
    plot_data['typical_price'] = plot_data.apply(lambda x : (float(x["open"])+ float(x["close"])+ float(x["high"]))/3, axis=1)
    
    fig = make_subplots(
        rows = 2,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing = 0.1,
        subplot_titles = (f'{ticker} Price ', 'Volume'),
        row_width = [0.3, 0.7]
    )

    fig.add_trace(
        go.Line(x = plot_data['time'], y = plot_data['typical_price'], name = f'{ticker}_typical_price'),
        row = 1,
        col = 1,
    )
    
    
    fig.add_trace(
        go.Line(x = plot_data['time'], y = plot_data[f'{ma1}_ma'], name = f'{ma1} SMA'),
        row = 1,
        col = 1,
    )
    
    fig.add_trace(
        go.Line(x = plot_data['time'], y = plot_data[f'{ma2}_ma'], name = f'{ma2} SMA'),
        row = 1,
        col = 1,
    )
    
    fig.add_trace(
        go.Bar(x = plot_data['time'], y = plot_data['volume'], name = 'Volume'),
        row = 2,
        col = 1,   
    )
    

    
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'Volume'
    
    fig.update_xaxes(
        rangeslider_visible = False,
    )
    
    return fig