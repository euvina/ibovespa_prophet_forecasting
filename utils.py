
## ----------------------------------------- ##
##                  Imports                  ##
## ----------------------------------------- ##

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker

import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

import itertools

from prophet import Prophet

import warnings
warnings.filterwarnings('ignore')

## ----------------------------------------- ##
##            Plots & Statistics             ##
## ----------------------------------------- ##

def plot_series(dataframe, series, same_ax=True,  
                title=None, xlabel=None, ylabel=None, 
                figsize=(15, 6), color='#1f3979',
                grid=True, legend=False):
    ''' Plots one or multiple time series.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing the time series.
    series : str or list
        Name of the column(s) to be plotted.
    same_ax : bool, optional
        If True, plots all series in the same axis. The default is True.
    title : str
        Title of the plot.
    xlabel : str
        Label of the x axis.
    ylabel : str
        Label of the y axis.
    figsize : tuple, optional
        Figure size. The default is (15, 6).
    color : str, optional
        Color of the plot. The default is '#1f3979'.
    grid : bool, optional
        If True, shows grid. The default is True.
    legend : bool, optional
        If True, shows legend. The default is False.
        
    '''
    # if series is a string, convert to list
    if type(series) == str:
        series = [series]
    
    # plot in the same axis
    if same_ax:
        plt.figure(figsize=figsize)
        for serie in series:
            plt.plot(dataframe[serie], color=color)
        plt.title(title, fontsize=20)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        if grid:
            plt.grid(True, alpha=0.5, linestyle='--')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
        plt.xticks(fontsize=12, rotation=30)
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        plt.yticks(fontsize=12)
        if legend:
            plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    # plot in individual subplots
    else:
        fig, ax = plt.subplots(len(series), 1, figsize=(15, 5*len(series)))
        for i, serie in enumerate(series):
            ax[i].plot(dataframe[serie], color=color)
            ax[i].set_title(serie)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
            ax[i].xaxis.set_major_locator(mdates.YearLocator(1))
            ax[i].tick_params(axis='x', labelrotation=30)
            if grid:
                ax[i].grid(True, alpha=0.5, linestyle='--')
        plt.tight_layout()
        plt.show()


def plot_acf_pacf(series, lags):
    ''' Plots the autocorrelation and 
    partial autocorrelation functions.
    
    Parameters
    ----------
    series : pandas.Series
        Time series.
    lags : int or list
        Number of lags to be plotted.
    
    '''
    if type(lags) == int:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sm.graphics.tsa.plot_acf(series, lags=lags, ax=ax[0], color='#1f3979')
        sm.graphics.tsa.plot_pacf(series, lags=lags, ax=ax[1], color='#e34592')
        plt.title(f'Correlação e Autocorrelação com lags: {lags}', fontsize=20)
        
    elif type(lags) == list:
        # plot in individual subplots
        fig, ax = plt.subplots(len(lags), 2, figsize=(15, 5*len(lags)))
        for i, lag in enumerate(lags):
            sm.graphics.tsa.plot_acf(series, lags=lag, ax=ax[i, 0])
            sm.graphics.tsa.plot_pacf(series, lags=lag, ax=ax[i, 1])
            ax[i, 0].set_title(f'Correlação com lags: {lag}')
            ax[i, 1].set_title(f'Autocorrelação com lags: {lag}')
    else:
        raise TypeError('lags must be an integer or a list of integers.')
    plt.tight_layout()
    plt.gca().spines['right'].set_visible(False) # colocar em folha de estilo
    plt.gca().spines['top'].set_visible(False)
    plt.show()


# função para decompor uma série temporal e plotar
def plot_decomposition(df, column, period=252, model='multiplicative'):
    ''' Decomposes a time series and plots it.
    
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame with time series.
    column : str
        Column name of the time series.
    period : int, optional
        Seasonal period. The default is 252.
    model : str, optional
        Model to be used in the decomposition. The default is 'multiplicative'.
    
    '''
    # decompose
    decomp = seasonal_decompose(df[column], model=model, period=period)
    
    # plot with subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
    decomp.observed.plot(ax=ax1, color='#1f3979')
    ax1.set_ylabel('Observado')
    decomp.trend.plot(ax=ax2, color='#1f3979')
    ax2.set_ylabel('Tendência')
    decomp.seasonal.plot(ax=ax3, color='#1f3979')
    ax3.set_ylabel('Sazonalidade')
    decomp.resid.plot(ax=ax4, color='#1f3979')
    ax4.set_ylabel('Resíduos')
    plt.suptitle(f'Decomposição de {column}', fontsize=20)
    plt.tight_layout()
    plt.show()


def stationarity_test(dataframe, column, window=252):
    ''' Performs the Dickey-Fuller test and plots 
    the rolling mean and rolling standard deviation.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing the time series.
    column : str
        Column name of the time series.
    window : int, optional
        Rolling window. The default is 252.
    
    Returns
    -------
    None.
    '''
    
    # perform Dickey-Fuller test
    dftest = adfuller(dataframe[column], autolag='AIC')
    # create a series with the results
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    # loop through the critical values
    for key, value in dftest[4].items():
        # add the critical values to the series
        dfoutput[r'Critical Value (%s)' %key] = value
    
    # plot rolling mean and standard deviation
    plt.figure(figsize=(15, 5))
    plt.plot(dataframe[column], color='#1f39f9', label='Original')
    plt.plot(dataframe[column].rolling(window=window).mean(), color='#ff7f0e', label='Rolling Mean')
    plt.plot(dataframe[column].rolling(window=window).std(), color='#2ca02c', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.xticks(pd.date_range(start=dataframe.index.min(), end=dataframe.index.max(), freq='24M'), rotation=0)
    plt.title(f'Média Móvel e Desvio Padrão de {column}', fontsize=20)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
    # print the results
    print(dfoutput)
    print('--'*20)
    print('Results of Dickey-Fuller Test:')
    if dfoutput['Test Statistic'] < dfoutput['Critical Value (1%)']:
        print('The Test Statistic is lower than the Critical Value (1%). The series is stationary.')
    elif dfoutput['Test Statistic'] < dfoutput['Critical Value (5%)']:
        print('The Test Statistic is lower than the Critical Value (5%). The series is stationary.')
    elif dfoutput['Test Statistic'] < dfoutput['Critical Value (10%)']:
        print('The Test Statistic is lower than the Critical Value (10%). The series is stationary.')
    else:
        print('The Test Statistic is higher than the Critical Values. The series is not stationary.')
        
    if dfoutput['p-value'] < 0.05:
        print('The p-value is lower than 0.05. The series is stationary.')
    else:
        print('The p-value is higher than 0.05. The series is not stationary.')



## ----------------------------------------- ##
## Data Transformation & Feature Engineering ##
## ----------------------------------------- ##

def map_lag(dataframe, column):
    ''' Maps exacly the same date from 
    the previous year, two and three years ago, 
    six months ago, three months ago and one month ago.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing the time series.
    column : str
        Column name of the time series.
    
    Returns
    -------
    pandas.DataFrame
        Dataframe containing the time series with lag features.
    '''
    
    df = dataframe.copy()
    
    target_map = df[column].to_dict()
    
    # calculate 1 year ago from date
    date_1year_ago = df.index - pd.DateOffset(years=1)
    df['lag_1year'] = date_1year_ago.map(target_map)
    # 2 years ago
    date_2years_ago = df.index - pd.DateOffset(years=2)
    df['lag_2years'] = date_2years_ago.map(target_map)
    # 3
    date_3years_ago = df.index - pd.DateOffset(years=3)
    df['lag_3years'] = date_3years_ago.map(target_map)
    # 6 months
    date_6months_ago = df.index - pd.DateOffset(months=6)
    df['lag_6months'] = date_6months_ago.map(target_map)
    # 3 months
    date_3months_ago = df.index - pd.DateOffset(months=3)
    df['lag_3months'] = date_3months_ago.map(target_map)
    # 1 month
    date_1month_ago = df.index - pd.DateOffset(months=1)
    df['lag_1month'] = date_1month_ago.map(target_map)
    
    return df


def date_features(dataframe):
    ''' Takes the index, changes his name to date,
    and creates datetime features for the index.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing the time series.
    
    Returns
    -------
    pandas.DataFrame
        Dataframe containing the time series with datetime features.
    '''
    
    df = dataframe.copy()
    df.index.rename('date', inplace=True)
    
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    df['quarter'] = df.index.quarter
    
    return df


def rolling_dataframe(dataframe, column, window, plot=True):
    ''' Calculates the rolling mean and 
    standard deviation of a time series.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing the time series.
    column : str
        Column name of the time series.
    window : int
        Window size.
    plot : bool, optional
    
    Returns
    -------
    pandas.DataFrame
        Dataframe containing the rolling mean and standard deviation.
    '''
    
    df_rolling = dataframe.copy()
    df_rolling['rolling_mean'] = df_rolling[column].rolling(window=window).mean()
    df_rolling['rolling_std'] = df_rolling[column].rolling(window=window).std()
    
    if plot:
        fig, ax = plt.subplots(figsize=(15, 5))
        df_rolling[column].plot(ax=ax, label='Original')
        df_rolling['rolling_mean'].plot(ax=ax, label='Rolling Mean')
        df_rolling['rolling_std'].plot(ax=ax, label='Rolling Std')
        plt.title(f'Média Móvel e Desvio Padrão de {column}', fontsize=20)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.legend(loc='best')
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    # drop NaNs
    df_rolling.dropna(inplace=True)
    
    return df_rolling


def transform_prophet(df, y, regressors=None):
    ''' Transform dataframe to be used in prophet.
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with date as index and y and regressors as columns.
    y : str
        Name of column with y.
    regressors : list
        List with names of columns with regressors.
    
    '''
    # create dataframe
    df_prophet = pd.DataFrame()
    
    # add date
    df_prophet['ds'] = df.index
    
    # add y
    df_prophet['y'] = df[y].values
    
    # add regressors
    if regressors is not None:
        for regressor in regressors:
            df_prophet[regressor] = df[regressor].values

    return df_prophet