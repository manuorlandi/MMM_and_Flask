#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import probplot, moment
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime 
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import plotly.express as px
import my_utilities
import custom_functions as cf
import pickle as pkl
import sys, os

if __name__ == '__main__':


    cfg = my_utilities.__cfg_reading()

    KPI_LIST        = ['adCost','clicksCounter','impressionsCounter']
    PROJECT_PATH    = eval(cfg['PROJECT_PATH'])
    DATA_PATH       = PROJECT_PATH/cfg['DATA_FOLDER']
    FILE            = cfg['FILE_TO_EXPLORE']
    OUTPUT_FILE     = cfg['OUTPUT_FILE']
    MODEL_LOCATION  = PROJECT_PATH/cfg['MODEL_LOCATION']
    MODEL_NAME      = cfg['PICKLE_NAME']


    amazon = pd.read_csv(DATA_PATH/'AMAZON_CLEANED.csv')
    amazon.sort_values(by='calDate', inplace=True, ignore_index=True)
    fb = pd.read_csv(DATA_PATH/'FACEBOOK_CLEANED.csv')
    fb.sort_values(by='calDate', inplace=True)
    google = pd.read_csv(DATA_PATH/'GOOGLE_CLEANED.csv')
    google.sort_values(by='calDate', inplace=True)
    revenues = pd.read_csv(DATA_PATH/'revenuesTest.csv')
    revenues.sort_values(by='calDate', inplace=True)

    amazon['calDate']   = pd.to_datetime(amazon['calDate']).dt.date
    fb['calDate']       = pd.to_datetime(fb['calDate']).dt.date
    google['calDate']   = pd.to_datetime(google['calDate']).dt.date
    revenues['calDate'] = pd.to_datetime(revenues['calDate']).dt.date


    print(f"AMAZON records: {len(amazon)}")
    print(f"FACEBOOK records: {len(fb)}")
    print(f"GOOGLE records: {len(google)}")
    print(f"REVENUES records: {len(revenues)}")

    print(f"Start Date: {amazon['calDate'].min()}, End Date: {amazon['calDate'].max()}")
    print(f"Start Date: {fb['calDate'].min()}, End Date: {fb['calDate'].max()}")
    print(f"Start Date: {google['calDate'].min()}, End Date: {google['calDate'].max()}")
    print(f"Start Date: {revenues['calDate'].min()}, End Date: {revenues['calDate'].max()}")

    min_date = min([amazon['calDate'].min(),fb['calDate'].min(),google['calDate'].min()])


    revenues_out = revenues[revenues['calDate']<min_date]

    print(revenues_out['totDailyGrossRevenue'].sum()/revenues['totDailyGrossRevenue'].sum())


    revenues = revenues[revenues['calDate']>=min_date].reset_index(drop=True)
    revenues = revenues.groupby('calDate')['totDailyGrossRevenue'].sum().reset_index()

    revenues['calDate'] = pd.to_datetime(revenues['calDate'])
    google['calDate']   = pd.to_datetime(google['calDate'])
    amazon['calDate']   = pd.to_datetime(amazon['calDate'])
    fb['calDate']       = pd.to_datetime(fb['calDate'])

    df_final = pd.merge(revenues, google, on='calDate', how='left', suffixes=[None,'GOOG'])
    df_final = pd.merge(df_final, fb, on='calDate', how='left',suffixes=[None,'FB'])
    df_final = pd.merge(df_final, amazon, on='calDate', how='left',suffixes=[None,'AMZ'])
    df_final.columns = ["calDate","totDailyGrossRevenue","adCostGOG", "clicksCounterGOG", "impressionsCounterGOG", "adCostFB", "clicksCounterFB", "impressionsCounterFB", "adCostAMZ", "clicksCounterAMZ", "impressionsCounterAMZ"]
    df_final.to_csv('inputs/final_dataframe.csv',index=False)


    df_final         = df_final.fillna(0)
    val, lags        = cf.get_relevant_lag(df_final, target='totDailyGrossRevenue', n_lags=30)
    _,importance_lag = zip(*sorted(zip(val, lags), reverse=True))


    relevant_lags = importance_lag[:6]
    relevant_lags

    df_cp   = df_final.copy()
    df_cp   = df_cp.set_index('calDate')
    dict_ts = cf.decompose_time_series(df_cp['totDailyGrossRevenue'])

    df_final['totalcost'] = df_final['adCostGOG'] + df_final['adCostFB'] + df_final['adCostAMZ'] 
    df_final.columns = ["calDate","totDailyGrossRevenue","adCostGOG", "clicksCounterGOG", "impressionsCounterGOG", "adCostFB", "clicksCounterFB", "impressionsCounterFB", "adCostAMZ", "clicksCounterAMZ", "impressionsCounterAMZ","totalcost"]

    #df_final['seasonal'] = dict_ts['seasonal'].reset_index(drop=*True)
    #df_final['trend'] = dict_ts['trend'].reset_index(drop=True).fillna(0)



    result = seasonal_decompose(df_cp['totDailyGrossRevenue'], model='additive', period=7)
    #df_final['seasonal'] = result.seasonal.reset_index(drop=True)
    #df_final['trend'] = result.trend.reset_index(drop=True).fillna(0)




    rol_cols = ['adCostGOG','clicksCounterGOG','impressionsCounterGOG','adCostFB','clicksCounterFB','impressionsCounterFB','adCostAMZ','clicksCounterAMZ','impressionsCounterAMZ','totalcost']
    df_final = cf.create_features(df_final, rol_cols, relevant_lags)



    df_final_scaled = df_final.copy()
    df_final_scaled = cf.scale_features(df_final_scaled, scaler_path=MODEL_LOCATION, scaler_filename = 'scaler.pkl', train=True)


    # Load the data into a Pandas DataFrame
    data = df_final_scaled.copy()
    data.fillna(0, inplace=True)

    cols = [el for el in data.columns if el not in ['totDailyGrossRevenue','calDate']]


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[cols], data['totDailyGrossRevenue'], test_size=0.1, random_state=42)

    # Fit a linear regression model to the training data
    model = Ridge().fit(X_train, y_train)

    # Evaluate the model on the test data
    r_squared = model.score(X_test, y_test)
    coefficients = dict(zip(cols, model.coef_))

    # Print the results
    print(f'R-squared: {r_squared:.2f}')

    y_pred = model.predict(X_test)
    #print(f'Mape {cf.mape(y_test, y_pred)}')
    print(f'Mae  {cf.mae(y_test, y_pred)}')
    print(f'Mse  {cf.mse(y_test, y_pred)}')
    pkl.dump(model, open(os.path.join(MODEL_LOCATION,MODEL_NAME), 'wb'))

