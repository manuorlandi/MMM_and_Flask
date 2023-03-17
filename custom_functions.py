import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import probplot, moment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime 
import plotly.express as px
import pickle as pkl
import sys, os


def decompose_time_series(data, model='additive', period=7):
    #Decomposes a time series into its seasonal, trend, and residual components.
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data, model=model, period=period)

    # Extract the seasonal, trend, and residual components
    seasonal = decomposition.seasonal
    trend = decomposition.trend
    residual = decomposition.resid

    # Return the components as a dictionary
    return {'seasonal': seasonal, 'trend': trend, 'residual': residual}



def add_missing_combinations(df_, date_column, columns):

    df = df_.copy()
    # Get all missing dates
    all_dates = pd.date_range(df[date_column].min(), df[date_column].max()).date
    existing_dates = df[date_column].unique()
    missing_dates = list(set(all_dates) - set(existing_dates))
    missing_dates.sort()

    # Create a new dataframe with missing dates and all possible combinations
    intersec_values = pd.DataFrame()

    for miss_date in missing_dates:
        offset=1
        found=False
        while not found:
        # Get the previous and next dates
            prev_date = miss_date - pd.DateOffset(days=offset)
            next_date = miss_date + pd.DateOffset(days=offset)

            # Get the values for the previous and next dates
            df_combinations = df[(df[date_column]>=prev_date) & (df[date_column]<=next_date)]
            if len(df_combinations!=0): found=True 
            else: offset+=1
        
        df_combinations[date_column] = miss_date
            
        # Get the intersection of the previous and next values
        intersec_values =  pd.concat([intersec_values,df_combinations[[date_column]+columns].drop_duplicates()], ignore_index=True)

    # Append new rows to the original dataframe
    #new_df = pd.concat([df,intersec_values], ignore_index=True)
    
    return intersec_values

def create_median_dict(df, columns, with_respect_to='campaignId'):

    d = {}

    for col in columns:
        inner_d = {}
        for campaign in df[with_respect_to].unique():
            df_smaller = df[df[with_respect_to]==campaign]
            inner_d[campaign] = np.percentile(df_smaller[col], 50)
        
        d[col] = inner_d

    return d


def not_common_elements(list1, list2):
    return list(set(list1) - set(list2))


def elements_droppable(df, element_nocost, which_el='campaignId', kpis = ['clicksCounter','impressionsCounter']):
    """
    Voglio capire se dato un elemento di cui non ho il costo (campagna o account che sia), se però hanno click/impression.
    Se non hanno click/impression allora posso eliminarle
    """     

    campaign_to_drop = []

    for cmp in element_nocost:
        sum_cmp=0
        for kpi in kpis:
                sum_cmp += df[(df[which_el] == cmp)][kpi].sum()
        if sum_cmp == 0:
            campaign_to_drop.append(cmp)
   
    return campaign_to_drop

def print_box_histplot(df, columns, percentile_list):

    fig, axs = plt.subplots(figsize=(15,18), nrows=len(columns), ncols=2)

    for col in columns:
        for perc in percentile_list:
            p = np.percentile(df[col], perc)
            print(f'{perc}th percentile for column {col} is {p}')
        print(f'Avg value for column {col} is {df[col].mean()}')
        sns.set_style('whitegrid')
        sns.boxplot(y=col,data=df, ax=axs[columns.index(col)][0])
        sns.histplot(data=df, x=col, binwidth=round(max(df[col])/50,0), ax=axs[columns.index(col)][1], stat='probability', common_norm=False)


def get_relevant_lag(data,  target='adCost', n_lags=12, fixed_lag=[1, 2, 3]):
    
    data_ = data[target].copy()
    
    # Calcolo PACF
    lag, ci = pacf(data_, nlags=n_lags, alpha=0.05)
    
    # Calcolo il limite (area blu in grafico pacf)
    boundaries = np.abs(np.array([ci - lag for lag, ci in zip(lag, ci)])[:, 1])
    
    # estraggo i lags maggiori dei limiti
    imp_lags = np.where(np.abs(lag) > boundaries)[0]
    
    imp_lags = list(set(imp_lags) - set([0] + fixed_lag))
    
    return imp_lags


def apply_adstock(x, L, P, D):
    '''
    params:
    x: original media variable, array
    L: length
    P: peak, delay in effect
    D: decay, retain rate
    returns:
    array, adstocked media variable
    '''
    x = np.append(np.zeros(L-1), x)
    
    weights = np.zeros(L)
    for l in range(L):
        weight = D**((l-P)**2)
        weights[L-1-l] = weight
    
    adstocked_x = []
    for i in range(L-1, len(x)):
        x_array = x[i-L+1:i+1]
        xi = sum(x_array * weights)/sum(weights)    
        adstocked_x.append(xi)
    adstocked_x = np.array(adstocked_x)
    return adstocked_x

def hill_transform(x, ec, slope):
    return 1 / (1 + (x / ec)**(-slope))



def get_relevant_lag(data,  target='adCost', n_lags=12, fixed_lag=[1, 2, 3]):
    
    data_ = data[target].copy()
    
    # Calcolo PACF
    lag, ci = pacf(data_, nlags=n_lags, alpha=0.05)
    
    # Calcolo il limite (area blu in grafico pacf)
    boundaries = np.abs(np.array([ci - lag for lag, ci in zip(lag, ci)])[:, 1])
    # estraggo i lags maggiori dei limiti
    imp_lags = np.where(np.abs(lag) > boundaries)[0]
    val_lags = np.abs(lag)
    imp_lags = list(set(imp_lags) - set([0] + fixed_lag))
    
    return val_lags[imp_lags], imp_lags


def mape(true, pred, EPSILON=1e-9):
    true, pred = np.array(true), np.array(pred)
    return np.mean(np.abs((true - pred) / (true + EPSILON))) * 100

def mse(true, pred):
    # true, pred = np.array(true), np.array(pred)
    return np.mean(np.square(true - pred))


def mae(true, pred):
    true, pred = np.array(true), np.array(pred)
    return np.mean(np.abs((true - pred)))


def create_features(df, rol_cols, relevant_lags):

    df_final = df.copy()

    for col in rol_cols:
        for lag in relevant_lags:
            #df_final['rol_avg_'+col+'_'+str(lag)+'d'] = df_final[col].rolling(lag, min_periods=1).mean()
            df_final['rol_sum_'+col+'_'+str(lag)+'d'] = df_final[col].rolling(lag, min_periods=1).sum()
            df_final['rol_ewm_'+col+'_'+str(lag)+'d'] = df_final[col].ewm(span=lag, min_periods=1).mean()
            df_final['rol_std_'+col+'_'+str(lag)+'d'] = df_final[col].rolling(lag, min_periods=1).std()
            df_final['rol_sqrd_dist_'+col+'_'+str(lag)+'d'] = np.abs(df_final['rol_ewm_'+col+'_'+str(lag)+'d'] - df_final['rol_std_'+col+'_'+str(lag)+'d'])**2
            df_final['SHIFT_'+col+'_'+str(lag)+'d'] = df_final[col].shift(lag)
            df_final['SHIFT_detrended_'+col] = df_final[col]- df_final[col].shift()
            df_final['SHIFT_sqrd_detrended_'+col] = df_final['SHIFT_detrended_'+col]**2

    return df_final


def scale_features(df, scaler_path, scaler_filename = 'scaler.pkl', train=True):

    df_final_scaled = df.copy()
    feature_cols    = [col for col in df_final_scaled.columns if col not in ['calDate']]

    if train:

        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(df_final_scaled[feature_cols])
        df_final_scaled[feature_cols] = transformed
        pkl.dump(scaler, open(os.path.join(scaler_path,scaler_filename), 'wb'))
    else:

        scaler = pkl.load(open(os.path.join(scaler_path,scaler_filename), 'rb'))
        transformed = scaler.fit_transform(df_final_scaled[feature_cols])
        df_final_scaled[feature_cols] = transformed
        
    return df_final_scaled