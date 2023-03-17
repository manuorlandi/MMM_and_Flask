#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import probplot, moment
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime 
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import plotly.express as px
import my_utilities
import custom_functions as cf
import pickle as pkl
import sys, os


cfg = my_utilities.__cfg_reading()

KPI_LIST = ['adCost','clicksCounter','impressionsCounter']
PROJECT_PATH = eval(cfg['PROJECT_PATH'])
DATA_PATH = PROJECT_PATH/cfg['DATA_FOLDER']
FILE = cfg['FILE_TO_EXPLORE']
OUTPUT_FILE = cfg['OUTPUT_FILE']
MODEL_LOCATION  = PROJECT_PATH/cfg['MODEL_LOCATION']
MODEL_NAME = cfg['PICKLE_NAME']


# In[46]:


amazon = pd.read_csv('AMAZON_CLEANED.csv')
amazon.sort_values(by='calDate', inplace=True, ignore_index=True)
fb = pd.read_csv('FACEBOOK_CLEANED.csv')
fb.sort_values(by='calDate', inplace=True)
google = pd.read_csv('GOOGLE_CLEANED.csv')
google.sort_values(by='calDate', inplace=True)
revenues = pd.read_csv('inputs/revenuesTest.csv')
revenues.sort_values(by='calDate', inplace=True)

amazon['calDate'] = pd.to_datetime(amazon['calDate']).dt.date
fb['calDate'] = pd.to_datetime(fb['calDate']).dt.date
google['calDate'] = pd.to_datetime(google['calDate']).dt.date
revenues['calDate'] = pd.to_datetime(revenues['calDate']).dt.date


print(f"AMAZON records: {len(amazon)}")
print(f"FACEBOOK records: {len(fb)}")
print(f"GOOGLE records: {len(google)}")
print(f"REVENUES records: {len(revenues)}")


# In[47]:


revenues = revenues.drop_duplicates().reset_index(drop=True)


# # HISTORICAL DEPTH

# In[48]:


print(f"Start Date: {amazon['calDate'].min()}, End Date: {amazon['calDate'].max()}")
print(f"Start Date: {fb['calDate'].min()}, End Date: {fb['calDate'].max()}")
print(f"Start Date: {google['calDate'].min()}, End Date: {google['calDate'].max()}")
print(f"Start Date: {revenues['calDate'].min()}, End Date: {revenues['calDate'].max()}")

min_date = min([amazon['calDate'].min(),fb['calDate'].min(),google['calDate'].min()])


# Ci sono delle revenues precedenti alla prima data di partenza delle prime campagne...errore? Andiamo ad analizzare quei dati

# In[49]:


revenues_out = revenues[revenues['calDate']<min_date]


# Sembrano trascurabili, quanto pesano sul totale?

# In[50]:


print(revenues_out['totDailyGrossRevenue'].sum()/revenues['totDailyGrossRevenue'].sum())


# Sono c.ca 5 basis point, rimuovo

# In[51]:


revenues = revenues[revenues['calDate']>=min_date].reset_index(drop=True)


# In[52]:


revenues = revenues.groupby('calDate')['totDailyGrossRevenue'].sum().reset_index()


# In[53]:


revenues['calDate'] = pd.to_datetime(revenues['calDate'])
google['calDate'] = pd.to_datetime(google['calDate'])
amazon['calDate'] = pd.to_datetime(amazon['calDate'])
fb['calDate'] = pd.to_datetime(fb['calDate'])


# In[54]:


df_final = pd.merge(revenues, google, on='calDate', how='left', suffixes=[None,'GOOG'])
print(df_final.columns)
df_final = pd.merge(df_final, fb, on='calDate', how='left',suffixes=[None,'FB'])
df_final = pd.merge(df_final, amazon, on='calDate', how='left',suffixes=[None,'AMZ'])


# In[55]:


df_final['adCost'].count()


# In[56]:


df_final = df_final.fillna(0)
# generate ACF and PACF plots
print(cf.get_relevant_lag(df_final, target='adCost', n_lags=30))


# In[57]:


df_cp = df_final.copy()
df_cp = df_cp.set_index('calDate')

dict_ts = cf.decompose_time_series(df_cp['adCost'])


# In[58]:


df_final['totalcost'] = df_final['adCost'] + df_final['adCostFB'] + df_final['adCostAMZ'] 
df_final.columns = ["calDate","totDailyGrossRevenue","adCostGOG", "clicksCounterGOG", "impressionsCounterGOG", "adCostFB", "clicksCounterFB", "impressionsCounterFB", "adCostAMZ", "clicksCounterAMZ", "impressionsCounterAMZ","totalcost"]


# In[59]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


#test = df_cp.copy()['totDailyGrossRevenue'][-300:]
# perform seasonal decomposition
result = seasonal_decompose(df_cp['totDailyGrossRevenue'], model='additive', period=7)
#result = seasonal_decompose(test, model='additive', period=100)

df_final['seasonal'] = result.seasonal.reset_index(drop=True)
df_final['trend'] = result.trend.reset_index(drop=True).fillna(0)


# In[61]:


rol_cols = ['adCostGOG','clicksCounterGOG','impressionsCounterGOG','adCostFB','clicksCounterFB','impressionsCounterFB','adCostAMZ','clicksCounterAMZ','impressionsCounterAMZ']

for col in rol_cols:
    df_final['rol_'+col+'_7d'] = df_final[col].rolling(7, min_periods=1).mean()


# In[62]:



feature_cols = [col for col in df_final.columns if col not in ['totDailyGrossRevenue','calDate']]
target = ['totDailyGrossRevenue']
print(feature_cols)
df_final_scaled = df_final.copy()

for feature in feature_cols:
    scaler = MinMaxScaler()
    original = df_final_scaled[feature].values.reshape(-1, 1)
    transformed = scaler.fit_transform(original)
    df_final_scaled[feature] = transformed

df_final_scaled['totDailyGrossRevenue'] = df_final_scaled['totDailyGrossRevenue']/1000


# In[63]:


# Load the data into a Pandas DataFrame
#data = df_final.drop(columns=['totalcost','clicksCounter','impressionsCounter', 'clicksCounterFB','impressionsCounterFB','clicksCounterAMZ','impressionsCounterAMZ'])
#data = df_final.drop(columns=['totalcost','adCost','impressionsCounter', 'adCostFB','impressionsCounterFB','adCostAMZ','impressionsCounterAMZ'])
data = df_final_scaled.copy()
data.fillna(0, inplace=True)

cols = [el for el in data.columns if el not in ['totDailyGrossRevenue','calDate']]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[cols], data['totDailyGrossRevenue'], test_size=0.2, random_state=42)

# Fit a linear regression model to the training data
model =  Ridge(alpha=2).fit(X_train, y_train)

# Evaluate the model on the test data
r_squared = model.score(X_test, y_test)
coefficients = dict(zip(cols, model.coef_))

# Print the results
print(f'R-squared: {r_squared:.2f}')

for k,v in coefficients.items():
    print(k,v)

print(model.intercept_)

pkl.dump(model, open(os.path.join(MODEL_LOCATION,MODEL_NAME), 'wb'))


# In[ ]:




