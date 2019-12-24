#!/usr/bin/env python
# coding: utf-8

# In[123]:


import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import os

# In[124]:


# Load and Prepare Data


# In[125]:
cwd = os.getcwd()

df = pd.read_csv(cwd +'/DataFiles/bike_data_london_merged.csv', header=None,index_col=False,names=['timestamp','cnt','t1','t2','hum','wind_speed','weather_code','is_holiday','is_weekend','season'])


data = df.drop("timestamp", axis=1)
data['weather_code'].unique()
weather_dict = {1 : 100,  2 : 100,  3 : 100,  4 : 100,
                7 : 200, 10 : 200, 26 : 200, 94 : 200}
data['weather_code']=data['weather_code'].replace(weather_dict)

#print(data)
data.head()


# # Checking what values columns contain

# In[80]:


# I check for the columns that contain not continuous values
#print(data.is_holiday.value_counts())

#print(data.is_weekend.value_counts())
#print(data.wind_speed.value_counts())


# # Import of libraries for scaling and splitting

# In[102]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Features and targets
X = data.loc[:, data.columns != 'cnt'].values # Feature
y = data.loc[:, data.columns == 'cnt'].values #Target

# Scaling of the output
#sc = StandardScaler()
sc = MinMaxScaler(feature_range=(0,10))
Y = sc.fit_transform(y[5:,])
Y=np.log1p(Y)

# Scaling of first four columns of features
Columns_scale =(X[5:,:4])
Xpart = sc.fit_transform(Columns_scale)
# OneHotEncoding of the last four
Columns_encode = (X[5:,4:]) # Onehot encode this
ohe = OneHotEncoder(sparse=False)
Xpart2 =ohe.fit_transform(Columns_encode)
#Then we just merge the two columns together
processed_data = np.concatenate([Xpart,Xpart2],axis = 1)


#The next thing we do is to split pur data to training and test
X_train, X_test, y_train, y_test = train_test_split(processed_data, Y, test_size=0.25,shuffle=True)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(2018)
