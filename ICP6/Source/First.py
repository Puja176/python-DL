1.Identify and delete
outliers for the “GarageArea” field in predicting SalePrice (House prices dataset)
Hint: 
Plot GaurageArea v/s SalePrice using scatter plot 
Then identify and eliminate outliers

Program:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('C:/Users/pooja/Documents/train.csv')



df_train.describe()

var = 'GarageArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
data.shape
data_remana = np.abs(stats.zscore(data))


data_remana.shape




data_remana[:5,:5]




data1 = data[(data_remana < 3).all(axis=1)]



var = 'GarageArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data1.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));



df = data[~data.SalePrice.isin(data[data_remana > 3].SalePrice)]

df.describe()


data1.shape
