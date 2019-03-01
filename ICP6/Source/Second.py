2.   Create Multiple Regression for the dataset
(Weather dataset)
Evaluate the model using RMSE and R2 score.
Weather dataset: 
https://umkc.box.com/s/60yr8e5p0x772ggtvfmaqysswjlph5y8
** You need to convert t
he categorical features to the numeric using the provided code in the slide
** You need to do the same with the Null values (missing data) in the data set

Program:
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/pooja/Documents/weatherHistory.csv')


corr = dataset.corr()

print (corr['Temperature (C)'].sort_values(ascending=False)[:5], '\n')
print (corr['Temperature (C)'].sort_values(ascending=False)[-5:])



#X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values
X = dataset.drop(['Summary','Daily Summary','Temperature (C)','Loud Cover'],axis=1)


#df = df_train.drop(['Summary','Daily Summary'],axis=1)

X = pd.get_dummies(X, columns=["Precip Type"])

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder = LabelEncoder()
#X.values[:, 1] = labelencoder.fit_transform(X.values[:, 1])
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
#X = X[:, 1:]



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
model = regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


#Evaluating the model

from sklearn.metrics import mean_squared_error, r2_score
print("Variance score: %.2f" % r2_score(y_test,y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))