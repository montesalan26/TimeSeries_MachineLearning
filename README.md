# TimeSeries_MachineLearning
In this project, I took data from a Time Series , created different features using lag features and rolling mean to train and test machine learning models like Moving Average, Autoregression, Linear Regression, Decision Tree and Random Forest models, to predict the taxi orders of a taxi company, and evaluating those predictions using the Root Mean Squared Error Metric. Before creating those models, I had to determine if the Data was stationary or Non-Stationary by looking at plots and using the Augmented Dickey-Fuller test. After determining the data was stationary, I created several plots seeing the different trends and seasonality using the seasonal decompose function. After analyzing the data, I determined what was the best model for the project by using RMSE(Root mean squared error) This project was created using Jupyter Notebook and Python.

The necessary python libraries used for this project are:

import pandas as pd  
from matplotlib import pyplot as plt  
from sklearn.model_selection import train_test_split  
import pmdarima as pm  
from statsmodels.tsa.seasonal import seasonal_decompose  
from statsmodels.tsa.stattools import adfuller  
import numpy as np  
from sklearn.metrics import mean_squared_error, mean_absolute_error  
from sklearn.preprocessing import StandardScaler  
from statsmodels.tsa.arima.model import ARIMA  
from sklearn.linear_model import LinearRegression  
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  
from statsmodels.tsa.ar_model import AutoReg, ar_select_order  
from statsmodels.tsa.stattools import arma_order_select_ic  
from statsmodels.tsa.statespace.sarimax import SARIMAX  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV  
