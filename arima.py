from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_excel("datasets/global-player-stats.xlsx")

df['num_attempts'].plot(figsize= (20,9))
series = df['num_attempts']

# now, train the model
potential_model = auto_arima(series, trace = True, suppress_warnings = True)
potential_model.summary()
#find the best index from the summary

#make the train data and verify data
training, validation = train_test_split(series, test_size=0.25, shuffle=False) 
arima_model: ARIMA = ARIMA(training, order = (1, 1, 0)).fit()  # order is the one we got from summary

arima_model.summary

#now, let's predict
prediction = arima_model.predict(len(training) , len(series) - 1,  typ = 'levels')
prediction_index = validation.index

plt.plot(prediction_index, prediction)

#let's find out the reliability of the model

# validation['num_attempts'].plot(legend = True)
RMSE = np.sqrt(mean_squared_error(prediction, validation))
mean = validation.mean()
print(RMSE)
print(mean)

#compare mean and rmse. precisely you can google how big of rmse is accpetable to mean

# predict 3.1 2023
predict_model = ARIMA(series, order = (1, 1, 0)).fit()
future = predict_model.get_prediction(len(df), len(df) + 60)
future_index = np.arange(len(df), len(df) + 61)
confi_interval = future.conf_int(alpha = .05 )
# the start and end date should be the duration that you want to predict
# predictdata.index = pd.date_range(start = '1/1/2023', end = '2/29/2023')
print(future)
plt.show()
