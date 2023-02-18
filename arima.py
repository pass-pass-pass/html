import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
df = pd.read_excel("global-player-stats.xlsx")

df['num_attempts'].plot(figsize= (20,9))
#  to test the data is staionary:
def printValue(data):
    result = adfuller(data, autolog='AIC')
    print("ADF",result[0])
    print("p value", result[1])
    print('lags ', result[2])
    print('rows for critical values and ADF regression ', result[3])

# if the p value is huge, then it's not staionary like .6 ....

# now, train the model
potential_model = auto_arima(df['num_attempts'], trace = True, suppress_warnings = True)
potential_model.summary()
#find the best index from the summary

#make the train data and verify data
num_training_points = 250
training = df.iloc[:num_training_points] #assume we use 250 rows to train
predict = df.iloc[num_training_points:] 
arima_model = ARIMA(training['num_attempts'], order = (1, 1, 0)).fit()  # order is the one we got from summary

arima_model.summary()
#now, let's predict
predict_data = arima_model.predict(num_training_points + 1 , len(df),  typ = 'levels')
# pred.index = df.index[len(training): len(df)]  use if no date

#let's find out the reliability of the model
predict_data.plot(legend = True)
predict['num_attempts'].plot(legend = True)
RMSE = np.sqrt(mean_squared_error(predict_data, predict['num_attempts']))
mean = predict['num_attempts'].mean()

#compare mean and rmse. precisely you can google how big of rmse is accpetable to mean

# predict 3.1 2023
predict_model = ARIMA(training['num_attempts'], order = (1, 1, 0)).fit()
predictdata= predict_model.predict(start = len(df), end = len(df) + 60)
# the start and end date should be the duration that you want to predict
# predictdata.index = pd.date_range(start = '1/1/2023', end = '2/29/2023')
print(predictdata)

