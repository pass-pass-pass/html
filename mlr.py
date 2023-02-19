import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np

df = pd.read_excel("datasets/global-player-stats.xlsx")

# drop unnecessary data

df = df.drop(columns = 'word')
df = df.drop(columns = 'num_attempts')
# slice data so that we can have all the relavant data except for all the tries
num_tries = np.tile(np.arange(1, 8), (df.shape[0], 1))
tries_probability = df.iloc[:, 3:10]
df['mean_num_tries'] = np.average(num_tries, weights=tries_probability, axis=1)

print(df)

X = df[('date', '')].values

# y is the data of six tries
y = df[1].values


# test the model
 
xtrain, xtest, ytrain,ytest = train_test_split(X,y, test_size = .3, random_state = 5)

model = LinearRegression()
model.fit(xtrain,ytrain)

print(model.coef_, model.intercept_)
# check the model 

y_predict = model.predict(xtrain)

# compare the data 
y_predict.plot(legend = True)
ytrain.plot(legend = True)

#now check the data it didn't be trained on to see the preformance 

y_test_predict = model.predict(xtest)
y_test_predict.plot(legend = True)
ytest.plot(legend = True)
# check for accuracy 
# check r squre, bigger, better

r2_score(ytest,y_test_predict)
# we can check for RMSE
print(np.sqrt(metrics.mean_squared_error(ytest,y_test_predict)))



# model looks good, now let's check for new data
x_for_real = pd.dataframe().values # create a new dataset for the data we want
model_for_real = LinearRegression()
model_for_real.fit(X,y)
print(model_for_real.coef_)

model_for_real.predict(x_for_real)

