import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics

from matplotlib import pyplot as plt

df = pd.read_excel("datasets/global-player-stats.xlsx")
df = df.dropna()
dataX = df['date'].values
datay = df['num_attempts'].values
print(df)
xtrain, xtest, ytrain, ytest = train_test_split(dataX, datay, test_size = .3) # for your need, you can set reandom_state = xxxx if you want randomly pick

print(ytrain)

xtrain = xtrain.reshape(-1, 1)
xtest = xtrain.reshape(-1, 1)
#ytrain = ytrain.reshape(-1, 1)
#ytest = ytest.reshape(-1, 1)

# print(xtrain, xtest)

model_test = RandomForestClassifier(n_estimators = 20)
model_test.fit(xtrain, ytrain)
predictOfTest = model_test.predict(xtest)

plt.plot(predictOfTest)
plt.plot(ytest)
# print(metrics.accuracy_score(ytest, predictOfTest))  # check for accuracy 

plt.show()

# predict 

model = RandomForestClassifier(n_estimators = 20)
model.fit(dataX,datay)
predict = model.predict(dataX)
predict.plot()
predict.scatter(dataX,predict)