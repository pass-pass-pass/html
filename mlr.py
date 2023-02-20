import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression

from sklearn.metrics import r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from mlinsights.mlmodel import IntervalRegressor


df = pd.read_excel("global-player-stats.xlsx")
date = df['contest_num']
df = df.drop(columns= 'date')
df["is_weekend"] = df["is_weekend"].astype(int)

df_pos = pd.read_excel('hardmode-stats.xlsx')
df_pos =df_pos.drop(columns = 'hardmode_percent')
df_pos = df_pos.drop(columns = 'num_dup_letters')
# drop unnecessary data
new_df = pd.get_dummies(df_pos, columns= ['part_of_speech'])
new_df = new_df.drop(columns= 'freq')

df = df.merge(new_df, on = ['word'])

df= df.drop(columns= 'num_hardmode_attempts')
df= df.drop(columns= 'hardmode_percent')
df= df.drop(columns= 'is_valid_word')
df= df.drop(columns= 'Unnamed: 0')
df= df.drop(columns= 'contest_num')
df['num_attempts2'] = df['num_attempts']
df = df.drop(columns= 'num_attempts')
df =df.drop(columns= 'date')
df['date'] = 
df.to_excel('test3.xlsx')
df.dropna()


# slice data so that we can have all the relavant data except for all the tries
# df['mean_num_tries'] = np.average(num_tries, weights=tries_probability, axis=1)
# num_tries = np.tile(np.arange(1, 8), (df.shape[0], 1))
# tries_probability = df.iloc[:, 3:10]


X = df.iloc[:,9: ].values

# # # y is the data of six tries
y = df.iloc[:,2:9 ].values

# # test the model
 
xtrain, xtest, ytrain,ytest = train_test_split(X,y, test_size = .3, random_state = 5)


model = LinearRegression()
model.fit(xtrain,ytrain)

print(model.coef_, model.intercept_)
# check the model 
y_predict = model.predict(xtrain)
print(y_predict[0, :])

# compare the data 
for y in range(len(y_predict)):
    dat = [1,2,3,4,5,6,7]
    plt.scatter(dat, y_predict[y,:])
plt.title('predicted trainging')
plt.ylabel("percent of every try")
plt.xlabel('tries')
# y_predict.plot(legend = True, label = 'predictd train data')
# ytrain.plot(legend = True, label = 'real data ')
plt.show()
for i in range(len(y_predict)):
    dat = [1,2,3,4,5,6,7]
    plt.scatter(dat,ytrain[i,:] )
plt.title('training y date')
plt.xlabel("tries")
plt.ylabel("percent of every try")
plt.show()
# # now let's calculate the prediction interval to see how confident we are on the model 





# # #now check the data it didn't be trained on to see the preformance 

y_test_predict = model.predict(xtest)

for i in range(len(y_test_predict)):
    dat = [1,2,3,4,5,6,7]
    plt.scatter(dat, y_test_predict[i,:])
plt.title('predicted test data')
plt.ylabel("percent of every try")
plt.xlabel('tries')
plt.show()
for i in range(len(y_test_predict)):
    dat = [1,2,3,4,5,6,7]
    plt.scatter(dat, ytest[i,:])
plt.title('test data')
plt.ylabel("percent of every try")
plt.xlabel('tries')
plt.show()


# # check for accuracy 
# # check r squre, bigger, better

r2 = r2_score(ytest,y_test_predict)
# # # we can check for RMSE
rmse = np.sqrt(metrics.mean_squared_error(ytest,y_test_predict))
number = [[r2, rmse]]
labels = [['r square','rmse' ]]

plt.table(number ,rowLabels= labels)
table = [['r square', r2],['rmse', rmse]]
print(tabulate(table))


# # model looks good, now let's check for new data
# x_for_real = pd.dataframe().values # create a new dataset for the data we want
# model_for_real = LinearRegression()
# model_for_real.fit(X,y)
# print(model_for_real.coef_)

# data = model_for_real.predict(x_for_real)





