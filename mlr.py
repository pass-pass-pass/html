import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np
from mlinsights.mlmodel import IntervalRegressor

df = pd.read_excel("global-player-stats.xlsx")
df["is_weekend"] = df["is_weekend"].astype(int)

df_pos = pd.read_excel('hardmode-stats.xlsx')
df_pos =df_pos.drop(columns = 'hardmode_percent')
df_pos = df_pos.drop(columns = 'num_dup_letters')
# drop unnecessary data
new_df = pd.get_dummies(df_pos, columns= ['part_of_speech'])
new_df = new_df.drop(columns= 'freq')
new_df = new_df.drop(columns= 'date')

df = df.merge(new_df, on = ['word'])

df= df.drop(columns= 'num_hardmode_attempts')
df= df.drop(columns= 'hardmode_percent')
df= df.drop(columns= 'norm_mean')
df= df.drop(columns= 'is_valid_word')
df= df.drop(columns= 'Unnamed: 0')
df= df.drop(columns= 'contest_num')

df.to_excel("test3.xlsx")

# slice data so that we can have all the relavant data except for all the tries
# df['mean_num_tries'] = np.average(num_tries, weights=tries_probability, axis=1)
# num_tries = np.tile(np.arange(1, 8), (df.shape[0], 1))
# tries_probability = df.iloc[:, 3:10]

# print(df)

X = df.iloc[:,9: ].values

# # # y is the data of six tries
y = df.iloc[:,2:9 ].values


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

