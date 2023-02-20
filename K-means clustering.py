from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('global-player-stats.xlsx')    # read into the data

df.iloc[:] = df.iloc[::-1].values

#preprocessing data


scaler = MinMaxScaler()
scaler.fit(df[['1_try_percent']])
df["1_try_percent"] = scaler.transform(df[['1_try_percent']])
scaler.fit(df[['2_try_percent']])
df["2_try_percent"] = scaler.transform(df[['2_try_percent']])
scaler.fit(df[['3_try_percent']])
df["3_try_percent"] = scaler.transform(df[['3_try_percent']])
scaler.fit(df[['4_try_percent']])
df['4_try_percent'] = scaler.transform(df[['4_try_percent']])
scaler.fit(df[['5_try_percent']])
df["5_try_percent"] = scaler.transform(df[['5_try_percent']])
scaler.fit(df[['6_try_percent']])
df["6_try_percent"] = scaler.transform(df[['6_try_percent']])

print(df)

# do elbow rule to find the best k 

SSE = []
for k in range(1,11):
    km = KMeans(n_clusters= k)
    km.fit(df.iloc[80:, 5:12])
    SSE.append(km.inertia_)
# plt.plot( range(1,11),SSE, marker = "*")
# plt.show()


# 3 should be the best

km = KMeans(n_clusters = 3)
predict = km.fit_predict(df.iloc[:, 5:])

df['cluster'] = predict

print(df)
df_easy = df[df.cluster == 0]
df_med = df[df.cluster == 1]
df_hard = df[df.cluster == 2]
plt.scatter(df_easy['contest_num'], df_easy['fail_percent'], color = 'black')
plt.scatter(df_med['contest_num'], df_med['fail_percent']  ,color = 'green')
plt.scatter(df_hard['contest_num'], df_hard['fail_percent'] ,color = 'red')



#calculate the reliability of the model
# The Calinski-Harabasz index (CH) is one of the clustering algorithms evaluation measures. 
# It is most commonly used to evaluate the goodness of split by a K-Means clustering algorithm for a given number of clusters.
# bigger, better

ch_ = calinski_harabasz_score(df.iloc[:, 5:12], predict)
print(ch_)





plt.show()


