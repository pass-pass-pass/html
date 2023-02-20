from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate
import pandas as pd
from sklearn import datasets
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('global-player-stats.xlsx')    # read into the data

df.iloc[:] = df.iloc[::-1].values
#preprocessing data
df = df.drop(columns= 'date')
df = df.dropna()
print(df.columns)
print(len(df.columns))
data = [ 359,359,354,'eerie', 0, 0, 0, 0.41882655, 10.78132608 ,31.5658326, 32.62737598 ,19.11441236, 5.43779202, 0,0,0,"false", 0, 'false']



df.loc[-1]  = data
df = df.drop(columns= 'hardmode_percent')

print(df)

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
# scaler.fit(df[['freq']])
# df["freq"] = scaler.transform(df[['freq']])


df = df.drop(columns= 'num_hardmode_attempts')


# # print(df.iloc[:,5:14])


# # # do elbow rule to find the best k 
print(df.columns)

SSE = []
for k in range(1,11):
    km = KMeans(n_clusters= k)
    km.fit(df.iloc[80:, 5:12])
    SSE.append(km.inertia_)
plt.plot( range(1,11),SSE, marker = "*")
plt.title("elbow rule")
plt.xlabel('k numbers')
plt.show()
# # # # # 3 should be the best
data_slice = df.iloc[:, 5:12]

km = KMeans(n_clusters = 3)
predict = km.fit_predict(data_slice)
print(data_slice.columns)
df['cluster'] = predict


df_easy = df[df.cluster == 0]
df_med = df[df.cluster == 1]
df_hard = df[df.cluster == 2]
df_hard.to_excel('hard.xlsx')
df_med.to_excel('med.xlsx')
df_easy.to_excel('easy.xlsx')
# centroids for the clusttering
# print(km.cluster_centers_)


plt.scatter(df_easy['contest_num'], df_easy['fail_percent'], color = 'purple',label = "easy")
plt.scatter(df_med['contest_num'], df_med['fail_percent']  ,color = 'green',  label = "med")
plt.scatter(df_hard['contest_num'], df_hard['fail_percent'] ,color = 'red', label = "hard")
plt.legend()

print(np.mean(df_easy['freq'].values), 'mean of freq')
print(np.mean(df_med['freq'].values), 'mean of freq')




plt.grid(True)
plt.xlabel("date")
plt.ylabel("fail percent")
plt.title("date against fail percent to view the general difficulty and model accuracy")




size = [len(df_easy), len(df_med),len(df_hard)]
labels = 'easy','med', 'hard'
fig,pie_ = plt.subplots()
pie_.pie(size, labels = labels, autopct='%1.1f%%')
plt.title("pie chart of the percentage of difficculty levels")

# calculate the reliability of the model
# The Calinski-Harabasz index (CH) is one of the clustering algorithms evaluation measures. 
# It is most commonly used to evaluate the goodness of split by a K-Means clustering algorithm for a given number of clusters.
# bigger, better

ch_ = calinski_harabasz_score(df.iloc[:, 5:12], predict)
print("ch index", ch_)

# #The Davies-Bouldin index (DBI) is one of the clustering algorithms evaluation measures.
# #  It is most commonly used to evaluate the goodness of split by a K-Means clustering algorithm for a given number of clusters.
# the small the better

db_ = davies_bouldin_score(df.iloc[:, 5:12], predict)
print(db_, "db index")
index = [[ch_, db_]]
labels2 = ["db index"]
figure2 = plt.subplot()
figure2.table(cellText= index, rowLabels= labels2)



plt.show()




