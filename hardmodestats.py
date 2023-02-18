import numpy as np
import pandas as pd
import seaborn as sb
from pandas import DataFrame, Series
from matplotlib import pyplot as plt

class HardmodeStats:

    def __init__(self, path):
        self.data : DataFrame = pd.read_excel(path)
        # self.data.set_index('word', inplace=True)

    def draw_percents(self):
        plt.figure()
        plt.xlabel("Date")
        plt.ylabel("Percentage of hardmode attemps")
        plt.title("Hardmode Attempt Percentage vs. Date")
        plt.plot(self.data['date'], self.data['hardmode_percent'])
        plt.show()

    def draw_dup_letters(self):
        print(self.data.loc[self.data['num_dup_letters'] == 2, 'word'])
        plt.figure()
        plt.hist(self.data['num_dup_letters'])
        plt.show()

    def draw_part_of_speech(self):
        plt.figure()
        parts, counts = np.unique(self.data['part_of_speech'], return_counts=True)
        plt.pie(counts, labels=parts, autopct='%1.1f%%')
        plt.show()

    def find_correlations(self):
        #for pos in self.data['part_of_speech'].unique():
        #    self.data['pos'] = self.data[self.data['part_of_speech'] == pos, ]
        data = self.data.loc[self.data['hardmode_percent'] < 40]
        data['freq'] = np.log(self.data['freq'])
        data = pd.get_dummies(data, columns=['part_of_speech'])

        corr_matrix = data.corr()
        print(corr_matrix)
        # visualization correlation with a heatmap
        color = sb.diverging_palette(250, 30, as_cmap = True)
        sb.heatmap(corr_matrix, annot = True, cmap = color)
        plt.figure()
        plt.scatter(data['freq'], data['hardmode_percent'], )
        plt.show()

hs = HardmodeStats('hardmode-stats.xlsx')
hs.draw_percents()
hs.find_correlations()