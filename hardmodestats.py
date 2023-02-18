import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt

class HardmodeStats:

    def __init__(self, path):
        self.data : DataFrame = pd.read_excel(path)
        self.data.set_index('word', inplace=True)

    def draw_percents(self):
        plt.figure()
        plt.xlabel("Date")
        plt.ylabel("Percentage of hardmode attemps")
        plt.title("Hardmode Attempt Percentage vs. Date")
        plt.plot(self.data['date'], self.data['hardmode_percent'])
        plt.show()

hs = HardmodeStats('hardmode-stats.xlsx')
hs.draw_percents()