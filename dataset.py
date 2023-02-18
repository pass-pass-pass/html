import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from pandas import DataFrame

class PlayerStat:

    def __init__(self, path) -> None:
        self.data: DataFrame = pd.read_excel(path)
        self.data.sort_values('date', inplace=True)

    def show_tries_distribution(self, index):
        tries = (1, 2, 3, 4, 5, 6, 7)
        percents = self.data.iloc[index].iloc[5:12].to_numpy()
        plt.bar(tries, percents)
        plt.show()

    def draw_num_attempts(self):
        plt.xlabel("Date")
        plt.ylabel("Number of attempts")
        plt.title("Number of Wordle Attempts per Day vs. Date")
        plt.plot(self.data['date'], self.data['num_attempts'])
        plt.show()

    def draw_diff_num_attempts(self):
        num_attempts = self.data['num_attempts'].to_numpy()
        diffs = num_attempts[1:] - num_attempts[:-1] # diff[i] is num_attempts[i + 1] - num_attempts[i]
        plt.xlabel("Date")
        plt.ylabel("Change in number of attempts")
        plt.title("Change in Number of Attempts per Day vs. Date")
        plt.plot(self.data['date'][1:], diffs)
        plt.show()

ps = PlayerStat("global-player-stats.xlsx")
ps.arima_num_attemps()
