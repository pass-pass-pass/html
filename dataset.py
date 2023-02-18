import warnings

import pandas as pd
import numpy as np

from scipy.signal import periodogram
from matplotlib import pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA

from pmdarima import auto_arima
from pmdarima.utils.visualization import decomposed_plot
from pmdarima.arima import decompose

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
        plt.figure()
        plt.xlabel("Date")
        plt.ylabel("Number of attempts")
        plt.title("Number of Wordle Attempts per Day vs. Date")
        plt.plot(self.data['date'], self.data['num_attempts'])
        plt.show()

    def decompose_num_attempts(self):
        num_attempts = self.data['num_attempts'].to_numpy()

        diffs = num_attempts
        order = 0
        while True:

            warnings.simplefilter("ignore") # kpss interpolation warning
            kpss_p = kpss(diffs)[1]
            adf_p = adfuller(diffs)[1]

            THRESHOLD = 0.05
            kpss_stationary = kpss_p > THRESHOLD
            adf_stationary = adf_p < THRESHOLD

            print(f'Order {order}:')
            print(f'KPSS stationary: {kpss_stationary}')
            print(f'ADF stationary: {adf_stationary}')

            if kpss_stationary and adf_stationary:
                break

            order += 1
            diffs = diffs[1:] - diffs[:-1]

        plt.figure()
        plt.xlabel("Date")
        plt.ylabel("Stationary Time Series")
        plt.title("Stationary Attempts vs. Date Series")
        plt.plot(self.data['date'][order:], diffs)
        plt.show()

        plot_pacf(num_attempts, lags=range(0, 20), method='ldb')
        freqs, powers = periodogram(diffs)
        freq_data = DataFrame({"freq" : freqs, "power": powers})
        freq_data.sort_values("power", ascending=False, inplace=True)
        print(freq_data.head(3))
        plt.figure()
        plt.scatter(freqs, powers)

        plt.show()

    def arima_num_attempts(self):
        num_attempts = self.data['num_attempts'].to_numpy()
        NUM_TRAINING = 300
        training_data = num_attempts[:NUM_TRAINING]
        validation_data = num_attempts[NUM_TRAINING:]

        arima = auto_arima(training_data, trace=True, suppress_warnings=True)
        figure_kwargs = {}
        decomposed_plot(decompose(training_data, type_='additive', m=4), figure_kwargs=figure_kwargs)
        

ps = PlayerStat("global-player-stats.xlsx")
# ps.draw_num_attempts()
ps.arima_num_attempts()