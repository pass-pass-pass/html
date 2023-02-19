import warnings
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

from scipy.signal import periodogram
from scipy.stats import norm, describe
from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.forecasting.stl import STLForecast, STLForecastResults

from pmdarima import auto_arima
from pmdarima.utils.visualization import decomposed_plot
from pmdarima.arima import decompose

from pandas import DataFrame, DateOffset, Series

from wordfreqs import WordFreqs
from partofspeech import PartOfSpeech

class PlayerStats:

    def __init__(self, path) -> None:
        self.path = path
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

    def explore_num_attempts(self):
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

    def naive_decompose_num_attempts(self):
        num_attempts = self.data['num_attempts'].to_numpy()
        decomposed = decompose(num_attempts, type_='multiplicative', m=7)
        decomposed_plot(decomposed, figure_kwargs={})
        _, trend, seasonal, random = decomposed

    def stl_decompose_num_attempts(self):
        num_attempts = self.data['num_attempts']
        num_attempts.index = self.data['date']
        # robust handles initial chaos better than non-robust
        # STL found a 7 day period
        result: DecomposeResult = STL(num_attempts, robust=True).fit()
        result.plot()
        # self.add_stl_plot(fig, result_auto_period, ["7 day period", "Auto period"])
        plt.show()

    def stl_predict_num_attempts(self):
        # https://www.statsmodels.org/dev/generated/statsmodels.tsa.forecasting.stl.STLForecast.html#statsmodels.tsa.forecasting.stl.STLForecast
        num_attempts = self.data['num_attempts']
        num_attempts.index = self.data['date']
        arima_params = dict(order=(1, 1, 0), trend='t')
        forecast_model = STLForecast(num_attempts, ARIMA, model_kwargs=arima_params, period=7, robust=True).fit()
        print(forecast_model.summary())
        prediction = forecast_model.get_prediction(datetime(2023, 1, 1), datetime(2023, 3, 1), dynamic=True)
        interval_95 = prediction.conf_int(0.05)
        interval_90 = prediction.conf_int(0.1)
        prediction_index = prediction.summary_frame().index

        plt.figure()
        plt.fill_between(prediction_index, interval_95['lower'], interval_95['upper'], color='b', alpha=0.1)
        plt.fill_between(prediction_index, interval_90['lower'], interval_90['upper'], color='b', alpha=0.2)
        plt.plot(num_attempts)
        plt.plot(prediction.predicted_mean)
        plt.show()

    def arima_num_attempts(self):
        num_attempts = self.data['num_attempts'].to_numpy()
        NUM_TRAINING = 250
        training_data = num_attempts[:NUM_TRAINING]
        validation_data = num_attempts[NUM_TRAINING:]

        # arima_auto = auto_arima(training_data, trace=True, m=7, suppress_warnings=True)
        # arima_auto.fit(training_data)
        # prediction = arima_auto.predict(len(validation_data))

        # abs_error = rmse(prediction, validation_data)
        # rel_error = abs_error / np.mean(validation_data)
        # print(rel_error)
        
        # plt.plot(validation_data)
        # plt.plot(prediction)
        # plt.show()

        arima_auto = auto_arima(num_attempts, trace=True, m=1, suppress_warnings=True)
        arima_auto.fit(num_attempts)
        prediction = arima_auto.predict(60)

        print(prediction)

        plt.plot(prediction)
        plt.show()
        
    def make_hardmode_table(self, path):
        wf = WordFreqs("datasets/word_freqs.csv")
        pos = PartOfSpeech("datasets/part_of_speech.csv")

        is_valid_word = wf.has_freqs(self.data['word']) & pos.has_parts(self.data['word'])
        valid_data = self.data.loc[is_valid_word]

        words = valid_data['word']
        freqs = wf.get_freqs(words)
        parts = pos.get_parts(words)
        hardmode_percents = valid_data['num_hardmode_attempts'] / valid_data['num_attempts'] * 100
        num_dup_letters = valid_data['word'].apply(lambda val: len(val) - len(set(val)))

        parts.index = words.index
        freqs.index = words.index
        self.data['freq'] = freqs
        self.data['hardmode_percent'] = hardmode_percents
        self.data['num_dup_letters'] = num_dup_letters
        self.data['is_valid_word'] = is_valid_word

        self.data.to_excel(path, index=False)
        
    def distribution_tries_norm_fit(self):
        tries_percents = self.data.iloc[:, 5:12]

        r2 = []
        params = []

        for i in range(0, 10):

            percents: Series = tries_percents.iloc[i] / np.sum(tries_percents.iloc[i])
            tries = np.arange(1, 8)

            # Guessed values of parameters
            n = len(tries)                  
            mean = sum(tries * percents)/n              
            sigma = sum(percents*(tries-mean)**2)/n

            popt, pcov = curve_fit(norm.pdf, tries, percents, p0=[mean,sigma])
            prediction = norm.pdf(tries, *popt)

            print(np.sqrt(np.trace(pcov)))
            print(r2_score(percents, prediction))
            print(popt)

            r2 = np.append(r2, r2_score(percents, prediction))
            params = np.append(params, popt)

            # Plot actual data vs. fitted curve
            # plt.plot(tries, percents, 'b+:',label='actual')
            # plt.plot(tries, prediction,'ro:',label='prediction')
            # plt.legend()
            # plt.title('# Tries Distribution vs. Gaussian Fit')
            # plt.xlabel('# Tries')
            # plt.ylabel('Percentage of Players')
            # plt.show()

        print(describe(r2))

    def add_norm_mean_col(self, path):
        
        def find_norm(row):
            num_tries_prob = row / np.sum(row)
            num_tries = np.arange(1, 8)
            popt, pcov = curve_fit(norm.pdf, num_tries, num_tries_prob, p0=[3, 1])
            return popt[0]

        self.data['norm_mean'] = self.data.iloc[:, 5:12].apply(find_norm, axis=1)
        print(self.data)
        self.data.to_excel(path)


if __name__ == "__main__":

    ps = PlayerStats("datasets/global-player-stats.xlsx")
    ps.add_norm_mean_col("datasets/global-player-stats.xlsx")