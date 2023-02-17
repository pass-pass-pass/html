import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from pandas import DataFrame

def main():
    player_stats: DataFrame = pd.read_excel("global-player-stats.xlsx")
    player_stats.sort_values('date', inplace=True)

    plt.xlabel("Date")
    plt.ylabel("Number of attempts")
    plt.title("Number of Wordle Attempts per Day from 2022/1/1 to 2023/1/1")
    plt.plot(player_stats['date'], player_stats['num_attempts'])
    plt.show()

main()