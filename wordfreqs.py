import pandas as pd
from pandas import DataFrame, Series

class WordFreqs:

    def __init__(self, path):
        self.data : DataFrame = pd.read_csv(path)
        self.data : DataFrame = self.data.loc[lambda data: data['word'].str.len() == 5, ]
        self.data.set_index('word', inplace=True)
        print(self.data)

    def get_freq(self, word):
        return self.data.at[word, 'count']

    def get_freqs(self, words):
        return self.data.loc[words, 'count']

    def has_freqs(self, words):
        return Series(words).isin(self.data.index)

wf = WordFreqs("word_freqs.csv")
print(wf.get_freq('about'))
words = ['about', 'value', 'moose']
print(wf.get_freqs(words))