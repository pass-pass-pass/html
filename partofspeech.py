import pandas as pd
from pandas import DataFrame, Series

class PartOfSpeech:

    def __init__(self, path):
        self.data : DataFrame = pd.read_csv(path)
        self.data : DataFrame = self.data.loc[lambda data: data['word'].str.len() == 5, ]
        self.data.set_index('word', inplace=True)

    def get_parts(self, words):
        return self.data.loc[words, 'pos_tag']

    def has_parts(self, words):
        return Series(words).isin(self.data.index)

if __name__ == "__main__":

    pos = PartOfSpeech("part_of_speech.csv")
    words = ['about', 'value', 'moose']
    print(pos.get_parts(words))