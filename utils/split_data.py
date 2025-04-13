from __future__ import  annotations
#import torch
#device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
#torch.manual_seed(0)
#from torch.utils.data import DataLoader, random_split, Dataset
import abc
import pandas as pd


class SplitData():
    def __init__(self, df:pd.core.frame.DataFrame):
        self.df = df
    @abc.abstractmethod
    def split(self):
        ...
    pass
class SpliterByHole(SplitData):
    def __init__(self, df, hole_name_train:list[int],hole_name_test: list[int]):

        super(SpliterByHole, self).__init__(df)

        self.hole_name_train = hole_name_train
        self.hole_name_test = hole_name_test
        self.df_train = []
        self.df_test = []
        self.split()
    def split(self):
        index_train = []
        index_test = []
        df = self.df
        for i in self.hole_name_train:
            index_train = index_train + list(df.index[df['Hole_name'] == i])
        for i in self.hole_name_test:
            index_test = index_test + list(df.index[df['Hole_name'] == i])
        self.df_train = df.iloc[index_train]
        self.df_test = df.iloc[index_test]