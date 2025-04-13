from __future__ import annotations
# num_workers=1
# from torch.utils.data import DataLoader, Dataset
import abc
import pandas as pd
import typing
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from utils.split_data import SplitData, SpliterByHole

# device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
# torch.manual_seed(0)
from torch.utils.data import DataLoader, Dataset, Sampler

class MyDataSet(Dataset):
    def __init__(self,x, y, **kwrgs):
        """
        build a dataset with predictors x and targets y

        :param x:
        :param y:
        :param kwrgs:
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx].float(), self.y[idx].float()



class BIDataSet(Dataset):
    def __init__(self, df, predictors: list[str], targets: list[str], transform: any, **kwargs) -> None:
        """
      read  csv data file from src as a  data frame, and return a data set object
      that separates predictors from targets
      :param src:
      :param transform:
      """
        # file_out = pd.read_csv(src, sep=",", encoding='cp1252')
        self.file_out = df  # file_out.loc[:, ~ file_out.columns.str.contains('^Unnamed')]
        if transform is not None:
            self.file_out = self.transform()
        self.x = torch.tensor(self.file_out[predictors].values, dtype=torch.float32)
        self.y = torch.tensor(self.file_out[targets].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx].float(), self.y[idx].float()

    @abc.abstractmethod
    def transform(self):
        ...

    pass


class ScaledBIDataSet(BIDataSet):
    def __init__(self, scale_predictors: bool, scale_targets: bool, mode: str, **kwargs):
        self.x = None
        self.y = None
        self.mode = mode
        self.scale_predictors = scale_predictors
        self.scale_targets = scale_targets
        self.file_out = None
        super(ScaledBIDataSet, self).__init__(**kwargs)
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scale()
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx].float(), self.y[idx].float()

    def transform(self):
        ...

    pass

    def scale(self):

        x = self.x
        y = self.y
        if self.mode == 'train':
            if self.scale_predictors:
                self.x = torch.tensor(self.scaler1.fit_transform(x))
                self.scale_predictors = False
            if self.scale_targets:
                self.y = torch.tensor(self.scaler2.fit_transform(y))
                self.scale_targets = False
        else:
            if self.scale_predictors:
                self.x = torch.tensor(self.scaler1.transform(x), dtype=torch.float32)
                self.scale_predictors = False
            if self.scale_targets:
                self.y = torch.tensor(self.scaler2.transform(y), dtype=torch.float32)
                self.scale_targets = False


class Bi2DDataFrame:
    def __init__(self, df, targets: list['str'], xy: list['str'], *args):
        self.df = df
        self.df2 = pd.DataFrame()
        self.targets = targets
        self.xy = xy
        d = {self.xy[0]: [], self.xy[1]: []}
        # print(d)
        for name in targets:
            d = {**d, **{'Average_' + name: []}}
       # self.hole_names = [i for i in range(1, 74) if i != 7]
        self.hole_names = args[0]
        print(self.hole_names)
        d = {**d, **{'HoleNames': self.hole_names}}
        self.d = d
        self.create_2d_data()

    def create_2d_data(self):
        for hole_name in self.hole_names:
            for target_name in self.targets:
                print(hole_name)
                hole_index = self.df.index[self.df['HoleNames'] == hole_name]
                self.d['Average_' + target_name].append(self.df['BI'][hole_index].mean())
            self.d[self.xy[0]].append(np.max(self.df[self.xy[0]][hole_index]))
            self.d[self.xy[1]].append(np.max(self.df[self.xy[1]][hole_index]))
        self.df2 = pd.DataFrame(self.d)
class SequentialData:
    def __init__(self, df, predictors:list['str'], targets: list['str'], hole_names):
        self.df = df
        self.predictors = predictors
        self.targets = targets
        self.hole_names = hole_names

        #self.get_sequential_data()
        #self.spliter = SpliterByHole()
    def forward(self):
        dataset_seq = dict()
        for h in self.hole_names:
            hole_name1 = [i for i in self.hole_names if i != h]
            spliter = SpliterByHole(self.df, hole_name1, [h])
            dataset_h = BIDataSet(spliter.df_test, predictors=self.predictors, targets=self.targets, transform=None)
            dataset_seq = {**dataset_seq, **{h: dataset_h}}
        return dataset_seq


class BIDataSetDask(Dataset):
    def __init__(self, df, predictors: list[str], targets: list[str], transform: any, **kwargs) -> None:
        """
      read  csv data file from src as a  data frame, and return a data set object
      that separates predictors from targets
      :param src:
      :param transform:
      """
        # file_out = pd.read_csv(src, sep=",", encoding='cp1252')
        self.file_out = df  # file_out.loc[:, ~ file_out.columns.str.contains('^Unnamed')]
        if transform is not None:
            self.file_out = self.transform()
        self.x = torch.tensor(self.file_out[predictors].values.compute(), dtype=torch.float32)
        self.y = torch.tensor(self.file_out[targets].values.compute(), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx].float(), self.y[idx].float()

    @abc.abstractmethod
    def transform(self):
        ...

    pass


class BatchSampler(Sampler):
    def __init__(self, df, num_batches, thred=2000):
        self.df = df
        self.num_batches = num_batches
        self.depths = np.arange(self.df['Depth'].min(), self.df['Depth'].max() + .01, step=.01)
        self.a = self.split_list(thred=thred)

    def split_list(self, thred):
        s = 0
        t = ()
        k, m = divmod(len(self.depths), self.num_batches)
        for i in range(self.num_batches):
            h = self.depths[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            h= [round(elm,2) for elm in h]
            if len(self.df[self.df['Depth'].isin(h)].index.tolist()) >= thred:
                t = t + (h,)
            else:
                s = s + 1
                print(k)
                # print(len(self.df[self.df['Depth'].isin(h)].index.tolist()))
                values = list(t)
                # print(t[len(t)-1] )
                # print(h)
                # print(list(t[len(t)-1]) + list(h))
                values[len(t) - 1] = list(t[len(t) - 1]) + list(h)
                #values=[round(elm,2 ) for elm in values]
                t = tuple(values)

        self.num_batches = len(t)
        print(self.num_batches)
        print(s)

        return t

    # return tuple(self.depths[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(self.num_batches))

    def __iter__(self):
        batch = []

        for l in self.a:
            batch = self.df[self.df['Depth'].isin(l)].index.tolist()

            yield batch
            batch = []

    def __len__(self):
        return self.num_batches


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, df, num_batches, thred=2000):
        self.batch_sampler = BatchSampler(df, num_batches, thred=thred)
        # print(batch_sampler.__len__())
        super().__init__(dataset, batch_sampler=self.batch_sampler)

# function used to reduce the resolution of the data by considering smaller depth-step
def reduce_bi_data(df, resol:float):
    depths =np.round(np.arange(df['Depth'].min(), df['Depth'].max() , step=resol),2)
    idx = df[df['Depth'].isin(depths)].index.tolist()
    df2 = df.loc[idx]
    df2.reset_index(inplace = True)
    return df2





