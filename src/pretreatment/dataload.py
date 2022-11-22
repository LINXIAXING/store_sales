"""
@PROJECT: StoreSales - dataload.py
@IDE: PyCharm
@DATE: 2022/11/9 下午1:09
@AUTHOR: lxx
"""
from enum import Enum

import torch
import numpy as np
import pandas as pd
from joblib import dump
from src.pretreatment.datamachine import format_data, scale, fixna, difference
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class DataloaderType(Enum):
    train = 0
    validate = 1
    test = 2
    prediction = 3


def load_store_data(config: EasyDict, data_type: DataloaderType):
    """

    :param data_type:
    :param config:
    :return:
    """

    """
    train = pd.read_csv('./src/dataset/train/train.csv')
    transaction = pd.read_csv('./src/dataset/train/transactions.csv')
    extra_holidays = pd.read_csv('./src/dataset/extra/holidays_events.csv')
    extra_oil = pd.read_csv('./src/dataset/extra/oil.csv')
    extra_store_inf = pd.read_csv('./src/dataset/extra/stores.csv')
    test = pd.read_csv('./src/dataset/test/test.csv')

    """
    transaction = pd.read_csv(config.load.train_path + 'transactions.csv')
    extra_holidays = pd.read_csv(config.load.extra_path + 'holidays_events.csv')
    extra_oil = pd.read_csv(config.load.extra_path + 'oil.csv')
    extra_store_inf = pd.read_csv(config.load.extra_path + 'stores.csv')

    if data_type == DataloaderType.prediction:
        dataset = pd.read_csv(config.load.test_path + 'test.csv')
        dataset.insert(loc=4, column='sales', value=0)
    else:
        dataset = pd.read_csv(config.load.train_path + 'train.csv')
        train_len = int(len(dataset) * config.train.train_scale)
        val_len = int(len(dataset) * config.train.val_scale)
        if data_type == DataloaderType.train:
            dataset = dataset.iloc[: train_len, :]
        elif data_type == DataloaderType.validate:
            dataset = dataset.iloc[train_len: train_len + val_len, :]
        elif data_type == DataloaderType.test:
            dataset = dataset.iloc[train_len + val_len:, :]
            pass
    # processing dataset
    format_dataset = format_data(dataset, transaction, extra_holidays, extra_oil, extra_store_inf)
    return format_dataset


def load_oil_data(path: str):
    oil = pd.read_csv(path + 'oil.csv')
    # 拼接油价数据
    fixna(oil)


def get_dataset(config: EasyDict, data_type: DataloaderType):
    ds = load_store_data(config, data_type)
    datasets = []
    for group, d in ds.groupby(['store_nbr', 'family']):
        new_dataset = StoreDataset(config=config, data_type=data_type)
        datasets.append(new_dataset)
    return datasets


class StoreDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config: EasyDict, data_type: DataloaderType):
        """

        :param config:
        :param data_type:
        """

        # load raw data file
        self.type = data_type
        # self.dl = dataset
        self.dl = load_store_data(config, data_type)
        self.transform = MinMaxScaler()
        self.dl_input = self.dl[['store_nbr', 'family', 'onpromotion', 'events', 'dcoilwtico']]
        self.dl_target = self.dl[['sales']]
        # self.dl_input = difference(self.dl[['store_nbr', 'family', 'onpromotion', 'events', 'dcoilwtico']].values.tolist())
        # self.dl_target = difference(self.dl[['sales']].values.tolist())
        self.window = config.train.training_length

    def __len__(self):
        # return number of store
        return len(self.dl) - self.window
        # return len(self.dl)

    # Will pull an index between 0 and __len__.
    def __getitem__(self, idx):
        # _input = torch.tensor(self.dl_input[idx: idx + self.window])  # [100, 5]
        # target = torch.tensor(self.dl_target[idx + self.window: idx + self.window + 1])  # [1, 1]
        _input = torch.tensor(self.dl_input.iloc[idx: idx + self.window].values)  # [100, 5]
        target = torch.tensor(self.dl_target.iloc[idx + self.window: idx + self.window + 1].values)  # [1, 1]
        return _input, target
