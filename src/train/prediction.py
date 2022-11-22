"""
@PROJECT: StoreSales - prediction.py
@IDE: PyCharm
@DATE: 2022/11/16 下午8:38
@AUTHOR: lxx
"""
import torch
from easydict import EasyDict
import pandas as pd
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from src.pretreatment.datamachine import format_data, scale, fixna
from tqdm import tqdm
from src.model.Transformer import TransAm


class Prediction(object):
    def __init__(self,
                 config: EasyDict,
                 model_name: str,
                 model: nn.Module,
                 dataloader: DataLoader):
        model_path = config.save.model_path + model_name

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.dataloader = dataloader
        self.model = model
        self.model.load_state_dict(state_dict=torch.load(model_path))  # map_location=torch.device(self.device)
        self.model.eval()
        self.model.to(self.device)

    def _inference(self) -> list:
        val_bar = tqdm(self.dataloader)
        val_bar.set_description(f'Epoch Inference')
        result = []
        for _input, target in val_bar:
            with torch.no_grad():
                pred = self.model(_input.double().to(self.device))
                pred = pred.reshape([len(pred) * len(pred[0])]).cpu()
                result += pred.double().numpy().tolist()
                # result += np.around(pred.double().numpy()).tolist()
        return result

    def get_infer(self):
        pred_data = pd.read_csv(self.config.load.test_path + 'test.csv')
        result = pd.DataFrame(self._inference(), columns=['sales'])
        pred_data.drop(axis=1, columns=['date', 'store_nbr', 'family', 'onpromotion'], inplace=True)
        result = pd.concat([pred_data, result], axis=1)
        fixna(result)
        return result

