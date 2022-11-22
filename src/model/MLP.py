"""
@PROJECT: StoreSales - MLP.py
@IDE: PyCharm
@DATE: 2022/11/17 下午5:36
@AUTHOR: lxx
"""
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, **kwargs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
