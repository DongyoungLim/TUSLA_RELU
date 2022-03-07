import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_wine, load_breast_cancer, load_iris, fetch_california_housing
import numpy as np
import pandas as pd

class UCI_Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.df[idx])
        return x

def get_data(dataset):

    input_size = 8
    output_size = 1
    data = pd.read_excel('./data/Concrete_Data.xls', header=0).values

    return data, input_size, output_size