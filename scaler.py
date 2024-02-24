import pandas as pd
import torch

class Scaler:
    def __init__(self, df, col_name="Log Returns"):
        self.df = df
        self.col_name = col_name


class Standard(Scaler):
    def __init__(self, df, col_name="Log Returns"):
        super().__init__(df, col_name)
        self.mean = self.df[self.col_name].mean()
        self.std = self.df[self.col_name].std()

    def fit(self, tensor):
        return (tensor - self.mean) / self.std

    def inverse_fit(self, tensor):
        return (tensor * self.std) + self.mean


class MinMax(Scaler):
    def __init__(self, df, col_name="Log Returns"):
        super().__init__(df, col_name)
        self.max_val = self.df[self.col_name].max()
        self.min_val = self.df[self.col_name].min()

    def fit(self, tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val)

    def inverse_fit(self, tensor):
        return tensor * (self.max_val - self.min_val) + self.min_val


class DoubleMinMax(Scaler):
    def __init__(self, df, col_name="Log Returns"):
        super().__init__(df, col_name)
        self.max_val = self.df[self.col_name].max()
        self.min_val = self.df[self.col_name].min()

    def fit(self, tensor):
        return 2 * ((tensor - self.min_val) / (self.max_val - self.min_val)) - 1

    def inverse_fit(self, tensor):
        return ((tensor + 1) * (self.max_val - self.min_val) / 2) + self.min_val


class No(Scaler):
    def fit(self, tensor):
        return tensor

    def inverse_fit(self, tensor):
        return tensor