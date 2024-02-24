import pandas as pd
import torch

class Scaler:
    def __init__(self, df, col_name="Log Returns"):
        self.df = df
        self.col_name = col_name
        
    
class Standard(Scaler):
    def __init__(self):
        super(Scaler, self).__init__(df, col_name)
        self.mean = df[col_name].mean()
        self.std = df[col_name].std()

    def fit(self, tensor):
        return (tensor-self.mean) / self.std
    
    def inverse_fit(self, tensor):
        return (tensor*self.std) + self.mean


class MinMax(Scaler):
    def __init__(self):
        super(Scaler, self).__init__(df, col_name)
        self.max = self.df.max()
        self.min = self.df.min()

    def fit(self, tensor):
        return (tensor - self.min) / (self.max - self.min) 
    
    def inverse_fit(self, tensor):
        return tensor * (self.max - self.min) + self.min
    
    
class DoubleMinMax(Scaler):
    def __init__(self):
        super(Scaler, self).__init__(df, col_name)
        self.max = self.df.max()
        self.min = self.df.min()

    def fit(self, tensor):
        return 2 * ((tensor - self.min) / (self.max - self.min)) - 1
    
    def inverse_fit(self, tensor):
        return ((tensor + 1) * (self.max - self.min) / 2) + self.min


class No(Scaler):
    def fit(self, tensor):
        return tensor

    def inverse_fit(self, tensor):
        return tensor