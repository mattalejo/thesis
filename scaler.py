import pandas as pd
import torch

class Scaler():
    def __init__(self, df, col_name="Log Returns"):
        self.df = df
        self.col_name = col_name
        self.mean = df[col_name].mean()
        self.std = df[col_name].std()
    
    def fit(self, tensor):
        return (tensor-self.mean) / self.std
    
    def inverse_fit(self, tensor):
        return (tensor*self.std) + self.mean
    