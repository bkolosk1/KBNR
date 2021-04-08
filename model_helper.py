# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:34:06 2020

@author: Bosec
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class ColumnarDataset(Dataset):
   def __init__(self, df, y):
       self.x = df
       self.y = y
        
   def __len__(self): 
       return len(self.y)
    
   def __getitem__(self, idx):
       row = self.x[idx, :]
       return np.array(row), self.y[idx] 
   
class ShallowNet(nn.Module):    
  def __init__(self, n_features):
    super(ShallowNet, self).__init__()
    self.a1 = nn.Linear(n_features, 2)
   
  def forward(self, x):
    return torch.sigmoid(self.a1(x))



class TwoNet(nn.Module):    
  def __init__(self, n_features, embedding_dim = 256):
    super(TwoNet, self).__init__()
    self.a1 = nn.Linear(n_features, embedding_dim)
    self.a2 = nn.Linear(embedding_dim, 2)
   
  def forward(self, x):
    x = torch.relu(self.a1(x))
    return torch.sigmoid(self.a2(x))



class ThreeNet(nn.Module):    
  def __init__(self, n_features, e1 = 2048, e2 = 1024, e3 = 640, e4 = 512, e5=216, p = 0.4):
    super(ThreeNet, self).__init__()
    self.a1 = nn.Linear(n_features, e1)
    self.a2 = nn.Linear(e1, e2)
    self.a3 = nn.Linear(e2, e3)
    self.a4 = nn.Linear(e3,2)
    self.dropout = nn.Dropout(p) 
    
  def forward(self, x):
    x = F.selu(self.dropout(self.a1(x)))
    x = F.selu(self.dropout(self.a2(x)))
    x = F.selu(self.dropout(self.a3(x)))
    x = torch.sigmoid(self.a4(x))
    return x


class SqrtNet(nn.Module):    
  def __init__(self, n_features, e_max = 8, p = 0.4):
    super(SqrtNet, self).__init__()
    layers = {}
    calculate_placements = [ 2**(e_max - i) for i in range(e_max) ]
    calculate_placements = [n_features] + calculate_placements
    self.e_max = e_max 
    for i in range(e_max):
        layers[i] = nn.Linear(calculate_placements[i], calculate_placements[i+1])    
    self.linears = nn.ModuleList(list(layers.values()))
    self.dropout = nn.Dropout(p) 
    
  def forward(self, x):
    for lyr in range(self.e_max-1):
        x = F.selu(self.dropout(self.linears[lyr](x)))
    x = torch.sigmoid(self.linears[self.e_max-1](x))
    return x




class FiveNet(nn.Module):    
  def __init__(self, n_features, e1 = 1024, e2 = 2048, e3 = 1024, e4 = 640, e5=512, p = 0.4):
    super(FiveNet, self).__init__()
    self.a1 = nn.Linear(n_features, e2)
    self.a2 = nn.Linear(e2, e3)
    self.a3 = nn.Linear(e3, e4)
    self.a4 = nn.Linear(e4, e5)
    self.a5 = nn.Linear(e5,2)
    self.dropout = nn.Dropout(p) 
    
  def forward(self, x):
    x = F.selu(self.dropout(self.a1(x)))
    x = F.selu(self.dropout(self.a2(x)))
    x = F.selu(self.dropout(self.a3(x)))
    x = F.selu(self.dropout(self.a4(x)))
    x = torch.sigmoid(self.a5(x))
    return x

