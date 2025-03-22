from torch.utils.data import Dataset
import torch.nn as nn
import torch

class MNISTDataset(Dataset):

  def __init__(self, X, y, device):
    self.X = X.type(torch.FloatTensor).to(device)
    self.y = nn.functional.one_hot(y).type(torch.FloatTensor).to(device)


  def __len__(self):
    return len(self.X)


  def __getitem__(self, idx):

    features = self.X[idx]
    target = self.y[idx]

    return features, target