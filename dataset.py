import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, features_list, labels):
        self.features_list = features_list
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [features[idx] for features in self.features_list], self.labels[idx]
