import pandas as pd
import torch

def load_data(file_paths, target_file='target.xlsx', label_column='BW'):
    datasets = []
    labels = None

    for path in file_paths:
        df = pd.read_excel(path)
        features = torch.tensor(df.values, dtype=torch.float32)
        datasets.append(features)

    target_df = pd.read_excel(target_file)
    labels = torch.tensor(target_df[label_column].values, dtype=torch.float32).unsqueeze(1)

    return datasets, labels
