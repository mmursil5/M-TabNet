import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureTransformer(nn.Module):
    def __init__(self, num_features):
        super(FeatureTransformer, self).__init__()
        self.fc = nn.Linear(num_features, 64)
        self.fc_gate = nn.Linear(num_features, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, 64)

    def forward(self, x):
        gate = torch.sigmoid(self.fc_gate(x))
        x = self.fc(x)
        x = self.bn(x * gate)
        x = F.relu(x)
        x = self.fc_out(x)
        return x
