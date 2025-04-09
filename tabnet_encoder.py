import torch
import torch.nn as nn
from sparsemax import Sparsemax
from feature_transformer import FeatureTransformer

class TabNetEncoder(nn.Module):
    def __init__(self, num_features, gamma=1.5):
        super(TabNetEncoder, self).__init__()
        self.gamma = gamma
        self.attention_fc = nn.Linear(num_features, num_features)
        self.sparsemax = Sparsemax()
        self.bn = nn.BatchNorm1d(num_features)
        self.transformer = FeatureTransformer(num_features)

    def forward(self, x, prior):
        attention_scores = self.attention_fc(x)
        mask = self.sparsemax(attention_scores * prior)
        x_masked = self.bn(x * mask)
        transformed = self.transformer(x_masked)
        return transformed, mask
