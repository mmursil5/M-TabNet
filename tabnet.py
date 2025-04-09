import torch
import torch.nn as nn
from tabnet_encoder import TabNetEncoder

class TabNet(nn.Module):
    def __init__(self, feature_dims, gamma=1.5):
        super(TabNet, self).__init__()
        self.gamma = gamma
        self.encoders = nn.ModuleList([TabNetEncoder(dim, gamma) for dim in feature_dims])
        self.final_layer = nn.Linear(64 * len(feature_dims), 1)

        self.input_bn_layers = nn.ModuleList([nn.BatchNorm1d(dim) for dim in feature_dims])

    def forward(self, inputs):
        encoded = []
        for encoder, input_modality, bn_layer in zip(self.encoders, inputs, self.input_bn_layers):
            input_modality = bn_layer(input_modality)
            prior = torch.ones_like(input_modality)
            transformed, mask = encoder(input_modality, prior)
            prior = prior * (self.gamma - mask)
            encoded.append(transformed)

        concatenated = torch.cat(encoded, dim=1)
        result = self.final_layer(concatenated)
        return result
