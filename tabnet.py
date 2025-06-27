import torch
import torch.nn as nn
import torch.nn.functional as F
from tabnet_encoder import TabNetEncoder
from sparsemax import Sparsemax

class TabNet(nn.Module):
    def __init__(self, feature_dims, gamma=1.5):
        super(TabNet, self).__init__()
        self.gamma = gamma
        self.encoders = nn.ModuleList([TabNetEncoder(dim, gamma) for dim in feature_dims])
        # Changed from 128 to 64 to match FeatureTransformer output
        self.final_layer = nn.Linear(64 * len(feature_dims), 1)
        self.input_bn_layers = nn.ModuleList([nn.BatchNorm1d(dim) for dim in feature_dims])

        # Define a learnable soft mask as part of the model
        self.soft_mask_fc = nn.ModuleList([nn.Linear(64, 64) for _ in feature_dims])

    def forward(self, inputs):
        encoded = []
        for encoder, input_modality, bn_layer, soft_mask_layer in zip(self.encoders, inputs, self.input_bn_layers, self.soft_mask_fc):
            # Apply Batch Normalization
            input_modality = bn_layer(input_modality)

            # Initialize prior for dynamic feature selection
            prior = torch.ones_like(input_modality)

            # Perform feature selection with Sparsemax
            transformed, mask = encoder(input_modality, prior)

            # Update prior to reflect feature usage
            prior = prior * (self.gamma - mask)

            # Apply a learnable soft mask
            soft_mask = soft_mask_layer(transformed)
            soft_mask = torch.sigmoid(soft_mask)
            transformed = transformed * soft_mask

            encoded.append(transformed)

        # Concatenate all modality-specific features
        concatenated = torch.cat(encoded, dim=1)

        # Final output prediction
        result = self.final_layer(concatenated)
        return result