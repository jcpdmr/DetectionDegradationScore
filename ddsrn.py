import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionWeightedChannelReductionBlock(nn.Module):
    def __init__(self, in_channels, reduction_factor=12, reduction_ratio_attention=16):
        super().__init__()
        reduced_channels = in_channels // reduction_factor
        self.channel_attention = ChannelAttention(
            in_channels, reduction_ratio_attention
        )
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        attention_weights = self.channel_attention(x)  # [B, C_in, H, W]
        weighted_input = x * attention_weights  # [B, C_in, H, W]
        reduced_features = self.reduction_conv(weighted_input)  # [B, C_reduced, H, W]
        return reduced_features


class SimpleBottleneckBlock(nn.Module):
    def __init__(self, channels, reduction_factor=2):
        super().__init__()
        hidden_channels = channels // reduction_factor

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            # nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        identity = x

        # Prima connessione residua
        conv1_out = self.conv1(x)

        # Percorso principale
        out = self.conv2(conv1_out)
        out = self.conv3(out)

        # Aggiungi il residuo interno e applica ReLU
        out = F.relu(out + conv1_out)

        # Percorso finale
        out = self.conv4(out)

        # Connessione residua principale
        return F.relu(out + identity)


class LightResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()

        expanded_dim = int(hidden_dim * 2)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.LayerNorm(expanded_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.post_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.post_norm(x + self.mlp(x))


class ChannelReductionBlock(nn.Module):
    def __init__(self, in_channels, reduction_factor=12):
        super().__init__()
        reduced_channels = in_channels // reduction_factor
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.reduce_conv(x)


class DDSRN(nn.Module):
    """Detection Degradation Score Regression Network"""

    def __init__(self, feature_channels: List[int], layer_indices: List[int]):
        super().__init__()
        self.layer_indices = layer_indices  # Store layer indices for reference
        self.layer_processors = nn.ModuleList(
            [
                self._make_layer_processor(feature_channels=feature_channels[i])
                for i in range(len(feature_channels))
            ]
        )  # One processor for each layer

        # Final channel reduction after concatenating layer features
        self.final_channel_reducer = nn.Sequential(
            nn.Conv2d(
                sum(self._get_pooled_feature_dims(feature_channels)) * 1,
                96,
                kernel_size=1,
            ),  # Multiply by 1 to account for (1, 1) pooling
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        # Modified MLP predictor to use LightResidualMLPBlock
        self.mlp_predictor = nn.Sequential(
            LightResidualMLPBlock(hidden_dim=96),
            LightResidualMLPBlock(hidden_dim=96),
            nn.Linear(96, 1),  # Input dim matches hidden_dim of LightResidualMLPBlock
            nn.Sigmoid(),
        )

    def _make_layer_processor(self, feature_channels):
        """Creates a layer processor module with ChannelReductionBlock and SimpleBottleneckBlock."""
        return nn.Sequential(
            ChannelReductionBlock(in_channels=feature_channels * 2, reduction_factor=8),
            SimpleBottleneckBlock(channels=feature_channels // 4),
            # SimpleBottleneckBlock(channels=feature_channels // 4),
        )

    def _get_pooled_feature_dims(self, layer_channels: List[int]) -> List[int]:
        """Calculates the output dimension after pooling for each layer. Now it's just channels after bottleneck as we use global pooling."""
        pooled_dims = []
        for i in range(len(layer_channels)):
            pooled_dim = layer_channels[i] // 4
            pooled_dims.append(pooled_dim)
        return pooled_dims

    def forward(
        self,
        gt_features_dict: Dict[int, torch.Tensor],
        mod_features_dict: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        layer_feature_vectors = []

        for i, layer_idx in enumerate(self.layer_indices):
            gt_feature = gt_features_dict[layer_idx]
            mod_feature = mod_features_dict[layer_idx]

            # 1. Concatenate GT and MOD features
            # features_diff = gt_feature - mod_feature  # [B, C_i, H_i, W_i]
            concatenated_features = torch.cat(
                [gt_feature, mod_feature], dim=1
            )  # [B, 2*C_i, H_i, W_i]

            # 2 & 3. Channel Reduction and Bottleneck Block
            processed_features = self.layer_processors[i](
                concatenated_features
            )  # [B, C_i/4, H_i, W_i]

            # 4. Global Max Spatial Pooling (directly to 1x1)
            pooled_features = F.adaptive_avg_pool2d(
                processed_features, output_size=(1, 1)
            )  # [B, C_i/4, 1, 1]

            # 5. Flatten
            flattened_features = torch.flatten(pooled_features, 1)  # [B, C_i/4 * 1 * 1]
            layer_feature_vectors.append(flattened_features)

        # Concatenate feature vectors from all layers
        combined_features_vector = torch.cat(
            layer_feature_vectors, dim=1
        )  # [B, sum(C_i/4 * 1 * 1)]

        # Optional final channel reduction with 1x1 conv
        reduced_features = (
            self.final_channel_reducer(
                combined_features_vector.unsqueeze(-1).unsqueeze(-1)
            )
            .squeeze(-1)
            .squeeze(-1)
        )  # [B, 96]

        # MLP Predictor
        distance_prediction = self.mlp_predictor(reduced_features).squeeze(-1)  # [B]

        return distance_prediction


def create_DDSRN_model(
    feature_channels: List[int],
    layer_indices: List[int],
) -> DDSRN:
    """Creates and initializes the DDSRN model"""
    return DDSRN(feature_channels=feature_channels, layer_indices=layer_indices)
