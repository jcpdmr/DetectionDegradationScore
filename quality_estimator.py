import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathAttention(nn.Module):
    """
    Enhanced attention module that processes features through parallel paths
    to capture both local and global relationships between GT and modified features.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Path 1: Standard cross-attention
        self.norm1 = nn.LayerNorm(in_channels)
        self.q_proj = nn.Linear(in_channels, num_heads * head_dim)
        self.k_proj = nn.Linear(in_channels, num_heads * head_dim)
        self.v_proj = nn.Linear(in_channels, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, in_channels)

        # Path 2: Local feature comparison
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim**-0.5

    def forward(
        self, gt_features: torch.Tensor, mod_features: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = gt_features.shape

        # Path 1: Global attention
        # Reshape and normalize
        gt_flat = gt_features.flatten(2).permute(0, 2, 1)
        mod_flat = mod_features.flatten(2).permute(0, 2, 1)
        gt_norm = self.norm1(gt_flat)
        mod_norm = self.norm1(mod_flat)

        # Compute attention
        q = (
            self.q_proj(gt_norm)
            .view(B, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(mod_norm)
            .view(B, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(mod_norm)
            .view(B, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        global_out = torch.matmul(attn, v)
        global_out = global_out.transpose(1, 2).contiguous().view(B, -1, C)
        global_out = self.out_proj(global_out)
        global_out = global_out.permute(0, 2, 1).view(B, C, H, W)

        # Path 2: Local comparison
        local_in = torch.cat([gt_features, mod_features], dim=1)
        local_out = self.local_conv(local_in)

        # Combine paths
        combined = self.fusion(torch.cat([global_out, local_out], dim=1))

        return combined


class QualityAssessmentModel(nn.Module):
    """
    Enhanced quality assessment model with dual-path attention and hierarchical feature processing.
    Uses multiple attention stages and residual connections for better feature extraction.
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Input normalization
        self.input_norm = nn.BatchNorm2d(in_channels)

        # Multi-stage attention
        self.attention_stages = nn.ModuleList(
            [
                DualPathAttention(in_channels, num_heads, head_dim, dropout),
                DualPathAttention(in_channels, num_heads, head_dim, dropout),
                DualPathAttention(in_channels, num_heads, head_dim, dropout),
            ]
        )

        # Feature processing path
        self.feature_processor = nn.ModuleList(
            [
                self._make_conv_block(in_channels, in_channels, dropout),
                self._make_conv_block(in_channels, in_channels // 2, dropout),
                self._make_conv_block(in_channels // 2, in_channels // 4, dropout),
            ]
        )

        # Spatial attention for weighted pooling
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels // 4, 1, 1), nn.Sigmoid()
        )

        # Final MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_channels // 4),
            self._make_mlp_block(in_channels // 4, 256, dropout),
            self._make_mlp_block(256, 128, dropout),
            self._make_mlp_block(128, 64, dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _make_conv_block(self, in_ch: int, out_ch: int, dropout: float) -> nn.Module:
        """Create a convolutional block with residual connection"""
        return nn.ModuleDict(
            {
                "main": nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                ),
                "shortcut": nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch)
                )
                if in_ch != out_ch
                else nn.Identity(),
            }
        )

    def _make_mlp_block(self, in_dim: int, out_dim: int, dropout: float) -> nn.Module:
        """Create an MLP block with normalization and dropout"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self, gt_features: torch.Tensor, mod_features: torch.Tensor
    ) -> torch.Tensor:
        # Normalize inputs
        gt_features = self.input_norm(gt_features)
        mod_features = self.input_norm(mod_features)

        # Multi-stage attention processing
        x = gt_features
        for attention in self.attention_stages:
            attended = attention(x, mod_features)
            x = x + attended  # Residual connection

        # Feature processing with residuals
        for block in self.feature_processor:
            main_out = block["main"](x)
            shortcut_out = block["shortcut"](x)
            x = main_out + shortcut_out

        # Compute attention weights for pooling
        attention_weights = self.spatial_attention(x)

        # Weighted global pooling
        weighted_features = x * attention_weights
        pooled = torch.sum(weighted_features, dim=(2, 3)) / attention_weights.sum(
            dim=(2, 3)
        )

        # Final prediction
        return self.mlp(pooled)


class SimpleBottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduced_channels = 32  # Reduced bottleneck dimension
        
        self.conv_block = nn.Sequential(
            # Reduction
            nn.Conv2d(channels, reduced_channels, 1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(),
            
            # 3x3 Processing
            nn.Conv2d(reduced_channels, reduced_channels, 3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(),
            
            # Expansion
            nn.Conv2d(reduced_channels, channels, 1),
            nn.BatchNorm2d(channels),
        )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv_block(x))
class LightResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        # Reduced expansion factor from 2 to 1.5
        expanded_dim = int(hidden_dim * 1.5)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.LayerNorm(expanded_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.post_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.post_norm(x + self.mlp(x))

class BaselineQualityModel(nn.Module):
    def __init__(self, in_channels=512, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = 64
        
        # Channel reduction with normalization and activation
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Two SimpleBottleneckBlocks for spatial processing
        self.process = nn.Sequential(
            SimpleBottleneckBlock(128),
            SimpleBottleneckBlock(128)
        )
        
        # Global pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP layers with normalizations
        self.mlp_layers = nn.Sequential(
            nn.Linear(128, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            LightResidualMLPBlock(self.hidden_dim, dropout),
            nn.Linear(self.hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, gt_features, mod_features):
        x = torch.cat([gt_features, mod_features], dim=1)
        x = self.reduce(x)
        x = self.process(x)
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)
        out = self.mlp_layers(x)
        return out

def create_baseline_quality_model() -> BaselineQualityModel:
    """Creates and initializes the baseline quality assessment model"""
    return BaselineQualityModel(in_channels=512)


def create_quality_model() -> QualityAssessmentModel:
    """Creates and initializes the enhanced quality assessment model"""
    return QualityAssessmentModel(
        in_channels=512, num_heads=8, head_dim=64, dropout=0.2
    )
