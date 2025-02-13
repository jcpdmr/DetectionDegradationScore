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
    def __init__(self, channels, reduction_factor=8):
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


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()

        self.mlp_expand = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mlp_compress = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.post_residual = nn.Sequential(nn.LayerNorm(hidden_dim), nn.ReLU())

    def forward(self, x):
        expanded = self.mlp_expand(x)
        compressed = self.mlp_compress(expanded)
        return self.post_residual(x + compressed)


class BaselineQualityModel(nn.Module):
    def __init__(self, in_channels=512, dropout=0.2, use_deep_residual=False):
        super().__init__()

        self.hidden_dim = in_channels // 2
        self.use_deep_residual = use_deep_residual

        # Riduzione immediata dei canali
        self.reduce = nn.Conv2d(in_channels * 2, in_channels, 1)

        # Blocchi residuali
        self.process = nn.Sequential(
            SimpleBottleneckBlock(in_channels),
            SimpleBottleneckBlock(in_channels),
        )

        # Dimensione ridotta iniziale
        self.initial_reduction = nn.Sequential(
            nn.Linear(in_channels * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )

        # Blocchi MLP residuali
        self.mlp_block1 = ResidualMLPBlock(self.hidden_dim, dropout)
        self.mlp_block2 = ResidualMLPBlock(self.hidden_dim, dropout)

        # Final prediction layers
        self.final_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.LayerNorm(self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 16),
            nn.LayerNorm(self.hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 16, 1),
            nn.Sigmoid(),
        )

    def forward(self, gt_features, mod_features):
        # Concatenazione e processing iniziale
        x = torch.cat([gt_features, mod_features], dim=1)
        x = self.reduce(x)
        x = self.process(x)

        # Global pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        max_pool = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Riduzione iniziale
        A = self.initial_reduction(x)

        # Primo blocco MLP
        D = self.mlp_block1(A)

        # Secondo blocco MLP
        if self.use_deep_residual:
            E = self.mlp_block2(D) + A  # Skip connection profonda
        else:
            E = self.mlp_block2(D)

        # Predizione finale
        out = self.final_layers(E)

        return out


def create_baseline_quality_model() -> BaselineQualityModel:
    """Creates and initializes the baseline quality assessment model"""
    return BaselineQualityModel(in_channels=512)


def create_quality_model() -> QualityAssessmentModel:
    """Creates and initializes the enhanced quality assessment model"""
    return QualityAssessmentModel(
        in_channels=512, num_heads=8, head_dim=64, dropout=0.2
    )
