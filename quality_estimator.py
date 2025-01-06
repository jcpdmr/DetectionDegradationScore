import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Cross-attention module that allows GT features to attend to Modified image features.
    Uses multi-head attention to capture different types of relationships between features.

    Args:
        in_channels: Number of input channels (512 from SPPF)
        num_heads: Number of attention heads for multi-head attention
        head_dim: Dimension of each attention head
        dropout: Dropout probability for attention weights
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Linear projections for Query (GT), Key and Value (Modified)
        self.q_proj = nn.Linear(in_channels, num_heads * head_dim)
        self.k_proj = nn.Linear(in_channels, num_heads * head_dim)
        self.v_proj = nn.Linear(in_channels, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, in_channels)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for dot product attention
        self.scale = head_dim**-0.5

    def forward(
        self, gt_features: torch.Tensor, mod_features: torch.Tensor
    ) -> torch.Tensor:
        batch_size, channels, height, width = gt_features.shape

        # Reshape features for attention computation
        # From [B, C, H, W] to [B, H*W, C]
        gt_features = gt_features.flatten(2).permute(0, 2, 1)
        mod_features = mod_features.flatten(2).permute(0, 2, 1)

        # Project and reshape for multi-head attention
        # From [B, H*W, C] to [B, H*W, num_heads, head_dim]
        q = self.q_proj(gt_features).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(mod_features).view(
            batch_size, -1, self.num_heads, self.head_dim
        )
        v = self.v_proj(mod_features).view(
            batch_size, -1, self.num_heads, self.head_dim
        )

        # Transpose for attention computation
        # From [B, H*W, num_heads, head_dim] to [B, num_heads, H*W, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back to original format
        # From [B, num_heads, H*W, head_dim] to [B, H*W, C]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, channels)
        out = self.out_proj(out)

        # Reshape back to spatial format [B, C, H, W]
        return out.permute(0, 2, 1).view(batch_size, channels, height, width)


class QualityAssessmentModel(nn.Module):
    """
    Model for assessing image quality by comparing GT and modified images through their SPPF features.
    Uses cross-attention to capture relationships between feature maps.

    For 384x384 input images, SPPF features will have spatial dimensions of 12x12
    (384/32 = 12, as YOLO downsamples by factor of 32 at SPPF level)

    Args:
        in_channels: Number of input channels from SPPF (512)
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
    """

    def __init__(self, in_channels: int = 512, num_heads: int = 8, head_dim: int = 64):
        super().__init__()

        # Cross-attention module
        self.cross_attention = CrossAttention(in_channels, num_heads, head_dim)

        # Feature processing after attention
        self.post_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # Global average pooling is implicit

        # Final MLP for error score prediction
        self.mlp = nn.Sequential(
            nn.Linear(in_channels // 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Bound output between 0 and 1 for error score
        )

    def forward(
        self, gt_features: torch.Tensor, mod_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the quality assessment model.

        Args:
            gt_features: Features from SPPF for GT image [B, 512, 12, 12]
            mod_features: Features from SPPF for modified image [B, 512, 12, 12]

        Returns:
            Predicted error score between 0 and 1
        """
        # Apply cross-attention
        attended_features = self.cross_attention(gt_features, mod_features)

        # Process attended features
        processed_features = self.post_attention(attended_features)

        # Global average pooling
        pooled_features = (
            F.adaptive_avg_pool2d(processed_features, 1).squeeze(-1).squeeze(-1)
        )

        # Predict error score
        error_score = self.mlp(pooled_features)

        return error_score


def create_quality_model() -> QualityAssessmentModel:
    """
    Creates and initializes the quality assessment model with default parameters.

    Returns:
        Initialized QualityAssessmentModel
    """
    return QualityAssessmentModel(
        in_channels=512,  # SPPF output channels
        num_heads=8,  # Number of attention heads
        head_dim=64,  # Dimension of each attention head
    )
