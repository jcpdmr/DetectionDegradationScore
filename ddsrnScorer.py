import torch
from ddsrn import create_ddsrn_model
from extractor import load_feature_extractor, FeatureExtractor
from backbones import Backbone
from torchvision.transforms import ToTensor
from PIL import Image


class ddsrnScorer(torch.nn.Module):
    def __init__(self, model_path: str, backbone: Backbone, weights_path: str, device="cuda"):
        super().__init__()
        self.device = device

        # Save backbone and model info
        self.backbone = backbone
        self.weights_path = weights_path

        # Load DDSRN model
        self.ddsrn = create_ddsrn_model(
            feature_channels=backbone.config.channels,
            layer_indices=backbone.config.indices,
        ).to(device).eval()

        checkpoint = torch.load(model_path, map_location=device)
        self.ddsrn.load_state_dict(checkpoint["model_state_dict"])
        self.ddsrn.eval()

        # Load feature extractor
        self.extractor: FeatureExtractor = load_feature_extractor(
            backbone_name=backbone,
            weights_path=weights_path
        ).to(device).eval()

    @torch.no_grad()
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1: Tensor [B, C, H, W] or [C, H, W]
            img2: Tensor [B, C, H, W] or [C, H, W]

        Returns:
            similarity score (lower means more similar): Tensor [B] or scalar
        """
        # Add batch dimension if necessary
        img1 = self._ensure_batch(img1).to(self.device)
        img2 = self._ensure_batch(img2).to(self.device)

        # Extract features
        feat1, feat2 = self.extractor.extract_features(img1, img2)

        # Predict similarity
        score = self.ddsrn(feat1, feat2).squeeze()

        return score
    
    @staticmethod
    def _ensure_batch(img: torch.Tensor) -> torch.Tensor:
        return img.unsqueeze(0) if img.dim() == 3 else img
    

def DDSRN(ref_img_path, deg_img_path):
    """
    Compute the DDS score using DDSRN (Detection Degradation Score Regression Network) between 
    a reference and a degraded image.

    Args:
        ref_img_path (str): Path to the reference (original/clean) image.
        deg_img_path (str): Path to the degraded (e.g., compressed, noisy) image.

    Returns:
        float: DDSRN score indicating the estimated degradation in object detection quality.
            A score closer to 0 indicates high similarity; closer to 1 indicates high degradation.
    """
    
    # Load model
    metric = ddsrnScorer(
        model_path="checkpoints/attempt38_40bins_point8_07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444/best_model.pt",
        backbone=Backbone.YOLO_V11_M,
        weights_path="yolo11m.pt",
        device="cuda"
    )
    # Load images
    ref_img = Image.open(ref_img_path).convert("RGB")
    deg_img = Image.open(deg_img_path).convert("RGB")

    # Compute DDSRN distance score
    return metric(ToTensor()(ref_img), ToTensor()(deg_img))
