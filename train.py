from ultralytics import YOLO
from yoloios import extract_multiple_features, LayerConfig, YOLOPerceptualLoss
import torch
import torch.nn as nn
from patches_loader import PatchesDataset
from torch.utils.data import DataLoader

def train_perceptual_loss(
    yolo_model: YOLO,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    data_path: str
):
    """
    Train the perceptual loss model
    """
    # Initialize models and move to GPU
    yolo_model.eval()  # Freeze YOLO weights
    percept_loss = YOLOPerceptualLoss()
    if torch.cuda.is_available():
        yolo_model = yolo_model.cuda()
        percept_loss = percept_loss.cuda()
    
    # Create data loader
    dataset = PatchesDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer and criterion
    optimizer = torch.optim.Adam(percept_loss.parameters(), lr=learning_rate)
    mse_criterion = nn.MSELoss()
    
    # Layer configs for feature extraction
    layer_configs = [
        LayerConfig(2, "02_C3k2_early"),
        LayerConfig(9, "09_SPPF"),
        LayerConfig(16, "16_C3k2_pre_detect")
    ]
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            gt_batch = batch['gt']
            modified_batch = batch['modified']
            
            if torch.cuda.is_available():
                gt_batch = gt_batch.cuda()
                modified_batch = modified_batch.cuda()
            
            # Calculate mAP between predictions
            maps = calculate_batch_map(yolo_model, gt_batch, modified_batch)
            
            # Extract features
            gt_features = extract_multiple_features(yolo_model, gt_batch, layer_configs)
            mod_features = extract_multiple_features(yolo_model, modified_batch, layer_configs)
            
            # Calculate perceptual distances
            distances = percept_loss(gt_features, mod_features)
            
            # Calculate loss (MSE between distances and mAP)
            loss = mse_criterion(distances, maps)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")