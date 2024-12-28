from ultralytics import YOLO
import os
import time
import torch
import torch.nn as nn
from patches_loader import create_dataloaders
from typing import Tuple, Dict
from mAP import calculate_batch_mAP
from yoloios import extract_multiple_features_and_predictions, LayerConfig, YOLOSimilarity

def train_perceptual_loss(
    yolo_model: YOLO,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    data_path: str,
    val_frequency: int = 1,
    patience: int = 5,
    output_dir: str = 'output',
    seed: int = 42
):
    """
    Train the perceptual loss model with validation and early stopping
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create log file with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    train_log_path = os.path.join(output_dir, f'log_{timestamp}_trn.txt')
    val_log_path = os.path.join(output_dir, f'log_{timestamp}_val.txt')
    
    with open(train_log_path, 'w') as log_file:
        # Write header
        log_file.write("epoch,loss,mAP,avg_distance,normalized_distance\n")
    with open(val_log_path, 'w') as log_file:
        # Write header
        log_file.write("epoch,loss,mAP,avg_distance,normalized_distance\n")
    
    # Initialize models
    yolo_model.eval()  # Freeze YOLO weights
    yolo_similarity_model = YOLOSimilarity()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        yolo_model = yolo_model.cuda()
        yolo_similarity_model = yolo_similarity_model.cuda()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(data_path, batch_size, seed=seed)
    
    # Setup training
    optimizer = torch.optim.Adam(yolo_similarity_model.parameters(), lr=learning_rate)
    mse_criterion = nn.MSELoss()
    layer_configs = [
        LayerConfig(2, "02_C3k2_early"),
        LayerConfig(9, "09_SPPF"),
        LayerConfig(16, "16_C3k2_pre_detect")
    ]
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        yolo_similarity_model.train()
        train_losses = []
        train_maps = []
        train_distances = []
        train_norm_distances = []
        
        for batch in train_loader:
            loss, mAP, distance = process_batch(
                yolo_model, yolo_similarity_model, batch,
                layer_configs, mse_criterion, device, optimizer, training=True
            )
            train_losses.append(loss)
            train_maps.append(mAP)
            train_distances.append(distance)
            train_norm_distances.append(torch.sigmoid(torch.tensor(distance)).item())
        
        # Log training metrics
        avg_train_metrics = {
            'loss': sum(train_losses) / len(train_losses),
            'mAP': sum(train_maps) / len(train_maps),
            'distance': sum(train_distances) / len(train_distances),
            'norm_distance': sum(train_norm_distances) / len(train_norm_distances)
        }
        
        with open(train_log_path, 'a') as log_file:
            log_file.write(f"{epoch+1},{avg_train_metrics['loss']:.6f},"
                         f"{avg_train_metrics['mAP']:.6f},{avg_train_metrics['distance']:.6f},"
                         f"{avg_train_metrics['norm_distance']:.6f}\n")
        
        # Validation phase if needed
        if (epoch + 1) % val_frequency == 0:
            yolo_similarity_model.eval()
            val_losses = []
            val_maps = []
            val_distances = []
            val_norm_distances = []
            
            with torch.no_grad():
                for batch in val_loader:
                    loss, mAP, distance = process_batch(
                        yolo_model, yolo_similarity_model, batch,
                        layer_configs, mse_criterion, device, training=False
                    )
                    val_losses.append(loss)
                    val_maps.append(mAP)
                    val_distances.append(distance)
                    val_norm_distances.append(torch.sigmoid(torch.tensor(distance)).item())
            
            # Log validation metrics
            avg_val_metrics = {
                'loss': sum(val_losses) / len(val_losses),
                'mAP': sum(val_maps) / len(val_maps),
                'distance': sum(val_distances) / len(val_distances),
                'norm_distance': sum(val_norm_distances) / len(val_norm_distances)
            }
            
            with open(val_log_path, 'a') as log_file:
                log_file.write(f"{epoch+1},{avg_val_metrics['loss']:.6f},"
                             f"{avg_val_metrics['mAP']:.6f},{avg_val_metrics['distance']:.6f},"
                             f"{avg_val_metrics['norm_distance']:.6f}\n")
            
            # Print current metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train - Loss: {avg_train_metrics['loss']:.4f}, mAP: {avg_train_metrics['mAP']:.4f}, "
                  f"Norm Distance: {avg_train_metrics['norm_distance']:.4f}")
            print(f"Val   - Loss: {avg_val_metrics['loss']:.4f}, mAP: {avg_val_metrics['mAP']:.4f}, "
                  f"Norm Distance: {avg_val_metrics['norm_distance']:.4f}\n")
            
            # Save best model and check for early stopping
            if avg_val_metrics['loss'] < best_val_loss:
                best_val_loss = avg_val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': yolo_similarity_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'layer_configs': layer_configs
                }, os.path.join(output_dir, 'best_model.pth'))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            # Print only training metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train - Loss: {avg_train_metrics['loss']:.4f}, mAP: {avg_train_metrics['mAP']:.4f}, "
                  f"Norm Distance: {avg_train_metrics['norm_distance']:.4f}\n")

def process_batch(
    yolo_model: YOLO,
    yolo_similarity_model: YOLOSimilarity,
    batch: Dict[str, torch.Tensor],
    layer_configs: list,
    mse_criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    training: bool = True
) -> Tuple[float, float, float]:
    """
    Process a single batch (training or validation)
    Returns loss, mAP and normalized distance for monitoring
    """
    gt_batch = batch['gt'].to(device)
    modified_batch = batch['modified'].to(device)
    
    # Extract features and predictions
    gt_features, gt_predictions = extract_multiple_features_and_predictions(
        yolo_model, gt_batch, layer_configs
    )
    mod_features, mod_predictions = extract_multiple_features_and_predictions(
        yolo_model, modified_batch, layer_configs
    )
    
    # Calculate metrics
    batch_mAP = calculate_batch_mAP(gt_predictions, mod_predictions)

    distances = yolo_similarity_model(gt_features, mod_features)
    avg_distance = distances.mean()
    
    # Normalize distance to [0,1] range. mAP is already in [0,1] no need to normalize
    normalized_distance = torch.sigmoid(avg_distance)
    
    # Move batch_mAP to correct device
    batch_mAP = torch.as_tensor(batch_mAP).to(device)

    # Compute loss (inverse correlation: low distance -> high mAP, high distance -> low mAP)
    loss = mse_criterion(1 - normalized_distance, batch_mAP)
    
    # Backpropagation if in training mode
    if training and optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item(), batch_mAP.item(), normalized_distance.item()