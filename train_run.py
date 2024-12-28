from ultralytics import YOLO
from train import train_perceptual_loss


def main():
    
    # Load models and move them to GPU
    yolo = YOLO('yolo11m.pt')
    train_perceptual_loss(yolo, 50, 16, 1e-4, 'patches', val_frequency=1, patience=10, output_dir='output', seed=100)

if __name__ == "__main__":
    main()