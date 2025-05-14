import torch
import json
from pathlib import Path
from tqdm import tqdm
from dataloader import create_dataloaders
from ddsrn import create_ddsrn_model
from extractor import load_feature_extractor, FeatureExtractor
import torch.nn.functional as F
from scipy.stats import spearmanr
import numpy as np
from scipy.stats import pearsonr
from backbones import Backbone
import time


def get_gpu_memory_usage(device_id=0):
    """Returns allocated and reserved GPU memory in megabytes for a given device ID."""
    if not torch.cuda.is_available():
        return 0, 0

    torch.cuda.synchronize(device_id)
    allocated = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved(device_id) / (1024 * 1024)
    return allocated, reserved


def predict_test_set(
    model_path: str,
    imgs_root: str,
    ddscores_root: str,
    output_path: str,
    backbone_name: Backbone,
    batch_size: int = 64,
    device: str = "cuda:0",
    weights_path: str = "yolo11m.pt",
    enable_bench_time: bool = False,
):
    """
    Make predictions on test set, calculate metrics, and save results to JSON.
    Can also benchmark inference time and memory usage if enable_bench_time is True.

    Args:
        model_path: Path to saved model checkpoint
        imgs_root: Root directory containing images
        ddscores_root: Root directory containing Detection Degradation scores
        output_path: Path where to save JSON results
        backbone_name: the Backbone to use
        batch_size: Batch size for inference
        device: Device to run inference on
        weights_path: Path to YOLO11m weights file, needed only for YOLO backbone
        enable_bench_time: Whether to benchmark inference time and memory usage
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    device_id = device.index if hasattr(device, "index") else 0

    # Baseline memory measurement before loading models
    if enable_bench_time and torch.cuda.is_available():
        torch.cuda.empty_cache()
        baseline_allocated, baseline_reserved = get_gpu_memory_usage(device_id)
        print(
            f"Baseline GPU memory - Allocated: {baseline_allocated:.2f} MB, Reserved: {baseline_reserved:.2f} MB"
        )

    layer_indices = backbone_name.config.indices
    feature_channels = backbone_name.config.channels

    # Load model
    model = create_ddsrn_model(
        feature_channels=feature_channels,
        layer_indices=layer_indices,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Initialize feature extractor
    extractor: FeatureExtractor = load_feature_extractor(
        backbone_name=backbone_name,
        weights_path=weights_path,
    ).to(device)

    # Memory measurement after loading models
    if enable_bench_time and torch.cuda.is_available():
        model_loaded_allocated, model_loaded_reserved = get_gpu_memory_usage(device_id)
        model_memory_overhead_allocated = model_loaded_allocated - baseline_allocated
        model_memory_overhead_reserved = model_loaded_reserved - baseline_reserved
        print(
            f"Model loading memory overhead - Allocated: {model_memory_overhead_allocated:.2f} MB, Reserved: {model_memory_overhead_reserved:.2f} MB"
        )

    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        backbone_name=backbone_name,
        dataset_root=imgs_root,
        ddscores_root=ddscores_root,
        batch_size=batch_size,
    )

    # Store results
    results = []
    all_predictions = []
    all_scores = []

    # For benchmarking
    if enable_bench_time:
        total_images = 0
        total_time = 0
        warmup_batches = 10
        inference_memory_allocated = 0
        inference_memory_reserved = 0
        peak_allocated = model_loaded_allocated if torch.cuda.is_available() else 0
        peak_reserved = model_loaded_reserved if torch.cuda.is_available() else 0

        print(f"Running inference benchmark with {backbone_name.value} backbone")
        print(f"Using device: {device}")
        print(f"Performing {warmup_batches} warmup batches...")

        # Warmup runs
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= warmup_batches:
                    break

                gt = batch["gt"].to(device)
                compressed = batch["compressed"].to(device)

                # Warmup extraction and prediction
                gt_features, mod_features = extractor.extract_features(
                    img_gt=gt, img_mod=compressed
                )
                _ = model(gt_features, mod_features)

        print("Starting timed inference...")

    # Make predictions
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Making predictions")):
            gt = batch["gt"].to(device)
            compressed = batch["compressed"].to(device)
            scores = batch["score"].to(device)
            names = batch["name"]

            if enable_bench_time and batch_idx >= warmup_batches:
                # For benchmarking, measure time for feature extraction and prediction
                torch.cuda.synchronize(device_id) if torch.cuda.is_available() else None
                start_time = time.time()

                gt_features, mod_features = extractor.extract_features(
                    img_gt=gt, img_mod=compressed
                )
                predictions = model(gt_features, mod_features).squeeze()

                torch.cuda.synchronize(device_id) if torch.cuda.is_available() else None
                end_time = time.time()

                batch_time = end_time - start_time
                total_time += batch_time
                total_images += len(gt)

                # Measure memory usage on first non-warmup batch
                if batch_idx == warmup_batches and torch.cuda.is_available():
                    current_allocated, current_reserved = get_gpu_memory_usage(
                        device_id
                    )
                    inference_memory_allocated = (
                        current_allocated - model_loaded_allocated
                    )
                    inference_memory_reserved = current_reserved - model_loaded_reserved
                    print(
                        f"Inference memory usage - Allocated: {inference_memory_allocated:.2f} MB, Reserved: {inference_memory_reserved:.2f} MB"
                    )
                    peak_allocated = current_allocated
                    peak_reserved = current_reserved

                # Track peak memory usage
                if torch.cuda.is_available():
                    current_allocated, current_reserved = get_gpu_memory_usage(
                        device_id
                    )
                    peak_allocated = max(peak_allocated, current_allocated)
                    peak_reserved = max(peak_reserved, current_reserved)
            else:
                # Normal prediction without timing
                gt_features, mod_features = extractor.extract_features(
                    img_gt=gt, img_mod=compressed
                )
                predictions = model(gt_features, mod_features).squeeze()

            # Store results
            for name, pred, score in zip(names, predictions, scores):
                # Convert .npy extension to .jpg while keeping the same numeric name
                img_name = f"{Path(name).stem}.jpg"

                results.append(
                    {
                        "filename": img_name,
                        "pred_ddscore": float(pred.cpu()),  # Predicted ddscore
                        "ddscore": float(score.cpu()),  # Ground truth ddscore
                    }
                )
                all_predictions.append(pred.cpu().numpy())
                all_scores.append(score.cpu().numpy())

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)

    # Calculate MAE
    mae = F.l1_loss(torch.tensor(all_predictions), torch.tensor(all_scores)).item()

    # Calculate Spearman's rank correlation
    spearman_corr, spearman_p = spearmanr(all_predictions, all_scores)

    # Calculate Pearson's correlation
    pearson_corr, pearson_p = pearsonr(all_predictions, all_scores)

    # Prepare statistics dictionary
    statistics = {
        "number_of_predictions": len(results),
        "average_predicted_distance": np.mean(all_predictions).item(),
        "MAE": mae,
        "Spearman_correlation": spearman_corr.item(),
        "Spearman_p_value": spearman_p.item(),
        "Pearson_correlation": pearson_corr.item(),
        "Pearson_p_value": pearson_p.item(),
    }

    # Add benchmark statistics if enabled
    if enable_bench_time and total_images > 0:
        avg_time_per_image = total_time / total_images
        fps = total_images / total_time

        benchmark_stats = {
            "total_inference_time_seconds": total_time,
            "total_images_processed": total_images,
            "average_time_per_image_seconds": avg_time_per_image,
            "frames_per_second": fps,
            "backbone": backbone_name.value,
            "device": str(device),
            "batch_size": batch_size,
        }

        # Add memory statistics if available
        if torch.cuda.is_available():
            memory_stats = {
                "baseline_memory_allocated_mb": baseline_allocated,
                "baseline_memory_reserved_mb": baseline_reserved,
                "model_memory_overhead_allocated_mb": model_memory_overhead_allocated,
                "model_memory_overhead_reserved_mb": model_memory_overhead_reserved,
                "inference_memory_usage_allocated_mb": inference_memory_allocated,
                "inference_memory_usage_reserved_mb": inference_memory_reserved,
                "peak_memory_allocated_mb": peak_allocated,
                "peak_memory_reserved_mb": peak_reserved,
            }
            benchmark_stats.update(memory_stats)

        statistics.update(benchmark_stats)

        print("\nBenchmark Results:")
        print(f"Backbone: {backbone_name.value}")
        print(f"Total images processed: {total_images}")
        print(f"Total inference time: {total_time:.4f} seconds")
        print(f"Average time per image: {avg_time_per_image * 1000:.4f} ms")
        print(f"Throughput: {fps:.2f} images/second")

        if torch.cuda.is_available():
            print("\nMemory Usage:")
            print(
                f"Model loading overhead - Allocated: {model_memory_overhead_allocated:.2f} MB, Reserved: {model_memory_overhead_reserved:.2f} MB"
            )
            print(
                f"Inference memory usage - Allocated: {inference_memory_allocated:.2f} MB, Reserved: {inference_memory_reserved:.2f} MB"
            )
            print(
                f"Peak memory usage - Allocated: {peak_allocated:.2f} MB, Reserved: {peak_reserved:.2f} MB"
            )

    # Save results to JSON
    output_data = {
        "statistics": statistics,
        "predictions": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Results saved to {output_path}")

    # Print statistics to console
    print("\nQuality Assessment Statistics:")
    quality_metrics = [
        "number_of_predictions",
        "average_predicted_distance",
        "MAE",
        "Spearman_correlation",
        "Pearson_correlation",
    ]
    for key in quality_metrics:
        value = statistics[key]
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def main():
    # Configuration
    GPU_ID = 0
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    ATTEMPT = 38
    DIR = "40bins_point8_07_coco17complete_320p_qual_20_25_30_35_40_45_50_subsamp_444"
    TRIAL = f"attempt{ATTEMPT}_{DIR}"
    MODEL_PATH = f"checkpoints/{TRIAL}/best_model.pt"
    IMGS_ROOT = "balanced_dataset_coco2017"
    DDSCORES_ROOT = "balanced_dataset_coco2017"
    OUTPUT_PATH = f"checkpoints/{TRIAL}/test_predictions.json"
    BACKBONE = Backbone.YOLO_V11_M

    # Benchmarking flag
    ENABLE_BENCH_TIME = True

    predict_test_set(
        model_path=MODEL_PATH,
        imgs_root=IMGS_ROOT,
        ddscores_root=DDSCORES_ROOT,
        output_path=OUTPUT_PATH,
        batch_size=128,
        weights_path="yolo11m.pt",
        backbone_name=BACKBONE,
        device=DEVICE,
        enable_bench_time=ENABLE_BENCH_TIME,
    )


if __name__ == "__main__":
    main()
