import torch
import json
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
import os
import sys
import subprocess
from torch.utils.data import DataLoader
from dataloader import ImagePairDataset
from ultralytics import YOLO
from score_metrics import match_predictions


def get_gpu_memory_usage(device_id=0):
    """Returns GPU memory usage in MiB using nvidia-smi command"""
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader', '--id=' + str(device_id)
            ], encoding='utf-8')
        return float(result.strip())  # Memory in MiB
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def measure_memory_variations(device_id=0, description=""):
    """Measure memory with multiple methods and print results"""
    nvidia_smi = get_gpu_memory_usage(device_id)
    
    # Try to get process-specific memory with pynvml if available
    process_memory = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        process_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        pid = os.getpid()
        for proc in process_info:
            if proc.pid == pid:
                process_memory = proc.usedGpuMemory / (1024 * 1024)  # Convert to MiB
                break
    except (ImportError, Exception):
        pass
    
    print(f"Memory check ({description}):")
    print(f"  - nvidia-smi total: {nvidia_smi:.2f} MiB")
    if process_memory > 0:
        print(f"  - Process-specific: {process_memory:.2f} MiB")
    
    return nvidia_smi


def benchmark_dds_calculation(
    model_path: str,
    imgs_root: str,
    output_path: str,
    batch_size: int = 64,
    device: str = "cuda:0",
    compression_level: int = 40,  # Pick one compression level
    warmup_batches: int = 10,
):
    """
    Benchmark the DDS calculation process (detector + CPU matching).

    Args:
        model_path: Path to the YOLO detector model
        imgs_root: Root directory containing images
        output_path: Path where to save JSON results
        batch_size: Batch size for detector inference
        device: Device to run detector on
        compression_level: Compression level to use for testing
        warmup_batches: Number of batches to use for warmup
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    device_id = device.index if hasattr(device, 'index') else 0
    
    # Baseline memory measurement before loading models
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    baseline_memory = measure_memory_variations(device_id, "baseline")

    # Load YOLO model
    print(f"Loading YOLO model from {model_path}")
    detector = YOLO(model_path)
    detector.to(device)
    
    # Memory measurement after loading model
    detector_loaded_memory = measure_memory_variations(device_id, "after loading YOLO")
    detector_memory_overhead = detector_loaded_memory - baseline_memory
    print(f"Detector loading memory overhead: {detector_memory_overhead:.2f} MiB")

    # Create dataset directly
    print(f"Creating dataloader with batch size {batch_size}")
    test_dataset = ImagePairDataset(
        root_path=imgs_root,
        split="test",
        scores_root=None,  # Not using scores from file
        preprocess=None    # No preprocessing needed for YOLO detector
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True
    )

    # Benchmarking variables
    total_images = 0
    total_detection_time = 0
    total_matching_time = 0
    total_time = 0
    results = []
    
    print(f"Performing {warmup_batches} warmup batches...")
    
    # Process all batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            gt_images = batch["gt"].to(device)
            compressed_images = batch["compressed"].to(device)
            names = batch["name"]
            
            is_benchmark_batch = batch_idx >= warmup_batches
            
            # Add a pause after warmup to manually check memory
            if batch_idx == warmup_batches:
                print("\nWarmup complete. Starting actual benchmark...")
                pre_detection_memory = measure_memory_variations(device_id, "pre-detection")
                
                # Optional pause for manual memory checking
                # print("Press Enter to continue...")
                # input()
            
            # Synchronize before timing
            if is_benchmark_batch:
                torch.cuda.synchronize(device_id) if torch.cuda.is_available() else None
                batch_start_time = time.time()
                detection_start_time = time.time()
            
            # Run detection on GT and compressed images
            gt_predictions = detector.predict(gt_images, verbose=False)
            compressed_predictions = detector.predict(compressed_images, verbose=False)
            
            # Timing for detection phase
            if is_benchmark_batch:
                torch.cuda.synchronize(device_id) if torch.cuda.is_available() else None
                detection_end_time = time.time()
                matching_start_time = time.time()
            
            # Memory after detection
            if is_benchmark_batch and batch_idx == warmup_batches:
                post_detection_memory = measure_memory_variations(device_id, "post-detection")
                detection_memory_usage = post_detection_memory - pre_detection_memory
                print(f"Detection memory usage: {detection_memory_usage:.2f} MiB")
                peak_memory = post_detection_memory
            
            # Calculate DDS scores (CPU matching process)
            batch_matches = match_predictions(gt_predictions, compressed_predictions)
            
            # Memory after matching (first benchmark batch)
            if is_benchmark_batch and batch_idx == warmup_batches:
                post_matching_memory = measure_memory_variations(device_id, "post-matching")
            
            # Timing for matching phase and total
            if is_benchmark_batch:
                torch.cuda.synchronize(device_id) if torch.cuda.is_available() else None
                matching_end_time = time.time()
                batch_end_time = time.time()
                
                # Calculate timing metrics
                batch_detection_time = detection_end_time - detection_start_time
                batch_matching_time = matching_end_time - matching_start_time
                batch_total_time = batch_end_time - batch_start_time
                
                # Accumulate metrics
                total_detection_time += batch_detection_time
                total_matching_time += batch_matching_time
                total_time += batch_total_time
                total_images += len(gt_images)
                
                # Update peak memory
                if torch.cuda.is_available():
                    current_memory = get_gpu_memory_usage(device_id)
                    peak_memory = max(peak_memory, current_memory)
            
            # Store results for benchmark batches (with proper conversion for JSON)
            if is_benchmark_batch:
                for name, match in zip(names, batch_matches):
                    # Ensure all values are JSON serializable
                    error_score = match["error_score"]
                    if isinstance(error_score, torch.Tensor):
                        error_score = float(error_score.item())
                    results.append({
                        "filename": name,
                        "error_score": error_score,
                    })
    
    # Final memory check
    final_memory = measure_memory_variations(device_id, "end of benchmark")
    
    # Calculate final metrics
    avg_detection_time_per_image = total_detection_time / total_images if total_images > 0 else 0
    avg_matching_time_per_image = total_matching_time / total_images if total_images > 0 else 0
    avg_total_time_per_image = total_time / total_images if total_images > 0 else 0
    detection_fps = total_images / total_detection_time if total_detection_time > 0 else 0
    matching_fps = total_images / total_matching_time if total_matching_time > 0 else 0
    total_fps = total_images / total_time if total_time > 0 else 0
    
    # Prepare benchmark statistics
    benchmark_stats = {
        "total_images_processed": total_images,
        "total_detector_inference_time_seconds": total_detection_time,
        "total_matching_time_seconds": total_matching_time,
        "total_dds_time_seconds": total_time,
        "average_detector_time_per_image_seconds": avg_detection_time_per_image,
        "average_matching_time_per_image_seconds": avg_matching_time_per_image,
        "average_dds_time_per_image_seconds": avg_total_time_per_image,
        "detector_inference_fps": detection_fps,
        "matching_fps": matching_fps,
        "dds_total_fps": total_fps,
        "detector_model": model_path,
        "device": str(device),
        "batch_size": batch_size,
        "compression_level": compression_level,
    }
    
    # Add memory statistics if available
    if torch.cuda.is_available():
        memory_stats = {
            "baseline_memory_mib": baseline_memory,
            "detector_memory_overhead_mib": detector_memory_overhead,
            "detection_memory_usage_mib": detection_memory_usage,
            "peak_memory_usage_mib": peak_memory,
            "final_memory_usage_mib": final_memory,
        }
        benchmark_stats.update(memory_stats)
    
    # Create output data
    output_data = {
        "benchmark_statistics": benchmark_stats,
        "dds_scores": results[:100],  # Just include the first 100 results for verification
    }
    
    # Save output to JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print("\nBenchmark Results:")
    print(f"Total images processed: {total_images}")
    print(f"Total detector inference time: {total_detection_time:.4f} seconds")
    print(f"Total matching time: {total_matching_time:.4f} seconds")
    print(f"Total DDS calculation time: {total_time:.4f} seconds")
    print(f"Average detector time per image: {avg_detection_time_per_image*1000:.4f} ms")
    print(f"Average matching time per image: {avg_matching_time_per_image*1000:.4f} ms")
    print(f"Average total time per image: {avg_total_time_per_image*1000:.4f} ms")
    print(f"Detector inference throughput: {detection_fps:.2f} images/second")
    print(f"Matching throughput: {matching_fps:.2f} images/second")
    print(f"DDS total throughput: {total_fps:.2f} images/second")
    
    if torch.cuda.is_available():
        print("\nMemory Usage:")
        print(f"Detector memory overhead: {detector_memory_overhead:.2f} MiB")
        print(f"Detection memory usage: {detection_memory_usage:.2f} MiB")
        print(f"Peak memory usage: {peak_memory:.2f} MiB")
    
    print(f"Results saved to {output_path}")


def main():
    # Configuration
    GPU_ID = 0
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "yolo11m.pt"
    IMGS_ROOT = "balanced_dataset_coco2017"
    OUTPUT_PATH = "benchmarks/dds_benchmark_results.json"
    BATCH_SIZE = 128
    COMPRESSION_LEVEL = 40  # Choose one compression level
    WARMUP_BATCHES = 10
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    benchmark_dds_calculation(
        model_path=MODEL_PATH,
        imgs_root=IMGS_ROOT,
        output_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        compression_level=COMPRESSION_LEVEL,
        warmup_batches=WARMUP_BATCHES,
    )


if __name__ == "__main__":
    main()