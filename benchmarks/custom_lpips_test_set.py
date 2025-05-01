import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import lpips
import torchvision.transforms as transforms
from scipy.stats import spearmanr, pearsonr
from pathlib import Path


def load_error_scores(json_path: str):
    """Carica i valori DDS target dal file JSON"""
    with open(json_path, "r") as f:
        return json.load(f)


def get_image_pairs(extracted_dir: str, compressed_dir: str, error_scores):
    """Trova le coppie di immagini valide con i relativi score DDS"""
    pairs = []

    for img_name in error_scores.keys():
        gt_path = os.path.join(extracted_dir, img_name)
        compressed_path = os.path.join(compressed_dir, img_name)

        # Verifica se entrambe le immagini esistono
        if os.path.exists(gt_path) and os.path.exists(compressed_path):
            pairs.append((gt_path, compressed_path, error_scores[img_name]))

    return pairs


def preprocess_image(img_path):
    """Carica e preelabora un'immagine per il modello LPIPS"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalizza in [-1, 1]
        ]
    )

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Aggiungi dimensione batch
    return img_tensor


def calculate_statistics(predictions, targets):
    """Calcola le statistiche di confronto tra predizioni e target"""
    predictions_array = np.array(predictions)
    targets_array = np.array(targets)

    # Calcola MAE (Mean Absolute Error)
    mae = np.mean(np.abs(predictions_array - targets_array))

    # Calcola correlazione di Spearman
    spearman_corr, spearman_p = spearmanr(predictions_array, targets_array)

    # Calcola correlazione di Pearson
    pearson_corr, pearson_p = pearsonr(predictions_array, targets_array)

    return {
        "number_of_predictions": len(predictions),
        "average_predicted_distance": np.mean(predictions_array),
        "average_target_score": np.mean(targets_array),
        "MAE": mae,
        "Spearman_correlation": spearman_corr,
        "Spearman_p_value": spearman_p,
        "Pearson_correlation": pearson_corr,
        "Pearson_p_value": pearson_p,
    }


def main():
    # Configurazione
    base_dir = "balanced_dataset_coco2017/test"
    extracted_dir = os.path.join(base_dir, "extracted")
    compressed_dir = os.path.join(base_dir, "compressed")
    error_scores_path = os.path.join(base_dir, "error_scores.json")
    custom_model_path = (
        "../PerceptualSimilarity/checkpoints/vgg_custom1/latest_net_.pth"
    )
    output_path = "custom_lpips_vs_dds_results.json"

    # Usa GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo dispositivo: {device}")

    # Carica i valori DDS target
    print(f"Caricamento dei valori DDS da {error_scores_path}")
    error_scores = load_error_scores(error_scores_path)
    print(f"Caricati {len(error_scores)} valori DDS")

    # Trova le coppie di immagini valide
    print("Ricerca delle coppie di immagini valide...")
    image_pairs = get_image_pairs(extracted_dir, compressed_dir, error_scores)
    print(f"Trovate {len(image_pairs)} coppie di immagini valide")

    # Inizializza il modello LPIPS personalizzato
    print(f"Caricamento del modello personalizzato da {custom_model_path}")
    trainer = lpips.Trainer()
    trainer.initialize(
        model="lpips",
        net="vgg",
        model_path=custom_model_path,
        use_gpu=(device.type == "cuda"),
    )

    # Calcola le distanze LPIPS personalizzate
    custom_lpips_scores = []
    dds_targets = []

    print("Calcolo delle distanze LPIPS personalizzate...")
    for gt_path, compressed_path, target_score in tqdm(image_pairs):
        # Carica e preelabora le immagini
        gt_img = preprocess_image(gt_path).to(device)
        compressed_img = preprocess_image(compressed_path).to(device)

        # Calcola la distanza LPIPS personalizzata
        with torch.no_grad():
            distance = trainer.forward(gt_img, compressed_img).item()

        custom_lpips_scores.append(distance)
        dds_targets.append(target_score)

    # Calcola e salva i risultati e le statistiche
    results = []
    for (gt_path, compressed_path, target), pred in zip(
        image_pairs, custom_lpips_scores
    ):
        img_name = os.path.basename(gt_path)
        results.append(
            {
                "image": img_name,
                "custom_lpips": pred,
                "dds_target": target,
                "difference": pred - target,
            }
        )

    statistics = calculate_statistics(custom_lpips_scores, dds_targets)

    # Salva i risultati su file JSON
    output_data = {"statistics": statistics, "results": results}

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    # Stampa le statistiche principali
    print("\nStatistiche LPIPS personalizzato (vgg_custom1) vs DDS:")
    print(f"Numero di predizioni: {statistics['number_of_predictions']}")
    print(
        f"Media distanze LPIPS predette: {statistics['average_predicted_distance']:.4f}"
    )
    print(f"Media target DDS: {statistics['average_target_score']:.4f}")
    print(f"MAE: {statistics['MAE']:.4f}")
    print(
        f"Correlazione di Spearman: {statistics['Spearman_correlation']:.4f} (p-value: {statistics['Spearman_p_value']:.4e})"
    )
    print(
        f"Correlazione di Pearson: {statistics['Pearson_correlation']:.4f} (p-value: {statistics['Pearson_p_value']:.4e})"
    )

    print(f"\nRisultati salvati in {output_path}")


if __name__ == "__main__":
    main()
