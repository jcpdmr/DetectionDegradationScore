import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Dati completi
all_models = [
    "Yolo11m (1.03M)",
    "Yolo11m-Light (0.31M)",
    "VGG16 (1.03M)",
    "EfficientNetV2M (0.86M)",
    "EfficientNetV2M-Light (0.27M)",
    "LPIPS-custom-train",
    "SSIM",
    "LPIPS",
    "DDS-Yolo11m-CPU",
]

# Creare una mappa di colori consistente per tutti i modelli
n = len(all_models)
color_map = {
    model: color for model, color in zip(all_models, cm.viridis(np.linspace(0, 1, n)))
}

# Dati per MAE
mae_models = [
    "Yolo11m (1.03M)",
    "Yolo11m-Light (0.31M)",
    "VGG16 (1.03M)",
    "EfficientNetV2M (0.86M)",
    "EfficientNetV2M-Light (0.27M)",
    "LPIPS-custom-train",
    "SSIM",
    "LPIPS",
]
mae_scores = [
    0.1379,
    0.1383,
    0.1648,
    0.1659,
    0.1723,
    0.2199,  # Aggiornato
    0.3039,  # Aggiornato
    0.3580,  # Nuovo valore
]
colors_mae = [color_map[model] for model in mae_models]

# Dati per coefficienti Pearson
pearson_models = [
    "Yolo11m (1.03M)",
    "Yolo11m-Light (0.31M)",
    "VGG16 (1.03M)",
    "EfficientNetV2M (0.86M)",
    "EfficientNetV2M-Light (0.27M)",
    "LPIPS-custom-train",
    "SSIM",
    "LPIPS",
]
pearson_corr = [
    0.6509,
    0.6465,
    0.5018,
    0.4851,
    0.4441,
    0.1389,  # Aggiornato
    0.0205,  # Aggiornato
    0.0637,  # Nuovo valore
]
colors_pearson = [color_map[model] for model in pearson_models]

# Dati per images/sec
speed_models = [
    "Yolo11m (1.03M)",
    "Yolo11m-Light (0.31M)",
    "VGG16 (1.03M)",
    "EfficientNetV2M (0.86M)",
    "EfficientNetV2M-Light (0.27M)",
    "DDS-Yolo11m-CPU",
]
images_per_sec = [
    203.71,
    218.59,
    268.04,
    243.73,
    246.20,
    72.31,
]
colors_speed = [color_map[model] for model in speed_models]

# 1. MAE GRAPH
plt.figure(figsize=(6, 6))
plt.bar(mae_models, mae_scores, color=colors_mae)
plt.ylabel("MAE")
plt.tick_params(axis="x", rotation=80)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparisons_mae.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. PEARSON CORRELATION GRAPH
plt.figure(figsize=(6, 6))
plt.bar(pearson_models, pearson_corr, color=colors_pearson)
plt.ylabel("Pearson Correlation")
plt.tick_params(axis="x", rotation=80)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparisons_pearson.png", dpi=300, bbox_inches="tight")
plt.close()

# 3. IMAGES PER SECOND GRAPH
plt.figure(figsize=(6, 6))
plt.bar(speed_models, images_per_sec, color=colors_speed)
plt.ylabel("Images/second")
plt.tick_params(axis="x", rotation=80)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparisons_speed.png", dpi=300, bbox_inches="tight")
plt.close()

# 4. INFERENCE TIME GRAPH
# Calcolare tempo di inferenza in millisecondi dai dati images/second
inference_models = [
    "Yolo11m (1.03M)",
    "Yolo11m-Light (0.31M)",
    "VGG16 (1.03M)",
    "EfficientNetV2M (0.86M)",
    "EfficientNetV2M-Light (0.27M)",
]
inference_speeds = [
    203.71,
    218.59,
    268.04,
    243.73,
    246.20,
]
# Convertire in millisecondi (1000 / img_per_sec)
inference_time_ms = [1000 / ips for ips in inference_speeds]
colors_inference = [color_map[model] for model in inference_models]

# Per DDS-Yolo11m-CPU, prepariamo i dati per la barra divisa
# Convertire i tempi da secondi a millisecondi
detector_time_ms = 0.007082948951784897 * 1000  # circa 7.08 ms
matching_time_ms = 0.00674583116427407 * 1000  # circa 6.75 ms

# Creare grafico con matplotlib
plt.figure(figsize=(8, 6))

# Graficare le barre singole per i modelli standard
plt.bar(inference_models, inference_time_ms, color=colors_inference)

# Aggiungere la barra divisa per DDS-Yolo11m-CPU
dds_position = len(inference_models)  # Posizione della barra DDS
plt.bar(
    dds_position,
    detector_time_ms,
    color=color_map["DDS-Yolo11m-CPU"],
    label="Detector Inference (GPU)",
)
plt.bar(
    dds_position,
    matching_time_ms,
    bottom=detector_time_ms,
    color=color_map["DDS-Yolo11m-CPU"],
    alpha=0.5,
    hatch="///",
    label="Matching (CPU)",
)

# Impostare l'etichetta dell'asse x per DDS-Yolo11m-CPU
plt.xticks(
    list(range(len(inference_models) + 1)), inference_models + ["DDS-Yolo11m-CPU"]
)

plt.ylabel("Inference Time (ms)")
plt.tick_params(axis="x", rotation=80)
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("model_comparisons_inference_time.png", dpi=300, bbox_inches="tight")
plt.close()
