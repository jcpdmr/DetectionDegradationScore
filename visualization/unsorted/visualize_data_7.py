import matplotlib.pyplot as plt
import numpy as np

# Dati dalla distribuzione di punteggi
score_distribution = {
    "0.00-0.02": 10747, "0.02-0.04": 16113, "0.04-0.06": 16636, "0.06-0.08": 15427,
    "0.08-0.10": 14213, "0.10-0.12": 12811, "0.12-0.14": 11790, "0.14-0.16": 10765,
    "0.16-0.18": 10173, "0.18-0.20": 10007, "0.20-0.22": 9941, "0.22-0.24": 10393,
    "0.24-0.26": 10914, "0.26-0.28": 12013, "0.28-0.30": 12929, "0.30-0.32": 13401,
    "0.32-0.34": 13684, "0.34-0.36": 16237, "0.36-0.38": 17193, "0.38-0.40": 16912,
    "0.40-0.42": 16505, "0.42-0.44": 16201, "0.44-0.46": 15744, "0.46-0.48": 15178,
    "0.48-0.50": 14432, "0.50-0.52": 20315, "0.52-0.54": 20610, "0.54-0.56": 18120,
    "0.56-0.58": 16021, "0.58-0.60": 14399, "0.60-0.62": 13271, "0.62-0.64": 12408,
    "0.64-0.66": 11517, "0.66-0.68": 13076, "0.68-0.70": 12848, "0.70-0.72": 10680,
    "0.72-0.74": 9655, "0.74-0.76": 9481, "0.76-0.78": 8985, "0.78-0.80": 7745,
    "0.80-0.82": 7799, "0.82-0.84": 6791, "0.84-0.86": 6078, "0.86-0.88": 5200,
    "0.88-0.90": 4556, "0.90-0.92": 3855, "0.92-0.94": 2625, "0.94-0.96": 1686,
    "0.96-0.98": 684, "0.98-1.00": 40
}

# Estrai i valori
counts = list(score_distribution.values())

# Crea i bin edges (i bordi degli intervalli)
bin_edges = [float(bin_range.split('-')[0]) for bin_range in score_distribution.keys()]
bin_edges.append(1.0)  # Aggiungi l'ultimo valore per completare gli intervalli

# Crea il grafico
plt.figure(figsize=(15, 7))

# Usa plt.bar con 'align="edge"' e 'width' uguale alla larghezza dell'intervallo (0.02)
plt.bar(bin_edges[:-1], counts, width=0.018, align='edge', alpha=0.7)

plt.xlabel("Score Range")
plt.ylabel("Number of Images")
plt.title("Distribution of Images Across Score Bins")
plt.grid(True, alpha=0.3)

# Imposta i tick ogni 0.04 (ogni 2 bin)
tick_positions = np.arange(0, 1.01, 0.04)
plt.xticks(tick_positions, [f"{x:.2f}" for x in tick_positions], rotation=45)

# Imposta i limiti dell'asse x
plt.xlim(0, 1.0)

plt.tight_layout()
plt.savefig("score_distribution.png", dpi=300)
print("Grafico salvato come 'score_distribution.png'")

# Se vuoi anche visualizzare il grafico
# plt.show()