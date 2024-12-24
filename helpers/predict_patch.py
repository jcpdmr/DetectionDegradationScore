from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Carica il modello YOLO
model = YOLO('yolo11m.pt')

# Crea la directory di output se non esiste
os.makedirs('predicted_patches', exist_ok=True)

# Processa tutte le patch
for i in range(50):
    # Leggi l'immagine
    input_path = f'extracted_patches/patch_{i}.jpg'
    output_path = f'predicted_patches/patch_{i}.jpg'
    
    # Verifica che il file esista
    if not os.path.exists(input_path):
        print(f"File {input_path} non trovato")
        continue
    
    # Leggi l'immagine
    img = cv2.imread(input_path)
    if img is None:
        print(f"Errore nella lettura di {input_path}")
        continue
    
    # Esegui la detection
    results = model(img, imgsz=256)
    
    # Disegna le bounding box e salva
    for r in results:
        # Disegna le box
        annotated_img = r.plot()
        
        # Stampa le predizioni
        for box in r.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"Patch {i}: Rilevato {class_name} con confidenza {conf:.2f}")
    
    # Salva l'immagine con le predizioni
    cv2.imwrite(output_path, annotated_img)
    print(f"Salvata predizione per patch_{i}")

print("Completato!")