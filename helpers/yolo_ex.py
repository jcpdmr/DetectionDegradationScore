from ultralytics import YOLO
import cv2

# Carica il modello YOLO
model = YOLO('yolo11m.pt')

# Apri il video
video_path = "VIRAT_S_040103_08_001475_001512.mp4"
cap = cv2.VideoCapture(video_path)

# Ottieni le dimensioni del video e crea un VideoWriter per il video di output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = 'output.mp4'
output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Processa il video frame per frame
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Esegui la prediction sul frame
    results = model(frame)
    
    # Disegna le bounding box e stampa le predizioni
    for r in results:
        annotated_frame = r.plot()  # disegna le box
        
        # Stampa le predizioni per ogni oggetto rilevato
        for box in r.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"Rilevato {class_name} con confidenza {conf:.2f}")
    
    # Salva il frame nel video di output
    output.write(annotated_frame)
    
    # Opzionale: mostra il frame in tempo reale
    cv2.imshow('YOLO Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia le risorse
cap.release()
output.release()
cv2.destroyAllWindows()