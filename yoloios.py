from ultralytics import YOLO
import torch
import numpy as np
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

# Creiamo una classe per gestire il salvataggio delle feature maps
class FeatureExtractor():
    def __init__(self):
        self.features = None
    
    def hook(self, module, input, output):
        # Questa funzione verrà chiamata ogni volta che i dati attraversano il layer
        self.features = output.detach()  # Salviamo l'output del layer

# Creiamo una funzione per registrare il hook sul layer desiderato
def register_feature_extractor(model):
    # Inizializziamo il nostro estrattore
    feature_extractor = FeatureExtractor()
    

    # In YOLO, dobbiamo accedere prima al modello base e poi al layer specifico
    # Il modello è strutturato come: model.model.model[2]
    target_layer = model.model.model[2]

    # Registriamo l'hook sul layer C3k2
    target_layer.register_forward_hook(feature_extractor.hook)
    
    return feature_extractor


# Esempio di utilizzo
def extract_features(model, image):
    # Registriamo il hook
    feature_extractor = register_feature_extractor(model)
    
    # Facciamo una forward pass
    with torch.no_grad():  # Non abbiamo bisogno dei gradienti
        model(image)
    
    # Ora feature_extractor.features contiene le feature maps del layer C3k2
    return feature_extractor.features

def load_image_for_yolo(image_path, input_size=(640, 640)):
    # Leggiamo l'immagine usando OpenCV
    # cv2.IMREAD_COLOR legge l'immagine in formato BGR
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convertiamo da BGR (formato OpenCV) a RGB (formato atteso da YOLO)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Creiamo le trasformazioni necessarie per preprocessare l'immagine
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte l'immagine in un tensor e normalizza da [0,255] a [0,1]
        transforms.Resize(input_size),  # Ridimensiona l'immagine alle dimensioni richieste
    ])
    
    # Applichiamo le trasformazioni
    image_tensor = transform(image)
    
    # Aggiungiamo la dimensione del batch (YOLO si aspetta input in formato BCHW)
    image_tensor = image_tensor.unsqueeze(0)
    
    # Se hai una GPU disponibile, sposta il tensor sulla GPU
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    return image_tensor

# Esempio di utilizzo
def process_image(model, image_path="example.jpeg"):
    # Carichiamo l'immagine
    image_tensor = load_image_for_yolo(image_path=image_path)
    
    # Facciamo una forward pass attraverso il modello
    with torch.no_grad():  # Disabilitiamo il calcolo dei gradienti per risparmiare memoria
        features = extract_features(model, image_tensor)
    
    print(f"Dimensione dell'immagine in input: {image_tensor.shape}")
    print(f"Dimensione delle feature maps: {features.shape}")
    
    return features

def visualize_feature_maps(features, num_features=16, figsize=(60, 20), output_path='feature_maps.png'):
    """
    Visualizza le feature maps e salva il risultato su file
    """
    # Spostiamo il tensor sulla CPU e lo convertiamo in numpy
    features = features.cpu().numpy()
    
    # Creiamo una figura con 3 subplot
    fig = plt.figure(figsize=figsize)
    
    # 1. Media di tutti i canali
    ax1 = fig.add_subplot(131)
    mean_activation = np.mean(features[0], axis=0)
    im1 = ax1.imshow(mean_activation, cmap='viridis')
    ax1.set_title('Media di tutti i canali')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Grid delle prime n feature maps
    ax2 = fig.add_subplot(132)
    grid_size = int(np.ceil(np.sqrt(num_features)))
    grid = np.zeros((grid_size * features.shape[2], grid_size * features.shape[3]))
    
    for idx in range(min(num_features, features.shape[1])):
        i = idx // grid_size
        j = idx % grid_size
        grid[i*features.shape[2]:(i+1)*features.shape[2], 
             j*features.shape[3]:(j+1)*features.shape[3]] = features[0, idx]
    
    im2 = ax2.imshow(grid, cmap='viridis')
    ax2.set_title(f'Prime {num_features} feature maps')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Mappa di attivazione massima
    ax3 = fig.add_subplot(133)
    max_activation = np.max(features[0], axis=0)
    im3 = ax3.imshow(max_activation, cmap='viridis')
    ax3.set_title('Attivazione massima')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    
    # Salviamo la figura invece di mostrarla
    plt.savefig(output_path)
    plt.close()
    
    # Creiamo e salviamo anche l'istogramma
    plt.figure(figsize=(10, 5))
    plt.hist(features[0].flatten(), bins=50)
    plt.title('Distribuzione delle attivazioni')
    plt.xlabel('Valore di attivazione')
    plt.ylabel('Frequenza')
    plt.savefig('activation_distribution.png')
    plt.close()

def analyze_features(features):
    print(f"Statistiche delle feature maps:")
    print(f"- Shape: {features.shape}")
    print(f"- Valore minimo: {features.min():.4f}")
    print(f"- Valore massimo: {features.max():.4f}")
    print(f"- Media: {features.mean():.4f}")
    print(f"- Deviazione standard: {features.std():.4f}")
    
    # Visualizziamo e salviamo le feature maps
    visualize_feature_maps(features, output_path='feature_maps.png')
    print("\nLe visualizzazioni sono state salvate come 'feature_maps.png' e 'activation_distribution.png'")


if __name__ == "__main__":

    output_dir = ''

    model = YOLO("yolo11x.pt")

    # Estraiamo le features dal layer
    features_2_c3k2 = process_image(model=model, image_path="example.jpeg")

    # Chiamata alla funzione con le tue feature maps
    visualize_feature_maps(features_2_c3k2, output_path=output_dir + 'feature_maps.png')
