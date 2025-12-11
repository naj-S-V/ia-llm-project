from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models_training_runs' / 'eco_sorter_v5' / 'weights' / 'best.pt'

# Classes détectables par le modèle
CLASS_NAMES = ['Cardboard', 'Garbage', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Traduction française pour l'affichage
CLASS_NAMES_FR = {
    'Cardboard': 'carton',
    'Garbage': 'ordure ménagère',
    'Glass': 'verre',
    'Metal': 'métal',
    'Paper': 'papier',
    'Plastic': 'plastique',
    'Trash': 'déchet'
}

class VisionModel:
    """
    Classe pour charger et utiliser le modèle YOLO pour la classification des déchets.
    """
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Le modèle est introuvable à : {MODEL_PATH}\n"
                f"Vérifie que le fichier 'best.pt' existe bien."
            )
        
        print(f"Chargement du modèle depuis : {MODEL_PATH}")
        self.model = YOLO(str(MODEL_PATH))
        print("✅ Modèle chargé avec succès")
    
    def predict(self, image, conf_threshold=0.5):
        """
        Fait une prédiction sur une image.
        
        Args:
            image: Image PIL ou chemin vers une image
            conf_threshold: Seuil de confiance minimum (0-1)
        
        Returns:
            dict: {
                'class_name': str,  # Nom de la classe en anglais
                'class_name_fr': str,  # Nom de la classe en français
                'confidence': float,  # Score de confiance
                'detected': bool  # True si quelque chose a été détecté
            }
        """
        # Si c'est une image PIL, on la convertit en array numpy
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Prédiction
        results = self.model.predict(
            source=image_array,
            conf=conf_threshold,
            verbose=False  # Pour éviter les logs dans la console
        )
        
        # Traitement du résultat
        if len(results) > 0 and len(results[0].boxes) > 0:
            # On prend la détection avec la plus haute confiance
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            
            # Index de la meilleure prédiction
            best_idx = np.argmax(confidences)
            best_class = classes[best_idx]
            best_conf = float(confidences[best_idx])
            
            class_name = CLASS_NAMES[best_class]
            class_name_fr = CLASS_NAMES_FR.get(class_name, class_name.lower())
            
            return {
                'class_name': class_name,
                'class_name_fr': class_name_fr,
                'confidence': best_conf,
                'detected': True
            }
        else:
            return {
                'class_name': None,
                'class_name_fr': 'aucun déchet détecté',
                'confidence': 0.0,
                'detected': False
            }

# Instance globale du modèle (singleton)
_model_instance = None

def get_model():
    """
    Retourne l'instance du modèle (singleton).
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = VisionModel()
    return _model_instance

def predict_waste_type(image):
    """
    Fonction simple pour prédire le type de déchet.
    Compatible avec l'interface existante de app.py.
    
    Args:
        image: Image PIL
    
    Returns:
        str: Description du déchet détecté en français
    """
    model = get_model()
    result = model.predict(image)
    
    if result['detected']:
        return f"{result['class_name_fr']} (confiance: {result['confidence']:.2%})"
    else:
        return "aucun déchet détecté"
