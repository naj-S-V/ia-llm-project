from ultralytics import YOLO
from pathlib import Path
import cv2
import sys

IMAGE_NAME = "paper.jpg"

# --- 1. CONFIGURATION ROBUSTE DES CHEMINS ---
# On prend le chemin du fichier actuel, et on remonte aux parents pour trouver la racine
# Supposons que ce fichier est dans eco-sorter/src/ ou eco-sorter/notebooks/
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent  # Remonte de 2 crans : src -> eco-sorter

# Définition des chemins clés
MODEL_PATH = PROJECT_ROOT / 'models_training_runs' / 'eco_sorter_v1' / 'weights' / 'best.pt'
IMAGE_PATH = PROJECT_ROOT / "data" / "test_images" / IMAGE_NAME

print(f"--- DÉBOGAGE DES CHEMINS ---")
print(f"Racine du projet détectée : {PROJECT_ROOT}")
print(f"Chemin modèle espéré      : {MODEL_PATH}")
print(f"Chemin image espéré       : {IMAGE_PATH}")
print(f"----------------------------")

# --- 2. VÉRIFICATIONS AVANT LANCEMENT ---
if not MODEL_PATH.exists():
    print(f"❌ ERREUR CRITIQUE : Le fichier modèle est introuvable !")
    print(f"   Vérifie que tu as bien déplacé 'best.pt' ici : {MODEL_PATH}")
    sys.exit(1)

if not IMAGE_PATH.exists():
    print(f"❌ ERREUR : L'image de test est introuvable !")
    print(f"   Vérifie que le fichier existe bien ici : {IMAGE_PATH}")
    sys.exit(1)

# --- 3. CHARGEMENT ET PRÉDICTION ---
print("✅ Fichiers trouvés. Chargement du modèle...")
try:
    model = YOLO(str(MODEL_PATH)) # YOLO préfère parfois les strings aux objets Path
except Exception as e:
    print(f"Erreur interne YOLO : {e}")
    sys.exit(1)

print(f"🔍 Analyse de l'image...")
# project & name permettent de forcer l'endroit où le résultat est sauvegardé
# On sauvegarde dans data/outputs/test_results pour que tu le trouves facilement
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" 

results = model.predict(
    source=str(IMAGE_PATH), 
    save=True, 
    conf=0.5,
    project=str(OUTPUT_DIR),  # On force le dossier de sauvegarde
    name='test_inference',    # Nom du sous-dossier
    exist_ok=True             # Écrase si existe déjà
)

# --- 4. AFFICHAGE DU RÉSULTAT ---
save_dir = Path(results[0].save_dir)
print(f"\n🎉 SUCCÈS !")
print(f"L'image avec les cadres a été sauvegardée ici :")
print(f"👉 {save_dir}")
print(f"Ouvre ce dossier dans ton explorateur de fichiers pour voir le résultat.")