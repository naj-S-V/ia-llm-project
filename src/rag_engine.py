import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Mise à jour de la librairie recommandée

# --- CONFIGURATION ---
# Chemins relatifs par rapport à la racine du projet
VECTORSTORE_PATH = "./data/vectorstore"
DOCUMENTS_PATH = "./data/documents"

# On instancie le modèle d'embedding UNE SEULE FOIS ici.
# Ce modèle tourne en LOCAL (CPU), gratuit et respectueux de la vie privée[cite: 139].
# Il transforme le texte en listes de nombres.
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_db(pdf_filename, region_name):
    """
    Fonction pour ingérer un PDF, le découper et le stocker dans la base vectorielle.
    """
    pdf_path = os.path.join(DOCUMENTS_PATH, pdf_filename)
    
    # 1. Chargement du document
    print(f"--- Chargement de {pdf_filename} ---")
    if not os.path.exists(pdf_path):
        print(f"ERREUR : Le fichier {pdf_path} est introuvable.")
        return

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. Ajout des métadonnées (CRUCIAL pour filtrer par région plus tard)
    for doc in docs:
        doc.metadata["region"] = region_name
        doc.metadata["source"] = pdf_filename
        
    # 3. Découpage (Splitting)
    # On découpe en morceaux de 1000 caractères avec un chevauchement de 200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Document découpé en {len(splits)} morceaux.")
    
    # 4. Stockage dans Chroma (Persistence)
    # C'est ici qu'on crée la base de données physique sur le disque
    Chroma.from_documents(
        documents=splits, 
        embedding=embedding_function, 
        persist_directory=VECTORSTORE_PATH
    )
    print(f"Succès ! Les données pour {region_name} sont stockées dans {VECTORSTORE_PATH}")

# --- ZONE D'EXÉCUTION (Pour tester directement ce fichier) ---
if __name__ == "__main__":
    # C'est ici que tu lances la création de ta base de données.
    # Modifie les noms de fichiers selon ce que tu as dans ton dossier data/documents/
    
    # Exemple pour Bruxelles (Assure-toi d'avoir un fichier nommé ainsi ou change le nom)
    create_vector_db(pdf_filename="guide_bruxelles.pdf", region_name="bruxelles")
    
    # Tu pourras décommenter les lignes suivantes quand tu auras les autres PDF
    # create_vector_db(pdf_filename="guide_namur.pdf", region_name="wallonie")
    # create_vector_db(pdf_filename="guide_anvers.pdf", region_name="flandre")