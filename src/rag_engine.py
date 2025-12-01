import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
# Chemins relatifs (adaptés à ta structure de dossier)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) # On remonte d'un cran vers 'ia-llm-project'
VECTORSTORE_PATH = os.path.join(root_dir, "data", "vectorstore")
DOCUMENTS_PATH = os.path.join(root_dir, "data", "documents")

# On instancie le modèle d'embedding (Tourne en LOCAL sur ton CPU)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_db(filename, region_name):
    """
    Fonction pour ingérer un document (PDF ou TXT), le découper et le stocker.
    """
    file_path = os.path.join(DOCUMENTS_PATH, filename)
    
    print(f"--- Chargement de {filename} ---")
    if not os.path.exists(file_path):
        print(f"ERREUR : Le fichier {file_path} est introuvable.")
        return

    # 1. Choix du chargeur selon l'extension
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        # encoding='utf-8' est important pour les accents français
        loader = TextLoader(file_path, encoding="utf-8")
        
    docs = loader.load()
    
    # 2. Ajout des métadonnées (CRUCIAL pour filtrer par région)
    for doc in docs:
        doc.metadata["region"] = region_name
        doc.metadata["source"] = filename
        
    # 3. Découpage (Splitting)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"Document découpé en {len(splits)} morceaux.")
    
    # 4. Stockage dans Chroma (Persistence)
    Chroma.from_documents(
        documents=splits, 
        embedding=embedding_function, 
        persist_directory=VECTORSTORE_PATH
    )
    print(f"Succès ! Base de données mise à jour dans {VECTORSTORE_PATH}")

def query_vector_db(question, region_name, n_results=4):
    """
    Fonction pour interroger la base vectorielle.
    Elle cherche les 'n_results' morceaux de textes les plus proches sémantiquement de la question.
    """
    # 1. On charge la base existante (pas besoin de recréer)
    db = Chroma(
        persist_directory=VECTORSTORE_PATH, 
        embedding_function=embedding_function
    )
    
    # 2. Recherche par similarité (Similarity Search)
    # Le filtre est CRUCIAL : on ne veut chercher QUE dans les documents de la région donnée
    print(f"\n--- Recherche pour : '{question}' ({region_name}) ---")
    results = db.similarity_search(
        query=question,
        k=n_results,
        filter={"region": region_name} # Filtre metadata
    )
    
    # 3. Affichage des résultats trouvés
    for i, doc in enumerate(results):
        print(f"\n[Résultat {i+1}] (Source: {doc.metadata.get('source', 'Inconnue')})")
        print(doc.page_content)
        print("-" * 20)
        
    return results

if __name__ == "__main__":
    # Test immédiat avec ton fichier texte
    #create_vector_db(filename="guide_bruxelles.txt", region_name="bruxelles")

    #create_vector_db(filename="guide_liege.txt", region_name="bruxelles")
    #create_vector_db(filename="guide_hainaut.txt", region_name="hainaut")
    #create_vector_db(filename="guide_namur.txt", region_name="namur")
    #create_vector_db(filename="guide_bw.txt", region_name="brabant_wallon")
    #create_vector_db(filename="guide_charleroi.txt", region_name="charleroi")
    #create_vector_db(filename="guide_mons.txt", region_name="mons")
    #create_vector_db(filename="guide_luxembourg.txt", region_name="luxembourg")
    
    # Tu pourras décommenter les lignes suivantes quand tu auras les autres PDF
    # create_vector_db(pdf_filename="guide_liege.pdf", region_name="liege")
    # create_vector_db(pdf_filename="guide_anvers.pdf", region_name="flandre")

    # 2. TEST DE RECUPERATION
    # Testons si le système comprend une question sur la pizza
    #query_vector_db("Dans quel sac mettre une bouteille en plastique ?", "bruxelles")

    # Testons une question piège (si tu as mis les règles sur les piles/produits chimiques)
    query_vector_db("Où jeter une bouteille de produit chimique ?", "bruxelles")
    