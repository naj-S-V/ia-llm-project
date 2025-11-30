import os
import sys
from dotenv import load_dotenv

# Chargement des variables d'environnement (récupère la clé depuis .env)
load_dotenv()

# Vérification de sécurité
if not os.getenv("MISTRAL_API_KEY"):
    print("❌ ERREUR : La clé MISTRAL_API_KEY est introuvable.")
    print("Assure-toi d'avoir créé le fichier .env à la racine du projet.")
    sys.exit(1)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION DES CHEMINS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
VECTORSTORE_PATH = os.path.join(root_dir, "data", "vectorstore")

# --- 1. CHARGEMENT DE LA MÉMOIRE (RAG) ---
print("Chargement de la base vectorielle...")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embedding_function)

# RETRIEVER : C'est ici qu'on règle la sensibilité !
# k=4: On récupère les 4 morceaux les plus proches pour donner un max de contexte au LLM
# Cela compense le fait que la "Javel" puisse arriver en 4ème.
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# --- 2. INITIALISATION DU CERVEAU (LLM) ---
# 'mistral-large-latest' est le plus intelligent. 
# Si tu veux économiser, utilise 'open-mistral-nemo' ou 'mistral-small-latest'.
llm = ChatMistralAI(
    model="mistral-small-latest", 
    temperature=0.1  # Faible température = Réponse factuelle et précise
)

# --- 3. DÉFINITION DE LA PERSONNALITÉ (Prompt) ---
# [cite_start]On utilise les sources [cite: 7, 50] pour définir un agent RAG strict.
template = """
Tu es Eco-Sorter, un assistant expert en gestion des déchets pour la région de Bruxelles.
Ta mission est d'aider les citoyens à trier correctement pour soutenir l'objectif de développement durable.

CONSIGNES STRICTES :
1. Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre.
2. Si la réponse se trouve dans le contexte, sois précis : dis exactement dans quel sac (Jaune, Bleu, Blanc, Orange, Vert) ou quel lieu (Proxy Chimik, Recypark, Bulles à verre) l'objet doit aller.
3. Si le contexte mentionne que c'est "INTERDIT" dans un sac, cherche dans le reste du contexte où c'est "AUTORISÉ".
4. Si tu ne trouves PAS la réponse dans le contexte, dis poliment : "Je n'ai pas l'information précise dans mon guide pour cet objet. Par précaution, vérifiez sur le site de Bruxelles-Propreté." (N'invente rien).

CONTEXTE ISSU DU GUIDE DE TRI :
{context}

QUESTION DE L'UTILISATEUR : 
{question}

RÉPONSE :
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 4. CRÉATION DE LA CHAÎNE (Pipeline) ---
def format_docs(docs):
    # Fonction pour "coller" les morceaux de texte ensemble
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- FONCTION D'INTERACTION ---
def ask_agent(user_input):
    print(f"\n👤 Utilisateur : {user_input}")
    print("⏳ Eco-Sorter réfléchit...")
    try:
        response = rag_chain.invoke(user_input)
        print(f"🤖 Eco-Sorter : {response}")
        return response
    except Exception as e:
        print(f"❌ Erreur technique : {e}")
        return "Désolé, une erreur est survenue."

if __name__ == "__main__":
    # --- ZONE DE TEST ---
    # Test 1 : Facile
    ask_agent("Où je mets mes épluchures d'orange ?")
    
    # Test 2 : Le piège sémantique (Javel)
    ask_agent("J'ai un vieux bidon d' eau de Javel vide, poubelle bleue ?")
    
    # Test 3 : Le piège de la négation (Plastique interdit)
    ask_agent("Où je jette un seau en plastique cassé ?")