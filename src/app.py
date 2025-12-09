import streamlit as st
import time
from agent_logic import ask_agent
import streamlit as st
import time

# --- OPTIMISATION VITESSE (CACHE) ---
# Cette fonction charge l'agent UNE SEULE FOIS. 
# Les rechargements suivants seront instantanés.
@st.cache_resource
def load_agent_engine():
    from agent_logic import ask_agent
    return ask_agent

# On récupère la fonction optimisée
ask_agent = load_agent_engine()

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Eco-Sorter AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS PERSONNALISÉ (Pour le style) ---
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; margin-bottom: 10px;}
    .reportview-container {background: #f0f2f6;}
    h1 {color: #2e7d32;} /* Vert écologie */
</style>
""", unsafe_allow_html=True)

# --- 3. BARRE LATÉRALE (Configuration) ---
with st.sidebar:
    st.image("https://sdgs.un.org/sites/default/files/goals/F_SDG_goals_icons-individual-rgb-11.png", width=100)
    st.title("Paramètres")
    
    # Sélecteur de Région (C'est ici qu'on pilote le RAG)
    selected_region = st.selectbox(
        "📍 Votre Région :",
        ("Anvers", "Bruxelles", "Brabant Wallon", "Charleroi", "Liège", "Namur", "Hainaut", "Luxembourg", "Mons"),
        index=0,
        help="Les règles de tri changent selon la région. L'IA adaptera ses réponses."
    )
    
    # Mapping pour convertir "Bruxelles" (Joli) -> "bruxelles" (Interne)
    region_mapping = {
        "Anvers": "anvers",
        "Brabant Wallon": "brabant_wallon",
        "Charleroi": "charleroi",
        "Liège": "liege",
        "Namur": "namur",
        "Hainaut": "hainaut",
        "Luxembourg": "luxembourg",
        "Mons": "mons",
        "Bruxelles": "bruxelles"
    }
    region_tag = region_mapping[selected_region]

    st.divider()
    st.info("💡 **Astuce :** Demandez 'Où jeter mes piles ?' ou 'Le carton à pizza ?'")
    
    # Bouton Reset
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# --- 4. GESTION DE L'HISTORIQUE (Session State) ---
# Streamlit recharge le script à chaque clic. Il faut sauvegarder l'historique.
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Message de bienvenue par défaut
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"Bonjour ! Je suis Eco-Sorter. Je suis configuré pour les règles de tri de **{selected_region}**. Quel déchet voulez-vous trier ?"
    })

# --- 5. ZONE PRINCIPALE (Chat) ---
st.title("♻️ Eco-Sorter : Assistant de Tri Intelligent")
st.caption("Projet IA Durable - SDG 11 : Villes et Communautés Durables")

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Si le message contient des métriques (c'est une réponse de l'assistant stockée avec métriques)
        if "metrics" in message:
            with st.expander("📊 Empreinte Numérique (Green IT)"):
                m = message["metrics"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Input", f"{m['input_tokens']} tok")
                c2.metric("Output", f"{m['output_tokens']} tok")
                c3.metric("Total", f"{m['total_tokens']} tok")

# --- 6. LOGIQUE D'INTERACTION ---
# Zone de saisie (en bas de page)
if prompt := st.chat_input("Décrivez votre déchet ici..."):
    
    # A. On affiche tout de suite la question de l'utilisateur
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. On appelle l'IA (Pendant que ça charge)
    with st.chat_message("assistant"):
        with st.spinner(f"Analyse des règles pour {selected_region}..."):
            
            # Appel au Backend (agent_logic.py)
            response_data = ask_agent(prompt, region=region_tag)
            
            # Affichage de la réponse texte
            st.markdown(response_data["answer"])
            
            # Affichage des métriques CO2 (Live)
            metrics = response_data["metrics"]
            with st.expander("📊 Détails de consommation (Live)"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Input", metrics['input_tokens'])
                c2.metric("Output", metrics['output_tokens'])
                co2_est = metrics['total_tokens'] * 0.0004 # Estimation fictive pédagogique
                c3.metric("Est. CO2", f"{co2_est:.4f} g")

    # C. Sauvegarde dans l'historique pour le prochain rechargement
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_data["answer"],
        "metrics": metrics # On sauvegarde aussi les métriques !
    })