import streamlit as st
from PIL import Image

# --- OPTIMISATION VITESSE (CACHE) ---
@st.cache_resource
def load_agent_engine():
    from agent_logic import ask_agent
    return ask_agent

@st.cache_resource
def load_vision_model():
    """Charge le modèle CNN une seule fois au démarrage"""
    from vision_model import predict_waste_type
    return predict_waste_type

def prepocess_image(image):
    """Prépare l'image pour le modèle (si besoin)"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

# On charge les moteurs IA
ask_agent = load_agent_engine()
predict_waste_type = load_vision_model()

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Eco-Sorter AI", page_icon="♻️", layout="wide")

# CSS pour le style
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; margin-bottom: 10px;}
    h1 {color: #2e7d32;}
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR (Régions) ---
with st.sidebar:
    #st.image("https://sdgs.un.org/sites/default/files/goals/F_SDG_goals_icons-individual-rgb-11.png", width=100)
    st.title("🌍 Ma Localisation")
    
    region_mapping = {
        "Bruxelles": "bruxelles",
        "Hainaut": "hainaut", 
        "Anvers": "antwerp",
        "Liège": "liege",
        "Namur": "namur", 
        "Brabant Wallon": "brabant_wallon",
        "Charleroi": "charleroi",
        "Luxembourg": "luxembourg",
        "Mons": "mons"
    }
    
    selected_label = st.selectbox(
        "Choix de la zone :",
        options=list(region_mapping.keys()),
        on_change=lambda: st.session_state.update({"messages": []}) 
    )
    region_tag = region_mapping[selected_label]
    
    st.divider()
    
    # ZONE UPLOAD IMAGE (Placée dans la sidebar pour la propreté)
    st.header("📸 Vision")
    uploaded_file = st.file_uploader("Prendre une photo", type=["jpg", "png", "jpeg"])

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Bonjour ! Quel déchet voulez-vous trier ?"})

# --- 4. AFFICHAGE HISTORIQUE (Avec CO2 corrigé) ---
st.title(f"♻️ Eco-Sorter ({selected_label})")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # AFFICHAGE METRIQUES (Format Uniformisé)
        if "metrics" in msg:
            with st.expander("📊 Empreinte CO2"):
                m = msg["metrics"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Input", f"{m['input_tokens']}")
                c2.metric("Output", f"{m['output_tokens']}")
                
                # Correction demandée : Afficher CO2 au lieu du Total
                co2_val = m['total_tokens'] * 0.00057  # 0.00057 g CO2 par token
                c3.metric("Est. CO2", f"{co2_val:.4f} g")

# --- 5. LOGIQUE IMAGE (Nouveau Flow Corrigé) ---
if uploaded_file:
    # On affiche l'image
    image = prepocess_image(Image.open(uploaded_file))
    st.image(image, caption="Image analysée", width=300)
    
    # Si on n'a pas encore validé cette image, on lance la prédiction (Mock)
    if "current_image_prediction" not in st.session_state:
        with st.spinner("🧠 Analyse visuelle en cours (CNN)..."):
            prediction = predict_waste_type(image)
            st.session_state.current_image_prediction = prediction
    
    # Récupération de la prédiction stockée
    prediction = st.session_state.current_image_prediction
    
    # Interface de validation
    st.info(f"Je pense voir : **{prediction}**")
    
    col_yes, col_no = st.columns(2)
    
    # BOUTON OUI
    if col_yes.button("✅ Oui, c'est ça"):
        
        question_mapping = {
            "carton": "Où jeter un déchet qui ressemble à du carton ?",
            "plastique": "Où jeter un déchet qui ressemble à du plastique ?",
            "papier": "Où jeter un déchet qui ressemble à du papier ?",
            "verre": "Où jeter un déchet qui ressemble à du verre ?",
            "métal": "Où jeter un déchet qui ressemble à du métal ?",
            "ordure ménagère": "Où jeter un déchet qui ressemble à une ordure ménagère ?",
            "déchet": "Où jeter ce déchet ?" #TODO : améliorer pour les ordures non identifiées
        }
        
        # 1. On affiche le message de l'utilisateur TOUT DE SUITE
        print("Prediction confirmée :", prediction)
        user_text = question_mapping.get(prediction.split()[0].lower(), "Où jeter ce déchet ?")
        st.chat_message("user").markdown(user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})
        
        # 2. On lance l'IA et on affiche la réponse TOUT DE SUITE (sans rerun)
        with st.chat_message("assistant"):
            with st.spinner("Consultation du guide de tri..."):
                response_data = ask_agent(user_text, region=region_tag)
                
                # Affichage texte
                st.markdown(response_data["answer"])
                
                # Affichage Métriques (CO2)
                metrics = response_data["metrics"]
                with st.expander("📊 Détails de consommation (Live)"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Input", metrics['input_tokens'])
                    c2.metric("Output", metrics['output_tokens'])
                    c3.metric("Est. CO2", f"{metrics['total_tokens'] * 0.0004:.4f} g")

                # 3. Sauvegarde dans l'historique
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_data["answer"],
                    "metrics": metrics
                })
        
        # 4. Nettoyage (On supprime la prédiction pour éviter de boucler)
        del st.session_state.current_image_prediction
        # IMPORTANT : J'ai enlevé st.rerun() ! 
        # Ainsi, la réponse reste affichée sous tes yeux.

    # BOUTON NON
    if col_no.button("❌ Non, corriger"):
        st.warning("D'accord, décrivez l'objet ci-dessous dans la zone de texte.")
        del st.session_state.current_image_prediction

# --- 6. LOGIQUE TEXTE (Classique) ---
if prompt := st.chat_input("Ex: Bouteille de Javel..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyse..."):
            response_data = ask_agent(prompt, region=region_tag)
            
            st.markdown(response_data["answer"])
            
            # Sauvegarde
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_data["answer"],
                "metrics": response_data["metrics"]
            })
            
            # Métriques Live
            with st.expander("📊 Empreinte CO2 (Live)"):
                m = response_data["metrics"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Input", m['input_tokens'])
                c2.metric("Output", m['output_tokens'])
                c3.metric("Est. CO2", f"{m['total_tokens'] * 0.0004:.4f} g")