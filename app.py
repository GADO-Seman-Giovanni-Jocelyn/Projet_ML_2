import streamlit as st
import pandas as pd
from visualisation import afficher_page_visualisation
from introduction import page_introduction
from modelisation import page_modelisation
from prediction import page_prediction  

# Définir la configuration de la page en premier
st.set_page_config(page_title="Classification des maladies cardiaques", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    /* Arrière-plan général */
    .main {
        background-color: #f0f6ff;
        background-image: linear-gradient(135deg, #e3f2fd, #ffffff);
    }

    /* Titres en bleu profond avec ombre légère */
    h1, h2, h3, h4, h5, h6 {
        color: #1565c0;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.05);
    }

    /* Sous-titres */
    .subheader {
        font-size: 20px;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }

    /* Boutons Streamlit */
    .stButton button {
        background-color: #1565c0;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 600;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .stButton button:hover {
        background-color: #0d47a1;
        transform: translateY(-2px);
    }

    /* Champs de saisie */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #b0bec5;
        padding: 10px;
        font-size: 16px;
        transition: border 0.3s ease, box-shadow 0.3s ease;
    }

    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #1565c0;
        box-shadow: 0 0 6px rgba(21, 101, 192, 0.4);
    }

    /* Sélecteurs */
    .stSelectbox select {
        color: #222222;
        background-color: #f7faff;
        border: 1px solid #90caf9;
        border-radius: 8px;
        padding: 8px;
    }

    /* Alertes/info */
    .stAlert {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        color: #0d47a1;
        padding: 12px;
        border-radius: 8px;
        font-weight: 500;
    }

    /* Conteneur principal */
    .block-container {
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        background-color: #ffffff;
    }

    /* Espacement */
    .stSlider, .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        font-family: 'Roboto', sans-serif;
        color: #333333;
    }

    .sidebar .sidebar-content h2 {
        color: #1565c0;
    }

    /* Titres centrés */
    .stMarkdown h1, .stMarkdown h2 {
        text-align: center;
    }

    /* Graphiques avec fond clair */
    .element-container svg {
        border-radius: 8px;
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)



# Ajouter un titre principal
st.title("Classification des maladies cardiaques 📊")
st.sidebar.image("img_dash.png", width=200)
st.markdown("""
    Bienvenue dans l'application de classification des maladies cardiaques !  
    Cette application vous permet d'explorer différentes techniques d'analyse, y compris l'introduction des données, la modélisation, et la visualisation des résultats.  
    Vous pouvez naviguer entre les différentes pages pour découvrir les insights cachés dans les données.
""")

PAGES = {
    "Introduction": page_introduction,
    "Modélisation": lambda: page_modelisation(),  # Passer data ici
    "Visualisation": lambda:afficher_page_visualisation(data),
    "Prédiction": page_prediction   # Pas besoin de data ici
}
# Sélection de la page via un menu déroulant dans la barre latérale
page = st.sidebar.selectbox(
    "Choisissez une page",
    options=list(PAGES.keys()),
    label_visibility="collapsed"  # Masquer l'étiquette de la sélection
)

# Ajout d'une section dans la barre latérale pour la description et les informations sur le projet
st.sidebar.header("Navigation")
st.sidebar.markdown("""
    L'application est structurée autour de trois principales sections :
    - **Introduction** : Un aperçu des données et du projet.
    - **Modélisation** : Utilisation de modèles de classification pour classer les individus.
    - **Visualisation** : Graphiques interactifs pour explorer les résultats.
    - **Classification** : Modèles de machine learning pour classer les individus .
""")

# Ajout d'un petit texte d'information dans la barre latérale (facultatif)
st.sidebar.markdown("### 📊 Classification des Maladies cardiaques")
st.sidebar.markdown("Cette analyse est basée sur les facteurs influençant le fait pour un individus d'etre atteint d'une maladie cardiaques.")
st.sidebar.markdown("---")

# Chargement des données
data = pd.read_csv("data/clean_heart_data.csv", sep=";")

# Appel de la fonction de la page sélectionnée
PAGES[page]()  