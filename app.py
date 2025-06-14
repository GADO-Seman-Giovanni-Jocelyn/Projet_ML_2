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
    /* ================ */
    /* VARIABLES COULEURS */
    /* ================ */
    :root {
        /* Palette médicale/santé */
        --bleu-hygiene: #1a73e8;       /* Couleur primaire */
        --bleu-light: #e8f0fe;         /* Arrière-plan */
        --vert-sante: #34a853;         /* Succès/positif */
        --rouge-alerte: #ea4335;       /* Erreur/danger */
        --orange-vitamine: #fb8c00;    /* Accent/attention */
        --gris-medical: #5f6368;       /* Texte secondaire */
        --blanc-pur: #ffffff;          /* Fond */
    }

    /* =================== */
    /* STRUCTURE PRINCIPALE */
    /* =================== */
    .main {
        background-color: var(--bleu-light);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(26, 115, 232, 0.05) 0%, transparent 20%);
    }

    .block-container {
        background-color: var(--blanc-pur);
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: 2rem;
        margin-top: 1.5rem;
    }

    /* ============= */
    /* TYPOGRAPHIE */
    /* ============= */
    h1 {
        color: var(--bleu-hygiene);
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        border-bottom: 3px solid var(--orange-vitamine);
        padding-bottom: 0.5rem;
        display: inline-block;
        margin-bottom: 1.5rem;
    }

    h2 {
        color: var(--bleu-hygiene);
        background-color: rgba(26, 115, 232, 0.1);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 1.8rem;
        margin-top: 2.5rem;
    }

    h3 {
        color: var(--gris-medical);
        font-size: 1.4rem;
        margin: 1.8rem 0 1rem 0;
    }

    /* ================= */
    /* COMPOSANTS CLAIRS */
    /* ================= */
    /* Boutons principaux */
    .stButton>button {
        background-color: var(--bleu-hygiene);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 2px 8px rgba(26, 115, 232, 0.2);
    }

    .stButton>button:hover {
        background-color: #0d5fd1;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.3);
    }

    /* Zones de saisie */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>select {
        border: 2px solid #dfe1e5;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        transition: all 0.3s;
    }

    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>select:focus {
        border-color: var(--bleu-hygiene);
        box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.1);
    }

    /* ================== */
    /* SECTIONS SPÉCIFIQUES */
    /* ================== */
    /* Section Résultats */
    .stAlert {
        border-left: 4px solid var(--bleu-hygiene);
        background-color: rgba(26, 115, 232, 0.05);
        border-radius: 0 12px 12px 0;
    }

    /* Section Paramètres */
    .stExpander {
        border: 1px solid rgba(26, 115, 232, 0.2);
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }

    .stExpander .st-emotion-cache-1q7spjk {
        background-color: rgba(26, 115, 232, 0.05);
    }

    /* Tableaux de résultats */
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* =============== */
    /* ÉLÉMENTS GRAPHIQUES */
    /* =============== */
    /* Graphiques */
    .element-container .stPlotlyChart {
        border-radius: 12px;
        background-color: white;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* Barre de progression */
    .stProgress>div>div>div {
        background-color: var(--bleu-hygiene);
    }

    /* ============= */
    /* SIDEBAR */
    /* ============= */
    .sidebar .sidebar-content {
        background-color: var(--blanc-pur);
        border-right: 1px solid rgba(26, 115, 232, 0.1);
    }

    /* ============= */
    /* RESPONSIVE */
    /* ============= */
    @media (max-width: 768px) {
        .block-container {
            padding: 1.5rem;
        }
        
        h1 {
            font-size: 2rem;
        }
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