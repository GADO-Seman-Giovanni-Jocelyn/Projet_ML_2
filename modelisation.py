import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# Configuration
DATA_PATH = 'clean_heart_data.csv'
TARGET = 'num'
PIPELINES_DIR = "Pipeline"

# Mapping des noms de modèles
model_name_map = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "mlp": "Multi-Layer Perceptron",
    "dt": "Decision Tree",
    "svm": "Support Vector Machine",
    "logreg": "Logistic Regression",
    "knn": "k-Nearest Neighbors",
    "nb": "Naive Bayes"
}

@st.cache_data
def load_test_data():
    """Charge les données de test une seule fois"""
    X_test = pd.read_csv('data/X_test.csv', sep=',')
    y_test = pd.read_csv('data/y_test.csv', sep=',')
    return X_test, y_test

@st.cache_data
def evaluate_models(pipelines_dir, X_test, y_test):
    """Évalue tous les modèles et retourne les performances"""
    performances = []
    
    for file in os.listdir(pipelines_dir):
        if file.endswith(".pkl"):
            model_short_name = file.replace("pipeline_", "").replace(".pkl", "")
            model_display_name = model_name_map.get(model_short_name, model_short_name)
            path = os.path.join(pipelines_dir, file)

            try:
                pipeline = joblib.load(path)
                y_pred = pipeline.predict(X_test)

                performances.append({
                    "Modèle": model_display_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Précision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "Rappel": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "F1-score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    "short_name": model_short_name
                })

            except Exception as e:
                st.error(f"❌ Erreur avec {model_display_name}: {str(e)}")
    
    return pd.DataFrame(performances).sort_values(by="F1-score", ascending=False)

def show_model_details(model_name, perf_df, X_test, y_test):
    """Affiche les détails d'un modèle spécifique"""
    selected_short_name = perf_df.loc[perf_df["Modèle"] == model_name, "short_name"].values[0]
    pipeline = joblib.load(os.path.join(PIPELINES_DIR, f"pipeline_{selected_short_name}.pkl"))
    y_pred = pipeline.predict(X_test)

    # Rapport de classification
    st.subheader("📊 Rapport de Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        st.metric("Précision", f"{precision_score(y_test, y_pred, average='weighted'):.2%}")
    with col2:
        st.metric("Rappel", f"{recall_score(y_test, y_pred, average='weighted'):.2%}")
        st.metric("F1-Score", f"{f1_score(y_test, y_pred, average='weighted'):.2%}")

    # Matrice de confusion
    st.subheader("🧮 Matrice de Confusion")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True, 
        fmt='d', 
        cmap='Blues',
        ax=ax
    )
    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Vérité Terrain")
    st.pyplot(fig)

def page_modelisation():
    st.title("📊 Modélisation - Évaluation des Modèles")

    # Vérification des dossiers
    if not os.path.exists(PIPELINES_DIR):
        st.error(f"❌ Dossier `{PIPELINES_DIR}` introuvable")
        return

    # Chargement des données
    X_test, y_test = load_test_data()

    # Évaluation des modèles
    st.subheader("📈 Comparaison des Modèles")
    perf_df = evaluate_models(PIPELINES_DIR, X_test, y_test)

    if perf_df.empty:
        st.warning("Aucun modèle valide trouvé")
        return

    # Affichage des résultats
    st.dataframe(
        perf_df.style
            .background_gradient(subset=['Accuracy', 'F1-score'], cmap='Blues')
            .format("{:.2%}", subset=['Accuracy', 'Précision', 'Rappel', 'F1-score']),
        use_container_width=True
    )

    # Meilleur modèle
    best_model = perf_df.iloc[0]
    st.success(f"🏆 Meilleur modèle: **{best_model['Modèle']}** (F1-Score: {best_model['F1-score']:.2%})")

    # Analyse détaillée
    st.subheader("🔍 Analyse par Modèle")
    selected_model = st.selectbox(
        "Choisir un modèle à analyser",
        perf_df["Modèle"],
        index=0
    )
    
    show_model_details(selected_model, perf_df, X_test, y_test)

# Pour tester directement le module
if __name__ == "__main__":
    page_modelisation()