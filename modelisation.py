import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix,
    classification_report
)

# Configuration
PIPELINES_DIR = "Pipeline"
MODEL_NAME_MAP = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "mlp": "MLP",
    "dt": "Decision Tree",
    "svm": "SVM",
    "logreg": "Logistic Regression",
    "knn": "k-NN",
    "nb": "Naive Bayes"
}

@st.cache_data
def load_test_data():
    """Charge les données de test"""
    try:
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv')
        return X_test, y_test
    except Exception as e:
        st.error(f"Erreur chargement données: {str(e)}")
        return None, None

@st.cache_resource
def load_model(model_path):
    """Charge un modèle avec cache"""
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Erreur chargement modèle: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test):
    """Évalue un modèle et retourne les métriques"""
    try:
        y_pred = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
    except Exception as e:
        st.error(f"Erreur évaluation: {str(e)}")
        return None

def show_model_metrics(metrics):
    """Affiche les métriques d'un modèle"""
    if not metrics:
        return

    st.subheader("📊 Performance du modèle")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with cols[1]:
        st.metric("Recall", f"{metrics['recall']:.2%}")
        st.metric("F1-Score", f"{metrics['f1']:.2%}")

    st.subheader("🧮 Matrice de confusion")
    fig, ax = plt.subplots()
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

def page_modelisation():
    """Page principale de modélisation"""
    st.title("🔍 Analyse des Modèles")

    # Chargement des données
    X_test, y_test = load_test_data()
    if X_test is None:
        return

    # Vérification des modèles disponibles
    if not os.path.exists(PIPELINES_DIR):
        st.error(f"Dossier {PIPELINES_DIR} introuvable!")
        return

    model_files = [f for f in os.listdir(PIPELINES_DIR) if f.endswith('.pkl')]
    if not model_files:
        st.warning("Aucun modèle trouvé!")
        return

    # Évaluation des modèles
    performances = []
    for file in model_files:
        model_name = MODEL_NAME_MAP.get(file.split('_')[1].split('.')[0], "Inconnu")
        model_path = os.path.join(PIPELINES_DIR, file)
        
        model = load_model(model_path)
        if model is None:
            continue
            
        metrics = evaluate_model(model, X_test, y_test)
        if metrics:
            performances.append({
                "Modèle": model_name,
                "Accuracy": metrics['accuracy'],
                "F1-Score": metrics['f1'],
                "path": model_path
            })

    # Affichage des résultats
    if not performances:
        st.error("Aucune performance calculée")
        return

    perf_df = pd.DataFrame(performances).sort_values("F1-Score", ascending=False)
    
    st.subheader("📈 Comparaison des modèles")
    st.dataframe(
        perf_df.style.format({
            "Accuracy": "{:.2%}",
            "F1-Score": "{:.2%}"
        }),
        use_container_width=True
    )

    # Analyse détaillée
    selected_model = st.selectbox(
        "Sélectionnez un modèle",
        perf_df["Modèle"]
    )
    
    selected_path = perf_df[perf_df["Modèle"] == selected_model]["path"].iloc[0]
    model = load_model(selected_path)
    metrics = evaluate_model(model, X_test, y_test)
    show_model_metrics(metrics)

# Pour tester indépendamment
if __name__ == "__main__":
    page_modelisation()