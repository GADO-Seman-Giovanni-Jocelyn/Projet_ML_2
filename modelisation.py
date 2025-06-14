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
from sklearn.model_selection import train_test_split

# Données
DATA_PATH = 'clean_heart_data.csv'
TARGET = 'num'

# Mapping noms abrégés -> noms complets
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

def page_modelisation():
    st.title("📊 Modélisation - Évaluation des Modèles de Classification")



    X_test = pd.read_csv('data/X_test.csv', sep=',')
    y_test = pd.read_csv('data/y_test.csv', sep=',')


    pipelines_dir = "Pipeline"
    if not os.path.exists(pipelines_dir):
        st.warning(f"📁 Dossier `{pipelines_dir}` introuvable. Assurez-vous d’avoir sauvegardé vos pipelines.")
        return

    pipeline_files = [f for f in os.listdir(pipelines_dir) if f.endswith(".pkl")]
    if not pipeline_files:
        st.warning("⚠️ Aucun fichier `.pkl` trouvé. Veuillez entraîner et sauvegarder vos modèles.")
        return

    st.subheader("🔄 Évaluation automatique des modèles")
    performances = []

    for file in pipeline_files:
        model_short_name = file.replace("pipeline_", "").replace(".pkl", "")
        model_display_name = model_name_map.get(model_short_name, model_short_name)
        path = os.path.join(pipelines_dir, file)

        try:
            pipeline = joblib.load(path)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            performances.append({
                "Modèle": model_display_name,
                "Accuracy": acc,
                "Précision": prec,
                "Rappel": rec,
                "F1-score": f1,
                "short_name": model_short_name  # Pour la sélection plus tard
            })

        except Exception as e:
            st.error(f"❌ Erreur lors de l’évaluation de `{model_display_name}` : {e}")

    if not performances:
        st.warning("⚠️ Aucune performance calculée. Vérifiez le contenu de vos fichiers modèles.")
        return

    perf_df = pd.DataFrame(performances).sort_values(by="F1-score", ascending=False)
    st.write("### 📈 Résultats comparatifs")
    st.dataframe(perf_df.drop(columns=['short_name']), use_container_width=True)

    best_model_name = perf_df.iloc[0]["Modèle"]
    st.success(f"🏆 Meilleur modèle (test set) : **{best_model_name}**")

    st.subheader("🔍 Analyse détaillée d’un modèle")
    # Utilisation des noms complets pour la sélection
    selected_display_name = st.selectbox("Sélectionner un modèle :", perf_df["Modèle"].tolist())
    # Récupérer le short_name correspondant
    selected_short_name = perf_df.loc[perf_df["Modèle"] == selected_display_name, "short_name"].values[0]

    selected_pipeline = joblib.load(os.path.join(pipelines_dir, f"pipeline_{selected_short_name}.pkl"))
    y_pred_selected = selected_pipeline.predict(X_test)

    st.write("### 📄 Rapport de Classification")
    report = classification_report(y_test, y_pred_selected, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.write("### 📊 Matrice de Confusion")
    cm = confusion_matrix(y_test, y_pred_selected)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
    ax.set_xlabel("Valeur Prédite")
    ax.set_ylabel("Valeur Réelle")
    st.pyplot(fig)
