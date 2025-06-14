import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 🔁 Dictionnaire de correspondance pour affichage lisible ===
dictionnaires_etiquettes = {
    "sex": {0: "Femme", 1: "Homme"},
    "cp": {0: "Typique", 1: "Atypique", 2: "Non-angineux", 3: "Asymptomatique"},
    "fbs": {0: "Non", 1: "Oui"},
    "restecg": {0: "Normal", 1: "Anomalie onde ST-T", 2: "Hypertrophie"},
    "num": {0: "Pas de maladie", 1: "Maladie présente"},
}

# === 📥 Chargement des données ===
@st.cache_data
def charger_donnees():
    try:
        data = pd.read_csv("data/clean_heart_data.csv", sep=";")
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()

# === 🔤 Conversion des colonnes à faible cardinalité en 'category' ===
def detecter_et_convertir_variables_qualitatives(data):
    for col in data.columns:
        if data[col].nunique() < 10 and data[col].dtype in ['int64', 'float64']:
            data[col] = data[col].astype('category')
    return data

# === 🏷️ Remplacement des codes par des étiquettes lisibles ===
def appliquer_etiquettes(data, dictionnaires):
    data_affichage = data.copy()
    for col, mapping in dictionnaires.items():
        if col in data_affichage.columns:
            data_affichage[col] = data_affichage[col].map(mapping).astype("category")
    return data_affichage

# === 📊 Statistiques générales stylées ===
def afficher_statistiques_generales(nb_patients, nb_classes, classe_dominante, part_classe_dominante, age_moyen):
    st.subheader("📊 Statistiques Générales")
    cols = st.columns(5)
    style_box = """
    style="
        padding: 15px; 
        border-radius: 10px; 
        background-color: #e6f0ff; 
        text-align: center;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        margin: 5px;
    "
    """

    with cols[0]:
        st.markdown(
            f'<div {style_box}><h3>👥 Patients</h3><p style="font-size:24px; font-weight:bold;">{nb_patients:,}</p></div>', 
            unsafe_allow_html=True
        )
    with cols[1]:
        st.markdown(
            f'<div {style_box}><h3>📂 Classes</h3><p style="font-size:24px; font-weight:bold;">{nb_classes}</p></div>', 
            unsafe_allow_html=True
        )
    with cols[2]:
        st.markdown(
            f'<div {style_box}><h3>🥇 Classe dominante</h3><p style="font-size:24px; font-weight:bold;">{classe_dominante}</p></div>', 
            unsafe_allow_html=True
        )
    with cols[3]:
        part_str = f"{part_classe_dominante:.1f} %" if isinstance(part_classe_dominante, (int, float)) else str(part_classe_dominante)
        st.markdown(
            f'<div {style_box}><h3>📈 % Dominante</h3><p style="font-size:24px; font-weight:bold;">{part_str}</p></div>', 
            unsafe_allow_html=True
        )
    with cols[4]:
        age_str = f"{age_moyen:.1f} ans" if isinstance(age_moyen, (int, float)) else str(age_moyen)
        st.markdown(
            f'<div {style_box}><h3>🎂 Âge moyen</h3><p style="font-size:24px; font-weight:bold;">{age_str}</p></div>', 
            unsafe_allow_html=True
        )

    st.divider()

# === 🎛️ Visualisation principale ===
def afficher_page_visualisation(data):
    st.title("🧠 Visualisation des Données Médicales")

    if data.empty:
        st.warning("Aucune donnée à afficher.")
        return

    data = appliquer_etiquettes(data, dictionnaires_etiquettes)
    data = detecter_et_convertir_variables_qualitatives(data)

    nb_patients = data.shape[0]
    if 'num' in data.columns:
        nb_classes = data['num'].nunique()
        classe_dominante = data['num'].mode()[0]
        part_classe_dominante = 100 * data['num'].value_counts(normalize=True).iloc[0]
    else:
        nb_classes = classe_dominante = part_classe_dominante = "?"

    age_moyen = data["age"].mean() if "age" in data.columns else "?"

    afficher_statistiques_generales(nb_patients, nb_classes, classe_dominante, part_classe_dominante, age_moyen)

    # Variables quantitatives
    st.subheader("📈 Analyse des Variables Quantitatives")
    quantitative_vars = data.select_dtypes(include=['int64', 'float64']).columns
    var_quant = st.selectbox("Choisissez une variable quantitative", quantitative_vars)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[var_quant], kde=True, color="steelblue", bins=30, ax=ax)
    plt.title(f"Distribution de {var_quant}")
    st.pyplot(fig)

    st.subheader("Boxplot de la variable sélectionnée")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=data[var_quant], color="skyblue", ax=ax)
    st.pyplot(fig)
    st.divider()

    # Variables qualitatives
    st.subheader("📊 Analyse des Variables Qualitatives")
    qualitative_vars = data.select_dtypes(include=['category', 'object']).columns
    if len(qualitative_vars) > 0:
        var_qual = st.selectbox("Choisissez une variable qualitative", qualitative_vars)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=var_qual, data=data, palette="Blues", ax=ax)
        plt.title(f"Répartition de {var_qual}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Diagramme Circulaire")
        fig, ax = plt.subplots(figsize=(8, 8))
        data[var_qual].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap="Blues", ax=ax)
        plt.ylabel("")
        st.pyplot(fig)
    else:
        st.info("Aucune variable qualitative détectée.")
    st.divider()

    # Analyse croisée
    st.subheader("🔄 Analyse Croisée : Qualitatif vs Quantitatif")
    quant_cross = st.selectbox("Variable Quantitative", quantitative_vars, key="quant_cross")
    qual_cross = st.selectbox("Variable Qualitative", qualitative_vars, key="qual_cross")

    st.subheader(f"Boxplot : {quant_cross} en fonction de {qual_cross}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=qual_cross, y=quant_cross, data=data, palette="cool", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader(f"Barplot : Moyenne de {quant_cross} par {qual_cross}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=qual_cross, y=quant_cross, data=data, palette="Blues", estimator="mean", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.divider()

    # Corrélation
    st.subheader("🔥 Matrice de Corrélation des Variables Quantitatives")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data[quantitative_vars].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title("Matrice de Corrélation")
    st.pyplot(fig)
    st.divider()

    # Statistiques détaillées
    st.subheader("📋 Statistiques Résumées")
    stats_var = st.multiselect("Sélectionnez les variables pour voir leurs statistiques", data.columns)
    if stats_var:
        st.write(data[stats_var].describe())
    else:
        st.warning("Aucune variable sélectionnée pour les statistiques.")

# === 🚀 Lancer la page ===
if __name__ == "__main__":
    data = charger_donnees()
    afficher_page_visualisation(data)
