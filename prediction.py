import streamlit as st
import requests

def page_prediction():
    st.title("🔬 Prédiction de Maladie Cardiaque")
    st.markdown("Remplissez les informations du patient pour prédire le risque de maladie cardiaque :")

    # Dictionnaires de correspondance
    sex_map = {"Femme": 0, "Homme": 1}
    cp_map = {
        "Angine typique": 0,
        "Angine atypique": 1,
        "Douleur non-angineuse": 2,
        "Asymptomatique": 3
    }
    fbs_map = {"≤ 120 mg/dl": 0, "> 120 mg/dl": 1}
    restecg_map = {
        "Normal": 0,
        "Anomalie onde ST-T": 1,
        "Hypertrophie ventriculaire gauche": 2
    }

    # Formulaire en colonnes
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("🎂 Âge", 20, 100, 50)
        sex_label = st.selectbox("🧍 Sexe", list(sex_map.keys()))
        cp_label = st.selectbox("💢 Type de douleur thoracique", list(cp_map.keys()))
        trestbps = st.number_input("🩺 Pression artérielle au repos", value=120)
        chol = st.number_input("🩸 Taux de cholestérol", value=200)

    with col2:
        fbs_label = st.selectbox("🍬 Sucre à jeun", list(fbs_map.keys()))
        restecg_label = st.selectbox("📈 Résultats ECG au repos", list(restecg_map.keys()))
        thalach = st.number_input("❤️ Fréquence cardiaque maximale", value=150)
        oldpeak = st.number_input("🌄 Oldpeak (dépression ST)", value=1.0, format="%.1f")
        ca = st.selectbox("🧪 Nombre de vaisseaux colorés (ca)", [0, 1, 2, 3, 4])

    # Transformation des modalités
    input_data = {
        "ca": ca,
        "age": age,
        "sex": sex_map[sex_label],
        "cp": cp_map[cp_label],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_map[fbs_label],
        "restecg": restecg_map[restecg_label],
        "thalach": thalach,
        "oldpeak": oldpeak
    }

    # Bouton de prédiction
    if st.button("📤 Lancer la prédiction"):
        with st.spinner("⏳ Envoi des données à l'API..."):
            try:
                response = requests.post("http://localhost:8000/predict", json=input_data)
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction")
                    confidence = result.get("confidence")

                    st.subheader("📦 Données envoyées au modèle :")
                    st.json(input_data)

                    if prediction == 1:
                        st.markdown("### 🛑 Résultat : **💔 Risque de maladie cardiaque**", unsafe_allow_html=True)
                    elif prediction == 0:
                        st.markdown("### ✅ Résultat : **💚 Aucun risque détecté**", unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ Résultat inattendu : {prediction}")

                    if confidence is not None:
                        st.info(f"🔒 Confiance du modèle : **{confidence * 100:.2f}%**")
                    else:
                        st.info("ℹ️ Confiance du modèle non disponible")

                else:
                    st.error(f"❌ Erreur API : {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"💥 Impossible de contacter l'API : {e}")
