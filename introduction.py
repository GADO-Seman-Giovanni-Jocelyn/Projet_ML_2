import streamlit as st

def page_introduction(data=None):
    # Titre principal
    st.title("❤️ Détection de Maladies Cardiaques par Machine Learning")

    # Bannière ou sous-titre
    st.markdown(
        """
        <h4 style="text-align: center; color: #D32F2F;">
        Un modèle prédictif au service de la prévention cardiovasculaire
        </h4>
        """, unsafe_allow_html=True
    )

    # Contexte
    st.subheader("🩺 Contexte et Justification")
    st.write("""
    Les maladies cardiovasculaires représentent la première cause de mortalité dans le monde. Leur détection précoce est essentielle pour 
    améliorer la qualité et l’espérance de vie des patients à risque.

    Grâce aux avancées en intelligence artificielle, il est désormais possible d'exploiter des données médicales pour :
    - Détecter les signaux faibles annonciateurs d’un risque cardiaque.
    - Appuyer les professionnels de santé dans leurs diagnostics.
    - Promouvoir une médecine plus prédictive et personnalisée.
    """)

    st.info("""
    *"Prévenir les maladies plutôt que les guérir est la clé d’un système de santé durable."*
    """)

    # Objectif général
    st.divider()
    st.subheader("🎯 Objectif Général")
    st.write("""
    Développer un modèle de classification performant capable de **prédire la présence d’une maladie cardiaque** à partir de données cliniques
    et biométriques.
    """)

    # Objectifs spécifiques
    st.subheader("🔍 Objectifs Spécifiques")
    st.markdown("""
    - Identifier les **facteurs de risque** les plus corrélés à la maladie cardiaque.
    - Construire un **pipeline de Machine Learning** de bout en bout.
    - Évaluer les performances à l’aide de métriques comme la précision, le rappel et le F1-score.
    - Intégrer ce modèle dans une **application Web interactive** à destination du public médical et grand public.
    """)

    # Plan du projet
    st.divider()
    st.subheader("🗺️ Plan du Projet")
    plan_steps = [
        "Chargement et exploration du dataset cardiaque.",
        "Analyse descriptive des variables médicales.",
        "Prétraitement des données (nettoyage, transformation, encodage).",
        "Construction d’un pipeline de classification optimisé.",
        "Évaluation des performances sur des données de test.",
        "Déploiement de l'application pour un usage interactif."
    ]
    for i, step in enumerate(plan_steps):
        st.markdown(f"**{i+1}.** {step}")

    # Conclusion
    st.divider()
    st.subheader("🚀 Vision et Impact")
    st.write("""
    Ce projet illustre comment l’IA peut contribuer activement à la **prévention médicale**, en offrant des outils rapides, accessibles et
    fiables pour anticiper les risques de maladies cardiaques. L’objectif est de favoriser une prise de décision plus éclairée dans le parcours 
    de soin.
    """)

    st.success("Prêt à tester notre modèle de classification ? Votre cœur vous dira merci ❤️")

    st.markdown(
        """
        <h5 style="text-align: center; color: #D32F2F;">
        Utilisez le menu à gauche pour naviguer entre les différentes étapes du projet.
        </h5>
        """, unsafe_allow_html=True
    )
