# 🫀 Projet Machine Learning - Classification des Maladies Cardiaques

![Machine Learning](https://img.shields.io/badge/Machine_Learning-Heart_Disease-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Un projet complet de machine learning pour la prédiction des maladies cardiaques avec interface interactive.

---

## 📝 Description

Ce projet vise à prédire la présence d'une maladie cardiaque à partir de données cliniques en utilisant différentes approches de machine learning. 

**Points clés :**
- 🧠 Comparaison de plusieurs algorithmes de classification
- 📊 Dashboard interactif pour l'analyse des résultats
- 🔍 Explicabilité des prédictions
- 🚀 Pipeline complet de A à Z

**Dataset utilisé :** [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (UCI Machine Learning Repository)

---

## 🎯 Fonctionnalités

### 🔧 Préprocessing
- Nettoyage des données (valeurs manquantes, outliers)
- Feature engineering
- Normalisation/Standardisation
- Split train/test stratifié

### 🤖 Algorithmes implémentés
| Modèle | Bibliothèque | Optimisé? |
|--------|--------------|-----------|
| Random Forest | scikit-learn | ✅ |
| XGBoost | xgboost | ✅ |
| SVM | scikit-learn | ✅ |
| Régression Logistique | scikit-learn | ✅ |
| KNN | scikit-learn | ✅ |

### 📈 Évaluation
- Métriques complètes :
  - ✅ Accuracy
  - ✅ Precision
  - ✅ Recall
  - ✅ F1-score
  - ✅ ROC-AUC
- Visualisations :
  - 📉 Matrice de confusion
  - 📊 Courbe ROC
  - 📈 Rapport de classification

### 🖥️ Interface
- Dashboard Streamlit interactif
- Sélection de modèle dynamique
- classification sur de nouveaux patients

---

## 🛠️ Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/OusseynouDIOP16/Projet_ML_2/tree/main
