import pandas as pd
import numpy as np
import joblib

# 1. PARAMÈTRES

CHEMIN_TABLE = "Table_analytique_ONCF_v1.xlsx"
CHEMIN_MODELE = "modele_xgboost_horizon_14j.joblib"
HORIZON = 14

SEUIL_MODERE = 0.20  
SEUIL_ELEVE = 0.40    

# 2. CHARGEMENT DE LA TABLE ET DU MODÈLE

df = pd.read_excel(CHEMIN_TABLE)
if "Iteration" in df.columns:
    df = df.drop(columns=["Iteration"])

model = joblib.load(CHEMIN_MODELE)
print(f"Modèle chargé : {CHEMIN_MODELE}")

target_col = f"Y_horizon_{HORIZON}j"
autres_horizons = [c for c in df.columns
                   if c.startswith("Y_horizon_") and c != target_col]
colonnes_a_exclure = ["Panne_ce_jour", "Date", "Engin"] + autres_horizons
feature_cols_brutes = [c for c in df.columns
                       if c not in colonnes_a_exclure + [target_col]]

# 3. SÉLECTION DE L'ÉTAT LE PLUS RÉCENT DE CHAQUE ENGIN

df["Date"] = pd.to_datetime(df["Date"])
derniere_date = df["Date"].max()
etat_actuel = df[df["Date"] == derniere_date].copy()

print(f"Date de référence (état le plus récent) : {derniere_date.date()}")
print(f"Nombre d'engins : {etat_actuel['Engin'].nunique()}\n")

# 4. PRÉPARATION DES FEATURES (identique au script d'entraînement)

X_actuel = etat_actuel[feature_cols_brutes].copy()
X_actuel = pd.get_dummies(X_actuel, columns=["Site", "Saison"], dummy_na=False)

# Alignement des colonnes sur celles vues à l'entraînement
# (garantit le même ordre et les mêmes colonnes one-hot, même si une
# catégorie n'apparaît pas dans les données du jour)
colonnes_modele = model.feature_names_in_
X_actuel = X_actuel.reindex(columns=colonnes_modele, fill_value=0)

# 5. PRÉDICTION

etat_actuel["Probabilite_panne_14j"] = model.predict_proba(X_actuel)[:, 1]

def classer_risque(p):
    if p >= SEUIL_ELEVE:
        return "Eleve"
    elif p >= SEUIL_MODERE:
        return "Modere"
    else:
        return "Faible"

etat_actuel["Niveau_risque"] = etat_actuel["Probabilite_panne_14j"].apply(classer_risque)

# 6. EXPORT POUR POWER BI

colonnes_export = [
    "Engin", "Date", "Site", "Age_jours", "Heures diesel",
    "Heures_travaillees_7j", "Heures_travaillees_14j", "Heures_travaillees_30j",
    "Nb_pannes_90j", "Nb_pannes_180j", "Jours_depuis_derniere_panne",
    "Probabilite_panne_14j", "Niveau_risque",
]

export = etat_actuel[colonnes_export].sort_values(
    "Probabilite_panne_14j", ascending=False
)

export.to_excel("Dashboard_risque_actuel.xlsx", index=False)

print("=" * 60)
print("Résultat (trié par risque décroissant) :")
print(export[["Engin", "Site", "Probabilite_panne_14j", "Niveau_risque"]]
      .to_string(index=False))
print("\nExport créé : Dashboard_risque_actuel.xlsx")
