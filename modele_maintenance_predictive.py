"""
ONCF - Maintenance prédictive des engins de transport de fret
Implémentation Python du modèle de classification (horizon 14 jours)

Ce script reproduit fidèlement, en Python, le pipeline de modélisation
validé dans KNIME à l'Étape 4 : préparation des features, entraînement
XGBoost avec gestion du déséquilibre de classes, découpage temporel
train/test, et évaluation (AUC, recall, précision) avec seuil de
décision optimisé.

Prérequis : la table analytique (Table_analytique_ONCF_v1.xlsx) doit déjà
exister, produite par le pipeline KNIME (Étapes 1 et 2). Ce script ne
refait PAS le nettoyage ni le calcul des features, uniquement la
modélisation.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score,
    roc_auc_score, accuracy_score, fbeta_score
)

# =============================================================================
# 1. CHARGEMENT DE LA TABLE ANALYTIQUE
# =============================================================================

CHEMIN_FICHIER = "Table_analytique_ONCF_v1.xlsx"
HORIZON = 14  # Horizon retenu (jours) - modifiable pour tester 3/7/30
SEUIL_DECISION = 0.20  # Seuil optimisé via recherche F2 (cf. KNIME)
DATE_COUPURE_TEST = "2026-03-01"  # Découpage temporel train/test

df = pd.read_excel(CHEMIN_FICHIER)

# Nettoyage : retrait de la colonne technique résiduelle d'une boucle KNIME
if "Iteration" in df.columns:
    df = df.drop(columns=["Iteration"])

target_col = f"Y_horizon_{HORIZON}j"
print(f"Table chargée : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
print(f"Horizon retenu : {HORIZON} jours (cible : {target_col})\n")

# =============================================================================
# 2. SÉLECTION DES FEATURES (exclusion des colonnes à risque de fuite)
# =============================================================================

# Colonnes exclues systématiquement des features, quel que soit l'horizon :
# - Panne_ce_jour : fuite de données (le jour même fait partie de la fenêtre Y)
# - Date : remplacée par Date (Month) / Saison, plus utiles pour un modèle
# - Engin : exclu pour éviter le sur-apprentissage (12 catégories seulement)
# - Toutes les colonnes Y_horizon_Xj SAUF celle de l'horizon choisi
autres_horizons = [c for c in df.columns
                   if c.startswith("Y_horizon_") and c != target_col]

colonnes_a_exclure = ["Panne_ce_jour", "Date", "Engin"] + autres_horizons

feature_cols_brutes = [c for c in df.columns
                       if c not in colonnes_a_exclure + [target_col]]

print("Features de base retenues :", feature_cols_brutes)
print()

# =============================================================================
# 3. ENCODAGE ONE-HOT (Site, Saison) -- équivalent du nœud "One to Many"
# =============================================================================

df_model = df[feature_cols_brutes + [target_col, "Date"]].copy()
df_model = pd.get_dummies(df_model, columns=["Site", "Saison"], dummy_na=False)

# =============================================================================
# 4. RETRAIT DES LIGNES SANS CIBLE CONNUE (équivalent "Row Filter Is not missing")
# =============================================================================

avant = len(df_model)
df_model = df_model[df_model[target_col].notna()].copy()
df_model[target_col] = df_model[target_col].astype(int)
print(f"Lignes retirées (cible manquante) : {avant - len(df_model)}")
print(f"Lignes restantes : {len(df_model)}\n")

# =============================================================================
# 5. DÉCOUPAGE TEMPOREL TRAIN / TEST (équivalent "Rule-based Row Splitter")
# =============================================================================

df_model["Date"] = pd.to_datetime(df_model["Date"])
mask_train = df_model["Date"] < pd.Timestamp(DATE_COUPURE_TEST)

train = df_model[mask_train].drop(columns=["Date"])
test = df_model[~mask_train].drop(columns=["Date"])

X_train = train.drop(columns=[target_col])
y_train = train[target_col]
X_test = test.drop(columns=[target_col])
y_test = test[target_col]

print(f"Entraînement : {len(train)} lignes (avant {DATE_COUPURE_TEST})")
print(f"Test         : {len(test)} lignes (à partir de {DATE_COUPURE_TEST})\n")

# =============================================================================
# 6. GESTION DU DÉSÉQUILIBRE DE CLASSES (scale_pos_weight)
# =============================================================================

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos

print(f"Répartition entraînement : {n_neg} négatifs (0), {n_pos} positifs (1)")
print(f"scale_pos_weight calculé : {scale_pos_weight:.3f}\n")

# =============================================================================
# 7. ENTRAÎNEMENT DU MODÈLE XGBOOST
# =============================================================================

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=100,          # équivalent "Boosting rounds"
    scale_pos_weight=scale_pos_weight,
    missing=np.nan,            # gestion native des valeurs manquantes
    eval_metric="logloss",
    random_state=42,
)

model.fit(X_train, y_train)
print("Modèle entraîné.\n")

# =============================================================================
# 8. PRÉDICTION SUR LE JEU DE TEST
# =============================================================================

y_proba = model.predict_proba(X_test)[:, 1]  # probabilité de la classe "1"
y_pred_defaut = (y_proba >= 0.5).astype(int)          # seuil par défaut
y_pred_optimise = (y_proba >= SEUIL_DECISION).astype(int)  # seuil optimisé

# =============================================================================
# 9. ÉVALUATION
# =============================================================================

def afficher_resultats(y_true, y_pred, y_proba, label):
    cm = confusion_matrix(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    print(f"--- {label} ---")
    print(f"Matrice de confusion :\n{cm}")
    print(f"Recall    : {recall:.3f}")
    print(f"Précision : {precision:.3f}")
    print(f"Accuracy  : {accuracy:.3f}")
    print(f"AUC       : {auc:.3f}\n")
    return {"recall": recall, "precision": precision,
            "accuracy": accuracy, "auc": auc}

print("=" * 60)
res_defaut = afficher_resultats(y_test, y_pred_defaut, y_proba, "Seuil 0.50 (défaut)")
res_optimise = afficher_resultats(y_test, y_pred_optimise, y_proba,
                                   f"Seuil {SEUIL_DECISION} (optimisé F2)")

# =============================================================================
# 10. IMPORTANCE DES FEATURES
# =============================================================================

importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)
print("=" * 60)
print("Importance des features (top 10) :")
print(importances.head(10).to_string())

# =============================================================================
# 11. SAUVEGARDE DU MODÈLE
# =============================================================================

import joblib
joblib.dump(model, f"modele_xgboost_horizon_{HORIZON}j.joblib")
print(f"\nModèle sauvegardé : modele_xgboost_horizon_{HORIZON}j.joblib")
