import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

model = joblib.load("modele_xgboost_horizon_14j.joblib")
df = pd.read_excel("Table_analytique_ONCF_v1.xlsx")

ENGIN_CIBLE = "TRR-04"  
DATE_CIBLE = df["Date"].max()  
ligne = df[(df["Engin"] == ENGIN_CIBLE) & (df["Date"] == DATE_CIBLE)].copy()
if ligne.empty:
    raise ValueError(f"Aucune ligne trouvée pour {ENGIN_CIBLE} à la date {DATE_CIBLE}")

colonnes_a_exclure = [
    "Engin", "Date", "Panne_ce_jour",
    "Y_horizon_3j", "Y_horizon_7j", "Y_horizon_14j", "Y_horizon_30j",
    "Iteration",
]
X_ligne = ligne.drop(columns=[c for c in colonnes_a_exclure if c in ligne.columns])
X_ligne = pd.get_dummies(X_ligne, columns=["Site", "Saison"])

colonnes_attendues = [
    "Heures diesel", "Age_jours", "Heures_travaillees_7j", "Heures_travaillees_14j",
    "Heures_travaillees_30j", "Nb_pannes_90j", "Nb_pannes_180j",
    "Jours_depuis_derniere_panne", "Date (Month)",
    "Site_Beni Ansar", "Site_Casa", "Site_Cimat", "Site_Cosumar", "Site_Jorf", "Site_Tanger",
    "Saison_Automne", "Saison_Ete", "Saison_Hiver", "Saison_Printemps",
]

for col in colonnes_attendues:
    if col not in X_ligne.columns:
        X_ligne[col] = 0

X_ligne = X_ligne[colonnes_attendues]
dmatrix = xgb.DMatrix(X_ligne)
contribs = model.get_booster().predict(dmatrix, pred_contribs=True)

contrib_series = pd.Series(contribs[0][:-1], index=X_ligne.columns)
contrib_series = contrib_series.sort_values()

plt.figure(figsize=(9, 6))
colors = ["#C0392B" if v > 0 else "#2E6DA4" for v in contrib_series.values]
plt.barh(contrib_series.index, contrib_series.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title(f"Contribution des features à la prédiction — {ENGIN_CIBLE} ({DATE_CIBLE.date()})")
plt.xlabel("Contribution à la probabilité de panne (log-odds)")
plt.tight_layout()
plt.savefig("contributions_xgboost.png", dpi=150)
plt.show()

print(contrib_series)
