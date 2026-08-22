# Maintenance prédictive des engins de transport de fret (ONCF)

Approche Machine Learning pour anticiper les pannes curatives d'un parc d'engins de transport rail-route (TRR), développée dans le cadre d'un stage au sein du Service Maintenance des Engins de l'Office National des Chemins de Fer (ONCF).

> **Note sur les données** : conformément à la clause de confidentialité de la convention de stage, aucune donnée réelle de l'ONCF n'est présente dans ce dépôt (historique de pannes, heures d'usage, résultats de prédiction par engin). Un jeu de données **synthétique** (`generate_synthetic_data.py`), reproduisant fidèlement la structure des données réelles avec des valeurs générées aléatoirement, est fourni pour permettre l'exécution complète du pipeline.

## Contexte

Le parc de 12 engins TRR assure la traction et la manœuvre des wagons de marchandises sur 5 sites de fret. Leur maintenance reposait jusqu'ici sur un mode réactif (intervention après panne). Ce projet développe un modèle de classification binaire estimant, pour un engin donné à une date donnée, la probabilité qu'une panne curative survienne dans un horizon de N jours.

## Résultats obtenus (sur données réelles, résumé anonymisé)

| Métrique | Valeur |
|---|---|
| Horizon retenu | 14 jours |
| AUC | 0,747 |
| Recall | 74,4 % |
| Précision | 71,2 % |
| Seuil de décision optimal | 0,20 |

## Stack technique

- **KNIME** : prototypage du pipeline (nettoyage, feature engineering, premiers modèles)
- **Python** (pandas, XGBoost, scikit-learn) : industrialisation et validation croisée
- **Power BI** : tableau de bord opérationnel (risque actuel, historique, usage/usure)

## Structure du dépôt

```
├── generate_synthetic_data.py     # Génère un jeu de données factice (structure identique aux vraies données)
├── modele_maintenance_predictive.py   # Pipeline d'entraînement du modèle XGBoost
├── prediction_dashboard.py        # Génère les prédictions pour alimenter le dashboard
├── analyse_pred_contribs.py       # Analyse d'interprétabilité (contributions XGBoost par prédiction)
├── data/
│   └── synthetic/                 # Données synthétiques générées (voir ci-dessus)
├── docs/
│   └── rapport_de_stage.pdf       # Rapport de stage complet (version anonymisée)
└── README.md
```

## Démarche méthodologique

1. **Nettoyage des données** : harmonisation de sources hétérogènes (noms d'engins, types de panne, sites), correction d'incidents (dates mal interprétées, doublons, pertes silencieuses de données lors de conversions de fichiers)
2. **Construction de la table analytique** : structuration au format engin × jour, calcul de features d'usage, d'usure et d'historique de pannes
3. **Choix de l'horizon de prédiction** : comparaison empirique de 4 horizons (3, 7, 14, 30 jours)
4. **Modélisation** : XGBoost, gestion du déséquilibre de classes (`scale_pos_weight`), découpage entraînement/test **temporel** (pas aléatoire, pour reproduire les conditions réelles d'usage)
5. **Optimisation du seuil de décision** : recherche sur 11 seuils candidats, optimisation du score F2 (priorité au rappel)
6. **Restitution opérationnelle** : tableau de bord Power BI à 3 pages (risque actuel, historique des pannes, usage/usure)

## Points méthodologiques notables

- Détection et correction d'une **fuite de données entre engins** dans la construction de la variable cible (fenêtre glissante ne respectant pas les frontières entre engins)
- Détection et correction d'une **fuite de données entre horizons** dans le prototypage (colonnes cibles concurrentes laissées par erreur dans les features)
- Investigation de robustesse des prédictions (analyse des contributions XGBoost) ayant permis d'identifier un risque de sur-apprentissage lié au volume de données limité, documenté comme limite du projet plutôt que dissimulé

## Utilisation

```bash
# 1. Générer les données synthétiques
python generate_synthetic_data.py

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Entraîner le modèle (sur données synthétiques)
python modele_maintenance_predictive.py

# 4. Analyser les contributions du modèle pour une prédiction donnée
python analyse_pred_contribs.py
```

## Auteur

Yassir Bouchnaf — Stage de fin de 2ème année, Ingénieur Génie Industriel, option Business & Data Management, ESITH.
Stage réalisé au sein de l'ONCF, Pôle Fret et Logistique.
