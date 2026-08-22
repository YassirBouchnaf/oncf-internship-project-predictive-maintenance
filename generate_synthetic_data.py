import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(42)

ENGINS = [f"TRR-{i:02d}" for i in range(1, 13)]
SITES = ["Tanger Med", "Jorf Lasfar", "Casa", "Cosumar", "Beni Ansar"]
SAISONS = ["Hiver", "Printemps", "Ete", "Automne"]
AGE_MISE_EN_SERVICE = {
    "TRR-01": 2014, "TRR-02": 2015, "TRR-03": 2015, "TRR-04": 2015,
    "TRR-05": 2015, "TRR-06": 2015, "TRR-07": 2016, "TRR-08": 2016,
    "TRR-09": 2016, "TRR-10": 2016, "TRR-12": 2017, "TRR-11": 2018,
}

DATE_DEBUT = pd.Timestamp("2024-06-01")
DATE_FIN = pd.Timestamp("2026-05-31")

def month_to_saison(month: int) -> str:
    if month in (12, 1, 2):
        return "Hiver"
    if month in (3, 4, 5):
        return "Printemps"
    if month in (6, 7, 8):
        return "Ete"
    return "Automne"
    
def generate():
    dates = pd.date_range(DATE_DEBUT, DATE_FIN, freq="D")
    rows = []
    for engin in ENGINS:
        site = np.random.choice(SITES)
        mise_en_service = AGE_MISE_EN_SERVICE[engin]
        base_panne_rate = np.random.uniform(0.01, 0.04)  
        heures_cumulees = np.random.uniform(2000, 6000)
        jours_depuis_panne = np.random.randint(0, 60)
        for date in dates:
            age_jours = (date - pd.Timestamp(f"{mise_en_service}-01-01")).days
            heures_cumulees += np.random.uniform(3, 9)  
            panne_ce_jour = 1 if np.random.random() < base_panne_rate else 0
            jours_depuis_panne = 0 if panne_ce_jour else jours_depuis_panne + 1
            rows.append({
                "Engin": engin,
                "Date": date,
                "Panne_ce_jour": panne_ce_jour,
                "Heures diesel": round(heures_cumulees, 1),
                "Age_jours": age_jours,
                "Site": site,
                "Heures_travaillees_7j": round(np.random.uniform(20, 60), 1),
                "Heures_travaillees_14j": round(np.random.uniform(40, 120), 1),
                "Heures_travaillees_30j": round(np.random.uniform(80, 250), 1),
                "Nb_pannes_90j": np.random.poisson(2),
                "Nb_pannes_180j": np.random.poisson(4),
                "Jours_depuis_derniere_panne": jours_depuis_panne,
                "Date (Month)": date.month,
                "Saison": month_to_saison(date.month),
            })

    df = pd.DataFrame(rows)
    for horizon in [3, 7, 14, 30]:
        col = f"Y_horizon_{horizon}j"
        df[col] = np.nan
        for engin in ENGINS:
            mask = df["Engin"] == engin
            sub = df.loc[mask, "Panne_ce_jour"].values
            y = np.full(len(sub), np.nan)
            for i in range(len(sub) - horizon):
                y[i] = 1 if sub[i + 1: i + 1 + horizon].sum() > 0 else 0
            df.loc[mask, col] = y
    df["Iteration"] = 0 
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Table_analytique_synthetic.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Données synthétiques générées : {out_path} ({len(df)} lignes)")
if __name__ == "__main__":
    generate()
