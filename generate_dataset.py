"""
generate_dataset.py
Genera dataset sintético para entrenar el modelo ML.
Correr UNA VEZ: python generate_dataset.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

def asignar_riesgo(iah, bmi, snore, events, hr, rr, sound, hours):
    puntaje = 0.0
    if iah >= 15:    puntaje += 4.0
    elif iah >= 5:   puntaje += 2.0
    if bmi >= 30:    puntaje += 2.0
    elif bmi >= 25:  puntaje += 1.0
    if snore:        puntaje += 1.5
    if events >= 15: puntaje += 1.5
    elif events >= 5: puntaje += 0.75
    if hr > 85:      puntaje += 0.5
    elif hr > 75:    puntaje += 0.25
    if sound > 65:   puntaje += 0.5
    if hours < 5:    puntaje += 0.5
    if puntaje >= 5.5:  return "Alto"
    elif puntaje >= 2.5: return "Moderado"
    else:               return "Bajo"

registros = []
for _ in range(N):
    cat = np.random.choice(["normal","leve","mod_severo"], p=[0.40,0.35,0.25])
    if cat == "normal":      iah = np.random.uniform(0, 4.9)
    elif cat == "leve":      iah = np.random.uniform(5, 14.9)
    else:                    iah = np.random.uniform(15, 45)

    bmi        = float(np.clip(np.random.normal(27.5, 5.0), 16, 50))
    prob_snore = min(0.20 + 0.30*(iah>5) + 0.20*(iah>15) + 0.15*(bmi>28), 0.95)
    snore      = bool(np.random.random() < prob_snore)
    sleep_h    = float(np.clip(np.random.normal(6.5, 1.0), 2, 10))
    events     = max(0, int(iah * sleep_h * np.random.uniform(0.6, 1.0)))
    hr         = float(np.clip(np.random.normal(62 + iah*0.3 + 5*(bmi>30), 7), 40, 110))
    rr         = float(np.clip(np.random.normal(15 + (iah>15)*2, 2.5), 6, 30))
    sound      = float(np.clip(np.random.normal(40 + 20*snore + iah*0.15, 7), 25, 90))

    registros.append({
        "heartRateAvg":       round(hr, 1),
        "respiratoryRateAvg": round(rr, 1),
        "soundLevelAvg":      round(sound, 1),
        "snoreDetected":      int(snore),
        "totalEvents":        events,
        "sleepHours":         round(sleep_h, 1),
        "iah":                round(iah, 2),
        "bmi":                round(bmi, 1),
        "riskLevel":          asignar_riesgo(iah,bmi,snore,events,hr,rr,sound,sleep_h)
    })

df = pd.DataFrame(registros).sample(frac=1, random_state=42).reset_index(drop=True)
print("=== DATASET GENERADO ===")
print(df["riskLevel"].value_counts())
df.to_csv("sleep_dataset_synthetic.csv", index=False)
print("✓ Guardado: sleep_dataset_synthetic.csv")