"""
train_model.py
Entrena el modelo Random Forest y guarda model.pkl
Correr UNA VEZ: python train_model.py
"""
import pickle, json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

print("=== SML — ENTRENAMIENTO ===")

# 1. Cargar datos
df = pd.read_csv("sleep_dataset_synthetic.csv")
print(f"Registros: {len(df)} | Clases: {df['riskLevel'].value_counts().to_dict()}")

FEATURES = [
    "heartRateAvg","respiratoryRateAvg","soundLevelAvg",
    "snoreDetected","totalEvents","sleepHours","iah","bmi"
]
X = df[FEATURES]
y = df["riskLevel"]

# 2. Codificar etiquetas (Alto=0, Bajo=1, Moderado=2)
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Mapeo: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

# 4. Modelo
modelo = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1
)

# 5. Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"CV Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

# 6. Entrenar y evaluar
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")
print(f"\nAccuracy test: {acc:.3f} | F1: {f1:.3f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Importancia de features
imp = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Importancia de features:")
for feat, val in imp.items():
    print(f"  {feat:25s} {val:.4f}  {'█'*int(val*40)}")

# 7. Guardar
VERSION = "1.0.0"
artefacto = {
    "model": modelo, "label_encoder": le, "features": FEATURES,
    "version": VERSION, "dataset_type": "synthetic",
    "disclaimer": "Prueba de concepto académica. No validado clínicamente.",
    "metrics": {
        "accuracy_test":    round(float(acc), 4),
        "f1_weighted_test": round(float(f1), 4),
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
    },
    "feature_importance": imp.round(4).to_dict()
}
with open("model.pkl", "wb") as f:
    pickle.dump(artefacto, f)

metricas = {k:v for k,v in artefacto.items() if k not in ("model","label_encoder")}
with open("model_metrics.json","w",encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

print(f"\n✓ model.pkl guardado (v{VERSION})")
print(f"✓ model_metrics.json guardado")
print(f"\n{'='*40}")
print("ENTRENAMIENTO COMPLETADO")