"""
ml_service.py
Carga model.pkl y expone predict_risk().
Si no hay model.pkl, usa reglas como fallback.
"""
import os, pickle, logging
logger = logging.getLogger(__name__)

RECOMENDACIONES = {
    "Bajo":     "No se detectan señales de riesgo significativo. Continúe con el monitoreo rutinario.",
    "Moderado": "Se detectaron señales que merecen atención. Se recomienda seguimiento clínico.",
    "Alto":     "Se detectaron señales de riesgo elevado. Consulte con especialista en medicina del sueño."
}

FEATURES = [
    "heartRateAvg","respiratoryRateAvg","soundLevelAvg",
    "snoreDetected","totalEvents","sleepHours","iah","bmi"
]

class MLService:
    def __init__(self, model_path=None):
        self.modelo   = None
        self.le       = None
        self.features = FEATURES
        self.version  = "rules-fallback"
        self.cargado  = False
        ruta = model_path or os.getenv("MODEL_PATH", "model.pkl")
        try:
            with open(ruta, "rb") as f:
                art = pickle.load(f)
            self.modelo   = art["model"]
            self.le       = art["label_encoder"]
            self.features = art["features"]
            self.version  = art["version"]
            self.cargado  = True
            logger.info(f"Modelo cargado: v{self.version}")
        except FileNotFoundError:
            logger.warning(f"No se encontró {ruta}. Usando reglas como fallback.")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}. Usando fallback.")

    def predict_risk(self, resumen: dict) -> dict:
        for f in self.features:
            if f not in resumen:
                resumen[f] = 0
        if self.cargado:
            return self._con_modelo(resumen)
        return self._con_reglas(resumen)

    def _con_modelo(self, resumen: dict) -> dict:
        try:
            X = [[
                float(resumen.get("heartRateAvg", 0)),
                float(resumen.get("respiratoryRateAvg", 0)),
                float(resumen.get("soundLevelAvg", 0)),
                int(bool(resumen.get("snoreDetected", False))),
                float(resumen.get("totalEvents", 0)),
                float(resumen.get("sleepHours", 0)),
                float(resumen.get("iah", 0)),
                float(resumen.get("bmi", 0))
            ]]
            pred_enc  = self.modelo.predict(X)[0]
            pred_prob = self.modelo.predict_proba(X)[0]
            riesgo    = self.le.inverse_transform([pred_enc])[0]
            confianza = round(float(max(pred_prob)), 4)
            probs = {c: round(float(p),4) for c,p in zip(self.le.classes_, pred_prob)}
            logger.info(f"[ML] {riesgo} (conf={confianza})")
            return {
                "mlRiskLevel": riesgo, "mlConfidence": confianza,
                "mlRecommendation": RECOMENDACIONES[riesgo],
                "modelVersion": self.version, "mlClassProbs": probs
            }
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return self._con_reglas(resumen)

    def _con_reglas(self, resumen: dict) -> dict:
        iah   = float(resumen.get("iah", 0))
        bmi   = float(resumen.get("bmi", 0))
        snore = bool(resumen.get("snoreDetected", False))
        ev    = float(resumen.get("totalEvents", 0))
        hr    = float(resumen.get("heartRateAvg", 0))
        snd   = float(resumen.get("soundLevelAvg", 0))
        p = 0.0
        if iah >= 15:   p += 4.0
        elif iah >= 5:  p += 2.0
        if bmi >= 30:   p += 2.0
        elif bmi >= 25: p += 1.0
        if snore:       p += 1.5
        if ev >= 15:    p += 1.5
        elif ev >= 5:   p += 0.75
        if hr > 85:     p += 0.5
        if snd > 65:    p += 0.5
        if p >= 5.5:   riesgo, conf = "Alto",     0.75
        elif p >= 2.5: riesgo, conf = "Moderado", 0.70
        else:          riesgo, conf = "Bajo",      0.80
        logger.info(f"[Rules] {riesgo} (p={p})")
        return {
            "mlRiskLevel": riesgo, "mlConfidence": conf,
            "mlRecommendation": RECOMENDACIONES[riesgo],
            "modelVersion": "rules-fallback", "mlClassProbs": {}
        }