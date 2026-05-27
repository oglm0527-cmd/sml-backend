"""
firebase_service.py
Inicializa Firebase Admin y guarda sesiones en Firestore.
Mantiene solo las últimas 5 noches por paciente.
"""
import os, json, logging
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)
MAX_SESIONES = 5

class FirebaseService:
    def __init__(self):
        self._inicializar()
        self.db = firestore.client()
        logger.info("FirebaseService listo")

    def _inicializar(self):
        if firebase_admin._apps:
            return
        # Modo 1: variable de entorno (Render)
        json_env = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if json_env:
            try:
                info = json.loads(json_env)
                firebase_admin.initialize_app(credentials.Certificate(info))
                logger.info("Firebase inicializado desde variable de entorno")
                return
            except Exception as e:
                logger.error(f"Error con FIREBASE_SERVICE_ACCOUNT_JSON: {e}")
                raise
        # Modo 2: archivo local (desarrollo)
        if os.path.exists("serviceAccountKey.json"):
            firebase_admin.initialize_app(credentials.Certificate("serviceAccountKey.json"))
            logger.info("Firebase inicializado desde serviceAccountKey.json")
            return
        raise RuntimeError(
            "Sin credenciales Firebase. "
            "Define FIREBASE_SERVICE_ACCOUNT_JSON o crea serviceAccountKey.json"
        )

    def uid_existe(self, uid: str) -> bool:
        try:
            doc = self.db.collection("users").document(uid).get()
            if not doc.exists:
                logger.warning(f"UID no encontrado: {uid}")
            return doc.exists
        except Exception as e:
            logger.error(f"Error verificando uid {uid}: {e}")
            return False
        
    def guardar_muestra(self, uid: str, datos: dict):
        try:
            session_id = datos.get("sessionId")
            if not session_id:
                fecha = datetime.now(timezone.utc).date().isoformat()
                session_id = f"session_{fecha}"

            sample_data = {
                "timestamp": datos.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "heartRate": float(datos.get("heartRate", 0)),
                "breathRate": float(datos.get("breathRate", 0)),
                "soundLevel": float(datos.get("soundLevel", 0)),
                "snoreDetected": bool(datos.get("snoreDetected", False)),
                "createdAt": firestore.SERVER_TIMESTAMP
            }

            ref = (
                self.db.collection("users").document(uid)
                       .collection("sleepHistory").document(session_id)
                       .collection("samples")
            )

            ref.add(sample_data)

            logger.info(
                f"✓ Muestra guardada: users/{uid}/sleepHistory/{session_id}/samples"
            )

            return True

        except Exception as e:
            logger.error(f"Error guardando muestra uid={uid}: {e}")
            return False

    def guardar_sesion_nocturna(self, uid: str, resumen: dict, prediccion: dict):
        try:
            fecha      = resumen.get("date", datetime.now(timezone.utc).date().isoformat())
            session_id = f"session_{fecha}"
            doc_ref    = (
                self.db.collection("users").document(uid)
                       .collection("sleepHistory").document(session_id)
            )
            iah = float(resumen.get("iah", 0))

            def iah_status(v):
                if v < 5:   return "Normal"
                if v < 15:  return "Leve"
                if v < 30:  return "Moderado"
                return "Severo"

            def sleep_quality(v, r):
                if r == "Alto" or v >= 15:  return "Mala"
                if r == "Moderado" or v>=5: return "Regular"
                return "Buena"

            datos = {
                "date":               fecha,
                "createdAt":          firestore.SERVER_TIMESTAMP,
                "heartRateAvg":       resumen.get("heartRateAvg", 0),
                "respiratoryRateAvg": resumen.get("respiratoryRateAvg", 0),
                "soundLevelAvg":      resumen.get("soundLevelAvg", 0),
                "snoreDetected":      bool(resumen.get("snoreDetected", False)),
                "totalEvents":        resumen.get("totalEvents", 0),
                "sleepHours":         resumen.get("sleepHours", 0),
                "iah":                iah,
                "iahStatus":          iah_status(iah),
                "mlRiskLevel":        prediccion.get("mlRiskLevel", ""),
                "mlConfidence":       prediccion.get("mlConfidence", 0),
                "mlRecommendation":   prediccion.get("mlRecommendation", ""),
                "modelVersion":       prediccion.get("modelVersion", ""),
                "mlClassProbs":       prediccion.get("mlClassProbs", {}),
                "riskLevel":          prediccion.get("mlRiskLevel", ""),
                "sleepQuality":       sleep_quality(iah, prediccion.get("mlRiskLevel","Bajo")),
                "mlDisclaimer":       "Resultado orientativo. No constituye diagnóstico médico."
            }
            doc_ref.set(datos, merge=True)
            logger.info(f"✓ Guardado: users/{uid}/sleepHistory/{session_id}")
            self._limpiar_antiguas(uid)
            return session_id
        except Exception as e:
            logger.error(f"Error guardando sesión uid={uid}: {e}")
            return None

    def _limpiar_antiguas(self, uid: str):
        try:
            ref = (self.db.collection("users").document(uid)
                          .collection("sleepHistory"))
            sesiones = ref.order_by("date", direction=firestore.Query.DESCENDING).get()
            if len(sesiones) <= MAX_SESIONES:
                return
            for s in sesiones[MAX_SESIONES:]:
                s.reference.delete()
                logger.info(f"Sesión antigua eliminada: {s.id}")
        except Exception as e:
            logger.warning(f"Error limpiando sesiones de uid={uid}: {e}")