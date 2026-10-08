"""
app.py
Servidor Flask principal. Inicializa servicios y arranca el subscriber MQTT.
IMPORTANTE: gunicorn debe usar --workers 1 para que MQTT funcione bien.
"""
import os, logging
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_service       import MLService
from firebase_service import FirebaseService
from mqtt_client      import MQTTClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
)
logger = logging.getLogger("SML")

app = Flask(__name__)
CORS(app)

# Inicializar los tres servicios UNA SOLA VEZ al arrancar
logger.info("Inicializando servicios SML...")
ml_service       = MLService()
firebase_service = FirebaseService()
mqtt             = MQTTClient(ml_service, firebase_service)
mqtt.iniciar()
logger.info("Backend SML listo ✓")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "service":      "SML Backend",
        "modelVersion": ml_service.version,
        "modelLoaded":  ml_service.cargado,
        "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint HTTP para predicción directa (sin MQTT).
    Útil para probar el modelo y para la demo.
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type debe ser application/json"}), 400
    datos = request.get_json(silent=True)
    if not datos:
        return jsonify({"error": "JSON inválido"}), 400
    uid = datos.get("uid")
    if not uid:
        return jsonify({"error": "Campo 'uid' requerido"}), 422
    if not firebase_service.uid_existe(uid):
        return jsonify({"error": f"uid '{uid}' no encontrado en Firestore"}), 404
    prediccion = ml_service.predict_risk(datos)
    session_id = firebase_service.guardar_sesion_nocturna(uid, datos, prediccion)
    return jsonify({**prediccion, "sessionId": session_id, "uid": uid}), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "modelVersion": ml_service.version,
        "modelLoaded":  ml_service.cargado,
        "features":     ml_service.features,
        "algorithm":    "Random Forest Classifier",
        "disclaimer":   "Prueba de concepto académica. No validado clínicamente."
    }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint no encontrado"}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Error interno: {e}")
    return jsonify({"error": "Error interno del servidor"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
