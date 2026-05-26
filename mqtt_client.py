"""
mqtt_client.py
Subscriber MQTT que escucha HiveMQ y delega a ml_service y firebase_service.
"""
import os, json, logging, ssl, time
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

class MQTTClient:
    def __init__(self, ml_service, firebase_service):
        self.ml  = ml_service
        self.db  = firebase_service
        self.host     = os.getenv("HIVEMQ_HOST", "")
        self.port     = int(os.getenv("HIVEMQ_PORT", "8883"))
        self.username = os.getenv("HIVEMQ_USERNAME", "")
        self.password = os.getenv("HIVEMQ_PASSWORD", "")
        if not all([self.host, self.username, self.password]):
            raise RuntimeError("Faltan variables HIVEMQ_HOST, HIVEMQ_USERNAME, HIVEMQ_PASSWORD")
        self.TOPIC_SUMMARY = "sml/patients/+/night-summary"
        self.TOPIC_SAMPLES = "sml/patients/+/samples"
        self.TOPIC_STATUS  = "sml/patients/+/status"
        self.client = self._crear_cliente()

    def _crear_cliente(self):
        c = mqtt.Client(client_id=f"sml-backend-{int(time.time())}", protocol=mqtt.MQTTv5)
        c.username_pw_set(self.username, self.password)
        c.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        c.on_connect    = self._on_connect
        c.on_disconnect = self._on_disconnect
        c.on_message    = self._on_message
        return c

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logger.info(f"Conectado a HiveMQ: {self.host}")
            client.subscribe(self.TOPIC_SUMMARY, qos=1)
            client.subscribe(self.TOPIC_SAMPLES, qos=1)
            client.subscribe(self.TOPIC_STATUS,  qos=1)
            logger.info("Suscripciones activas")
        else:
            logger.error(f"Error conexión MQTT código: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            logger.warning(f"Desconectado (código {reason_code}). Reconectando...")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            datos = json.loads(msg.payload.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            logger.error(f"JSON inválido en {topic}: {e}")
            return
        partes = topic.split("/")
        if len(partes) < 4:
            logger.error(f"Topic inválido: {topic}")
            return
        uid  = partes[2]
        tipo = partes[3]
        if not uid or uid == "undefined":
            logger.error(f"UID inválido en topic: {topic}")
            return
        if tipo == "night-summary":
            self._procesar_resumen(uid, datos)
        elif tipo == "samples":
            logger.debug(f"[sample] uid={uid} HR={datos.get('heartRate')} BR={datos.get('breathRate')}")
        elif tipo == "status":
            logger.info(f"[status] uid={uid} | {datos}")

    def _procesar_resumen(self, uid: str, datos: dict):
        logger.info(f"[night-summary] uid={uid}")
        if not self.db.uid_existe(uid):
            logger.error(f"uid={uid} no existe en Firestore. Registrar el paciente primero.")
            return
        prediccion = self.ml.predict_risk(datos)
        logger.info(f"[ML] {prediccion['mlRiskLevel']} conf={prediccion['mlConfidence']}")
        session_id = self.db.guardar_sesion_nocturna(uid, datos, prediccion)
        if session_id:
            logger.info(f"✓ Sesión guardada: {session_id}")
        else:
            logger.error(f"Error guardando sesión uid={uid}")

    def iniciar(self):
        logger.info(f"Conectando a {self.host}:{self.port}...")
        self.client.connect(self.host, self.port, keepalive=60)
        self.client.loop_start()
        logger.info("Loop MQTT iniciado en background")

    def detener(self):
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("MQTT detenido")