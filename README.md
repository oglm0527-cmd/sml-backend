# Sleep Monitoring Lab (SML)

Sleep Monitoring Lab is an academic IoT-based system for monitoring
sleep-related physiological variables and analyzing patterns associated
with sleep apnea risk.

The system integrates sensor acquisition, MQTT communication, a Flask
backend, cloud services, machine learning and a web application.

## System Architecture

![SML System Architecture](docs/architecture.png)

## Backend Features

- Receives physiological data through MQTT
- Processes nightly sleep summaries and real-time samples
- Stores information in Cloud Firestore
- Provides REST endpoints using Flask
- Performs sleep-risk classification using a Random Forest model
- Integrates with the web application for data visualization
- Supports radar and microphone-based sensing

## Technologies

- Python
- Flask
- MQTT / HiveMQ
- Firebase Authentication
- Cloud Firestore
- Scikit-learn
- Random Forest
- Raspberry Pi
- Render

## Machine Learning

The project includes a Random Forest classifier used as an academic
proof of concept for sleep-risk classification.

The training dataset included in this repository is synthetically
generated and does not contain real patient information.

Current model performance was evaluated only on synthetic data and
must not be interpreted as clinical performance.

## API Endpoints

### GET `/health`

Checks backend status and model availability.

### GET `/model-info`

Returns information about the machine-learning model.

### POST `/predict`

Processes a nightly summary, obtains a risk classification and stores
the result in Firestore.

## Project Status

Academic prototype developed as part of the Bioengineering program at
Pontificia Universidad Javeriana.

> This project is an academic proof of concept and has not been
> clinically validated. It is not intended for medical diagnosis.
