# 🕵️ ProSpy — Fake Account Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/Flask-3.0.3-black?logo=flask" />
  <img src="https://img.shields.io/badge/TensorFlow-2.16.1-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Binary%20Classification-green" />
  <img src="https://img.shields.io/badge/Deployment-Render-purple?logo=render" />
</p>

<p align="center">
  <b>AI-powered fake social media account detection using Machine Learning.</b>
</p>

---

## 📌 About The Project

**ProSpy** is a Machine Learning-powered web application designed to analyze social media account information and predict whether an account is likely to be **Real** or **Fake**.

The project combines:

- 🤖 Machine Learning
- 🧠 TensorFlow/Keras
- 🌐 Flask
- 🎨 HTML/CSS/JavaScript
- 📊 Feature preprocessing
- 🚀 Cloud deployment

Users provide account/profile information through the web interface, and the trained ML model analyzes the selected features to generate a prediction.

---

## ✨ Features

- 🔍 Fake account detection
- 🤖 Trained TensorFlow/Keras model
- 📊 Feature preprocessing
- ⚙️ Feature scaling using a saved scaler
- 🎯 Selected feature support
- 🌐 Flask REST API
- 🎨 Simple web interface
- 🔄 Frontend → Backend → ML model pipeline
- 🚀 Ready for cloud deployment
- 📦 Reproducible environment using `requirements.txt`

---

## 🧠 How It Works

```text
                User
                 │
                 ▼
        ┌─────────────────┐
        │  Web Interface  │
        │   index.html    │
        └────────┬────────┘
                 │
                 │ Profile Data
                 ▼
        ┌─────────────────┐
        │  Flask Backend  │
        │     app.py      │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Feature         │
        │ Engineering     │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Feature Scaler  │
        │   scaler.pkl    │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ TensorFlow      │
        │ ML Model        │
        │ best_model.keras │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Prediction    │
        │  Real / Fake    │
        └─────────────────┘