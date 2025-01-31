# 🌾 Crop Recommendation System (CRS) - Machine Learning API

## 🚀 Overview

The **Crop Recommendation System (CRS)** is a Machine Learning (ML) API designed to suggest the most suitable crop based on environmental factors like **temperature, humidity, soil pH, and rainfall**. The model is trained using **RandomForestClassifier** and deployed using **Flask**.

## 📌 Features

- 📡 **REST API** for predicting the best crop.
- 🔐 **CORS Protection** to restrict access to only the frontend.
- 📊 **Confidence Score** for each prediction.
- 🛠 **Scalable & Secure** backend.

---

## 🏗 Tech Stack

- **Backend:** Flask, Flask-CORS
- **ML Model:** Scikit-learn, NumPy, Pandas
- **Database (optional):** PostgreSQL/MySQL for user & farm data
- **Frontend:** Next.js (for UI, if applicable)

---

## ⚙️ Installation & Setup

### Create a Virtual Environment for python

```sh
py -m venv .venv
```

### Activate the virtual Environment

# for linux based cmd

```sh
source .venv/Scripts/activate
```

### 1️⃣ Install Dependencies

```sh
pip install flask flask-cors scikit-learn numpy pandas joblib dotenv
```

### 2️⃣ Set Environment Variables

Create a `.env` file in the root directory:

```ini
FRONT_END_URL=http://localhost:3000  # Change for production
```

### 3️⃣ Train the ML Model

Run the training script to generate the model file:

```sh
python models/train_model.py
```

This will create `model.pkl`, which will be used for predictions.

### 4️⃣ Start the Server

Run the Flask API server:

```sh
flask run
```

---

## 📌 API Endpoints

### **1️⃣ Predict Crop**

**Endpoint:** `POST /predict`

**Request Body (JSON):**

```json
{
    "district":"Shirala",
    "village":"Ambewadi",
    "ph":6.7
    "n":262.4,
    "p":30.0,
    "k":224.4,
    "Soil_Quality":"Fertile Soil",
}
```

**Response:**

```json
{
  "crop": "GroundNut"
  //   "confidence": 0.92
}
```

<!-- ### **2️⃣ Health Check**
**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "API is running"
}
```

---

## 🔐 CORS Security
This API allows access **only** from the frontend specified in the `.env` file:
```python
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": os.getenv("FRONT_END_URL")}})
```
-->

---

## 📚 Model Training (train_model.py)

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv("crop_data.csv")
X = data[["pH", 'N', 'P', 'K', 'District', 'Village', 'Soil_Quality']]
y = data["Crop"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
```

---

## 💡 Future Improvements

- ✅ Add **real-time weather data integration**.
- 📈 Improve model accuracy using **deep learning**.
- 🌍 Deploy using **Docker & Kubernetes** for scalability.

---

## 📝 License

This project is **open-source** under the **MIT License**.
