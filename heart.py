from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import joblib
import sqlite3
import datetime
import os
import json

# Load the models
tachycardia_model = joblib.load('model/tachycardia_model.joblib')
hypertrophy_model = joblib.load('model/hypertrophy_model.joblib')
cholesterol_model = joblib.load('model/cholesterol_model.joblib')

# Initialize Flask app
app = Flask(__name__)

# Database setup
DB_PATH = 'heart_monitor.db'

def init_db():
    """Initialize the SQLite database with required tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for storing sensor readings and predictions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS heart_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        heart_rate INTEGER NOT NULL,
        hrv REAL NOT NULL,
        spo2 REAL NOT NULL,
        systolic INTEGER NOT NULL,
        diastolic INTEGER NOT NULL,
        body_temp REAL NOT NULL,
        tachycardia_pred INTEGER NOT NULL,
        hypertrophy_pred INTEGER NOT NULL,
        cholesterol_pred INTEGER NOT NULL,
        tachycardia_prob REAL,
        hypertrophy_prob REAL,
        cholesterol_prob REAL
    )
    ''')
    
    conn.commit()
    conn.close()

# Function to calculate additional parameters and make ML predictions
def calculate_health_metrics(heart_rate):
    # Calculate health metrics based on heart rate
    hrv = max(20, 100 - heart_rate)  # HRV tends to decrease with higher HR
    spo2 = 98 if heart_rate < 100 else 95  # SpO2 drops slightly with high HR
    
    # Simulate more features that the model expects
    systolic = 110 + (heart_rate // 10)  # Approximation based on HR
    diastolic = 70 + (heart_rate // 20)  # Approximation based on HR
    body_temp = 36.6 + (heart_rate - 70) * 0.01  # Slight increase with HR
    
    blood_pressure = f"{systolic}/{diastolic}"
    
    # Create feature array for ML models
    # Adjusted to provide 6 features as expected by the model
    features = np.array([[heart_rate, hrv, spo2, systolic, diastolic, body_temp]])
    
    # Make predictions using the ML models
    tachycardia_pred = tachycardia_model.predict(features)[0]
    hypertrophy_pred = hypertrophy_model.predict(features)[0]
    cholesterol_pred = cholesterol_model.predict(features)[0]
    
    # Get probability scores if models support predict_proba
    try:
        tachycardia_prob = tachycardia_model.predict_proba(features)[0][1]
        hypertrophy_prob = hypertrophy_model.predict_proba(features)[0][1]
        cholesterol_prob = cholesterol_model.predict_proba(features)[0][1]
    except:
        tachycardia_prob = hypertrophy_prob = cholesterol_prob = None
    
    # Current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Store data in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO heart_readings 
    (timestamp, heart_rate, hrv, spo2, systolic, diastolic, body_temp,
    tachycardia_pred, hypertrophy_pred, cholesterol_pred,
    tachycardia_prob, hypertrophy_prob, cholesterol_prob)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, heart_rate, hrv, spo2, systolic, diastolic, body_temp,
        int(tachycardia_pred), int(hypertrophy_pred), int(cholesterol_pred),
        tachycardia_prob, hypertrophy_prob, cholesterol_prob
    ))
    conn.commit()
    conn.close()
    
    return {
        "timestamp": timestamp,
        "Tachycardia Prediction": int(tachycardia_pred),
        "Hypertrophy Prediction": int(hypertrophy_pred),
        "High Cholesterol Prediction": int(cholesterol_pred),
        "Prediction Probabilities": {
            "Tachycardia Probability": float(tachycardia_prob) if tachycardia_prob is not None else None,
            "Hypertrophy Probability": float(hypertrophy_prob) if hypertrophy_prob is not None else None,
            "High Cholesterol Probability": float(cholesterol_prob) if cholesterol_prob is not None else None
        },
        "Calculated Features": {
            "Heart Rate (BPM)": heart_rate,
            "HRV (ms)": hrv,
            "SpO2 (%)": spo2,
            "Blood Pressure": blood_pressure,
            "Body Temperature (°C)": round(body_temp, 1)
        }
    }

def get_recent_readings(limit=10):
    """Get recent readings from the database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM heart_readings
    ORDER BY id DESC
    LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to list of dicts for JSON serialization
    result = []
    for row in rows:
        result.append({
            "id": row['id'],
            "timestamp": row['timestamp'],
            "heart_rate": row['heart_rate'],
            "hrv": row['hrv'],
            "spo2": row['spo2'],
            "systolic": row['systolic'],
            "diastolic": row['diastolic'],
            "body_temp": row['body_temp'],
            "blood_pressure": f"{row['systolic']}/{row['diastolic']}",
            "tachycardia_pred": row['tachycardia_pred'],
            "hypertrophy_pred": row['hypertrophy_pred'],
            "cholesterol_pred": row['cholesterol_pred'],
            "tachycardia_prob": row['tachycardia_prob'],
            "hypertrophy_prob": row['hypertrophy_prob'],
            "cholesterol_prob": row['cholesterol_prob']
        })
    
    return result

@app.route("/", methods=["GET"])
def index():
    # Get recent readings
    recent_readings = get_recent_readings(10)
    return render_template("index.html", readings=recent_readings)

@app.route("/api/data", methods=["GET"])
def get_data():
    """API endpoint to get recent readings for real-time updates"""
    recent_readings = get_recent_readings(10)
    return jsonify({"readings": recent_readings})

@app.route("/api/add_reading", methods=["POST"])
def add_reading():
    """API endpoint to add a new sensor reading"""
    if request.method == "POST":
        try:
            data = request.json
            heart_rate = int(data.get("heart_rate", 0))
            
            if heart_rate <= 0:
                return jsonify({"error": "Invalid heart rate"}), 400
                
            result = calculate_health_metrics(heart_rate)
            return jsonify({"success": True, "data": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/submit_reading", methods=["POST"])
def submit_reading():
    """Form endpoint to add a new reading through web interface"""
    if request.method == "POST":
        try:
            heart_rate = int(request.form["heart_rate"])
            calculate_health_metrics(heart_rate)
            return redirect(url_for('index'))
        except Exception as e:
            return f"Error: {str(e)}", 500

# Initialize database before running app
init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)