from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import numpy as np
import joblib
import sqlite3
import datetime
import os
import json
from flask_mail import Mail, Message
# FDI SMS Configuration
FDI_USERNAME="5BBEC534-9DC0-4DC9-A2D0-D33F5537AEC4"
FDI_PASSWORD="6335681A-074B-4FCF-81C1-FBFEB8C45579"
FDI_BASE_URL="https://messaging.fdibiz.com/api/v1/"
FDI_SENDER_ID="FDI"
# Load the models
tachycardia_model = joblib.load('model/tachycardia_model.joblib')
hypertrophy_model = joblib.load('model/hypertrophy_model.joblib')
cholesterol_model = joblib.load('model/cholesterol_model.joblib')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flash messages

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'heart-monitor@example.com')

mail = Mail(app)

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
        patient_id INTEGER NOT NULL,
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
        cholesterol_prob REAL,
        notification_sent INTEGER DEFAULT 0,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    )
    ''')
    
    # Create table for patients
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        medical_history TEXT,
        created_at TEXT NOT NULL
    )
    ''')
    
    # Create table for caregivers
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS caregivers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        phone TEXT,
        created_at TEXT NOT NULL
    )
    ''')
    
    # Create table for patient-caregiver relationships
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_caregivers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        caregiver_id INTEGER NOT NULL,
        relationship TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients (id),
        FOREIGN KEY (caregiver_id) REFERENCES caregivers (id),
        UNIQUE(patient_id, caregiver_id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Function to calculate additional parameters and make ML predictions
def calculate_health_metrics(heart_rate, patient_id):
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
    (timestamp, patient_id, heart_rate, hrv, spo2, systolic, diastolic, body_temp,
    tachycardia_pred, hypertrophy_pred, cholesterol_pred,
    tachycardia_prob, hypertrophy_prob, cholesterol_prob)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, patient_id, heart_rate, hrv, spo2, systolic, diastolic, body_temp,
        int(tachycardia_pred), int(hypertrophy_pred), int(cholesterol_pred),
        tachycardia_prob, hypertrophy_prob, cholesterol_prob
    ))
    
    reading_id = cursor.lastrowid
    conn.commit()
    
    # Check if any dangerous condition is detected
    is_dangerous = (int(tachycardia_pred) == 1 or 
                   int(hypertrophy_pred) == 1 or 
                   int(cholesterol_pred) == 1 or
                   heart_rate > 120 or 
                   systolic > 160 or 
                   diastolic > 100 or
                   spo2 < 92)
    
    # If dangerous condition detected, notify caregivers
    if is_dangerous:
        # Get patient details
        cursor.execute('SELECT name FROM patients WHERE id = ?', (patient_id,))
        patient_name = cursor.fetchone()[0]
        
        # Get caregivers for this patient
        cursor.execute('''
        SELECT c.name, c.email, pc.relationship 
        FROM caregivers c
        JOIN patient_caregivers pc ON c.id = pc.caregiver_id
        WHERE pc.patient_id = ?
        ''', (patient_id,))
        
        caregivers = cursor.fetchall()
        
        # Send notification to each caregiver
        for caregiver in caregivers:
            caregiver_name, caregiver_email, relationship = caregiver
            send_danger_notification(
                caregiver_email, 
                caregiver_name,
                patient_name, 
                relationship,
                {
                    "heart_rate": heart_rate,
                    "blood_pressure": blood_pressure,
                    "spo2": spo2,
                    "body_temp": round(body_temp, 1),
                    "tachycardia": bool(tachycardia_pred),
                    "hypertrophy": bool(hypertrophy_pred),
                    "high_cholesterol": bool(cholesterol_pred)
                }
            )
        
        # Mark notification as sent
        cursor.execute('UPDATE heart_readings SET notification_sent = 1 WHERE id = ?', (reading_id,))
        conn.commit()
    
    conn.close()
    
    return {
        "timestamp": timestamp,
        "patient_id": patient_id,
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
        },
        "is_dangerous": is_dangerous
    }

def send_danger_notification(caregiver_email, caregiver_name, patient_name, relationship, health_data):
    """Send email notification to caregiver about dangerous health condition"""
    subject = f"URGENT: Health Alert for {patient_name}"
    
    # Create message body
    body = f"""
    Dear {caregiver_name},
    
    This is an automated alert from the Heart Monitoring System.
    
    Your {relationship}, {patient_name}, has shown concerning health readings that may require attention:
    
    - Heart Rate: {health_data['heart_rate']} BPM
    - Blood Pressure: {health_data['blood_pressure']} mmHg
    - SpO2: {health_data['spo2']}%
    - Body Temperature: {health_data['body_temp']}°C
    
    Detected Conditions:
    - Tachycardia: {"Yes" if health_data['tachycardia'] else "No"}
    - Cardiac Hypertrophy: {"Yes" if health_data['hypertrophy'] else "No"}
    - High Cholesterol Risk: {"Yes" if health_data['high_cholesterol'] else "No"}
    
    Please check on the patient or contact their healthcare provider if needed.
    
    This is an automated message. Please do not reply.
    
    Heart Monitoring System
    """
    
    try:
        msg = Message(
            subject=subject,
            recipients=[caregiver_email],
            body=body
        )
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

def get_recent_readings(patient_id=None, limit=10):
    """Get recent readings from the database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    if patient_id:
        cursor.execute('''
        SELECT r.*, p.name as patient_name 
        FROM heart_readings r
        JOIN patients p ON r.patient_id = p.id
        WHERE r.patient_id = ?
        ORDER BY r.id DESC
        LIMIT ?
        ''', (patient_id, limit))
    else:
        cursor.execute('''
        SELECT r.*, p.name as patient_name 
        FROM heart_readings r
        JOIN patients p ON r.patient_id = p.id
        ORDER BY r.id DESC
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
            "patient_id": row['patient_id'],
            "patient_name": row['patient_name'],
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
            "cholesterol_prob": row['cholesterol_prob'],
            "notification_sent": row['notification_sent']
        })
    
    return result

def get_all_patients():
    """Get all patients from the database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM patients ORDER BY name')
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        result.append(dict(row))
    
    return result

def get_patient(patient_id):
    """Get a specific patient from the database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    row = cursor.fetchone()
    
    if row:
        patient = dict(row)
        
        # Get associated caregivers
        cursor.execute('''
        SELECT c.*, pc.relationship 
        FROM caregivers c
        JOIN patient_caregivers pc ON c.id = pc.caregiver_id
        WHERE pc.patient_id = ?
        ''', (patient_id,))
        
        caregivers = []
        for caregiver_row in cursor.fetchall():
            caregiver = dict(caregiver_row)
            caregivers.append(caregiver)
        
        patient['caregivers'] = caregivers
        conn.close()
        return patient
    
    conn.close()
    return None

def get_all_caregivers():
    """Get all caregivers from the database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM caregivers ORDER BY name')
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        result.append(dict(row))
    
    return result
@app.route("/home")
def home():
    """Home/Landing page"""
    return render_template("home.html")

@app.route("/", methods=["GET"])
def index():
    # Get all patients for the dropdown
    patients = get_all_patients()
    
    # Get recent readings (for all patients if none selected)
    patient_id = request.args.get('patient_id', type=int)
    recent_readings = get_recent_readings(patient_id, 10)
    
    selected_patient = None
    if patient_id:
        selected_patient = get_patient(patient_id)
    
    return render_template(
        "index.html", 
        readings=recent_readings, 
        patients=patients,
        selected_patient=selected_patient
    )

@app.route("/api/data", methods=["GET"])
def get_data():
    """API endpoint to get recent readings for real-time updates"""
    patient_id = request.args.get('patient_id', type=int)
    recent_readings = get_recent_readings(patient_id, 10)
    return jsonify({"readings": recent_readings})

@app.route("/api/add_reading", methods=["POST"])
def add_reading():
    """API endpoint to add a new sensor reading"""
    if request.method == "POST":
        try:
            data = request.json
            heart_rate = int(data.get("heart_rate", 0))
            patient_id = int(data.get("patient_id", 0))
            
            if heart_rate <= 0:
                return jsonify({"error": "Invalid heart rate"}), 400
                
            if patient_id <= 0:
                return jsonify({"error": "Invalid patient ID"}), 400
                
            result = calculate_health_metrics(heart_rate, patient_id)
            return jsonify({"success": True, "data": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/submit_reading", methods=["POST"])
def submit_reading():
    """Form endpoint to add a new reading through web interface"""
    if request.method == "POST":
        try:
            heart_rate = int(request.form["heart_rate"])
            patient_id = int(request.form["patient_id"])
            
            if patient_id <= 0:
                flash("Please select a patient", "error")
                return redirect(url_for('index'))
                
            result = calculate_health_metrics(heart_rate, patient_id)
            
            if result["is_dangerous"]:
                flash("Warning: Dangerous health condition detected! Caregivers have been notified.", "warning")
            else:
                flash("Reading added successfully", "success")
                
            return redirect(url_for('index', patient_id=patient_id))
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for('index'))

# Patient Management Routes
@app.route("/patients", methods=["GET"])
def list_patients():
    """List all patients"""
    patients = get_all_patients()
    return render_template("patients.html", patients=patients)

@app.route("/patients/add", methods=["GET", "POST"])
def add_patient():
    """Add a new patient"""
    if request.method == "POST":
        try:
            name = request.form["name"]
            age = int(request.form["age"])
            gender = request.form["gender"]
            medical_history = request.form.get("medical_history", "")
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO patients (name, age, gender, medical_history, created_at) VALUES (?, ?, ?, ?, ?)',
                (name, age, gender, medical_history, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()
            
            flash("Patient added successfully", "success")
            return redirect(url_for('list_patients'))
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for('add_patient'))
    
    return render_template("add_patient.html")

@app.route("/patients/<int:patient_id>", methods=["GET"])
def view_patient(patient_id):
    """View a specific patient"""
    patient = get_patient(patient_id)
    if not patient:
        flash("Patient not found", "error")
        return redirect(url_for('list_patients'))
    
    # Get recent readings for this patient
    readings = get_recent_readings(patient_id, 20)
    
    # Get all caregivers for assignment
    all_caregivers = get_all_caregivers()
    
    return render_template(
        "view_patient.html", 
        patient=patient, 
        readings=readings,
        all_caregivers=all_caregivers
    )

@app.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
def edit_patient(patient_id):
    """Edit a patient"""
    patient = get_patient(patient_id)
    if not patient:
        flash("Patient not found", "error")
        return redirect(url_for('list_patients'))
    
    if request.method == "POST":
        try:
            name = request.form["name"]
            age = int(request.form["age"])
            gender = request.form["gender"]
            medical_history = request.form.get("medical_history", "")
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE patients SET name = ?, age = ?, gender = ?, medical_history = ? WHERE id = ?',
                (name, age, gender, medical_history, patient_id)
            )
            conn.commit()
            conn.close()
            
            flash("Patient updated successfully", "success")
            return redirect(url_for('view_patient', patient_id=patient_id))
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for('edit_patient', patient_id=patient_id))
    
    return render_template("edit_patient.html", patient=patient)

# Caregiver Management Routes
@app.route("/caregivers", methods=["GET"])
def list_caregivers():
    """List all caregivers"""
    caregivers = get_all_caregivers()
    return render_template("caregivers.html", caregivers=caregivers)

@app.route("/caregivers/add", methods=["GET", "POST"])
def add_caregiver():
    """Add a new caregiver"""
    if request.method == "POST":
        try:
            name = request.form["name"]
            email = request.form["email"]
            phone = request.form.get("phone", "")
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO caregivers (name, email, phone, created_at) VALUES (?, ?, ?, ?)',
                (name, email, phone, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()
            
            flash("Caregiver added successfully", "success")
            return redirect(url_for('list_caregivers'))
        except sqlite3.IntegrityError:
            flash("A caregiver with this email already exists", "error")
            return redirect(url_for('add_caregiver'))
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for('add_caregiver'))
    
    return render_template("add_caregiver.html")

@app.route("/caregivers/<int:caregiver_id>/edit", methods=["GET", "POST"])
def edit_caregiver(caregiver_id):
    """Edit a caregiver"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM caregivers WHERE id = ?', (caregiver_id,))
    caregiver = dict(cursor.fetchone() or {})
    
    if not caregiver:
        conn.close()
        flash("Caregiver not found", "error")
        return redirect(url_for('list_caregivers'))
    
    if request.method == "POST":
        try:
            name = request.form["name"]
            email = request.form["email"]
            phone = request.form.get("phone", "")
            
            cursor.execute(
                'UPDATE caregivers SET name = ?, email = ?, phone = ? WHERE id = ?',
                (name, email, phone, caregiver_id)
            )
            conn.commit()
            conn.close()
            
            flash("Caregiver updated successfully", "success")
            return redirect(url_for('list_caregivers'))
        except sqlite3.IntegrityError:
            flash("A caregiver with this email already exists", "error")
            return redirect(url_for('edit_caregiver', caregiver_id=caregiver_id))
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for('edit_caregiver', caregiver_id=caregiver_id))
    
    conn.close()
    return render_template("edit_caregiver.html", caregiver=caregiver)

# Patient-Caregiver Association Routes
@app.route("/patients/<int:patient_id>/add_caregiver", methods=["POST"])
def add_patient_caregiver(patient_id):
    """Associate a caregiver with a patient"""
    try:
        caregiver_id = int(request.form["caregiver_id"])
        relationship = request.form["relationship"]
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if patient exists
        cursor.execute('SELECT id FROM patients WHERE id = ?', (patient_id,))
        if not cursor.fetchone():
            conn.close()
            flash("Patient not found", "error")
            return redirect(url_for('list_patients'))
        
        # Check if caregiver exists
        cursor.execute('SELECT id FROM caregivers WHERE id = ?', (caregiver_id,))
        if not cursor.fetchone():
            conn.close()
            flash("Caregiver not found", "error")
            return redirect(url_for('view_patient', patient_id=patient_id))
        
        # Add association
        try:
            cursor.execute(
                'INSERT INTO patient_caregivers (patient_id, caregiver_id, relationship, created_at) VALUES (?, ?, ?, ?)',
                (patient_id, caregiver_id, relationship, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            flash("Caregiver associated with patient successfully", "success")
        except sqlite3.IntegrityError:
            flash("This caregiver is already associated with this patient", "error")
        
        conn.close()
        return redirect(url_for('view_patient', patient_id=patient_id))
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for('view_patient', patient_id=patient_id))

@app.route("/patients/<int:patient_id>/remove_caregiver/<int:caregiver_id>", methods=["POST"])
def remove_patient_caregiver(patient_id, caregiver_id):
    """Remove a caregiver association from a patient"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            'DELETE FROM patient_caregivers WHERE patient_id = ? AND caregiver_id = ?',
            (patient_id, caregiver_id)
        )
        conn.commit()
        conn.close()
        
        flash("Caregiver removed from patient successfully", "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
    
    return redirect(url_for('view_patient', patient_id=patient_id))
@app.context_processor
def utility_processor():
    def get_reading_status(reading):
        # Check heart rate
        if hasattr(reading, 'heart_rate') and reading.heart_rate:
            if reading.heart_rate < 50 or reading.heart_rate > 120:
                return 'critical'
            elif reading.heart_rate < 60 or reading.heart_rate > 100:
                return 'warning'
        
        # Check O2 saturation
        if hasattr(reading, 'o2_saturation') and reading.o2_saturation:
            if reading.o2_saturation < 88:
                return 'critical'
            elif reading.o2_saturation < 92:
                return 'warning'
        
        # Check temperature
        if hasattr(reading, 'temperature') and reading.temperature:
            if reading.temperature < 35.0 or reading.temperature > 39.0:
                return 'critical'
            elif reading.temperature < 36.0 or reading.temperature > 38.5:
                return 'warning'
        
        return 'normal'
    
    return dict(get_reading_status=get_reading_status)@app.route("/sms_settings")
# @login_required
# @role_required('admin')
def sms_settings():
    """SMS configuration page"""
    return render_template("sms_settings.html", 
                         SMS_USERNAME=SMS_USERNAME,
                         SMS_BASE_URL=SMS_BASE_URL,
                         SMS_SENDER_ID=SMS_SENDER_ID)
# Initialize database before running app
init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
