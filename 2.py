from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, Response, stream_with_context
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import numpy as np
import joblib
import os
import re
from functools import wraps
import requests
import json
import secrets
import threading
import queue
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== Initialize Flask App ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_monitor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# ==================== FDI SMS Configuration ====================
SMS_USERNAME = os.getenv("FDI_USERNAME", "5BBEC534-9DC0-4DC9-A2D0-D33F5537AEC4")
SMS_PASSWORD = os.getenv("FDI_PASSWORD", "6335681A-074B-4FCF-81C1-FBFEB8C45579")
SMS_BASE_URL = os.getenv("FDI_BASE_URL", "https://messaging.fdibiz.com/api/v1/")
SMS_SENDER_ID = os.getenv("FDI_SENDER_ID", "FDI")
SMS_ENABLED = bool(SMS_USERNAME and SMS_PASSWORD)

# ==================== Load ML Models ====================
models_loaded = False
tachycardia_model = None
hypertrophy_model = None
cholesterol_model = None
scaler = None

try:
    tachycardia_model = joblib.load('model/tachycardia_model.joblib')
    hypertrophy_model = joblib.load('model/hypertrophy_model.joblib')
    cholesterol_model = joblib.load('model/cholesterol_model.joblib')
    scaler = joblib.load('model/scaler.pkl')
    models_loaded = True
    print("✓ ML models loaded successfully")
except Exception as e:
    print(f"⚠️ ML models not loaded: {str(e)}")
    print("   Running in demo mode with rule-based predictions")

# ==================== Real-time Data Streaming ====================
realtime_data = {}
data_lock = threading.Lock()

# ==================== Database Models ====================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    full_name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='caregiver')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    medical_history = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    readings = db.relationship('HeartReading', backref='patient', lazy=True)

class PatientCaregiver(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    caregiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    relationship = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('patient_id', 'caregiver_id', name='unique_patient_caregiver'),)

class HeartReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    heart_rate = db.Column(db.Integer, nullable=False)
    hrv = db.Column(db.Float)
    spo2 = db.Column(db.Float)
    systolic = db.Column(db.Integer)
    diastolic = db.Column(db.Integer)
    body_temp = db.Column(db.Float)
    
    tachycardia_pred = db.Column(db.Integer, default=0)
    hypertrophy_pred = db.Column(db.Integer, default=0)
    cholesterol_pred = db.Column(db.Integer, default=0)
    
    tachycardia_prob = db.Column(db.Float)
    hypertrophy_prob = db.Column(db.Float)
    cholesterol_prob = db.Column(db.Float)
    
    notification_sent = db.Column(db.Boolean, default=False)
    device_id = db.Column(db.String(50))

class IoTDevice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(50), unique=True, nullable=False)
    device_name = db.Column(db.String(100), nullable=False)
    device_type = db.Column(db.String(50))
    api_key = db.Column(db.String(100), unique=True, nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'))
    status = db.Column(db.String(20), default='active')
    last_seen = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ==================== FDI SMS Functions ====================

def send_sms_fdi(phone_number, message):
    """Send SMS using FDI API"""
    if not SMS_ENABLED:
        print(f"[SMS Demo] Would send to {phone_number}: {message[:100]}...")
        return True, "Demo mode - SMS not actually sent"
    
    try:
        # Clean phone number
        phone_number = re.sub(r'[^0-9+]', '', phone_number)
        
        # Ensure phone number has country code
        if not phone_number.startswith('+'):
            if phone_number.startswith('0'):
                phone_number = '+250' + phone_number[1:]
            else:
                phone_number = '+250' + phone_number
        
        url = SMS_BASE_URL + "send"
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "username": SMS_USERNAME,
            "password": SMS_PASSWORD,
            "sender_id": SMS_SENDER_ID,
            "to": phone_number,
            "text": message[:160]
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SMS sent successfully to {phone_number}")
            return True, "SMS sent successfully"
        else:
            print(f"❌ SMS failed: {response.status_code}")
            return False, f"SMS failed: {response.status_code}"
            
    except Exception as e:
        print(f"❌ SMS error: {str(e)}")
        return False, f"SMS error: {str(e)}"

def send_health_alert_sms(patient_name, caregiver_phone, caregiver_name, health_data):
    """Send health alert SMS to caregiver"""
    conditions_text = ', '.join(health_data['conditions'][:3])
    
    message = f"""URGENT HEALTH ALERT

Patient: {patient_name}
HR: {health_data['heart_rate']} BPM
BP: {health_data['blood_pressure']}
SpO2: {health_data['spo2']}%
Temp: {health_data['body_temp']}°C

Detected: {conditions_text}

Please check on patient immediately."""

    if len(message) > 160:
        message = f"ALERT: {patient_name} HR:{health_data['heart_rate']} BP:{health_data['blood_pressure']} SpO2:{health_data['spo2']}% Conditions:{conditions_text[:50]}"
    
    return send_sms_fdi(caregiver_phone, message)

def send_health_alert_to_caregivers(patient_id, health_data):
    """Send health alerts to all caregivers of a patient"""
    patient = Patient.query.get(patient_id)
    if not patient:
        return False
    
    caregivers = db.session.query(User).join(
        PatientCaregiver, User.id == PatientCaregiver.caregiver_id
    ).filter(PatientCaregiver.patient_id == patient_id).all()
    
    if not caregivers:
        return False
    
    alert_sent = False
    for caregiver in caregivers:
        if caregiver.phone:
            success, message = send_health_alert_sms(
                patient.name, caregiver.phone, caregiver.full_name, health_data
            )
            if success:
                alert_sent = True
    
    return alert_sent

# ==================== Real-time Streaming Functions ====================

def broadcast_realtime_data(device_id, data):
    """Broadcast real-time data to all connected clients"""
    with data_lock:
        if device_id in realtime_data:
            json_data = json.dumps(data)
            for client_queue in realtime_data[device_id]:
                try:
                    client_queue.put_nowait(json_data)
                except:
                    pass

# ==================== Health Prediction Functions ====================

def calculate_health_metrics(heart_rate, patient_id=None, device_id=None):
    """Calculate health metrics and make ML predictions"""
    
    # Calculate derived metrics
    hrv = max(20, min(100, 100 - heart_rate * 0.5))
    spo2 = 98 if heart_rate < 100 else 96 if heart_rate < 120 else 94
    systolic = 110 + (heart_rate // 10)
    diastolic = 70 + (heart_rate // 20)
    body_temp = 36.6 + (heart_rate - 70) * 0.01
    
    blood_pressure = f"{systolic}/{diastolic}"
    
    # Make predictions
    if models_loaded and scaler:
        try:
            features = np.array([[heart_rate, hrv, spo2, systolic, diastolic, body_temp]])
            features_scaled = scaler.transform(features)
            
            tachycardia_pred = int(tachycardia_model.predict(features_scaled)[0])
            hypertrophy_pred = int(hypertrophy_model.predict(features_scaled)[0])
            cholesterol_pred = int(cholesterol_model.predict(features_scaled)[0])
            
            if hasattr(tachycardia_model, 'predict_proba'):
                tachycardia_prob = float(tachycardia_model.predict_proba(features_scaled)[0][1])
                hypertrophy_prob = float(hypertrophy_model.predict_proba(features_scaled)[0][1])
                cholesterol_prob = float(cholesterol_model.predict_proba(features_scaled)[0][1])
            else:
                tachycardia_prob = tachycardia_pred
                hypertrophy_prob = hypertrophy_pred
                cholesterol_prob = cholesterol_pred
        except:
            tachycardia_pred = 1 if heart_rate > 100 else 0
            hypertrophy_pred = 1 if 90 < heart_rate < 110 else 0
            cholesterol_pred = 0
            tachycardia_prob = min(1.0, max(0, (heart_rate - 60) / 100))
            hypertrophy_prob = 0.5 if 90 < heart_rate < 110 else 0.1
            cholesterol_prob = 0.2
    else:
        tachycardia_pred = 1 if heart_rate > 100 else 0
        hypertrophy_pred = 1 if 90 < heart_rate < 110 else 0
        cholesterol_pred = 0
        tachycardia_prob = min(1.0, max(0, (heart_rate - 60) / 100))
        hypertrophy_prob = 0.5 if 90 < heart_rate < 110 else 0.1
        cholesterol_prob = 0.2
    
    reading_id = None
    if patient_id:
        reading = HeartReading(
            patient_id=patient_id,
            heart_rate=heart_rate,
            hrv=round(hrv, 1),
            spo2=round(spo2, 1),
            systolic=systolic,
            diastolic=diastolic,
            body_temp=round(body_temp, 1),
            tachycardia_pred=tachycardia_pred,
            hypertrophy_pred=hypertrophy_pred,
            cholesterol_pred=cholesterol_pred,
            tachycardia_prob=round(tachycardia_prob, 3),
            hypertrophy_prob=round(hypertrophy_prob, 3),
            cholesterol_prob=round(cholesterol_prob, 3),
            device_id=device_id
        )
        db.session.add(reading)
        db.session.commit()
        reading_id = reading.id
        
        conditions = []
        if tachycardia_pred == 1:
            conditions.append("Tachycardia")
        if hypertrophy_pred == 1:
            conditions.append("Cardiac Hypertrophy")
        if cholesterol_pred == 1:
            conditions.append("High Cholesterol Risk")
        if heart_rate > 120:
            conditions.append("Severe Tachycardia")
        if heart_rate < 50:
            conditions.append("Bradycardia")
        if spo2 < 92:
            conditions.append("Low Oxygen")
        
        is_dangerous = len(conditions) > 0
        
        if is_dangerous and not reading.notification_sent:
            health_data = {
                'heart_rate': heart_rate,
                'blood_pressure': blood_pressure,
                'spo2': round(spo2, 1),
                'body_temp': round(body_temp, 1),
                'conditions': conditions
            }
            
            if send_health_alert_to_caregivers(patient_id, health_data):
                reading.notification_sent = True
                db.session.commit()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "heart_rate": heart_rate,
        "hrv": round(hrv, 1),
        "spo2": round(spo2, 1),
        "blood_pressure": blood_pressure,
        "body_temp": round(body_temp, 1),
        "tachycardia_prediction": tachycardia_pred,
        "hypertrophy_prediction": hypertrophy_pred,
        "cholesterol_prediction": cholesterol_pred,
        "tachycardia_probability": round(tachycardia_prob * 100, 1),
        "hypertrophy_probability": round(hypertrophy_prob * 100, 1),
        "cholesterol_probability": round(cholesterol_prob * 100, 1),
        "is_dangerous": len(conditions) > 0 if patient_id else False,
        "conditions_detected": conditions if patient_id else [],
        "reading_id": reading_id
    }

# ==================== Authentication Functions ====================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                flash('Please log in to access this page.', 'warning')
                return redirect(url_for('login'))
            if current_user.role not in roles:
                flash('You do not have permission to access this page.', 'error')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ==================== Main Routes ====================

@app.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            patients = Patient.query.all()
        else:
            patients = db.session.query(Patient).join(
                PatientCaregiver, Patient.id == PatientCaregiver.patient_id
            ).filter(PatientCaregiver.caregiver_id == current_user.id).all()
        
        patient_id = request.args.get('patient_id', type=int)
        if patient_id:
            readings = HeartReading.query.filter_by(patient_id=patient_id).order_by(HeartReading.id.desc()).limit(20).all()
            selected_patient = Patient.query.get(patient_id)
        else:
            patient_ids = [p.id for p in patients]
            if patient_ids:
                readings = HeartReading.query.filter(HeartReading.patient_id.in_(patient_ids)).order_by(HeartReading.id.desc()).limit(20).all()
            else:
                readings = []
            selected_patient = None
        
        # Convert datetime for template
        for reading in readings:
            if reading.timestamp:
                reading.timestamp_str = reading.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        return render_template("dashboard.html", 
                             patients=patients, 
                             readings=readings,
                             selected_patient=selected_patient)
    else:
        return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if not user.is_active:
                flash('Your account has been deactivated.', 'error')
                return redirect(url_for('login'))
            
            login_user(user)
            flash(f'Welcome back, {user.full_name}!', 'success')
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        phone = request.form.get("phone")
        full_name = request.form.get("full_name")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        role = request.form.get("role", "caregiver")
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            email=email,
            phone=phone,
            full_name=full_name,
            role=role
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        if phone and SMS_ENABLED:
            send_sms_fdi(phone, f"Welcome {full_name}! Your Heart Monitor account is ready.")
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ==================== Patient Management ====================

@app.route("/patients")
@login_required
def list_patients():
    if current_user.role == 'admin':
        patients = Patient.query.all()
    else:
        patients = db.session.query(Patient).join(
            PatientCaregiver, Patient.id == PatientCaregiver.patient_id
        ).filter(PatientCaregiver.caregiver_id == current_user.id).all()
    
    return render_template("patients.html", patients=patients)

@app.route("/patients/add", methods=["GET", "POST"])
@login_required
@role_required('admin', 'doctor')
def add_patient():
    if request.method == "POST":
        name = request.form.get("name")
        age = int(request.form.get("age"))
        gender = request.form.get("gender")
        medical_history = request.form.get("medical_history", "")
        
        patient = Patient(
            name=name,
            age=age,
            gender=gender,
            medical_history=medical_history,
            created_by=current_user.id
        )
        
        db.session.add(patient)
        db.session.commit()
        
        flash(f'Patient {name} added successfully!', 'success')
        return redirect(url_for('list_patients'))
    
    return render_template("add_patient.html")

@app.route("/patients/<int:patient_id>")
@login_required
def view_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    if current_user.role != 'admin':
        assignment = PatientCaregiver.query.filter_by(
            patient_id=patient_id, caregiver_id=current_user.id
        ).first()
        if not assignment:
            flash('You do not have access to this patient.', 'error')
            return redirect(url_for('list_patients'))
    
    readings = HeartReading.query.filter_by(patient_id=patient_id).order_by(HeartReading.id.desc()).limit(50).all()
    
    # Serialize readings for charts
    serializable_readings = []
    for r in readings:
        serializable_readings.append({
            'id': r.id,
            'timestamp': r.timestamp.isoformat() if r.timestamp else None,
            'heart_rate': r.heart_rate,
            'hrv': r.hrv,
            'spo2': r.spo2,
            'systolic': r.systolic,
            'diastolic': r.diastolic,
            'body_temp': r.body_temp,
            'tachycardia_pred': r.tachycardia_pred,
            'hypertrophy_pred': r.hypertrophy_pred,
            'cholesterol_pred': r.cholesterol_pred,
            'notification_sent': r.notification_sent
        })
    
    caregivers = db.session.query(User).join(
        PatientCaregiver, User.id == PatientCaregiver.caregiver_id
    ).filter(PatientCaregiver.patient_id == patient_id).all()
    
    all_caregivers = User.query.filter(User.role == 'caregiver').all()
    
    return render_template("view_patient.html", 
                         patient=patient, 
                         readings=readings,
                         serializable_readings=serializable_readings,
                         caregivers=caregivers,
                         all_caregivers=all_caregivers)

@app.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@login_required
@role_required('admin', 'doctor')
def edit_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    if request.method == "POST":
        patient.name = request.form.get("name")
        patient.age = int(request.form.get("age"))
        patient.gender = request.form.get("gender")
        patient.medical_history = request.form.get("medical_history", "")
        
        db.session.commit()
        flash('Patient updated successfully!', 'success')
        return redirect(url_for('view_patient', patient_id=patient_id))
    
    return render_template("edit_patient.html", patient=patient)

@app.route("/patients/<int:patient_id>/add_caregiver", methods=["POST"])
@login_required
@role_required('admin', 'doctor')
def add_patient_caregiver(patient_id):
    caregiver_id = request.form.get("caregiver_id")
    relationship = request.form.get("relationship")
    
    existing = PatientCaregiver.query.filter_by(
        patient_id=patient_id, caregiver_id=caregiver_id
    ).first()
    
    if existing:
        flash('Caregiver already assigned to this patient.', 'warning')
    else:
        assignment = PatientCaregiver(
            patient_id=patient_id,
            caregiver_id=caregiver_id,
            relationship=relationship
        )
        db.session.add(assignment)
        db.session.commit()
        flash('Caregiver assigned successfully!', 'success')
    
    return redirect(url_for('view_patient', patient_id=patient_id))

@app.route("/patients/<int:patient_id>/remove_caregiver/<int:caregiver_id>", methods=["POST"])
@login_required
@role_required('admin', 'doctor')
def remove_patient_caregiver(patient_id, caregiver_id):
    assignment = PatientCaregiver.query.filter_by(
        patient_id=patient_id, caregiver_id=caregiver_id
    ).first_or_404()
    
    db.session.delete(assignment)
    db.session.commit()
    flash('Caregiver removed successfully.', 'success')
    
    return redirect(url_for('view_patient', patient_id=patient_id))

# ==================== Health Readings ====================

@app.route("/add_reading", methods=["POST"])
@login_required
def add_reading():
    patient_id = request.form.get("patient_id", type=int)
    heart_rate = request.form.get("heart_rate", type=int)
    
    if not patient_id or not heart_rate:
        flash('Please provide both patient ID and heart rate.', 'error')
        return redirect(url_for('index'))
    
    if current_user.role != 'admin':
        assignment = PatientCaregiver.query.filter_by(
            patient_id=patient_id, caregiver_id=current_user.id
        ).first()
        if not assignment:
            flash('You do not have access to this patient.', 'error')
            return redirect(url_for('index'))
    
    result = calculate_health_metrics(heart_rate, patient_id)
    
    if result['is_dangerous']:
        flash(f'⚠️ DANGEROUS CONDITION: {", ".join(result["conditions_detected"])}. Caregivers alerted!', 'warning')
    else:
        flash('Reading added successfully.', 'success')
    
    return redirect(url_for('index', patient_id=patient_id))

# ==================== API Endpoints ====================

@app.route("/api/reading", methods=["POST"])
def api_add_reading():
    """API endpoint for IoT devices"""
    try:
        data = request.json
        print(f"Received data: {data}")
        
        device_id = data.get('device_id')
        api_key = data.get('api_key')
        heart_rate = data.get('heart_rate')
        
        device = IoTDevice.query.filter_by(device_id=device_id, api_key=api_key).first()
        if not device:
            return jsonify({"error": "Invalid device credentials"}), 401
        
        if device.status != 'active':
            return jsonify({"error": "Device is not active"}), 403
        
        if not heart_rate or heart_rate < 30 or heart_rate > 200:
            return jsonify({"error": f"Invalid heart rate: {heart_rate}"}), 400
        
        device.last_seen = datetime.utcnow()
        db.session.commit()
        
        patient_id = device.patient_id
        result = calculate_health_metrics(heart_rate, patient_id, device_id)
        
        # Broadcast real-time data
        realtime_payload = {
            "heart_rate": heart_rate,
            "timestamp": datetime.now().isoformat(),
            "predictions": {
                "tachycardia": result['tachycardia_prediction'],
                "hypertrophy": result['hypertrophy_prediction'],
                "cholesterol": result['cholesterol_prediction']
            },
            "is_dangerous": result['is_dangerous']
        }
        broadcast_realtime_data(device_id, realtime_payload)
        
        return jsonify({
            "success": True,
            "reading_id": result.get('reading_id'),
            "heart_rate": heart_rate,
            "predictions": result['tachycardia_prediction'],
            "is_dangerous": result['is_dangerous']
        }), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/stream/<device_id>")
def stream_data(device_id):
    """Server-Sent Events endpoint for real-time streaming"""
    def generate():
        client_queue = queue.Queue()
        
        with data_lock:
            if device_id not in realtime_data:
                realtime_data[device_id] = []
            realtime_data[device_id].append(client_queue)
        
        try:
            while True:
                try:
                    data = client_queue.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield f": heartbeat\n\n"
        finally:
            with data_lock:
                if device_id in realtime_data:
                    realtime_data[device_id].remove(client_queue)
                    if not realtime_data[device_id]:
                        del realtime_data[device_id]
    
    return Response(stream_with_context(generate()), 
                   mimetype="text/event-stream",
                   headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ==================== Device Management ====================

def generate_device_id():
    return f"DEV_{secrets.token_hex(8).upper()}"

def generate_api_key():
    return secrets.token_urlsafe(32)

@app.route("/devices")
@login_required
def list_devices():
    if current_user.role == 'admin':
        devices = IoTDevice.query.all()
    else:
        patient_ids = db.session.query(PatientCaregiver.patient_id).filter_by(caregiver_id=current_user.id).subquery()
        devices = IoTDevice.query.filter(IoTDevice.patient_id.in_(patient_ids)).all()
    
    return render_template("devices.html", devices=devices)

@app.route("/devices/add", methods=["GET", "POST"])
@login_required
@role_required('admin')
def add_device():
    patients = Patient.query.all()
    
    if request.method == "POST":
        device_name = request.form.get("device_name")
        device_type = request.form.get("device_type")
        patient_id = request.form.get("patient_id", type=int)
        
        device = IoTDevice(
            device_id=generate_device_id(),
            device_name=device_name,
            device_type=device_type,
            api_key=generate_api_key(),
            patient_id=patient_id if patient_id else None
        )
        
        db.session.add(device)
        db.session.commit()
        
        flash(f'Device {device_name} registered! ID: {device.device_id}', 'success')
        return redirect(url_for('list_devices'))
    
    return render_template("add_device.html", patients=patients)

@app.route("/devices/<device_id>")
@login_required
def view_device(device_id):
    device = IoTDevice.query.filter_by(device_id=device_id).first_or_404()
    
    if current_user.role != 'admin' and device.patient_id:
        assignment = PatientCaregiver.query.filter_by(
            patient_id=device.patient_id, caregiver_id=current_user.id
        ).first()
        if not assignment:
            flash('You do not have access to this device.', 'error')
            return redirect(url_for('list_devices'))
    
    readings = HeartReading.query.filter_by(device_id=device_id).order_by(HeartReading.id.desc()).limit(50).all()
    patients = Patient.query.all()
    
    return render_template("view_device.html", device=device, readings=readings, patients=patients)

@app.route("/devices/<device_id>/edit", methods=["POST"])
@login_required
@role_required('admin')
def edit_device(device_id):
    device = IoTDevice.query.filter_by(device_id=device_id).first_or_404()
    
    device.device_name = request.form.get("device_name")
    device.device_type = request.form.get("device_type")
    device.patient_id = request.form.get("patient_id", type=int) or None
    device.status = request.form.get("status")
    
    db.session.commit()
    flash('Device updated successfully!', 'success')
    
    return redirect(url_for('view_device', device_id=device_id))

@app.route("/devices/<device_id>/regenerate_key", methods=["POST"])
@login_required
@role_required('admin')
def regenerate_device_key(device_id):
    device = IoTDevice.query.filter_by(device_id=device_id).first_or_404()
    device.api_key = generate_api_key()
    db.session.commit()
    
    flash('API key regenerated successfully!', 'success')
    return redirect(url_for('view_device', device_id=device_id))

# ==================== User Management ====================

@app.route("/users")
@login_required
@role_required('admin')
def list_users():
    users = User.query.all()
    return render_template("users.html", users=users)

@app.route("/users/add", methods=["GET", "POST"])
@login_required
@role_required('admin')
def add_user():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        phone = request.form.get("phone")
        full_name = request.form.get("full_name")
        password = request.form.get("password")
        role = request.form.get("role")
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('add_user'))
        
        user = User(
            username=username,
            email=email,
            phone=phone,
            full_name=full_name,
            role=role
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash(f'User {username} created successfully!', 'success')
        return redirect(url_for('list_users'))
    
    return render_template("add_user.html")

@app.route("/users/<int:user_id>/edit", methods=["GET", "POST"])
@login_required
@role_required('admin')
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if request.method == "POST":
        user.full_name = request.form.get("full_name")
        user.email = request.form.get("email")
        user.phone = request.form.get("phone")
        user.role = request.form.get("role")
        user.is_active = 'is_active' in request.form
        
        new_password = request.form.get("new_password")
        if new_password:
            user.set_password(new_password)
        
        db.session.commit()
        flash('User updated successfully!', 'success')
        return redirect(url_for('list_users'))
    
    return render_template("edit_user.html", user=user)

@app.route("/users/<int:user_id>/delete", methods=["POST"])
@login_required
@role_required('admin')
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'error')
        return redirect(url_for('list_users'))
    
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully.', 'success')
    
    return redirect(url_for('list_users'))

# ==================== Caregiver Management ====================

@app.route("/caregivers")
@login_required
def list_caregivers():
    if current_user.role == 'admin':
        caregivers = User.query.filter_by(role='caregiver').all()
    elif current_user.role == 'doctor':
        caregivers = User.query.filter_by(role='caregiver').all()
    else:
        caregivers = [current_user]
    
    for caregiver in caregivers:
        caregiver.patient_count = PatientCaregiver.query.filter_by(caregiver_id=caregiver.id).count()
    
    return render_template("caregivers.html", caregivers=caregivers)

@app.route("/caregivers/add", methods=["GET", "POST"])
@login_required
@role_required('admin', 'doctor')
def add_caregiver():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        
        if not name or not email:
            flash('Name and email are required.', 'error')
            return redirect(url_for('add_caregiver'))
        
        if User.query.filter_by(email=email).first():
            flash('A user with this email already exists.', 'error')
            return redirect(url_for('add_caregiver'))
        
        username = email.split('@')[0]
        base_username = username
        counter = 1
        while User.query.filter_by(username=username).first():
            username = f"{base_username}{counter}"
            counter += 1
        
        import string
        alphabet = string.ascii_letters + string.digits
        password = ''.join(secrets.choice(alphabet) for i in range(10))
        
        caregiver = User(
            username=username,
            email=email,
            phone=phone,
            full_name=name,
            role='caregiver',
            is_active=True
        )
        caregiver.set_password(password)
        
        db.session.add(caregiver)
        db.session.commit()
        
        flash(f'Caregiver {name} added! Username: {username}, Password: {password}', 'success')
        return redirect(url_for('list_caregivers'))
    
    return render_template("add_caregiver.html")

@app.route("/caregivers/<int:caregiver_id>/edit", methods=["GET", "POST"])
@login_required
@role_required('admin', 'doctor')
def edit_caregiver(caregiver_id):
    caregiver = User.query.get_or_404(caregiver_id)
    
    if caregiver.role != 'caregiver':
        flash('This user is not a caregiver.', 'error')
        return redirect(url_for('list_caregivers'))
    
    if request.method == "POST":
        caregiver.full_name = request.form.get("name")
        caregiver.email = request.form.get("email")
        caregiver.phone = request.form.get("phone")
        
        new_password = request.form.get("new_password")
        if new_password and len(new_password) >= 6:
            caregiver.set_password(new_password)
            flash('Password updated successfully.', 'success')
        
        db.session.commit()
        flash('Caregiver updated successfully!', 'success')
        return redirect(url_for('list_caregivers'))
    
    return render_template("edit_caregiver.html", caregiver=caregiver)

# ==================== SMS Test Route ====================

@app.route("/test_sms", methods=["GET", "POST"])
@login_required
@role_required('admin')
def test_sms():
    if request.method == "POST":
        phone_number = request.form.get("phone_number")
        if phone_number:
            success, message = send_sms_fdi(phone_number, "Test SMS from Heart Monitor System. Your SMS configuration is working correctly!")
            if success:
                flash(f"Test SMS sent successfully to {phone_number}!", "success")
            else:
                flash(f"Failed to send SMS: {message}", "error")
        else:
            flash("Please enter a phone number", "error")
        return redirect(url_for('test_sms'))
    
    return render_template("test_sms.html", 
                         SMS_USERNAME=SMS_USERNAME,
                         SMS_BASE_URL=SMS_BASE_URL,
                         SMS_SENDER_ID=SMS_SENDER_ID)

# ==================== Real-time Dashboard ====================

@app.route("/realtime")
@login_required
def realtime_dashboard():
    """Real-time ECG monitoring dashboard"""
    devices = IoTDevice.query.filter_by(status='active').all()
    return render_template("realtime_dashboard.html", devices=devices)

# ==================== Database Initialization ====================

def init_db():
    """Initialize database and create default users"""
    db.create_all()
    
    # Create default admin
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@heartmonitor.com',
            phone='+250788123456',
            full_name='System Administrator',
            role='admin'
        )
        admin.set_password('admin123')
        db.session.add(admin)
        print("✓ Default admin created: admin / admin123")
    
    # Create demo doctor
    if not User.query.filter_by(username='doctor').first():
        doctor = User(
            username='doctor',
            email='doctor@heartmonitor.com',
            phone='+250788123457',
            full_name='Dr. John Smith',
            role='doctor'
        )
        doctor.set_password('doctor123')
        db.session.add(doctor)
        print("✓ Demo doctor created: doctor / doctor123")
    
    # Create demo caregiver
    if not User.query.filter_by(username='caregiver').first():
        caregiver = User(
            username='caregiver',
            email='caregiver@heartmonitor.com',
            phone='+250788123458',
            full_name='Jane Caregiver',
            role='caregiver'
        )
        caregiver.set_password('caregiver123')
        db.session.add(caregiver)
        print("✓ Demo caregiver created: caregiver / caregiver123")
    
    # Create demo patient
    if Patient.query.count() == 0:
        patient = Patient(
            name='John Patient',
            age=65,
            gender='Male',
            medical_history='Hypertension, Type 2 Diabetes',
            created_by=1
        )
        db.session.add(patient)
        db.session.commit()
        
        assignment = PatientCaregiver(
            patient_id=patient.id,
            caregiver_id=3,
            relationship='Primary Caregiver'
        )
        db.session.add(assignment)
        db.session.commit()
        print("✓ Demo patient created with caregiver assignment")
    
    # Create demo IoT device
    if IoTDevice.query.count() == 0:
        device = IoTDevice(
            device_id=generate_device_id(),
            device_name='Demo Heart Monitor',
            device_type='Heart Monitor',
            api_key=generate_api_key(),
            patient_id=1,
            status='active'
        )
        db.session.add(device)
        db.session.commit()
        print(f"✓ Demo IoT device created: {device.device_id}")
    
    db.session.commit()
    print("\n✓ Database initialization complete!")
    print("=" * 50)
    print("Login Credentials:")
    print("  Admin:     admin / admin123")
    print("  Doctor:    doctor / doctor123")
    print("  Caregiver: caregiver / caregiver123")
    print("=" * 50)
    print(f"SMS Status: {'ENABLED' if SMS_ENABLED else 'DISABLED (Demo Mode)'}")
    print(f"ML Models:  {'LOADED' if models_loaded else 'DEMO MODE'}")
    print("=" * 50)

# ==================== Run App ====================

if __name__ == '__main__':
    with app.app_context():
        init_db()
    print(f"\n🚀 Starting Heart Monitor System...")
    print(f"🌐 Server running at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)