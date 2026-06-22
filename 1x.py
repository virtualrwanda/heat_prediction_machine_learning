from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
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

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_monitor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Load ML models
try:
    tachycardia_model = joblib.load('model/tachycardia_model.joblib')
    hypertrophy_model = joblib.load('model/hypertrophy_model.joblib')
    cholesterol_model = joblib.load('model/cholesterol_model.joblib')
    scaler = joblib.load('model/scaler.pkl')
    models_loaded = True
except:
    models_loaded = False
    print("Warning: ML models not found. Running in demo mode.")

# Africa's Talking SMS Configuration
AFRICASTALKING_USERNAME = os.environ.get('AFRICASTALKING_USERNAME', 'sandbox')
AFRICASTALKING_API_KEY = os.environ.get('AFRICASTALKING_API_KEY', 'your_sandbox_api_key')
SMS_SENDER_ID = os.environ.get('SMS_SENDER_ID', 'HeartMonitor')

# ==================== DATABASE MODELS ====================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    full_name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='caregiver')  # admin, doctor, caregiver
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    patients = db.relationship('PatientCaregiver', foreign_keys='PatientCaregiver.caregiver_id', backref='caregiver')
    
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
    
    # Relationships
    readings = db.relationship('HeartReading', backref='patient', lazy=True)
    caregivers = db.relationship('PatientCaregiver', backref='patient_assigned', lazy=True)

class PatientCaregiver(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    caregiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    relationship = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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
    
    # ML Predictions
    tachycardia_pred = db.Column(db.Integer, default=0)
    hypertrophy_pred = db.Column(db.Integer, default=0)
    cholesterol_pred = db.Column(db.Integer, default=0)
    
    # Probability scores
    tachycardia_prob = db.Column(db.Float)
    hypertrophy_prob = db.Column(db.Float)
    cholesterol_prob = db.Column(db.Float)
    
    # Alert status
    notification_sent = db.Column(db.Boolean, default=False)
    device_id = db.Column(db.String(50))  # For IoT devices

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

# ==================== SMS FUNCTIONS ====================

def send_sms_africastalking(phone_number, message):
    """Send SMS using Africa's Talking API"""
    try:
        url = "https://api.sandbox.africastalking.com/version1/messaging"
        headers = {
            "ApiKey": AFRICASTALKING_API_KEY,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        # Format phone number (ensure it starts with country code)
        if not phone_number.startswith('+'):
            phone_number = '+27' + phone_number.lstrip('0')  # Default to South Africa
        
        data = {
            "username": AFRICASTALKING_USERNAME,
            "to": phone_number,
            "message": message,
            "from": SMS_SENDER_ID
        }
        
        response = requests.post(url, headers=headers, data=data)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('SMSMessageData', {}).get('Recipients'):
                return True, "SMS sent successfully"
        return False, f"SMS failed: {response.text}"
    
    except Exception as e:
        return False, f"SMS error: {str(e)}"

def send_health_alert(patient_name, caregiver_phone, health_data):
    """Send health alert SMS to caregiver"""
    message = f"""URGENT: Health Alert for {patient_name}
HR: {health_data['heart_rate']} BPM
BP: {health_data['blood_pressure']}
SpO2: {health_data['spo2']}%
Temp: {health_data['body_temp']}°C

Detected: {', '.join(health_data['conditions'])}
Please check on patient immediately.

Heart Monitor System"""
    
    # Truncate message if too long (160 char limit per SMS)
    if len(message) > 160:
        message = message[:157] + "..."
    
    return send_sms_africastalking(caregiver_phone, message)

def send_welcome_sms(phone_number, name):
    """Send welcome SMS to new user"""
    message = f"Welcome to Heart Monitor System, {name}! Your account has been created successfully. Login to start monitoring."
    return send_sms_africastalking(phone_number, message)

# ==================== HEALTH PREDICTION FUNCTIONS ====================

def calculate_health_metrics(heart_rate, patient_id=None, device_id=None):
    """Calculate health metrics and make ML predictions"""
    
    # Calculate derived metrics
    hrv = max(20, 100 - heart_rate)  # HRV decreases with higher HR
    spo2 = 98 if heart_rate < 100 else 95 if heart_rate < 120 else 92
    systolic = 110 + (heart_rate // 10)
    diastolic = 70 + (heart_rate // 20)
    body_temp = 36.6 + (heart_rate - 70) * 0.01
    
    blood_pressure = f"{systolic}/{diastolic}"
    
    # Make predictions using ML models
    if models_loaded:
        try:
            # Create feature array
            features = np.array([[heart_rate, hrv, spo2, systolic, diastolic, body_temp]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Get predictions
            tachycardia_pred = int(tachycardia_model.predict(features_scaled)[0])
            hypertrophy_pred = int(hypertrophy_model.predict(features_scaled)[0])
            cholesterol_pred = int(cholesterol_model.predict(features_scaled)[0])
            
            # Get probabilities
            tachycardia_prob = float(tachycardia_model.predict_proba(features_scaled)[0][1])
            hypertrophy_prob = float(hypertrophy_model.predict_proba(features_scaled)[0][1])
            cholesterol_prob = float(cholesterol_model.predict_proba(features_scaled)[0][1])
        except:
            # Fallback to rule-based predictions
            tachycardia_pred = 1 if heart_rate > 100 else 0
            hypertrophy_pred = 1 if heart_rate > 90 and heart_rate < 110 else 0
            cholesterol_pred = 0  # Can't predict from HR alone
            tachycardia_prob = min(1.0, (heart_rate - 60) / 100)
            hypertrophy_prob = 0.5 if 90 < heart_rate < 110 else 0.1
            cholesterol_prob = 0.2
    else:
        # Rule-based predictions (demo mode)
        tachycardia_pred = 1 if heart_rate > 100 else 0
        hypertrophy_pred = 1 if 90 < heart_rate < 110 else 0
        cholesterol_pred = 0
        tachycardia_prob = min(1.0, (heart_rate - 60) / 100)
        hypertrophy_prob = 0.5 if 90 < heart_rate < 110 else 0.1
        cholesterol_prob = 0.2
    
    # Save to database if patient_id provided
    reading_id = None
    if patient_id:
        reading = HeartReading(
            patient_id=patient_id,
            heart_rate=heart_rate,
            hrv=hrv,
            spo2=spo2,
            systolic=systolic,
            diastolic=diastolic,
            body_temp=body_temp,
            tachycardia_pred=tachycardia_pred,
            hypertrophy_pred=hypertrophy_pred,
            cholesterol_pred=cholesterol_pred,
            tachycardia_prob=tachycardia_prob,
            hypertrophy_prob=hypertrophy_prob,
            cholesterol_prob=cholesterol_prob,
            device_id=device_id
        )
        db.session.add(reading)
        db.session.commit()
        reading_id = reading.id
        
        # Check for dangerous conditions and send alerts
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
            conditions.append("Low Oxygen Saturation")
        
        is_dangerous = len(conditions) > 0
        
        if is_dangerous and not reading.notification_sent:
            # Get patient details
            patient = Patient.query.get(patient_id)
            
            # Get all caregivers for this patient
            caregivers = db.session.query(User).join(
                PatientCaregiver, User.id == PatientCaregiver.caregiver_id
            ).filter(PatientCaregiver.patient_id == patient_id).all()
            
            # Send SMS alerts to caregivers
            for caregiver in caregivers:
                if caregiver.phone:
                    health_data = {
                        'heart_rate': heart_rate,
                        'blood_pressure': blood_pressure,
                        'spo2': spo2,
                        'body_temp': round(body_temp, 1),
                        'conditions': conditions
                    }
                    success, msg = send_health_alert(patient.name, caregiver.phone, health_data)
                    if success:
                        print(f"Alert sent to {caregiver.phone}")
            
            # Mark notification as sent
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
        "is_dangerous": is_dangerous if patient_id else False,
        "conditions_detected": conditions if patient_id else []
    }

# ==================== AUTHENTICATION DECORATORS ====================

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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==================== ROUTES ====================

@app.route("/")
def index():
    if current_user.is_authenticated:
        # Get all patients for the dropdown
        if current_user.role == 'admin':
            patients = Patient.query.all()
        else:
            # Get patients assigned to this caregiver
            patients = db.session.query(Patient).join(
                PatientCaregiver, Patient.id == PatientCaregiver.patient_id
            ).filter(PatientCaregiver.caregiver_id == current_user.id).all()
        
        # Get recent readings
        patient_id = request.args.get('patient_id', type=int)
        if patient_id:
            readings = HeartReading.query.filter_by(patient_id=patient_id).order_by(HeartReading.id.desc()).limit(20).all()
            selected_patient = Patient.query.get(patient_id)
        else:
            # Get readings for accessible patients
            patient_ids = [p.id for p in patients]
            if patient_ids:
                readings = HeartReading.query.filter(HeartReading.patient_id.in_(patient_ids)).order_by(HeartReading.id.desc()).limit(20).all()
            else:
                readings = []
            selected_patient = None
        
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
        
        # Validation
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
        
        # Create new user
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
        
        # Send welcome SMS
        if phone:
            send_welcome_sms(phone, full_name)
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ==================== PATIENT MANAGEMENT ====================

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
    
    # Convert readings to serializable dictionaries for JSON
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
    
    # Check if already assigned
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

# ==================== HEALTH READINGS ====================

@app.route("/add_reading", methods=["POST"])
@login_required
def add_reading():
    patient_id = request.form.get("patient_id", type=int)
    heart_rate = request.form.get("heart_rate", type=int)
    
    if not patient_id or not heart_rate:
        flash('Please provide both patient ID and heart rate.', 'error')
        return redirect(url_for('index'))
    
    # Check permission
    if current_user.role != 'admin':
        assignment = PatientCaregiver.query.filter_by(
            patient_id=patient_id, caregiver_id=current_user.id
        ).first()
        if not assignment:
            flash('You do not have access to this patient.', 'error')
            return redirect(url_for('index'))
    
    # Calculate predictions
    result = calculate_health_metrics(heart_rate, patient_id)
    
    if result['is_dangerous']:
        flash(f'⚠️ DANGEROUS CONDITION DETECTED: {", ".join(result["conditions_detected"])}. Caregivers have been alerted!', 'warning')
    else:
        flash('Reading added successfully. No dangerous conditions detected.', 'success')
    
    return redirect(url_for('index', patient_id=patient_id))

# @app.route("/api/reading", methods=["POST"])
# def api_add_reading():
#     """API endpoint for IoT devices"""
#     data = request.json
    
#     device_id = data.get('device_id')
#     api_key = data.get('api_key')
#     heart_rate = data.get('heart_rate')
    
#     # Validate device
#     device = IoTDevice.query.filter_by(device_id=device_id, api_key=api_key).first()
#     if not device:
#         return jsonify({"error": "Invalid device credentials"}), 401
    
#     if device.status != 'active':
#         return jsonify({"error": "Device is not active"}), 403
    
#     if not heart_rate or heart_rate < 30 or heart_rate > 220:
#         return jsonify({"error": "Invalid heart rate"}), 400
    
#     # Update last seen
#     device.last_seen = datetime.utcnow()
#     db.session.commit()
    
#     # Calculate predictions
#     patient_id = device.patient_id
#     result = calculate_health_metrics(heart_rate, patient_id, device_id)
    
#     return jsonify({
#         "success": True,
#         "reading_id": result.get('reading_id'),
#         "predictions": {
#             "tachycardia": result['tachycardia_prediction'],
#             "hypertrophy": result['hypertrophy_prediction'],
#             "cholesterol": result['cholesterol_prediction']
#         },
#         "probabilities": {
#             "tachycardia": result['tachycardia_probability'],
#             "hypertrophy": result['hypertrophy_probability'],
#             "cholesterol": result['cholesterol_probability']
#         },
#         "is_dangerous": result['is_dangerous']
#     })

# # ==================== USER MANAGEMENT (Admin) ====================
@app.route("/api/reading", methods=["POST"])
def api_add_reading():
    """API endpoint for IoT devices - Accepts ECG/Heart Rate data"""
    try:
        data = request.json
        print(f"Received data: {data}")
        
        device_id = data.get('device_id')
        api_key = data.get('api_key')
        heart_rate = data.get('heart_rate')
        ecg_value = data.get('ecg_value')
        leads_off = data.get('leads_off', False)
        
        # Validate device
        device = IoTDevice.query.filter_by(device_id=device_id, api_key=api_key).first()
        if not device:
            return jsonify({"error": "Invalid device credentials"}), 401
        
        if device.status != 'active':
            return jsonify({"error": "Device is not active"}), 403
        
        # If ECG value is provided, convert to approximate heart rate
        if ecg_value and not heart_rate:
            # Simple conversion: ECG value typically 0-1024
            # Normalize to heart rate range 40-180 BPM
            heart_rate = 40 + (ecg_value * 140 / 1024)
            heart_rate = int(heart_rate)
        
        if not heart_rate or heart_rate < 30 or heart_rate > 220:
            return jsonify({"error": "Invalid heart rate"}), 400
        
        # Update last seen
        device.last_seen = datetime.utcnow()
        db.session.commit()
        
        # Calculate predictions
        patient_id = device.patient_id
        result = calculate_health_metrics(heart_rate, patient_id, device_id)
        
        response_data = {
            "success": True,
            "reading_id": result.get('reading_id'),
            "heart_rate": heart_rate,
            "predictions": {
                "tachycardia": result['tachycardia_prediction'],
                "hypertrophy": result['hypertrophy_prediction'],
                "cholesterol": result['cholesterol_prediction']
            },
            "probabilities": {
                "tachycardia": result['tachycardia_probability'],
                "hypertrophy": result['hypertrophy_probability'],
                "cholesterol": result['cholesterol_probability']
            },
            "is_dangerous": result['is_dangerous'],
            "message": "Reading processed successfully"
        }
        
        if result['is_dangerous']:
            response_data['alert'] = "Dangerous condition detected! Caregivers notified."
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error in api_add_reading: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ecg", methods=["POST"])
def api_ecg_reading():
    """Specialized endpoint for ECG data"""
    try:
        data = request.json
        print(f"Received ECG data: {data}")
        
        device_id = data.get('device_id')
        api_key = data.get('api_key')
        ecg_value = data.get('ecg_value')
        leads_off = data.get('leads_off', False)
        
        # Validate device
        device = IoTDevice.query.filter_by(device_id=device_id, api_key=api_key).first()
        if not device:
            return jsonify({"error": "Invalid device credentials"}), 401
        
        if device.status != 'active':
            return jsonify({"error": "Device is not active"}), 403
        
        if leads_off:
            return jsonify({"error": "Leads are disconnected", "status": "error"}), 400
        
        # Convert ECG value to heart rate
        # ECG value typically ranges 0-1024, with 512 being baseline
        # Heart rate calculation based on peak detection would be better
        # For now, use a simple conversion
        if ecg_value:
            # Normalize ECG value (0-1024) to approximate heart rate (40-180)
            heart_rate = 40 + (abs(ecg_value - 512) * 140 / 512)
            heart_rate = int(heart_rate)
            heart_rate = max(40, min(180, heart_rate))  # Clamp to reasonable range
        else:
            heart_rate = 72  # Default if no value
        
        # Update last seen
        device.last_seen = datetime.utcnow()
        db.session.commit()
        
        # Calculate predictions
        patient_id = device.patient_id
        result = calculate_health_metrics(heart_rate, patient_id, device_id)
        
        return jsonify({
            "success": True,
            "heart_rate": heart_rate,
            "ecg_value": ecg_value,
            "predictions": {
                "tachycardia": result['tachycardia_prediction'],
                "hypertrophy": result['hypertrophy_prediction'],
                "cholesterol": result['cholesterol_prediction']
            },
            "is_dangerous": result['is_dangerous']
        }), 200
        
    except Exception as e:
        print(f"Error in api_ecg: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/device/status", methods=["GET"])
def device_status():
    """Check device status"""
    device_id = request.args.get('device_id')
    api_key = request.args.get('api_key')
    
    device = IoTDevice.query.filter_by(device_id=device_id, api_key=api_key).first()
    if not device:
        return jsonify({"status": "invalid", "message": "Invalid credentials"}), 401
    
    return jsonify({
        "status": device.status,
        "device_name": device.device_name,
        "device_type": device.device_type,
        "last_seen": device.last_seen.isoformat() if device.last_seen else None,
        "patient_assigned": device.patient_id is not None
    }), 200
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

# ==================== IOT DEVICE MANAGEMENT ====================

import secrets

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
        # Show devices assigned to patients this caregiver has access to
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
        
        flash(f'Device {device_name} registered successfully! Device ID: {device.device_id}', 'success')
        return redirect(url_for('list_devices'))
    
    return render_template("add_device.html", patients=patients)

@app.route("/devices/<device_id>")
@login_required
def view_device(device_id):
    device = IoTDevice.query.filter_by(device_id=device_id).first_or_404()
    
    # Check permission
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

# ==================== STATISTICS & REPORTS ====================

@app.route("/stats")
@login_required
@role_required('admin', 'doctor')
def view_stats():
    total_patients = Patient.query.count()
    total_readings = HeartReading.query.count()
    total_devices = IoTDevice.query.count()
    total_caregivers = User.query.filter_by(role='caregiver').count()
    
    # Recent alerts (dangerous readings)
    recent_alerts = HeartReading.query.filter(
        HeartReading.notification_sent == True
    ).order_by(HeartReading.id.desc()).limit(20).all()
    
    # Readings by condition
    tachycardia_count = HeartReading.query.filter_by(tachycardia_pred=1).count()
    hypertrophy_count = HeartReading.query.filter_by(hypertrophy_pred=1).count()
    cholesterol_count = HeartReading.query.filter_by(cholesterol_pred=1).count()
    
    return render_template("stats.html",
                         total_patients=total_patients,
                         total_readings=total_readings,
                         total_devices=total_devices,
                         total_caregivers=total_caregivers,
                         recent_alerts=recent_alerts,
                         tachycardia_count=tachycardia_count,
                         hypertrophy_count=hypertrophy_count,
                         cholesterol_count=cholesterol_count)

# ==================== INITIALIZATION ====================
# ==================== CAREGIVER MANAGEMENT ROUTES ====================

@app.route("/caregivers")
@login_required
def list_caregivers():
    """List all caregivers"""
    if current_user.role == 'admin':
        caregivers = User.query.filter_by(role='caregiver').all()
    elif current_user.role == 'doctor':
        caregivers = User.query.filter_by(role='caregiver').all()
    else:
        # Caregivers can only see themselves
        caregivers = [current_user]
    
    # Add patient count for each caregiver
    for caregiver in caregivers:
        patient_count = db.session.query(PatientCaregiver).filter_by(caregiver_id=caregiver.id).count()
        caregiver.patient_count = patient_count
    
    return render_template("caregivers.html", caregivers=caregivers)

@app.route("/caregivers/add", methods=["GET", "POST"])
@login_required
@role_required('admin', 'doctor')
def add_caregiver():
    """Add a new caregiver"""
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        
        # Validate input
        if not name or not email:
            flash('Name and email are required.', 'error')
            return redirect(url_for('add_caregiver'))
        
        # Check if email already exists
        existing = User.query.filter_by(email=email).first()
        if existing:
            flash('A user with this email already exists.', 'error')
            return redirect(url_for('add_caregiver'))
        
        # Generate username from email
        username = email.split('@')[0]
        # Make username unique
        base_username = username
        counter = 1
        while User.query.filter_by(username=username).first():
            username = f"{base_username}{counter}"
            counter += 1
        
        # Create password (send to email in production)
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits
        password = ''.join(secrets.choice(alphabet) for i in range(10))
        
        # Create new user as caregiver
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
        
        # In production, send email with credentials
        flash(f'Caregiver {name} added successfully! Username: {username}, Password: {password}', 'success')
        flash('Please share these credentials with the caregiver securely.', 'info')
        
        return redirect(url_for('list_caregivers'))
    
    return render_template("add_caregiver.html")

@app.route("/caregivers/<int:caregiver_id>/edit", methods=["GET", "POST"])
@login_required
@role_required('admin', 'doctor')
def edit_caregiver(caregiver_id):
    """Edit a caregiver"""
    caregiver = User.query.get_or_404(caregiver_id)
    
    # Check if this is actually a caregiver
    if caregiver.role != 'caregiver':
        flash('This user is not a caregiver.', 'error')
        return redirect(url_for('list_caregivers'))
    
    if request.method == "POST":
        caregiver.full_name = request.form.get("name")
        caregiver.email = request.form.get("email")
        caregiver.phone = request.form.get("phone")
        
        # Update password if provided
        new_password = request.form.get("new_password")
        if new_password and len(new_password) >= 6:
            caregiver.set_password(new_password)
            flash('Password updated successfully.', 'success')
        
        db.session.commit()
        flash('Caregiver updated successfully!', 'success')
        return redirect(url_for('list_caregivers'))
    
    return render_template("edit_caregiver.html", caregiver=caregiver)

@app.route("/caregivers/<int:caregiver_id>/delete", methods=["POST"])
@login_required
@role_required('admin')
def delete_caregiver(caregiver_id):
    """Delete a caregiver"""
    caregiver = User.query.get_or_404(caregiver_id)
    
    if caregiver.role != 'caregiver':
        flash('This user is not a caregiver.', 'error')
        return redirect(url_for('list_caregivers'))
    
    # Check if caregiver has assigned patients
    assignments = PatientCaregiver.query.filter_by(caregiver_id=caregiver_id).count()
    if assignments > 0:
        flash(f'Cannot delete caregiver with {assignments} assigned patient(s). Remove assignments first.', 'error')
        return redirect(url_for('list_caregivers'))
    
    db.session.delete(caregiver)
    db.session.commit()
    
    flash('Caregiver deleted successfully.', 'success')
    return redirect(url_for('list_caregivers'))

@app.route("/caregivers/<int:caregiver_id>/assignments")
@login_required
def caregiver_assignments(caregiver_id):
    """View patients assigned to a caregiver"""
    caregiver = User.query.get_or_404(caregiver_id)
    
    if caregiver.role != 'caregiver':
        flash('This user is not a caregiver.', 'error')
        return redirect(url_for('list_caregivers'))
    
    # Get all patients assigned to this caregiver
    patients = db.session.query(Patient).join(
        PatientCaregiver, Patient.id == PatientCaregiver.patient_id
    ).filter(PatientCaregiver.caregiver_id == caregiver_id).all()
    
    return render_template("caregiver_assignments.html", caregiver=caregiver, patients=patients)
def init_db():
    """Initialize database and create default admin user"""
    db.create_all()
    
    # Create default admin user if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@heartmonitor.com',
            phone='+1234567890',
            full_name='System Administrator',
            role='admin'
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("Default admin user created: admin / admin123")
@app.template_filter('datetime')
def format_datetime(value, format='%Y-%m-%d %H:%M'):
    """Format a datetime object."""
    if value is None:
        return ''
    if isinstance(value, str):
        return value[:16]
    return value.strftime(format)
if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=False, host='0.0.0.0', port=5000)