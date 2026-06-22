from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import numpy as np
import joblib
import sqlite3
from datetime import datetime, timedelta
import os
import json
from flask_mail import Mail, Message
import uuid
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import requests

# Load the models
tachycardia_model = joblib.load('model/tachycardia_model.joblib')
hypertrophy_model = joblib.load('model/hypertrophy_model.joblib')
cholesterol_model = joblib.load('model/cholesterol_model.joblib')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'heart-monitor-clinical-secret-2024')

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'heart-monitor@example.com')

mail = Mail(app)

DB_PATH = 'heart_monitor.db'

# ── FDI Biz SMS ───────────────────────────────────────────────────────────────
SMS_USERNAME  = os.getenv("FDI_USERNAME",  "5BBEC534-9DC0-4DC9-A2D0-D33F5537AEC4")
SMS_PASSWORD  = os.getenv("FDI_PASSWORD",  "6335681A-074B-4FCF-81C1-FBFEB8C45579")
SMS_BASE_URL  = os.getenv("FDI_BASE_URL",  "https://messaging.fdibiz.com/api/v1/")
SMS_SENDER_ID = os.getenv("FDI_SENDER_ID", "FDI")

# ── Database ─────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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
        device_id TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (id),
        FOREIGN KEY (device_id) REFERENCES devices (device_id)
    )
    ''')

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

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS caregivers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        phone TEXT,
        created_at TEXT NOT NULL
    )
    ''')

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

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS devices (
        device_id TEXT PRIMARY KEY,
        device_name TEXT NOT NULL,
        device_type TEXT NOT NULL,
        patient_id INTEGER,
        api_key TEXT NOT NULL,
        last_seen TEXT,
        status TEXT DEFAULT 'active',
        created_at TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'caregiver',
        full_name TEXT NOT NULL,
        phone TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        is_active INTEGER DEFAULT 1
    )
    ''')

    # Migrate existing databases: add phone column if missing
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN phone TEXT DEFAULT ''")
        conn.commit()
    except Exception:
        pass

    # Seed a default admin account if no users exist
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            'INSERT INTO users (username, email, password_hash, role, full_name, created_at) VALUES (?, ?, ?, ?, ?, ?)',
            ('admin', 'admin@hospital.com', generate_password_hash('admin123'),
             'admin', 'System Administrator', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

    conn.commit()
    conn.close()

# ── Auth decorators ───────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def role_required(*roles):
    """Must be stacked BELOW @login_required."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if session.get('user_role') not in roles:
                flash('Access denied — insufficient permissions.', 'error')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated
    return decorator

# ── Context processor — injects current_user into every template ──────────────

@app.context_processor
def inject_user():
    return {
        'current_user': {
            'id':               session.get('user_id'),
            'username':         session.get('username'),
            'role':             session.get('user_role'),
            'full_name':        session.get('full_name'),
            'is_authenticated': 'user_id' in session,
        }
    }

# ── ML prediction helpers ─────────────────────────────────────────────────────

def calculate_health_metrics(heart_rate, patient_id):
    hrv      = max(20, 100 - heart_rate)
    spo2     = 98 if heart_rate < 100 else 95
    systolic  = 110 + (heart_rate // 10)
    diastolic = 70  + (heart_rate // 20)
    body_temp = 36.6 + (heart_rate - 70) * 0.01
    blood_pressure = f"{systolic}/{diastolic}"

    features = np.array([[heart_rate, hrv, spo2, systolic, diastolic, body_temp]])

    tachycardia_pred  = tachycardia_model.predict(features)[0]
    hypertrophy_pred  = hypertrophy_model.predict(features)[0]
    cholesterol_pred  = cholesterol_model.predict(features)[0]

    try:
        tachycardia_prob  = tachycardia_model.predict_proba(features)[0][1]
        hypertrophy_prob  = hypertrophy_model.predict_proba(features)[0][1]
        cholesterol_prob  = cholesterol_model.predict_proba(features)[0][1]
    except Exception:
        tachycardia_prob = hypertrophy_prob = cholesterol_prob = None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn   = sqlite3.connect(DB_PATH)
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

    is_dangerous = (
        int(tachycardia_pred) == 1 or int(hypertrophy_pred) == 1 or int(cholesterol_pred) == 1
        or heart_rate > 120 or systolic > 160 or diastolic > 100 or spo2 < 92
    )

    if is_dangerous:
        cursor.execute('SELECT name FROM patients WHERE id = ?', (patient_id,))
        row = cursor.fetchone()
        patient_name = row[0] if row else 'Unknown'

        conditions = []
        if int(tachycardia_pred) == 1: conditions.append("Tachycardia")
        if int(hypertrophy_pred) == 1: conditions.append("Hypertrophy")
        if int(cholesterol_pred) == 1: conditions.append("High Cholesterol")
        sms_alert = (
            f"CLINICAL ALERT — {patient_name}\n"
            f"HR:{heart_rate}BPM BP:{blood_pressure} SpO2:{spo2}%\n"
            f"Temp:{round(body_temp,1)}C\n"
            f"{', '.join(conditions) if conditions else 'Abnormal vitals'}\n"
            f"— Clinical Dashboard"
        )

        health_data = {
            "heart_rate":       heart_rate,
            "blood_pressure":   blood_pressure,
            "spo2":             spo2,
            "body_temp":        round(body_temp, 1),
            "tachycardia":      bool(tachycardia_pred),
            "hypertrophy":      bool(hypertrophy_pred),
            "high_cholesterol": bool(cholesterol_pred),
        }

        # Notify assigned caregivers (email + SMS)
        cursor.execute('''
        SELECT c.name, c.email, c.phone, pc.relationship
        FROM caregivers c
        JOIN patient_caregivers pc ON c.id = pc.caregiver_id
        WHERE pc.patient_id = ?
        ''', (patient_id,))
        for cg in cursor.fetchall():
            cg_name, cg_email, cg_phone, relationship = cg
            send_danger_notification(cg_email, cg_name, patient_name, relationship, health_data)
            if cg_phone:
                send_sms_fdi(cg_phone, sms_alert)

        # Notify all active doctors via SMS
        cursor.execute('''
        SELECT full_name, phone FROM users
        WHERE role = 'doctor' AND is_active = 1
          AND phone IS NOT NULL AND phone != ''
        ''')
        for dr_name, dr_phone in cursor.fetchall():
            send_sms_fdi(dr_phone, sms_alert)

        cursor.execute('UPDATE heart_readings SET notification_sent = 1 WHERE id = ?', (reading_id,))
        conn.commit()

    conn.close()

    return {
        "timestamp":  timestamp,
        "patient_id": patient_id,
        "Tachycardia Prediction":       int(tachycardia_pred),
        "Hypertrophy Prediction":       int(hypertrophy_pred),
        "High Cholesterol Prediction":  int(cholesterol_pred),
        "Prediction Probabilities": {
            "Tachycardia Probability":      float(tachycardia_prob)  if tachycardia_prob  is not None else None,
            "Hypertrophy Probability":      float(hypertrophy_prob)  if hypertrophy_prob  is not None else None,
            "High Cholesterol Probability": float(cholesterol_prob)  if cholesterol_prob  is not None else None,
        },
        "Calculated Features": {
            "Heart Rate (BPM)":       heart_rate,
            "HRV (ms)":               hrv,
            "SpO2 (%)":               spo2,
            "Blood Pressure":         blood_pressure,
            "Body Temperature (°C)":  round(body_temp, 1),
        },
        "is_dangerous": is_dangerous,
    }

def send_danger_notification(caregiver_email, caregiver_name, patient_name, relationship, health_data):
    subject = f"URGENT: Health Alert for {patient_name}"
    body = f"""
Dear {caregiver_name},

This is an automated alert from the Clinical Heart Monitoring System.

Your {relationship}, {patient_name}, has shown concerning readings:

  Heart Rate:       {health_data['heart_rate']} BPM
  Blood Pressure:   {health_data['blood_pressure']} mmHg
  SpO2:             {health_data['spo2']}%
  Body Temperature: {health_data['body_temp']}°C

Detected Conditions:
  Tachycardia:        {"Yes" if health_data['tachycardia']       else "No"}
  Cardiac Hypertrophy:{"Yes" if health_data['hypertrophy']       else "No"}
  High Cholesterol:   {"Yes" if health_data['high_cholesterol']  else "No"}

Please check on the patient or contact their healthcare provider.

— Clinical Heart Monitor (automated message, do not reply)
"""
    try:
        msg = Message(subject=subject, recipients=[caregiver_email], body=body)
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

def send_sms_fdi(phone, message):
    """Send SMS via FDI Biz API. Returns True on success."""
    if not phone:
        return False
    phone = phone.strip().replace(' ', '').replace('-', '')
    try:
        resp = requests.post(
            SMS_BASE_URL + "sms/send",
            auth=(SMS_USERNAME, SMS_PASSWORD),
            json={"to": phone, "message": message, "sender_id": SMS_SENDER_ID},
            timeout=10,
        )
        success = resp.status_code in (200, 201)
        print(f"[FDI SMS] {'OK' if success else 'FAIL'} → {phone} ({resp.status_code})")
        return success
    except Exception as exc:
        print(f"[FDI SMS] Error → {phone}: {exc}")
        return False

def get_recent_readings(patient_id=None, limit=10):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if patient_id:
        cursor.execute('''
        SELECT r.*, p.name as patient_name
        FROM heart_readings r JOIN patients p ON r.patient_id = p.id
        WHERE r.patient_id = ? ORDER BY r.id DESC LIMIT ?
        ''', (patient_id, limit))
    else:
        cursor.execute('''
        SELECT r.*, p.name as patient_name
        FROM heart_readings r JOIN patients p ON r.patient_id = p.id
        ORDER BY r.id DESC LIMIT ?
        ''', (limit,))

    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        result.append({
            "id":               row['id'],
            "timestamp":        row['timestamp'],
            "patient_id":       row['patient_id'],
            "patient_name":     row['patient_name'],
            "heart_rate":       row['heart_rate'],
            "hrv":              row['hrv'],
            "spo2":             row['spo2'],
            "systolic":         row['systolic'],
            "diastolic":        row['diastolic'],
            "body_temp":        row['body_temp'],
            "blood_pressure":   f"{row['systolic']}/{row['diastolic']}",
            "tachycardia_pred": row['tachycardia_pred'],
            "hypertrophy_pred": row['hypertrophy_pred'],
            "cholesterol_pred": row['cholesterol_pred'],
            "tachycardia_prob": row['tachycardia_prob'],
            "hypertrophy_prob": row['hypertrophy_prob'],
            "cholesterol_prob": row['cholesterol_prob'],
            "notification_sent": row['notification_sent'],
        })
    return result

def get_all_patients():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients ORDER BY name')
    result = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return result

def get_patient(patient_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    patient = dict(row)
    cursor.execute('''
    SELECT c.*, pc.relationship
    FROM caregivers c JOIN patient_caregivers pc ON c.id = pc.caregiver_id
    WHERE pc.patient_id = ?
    ''', (patient_id,))
    patient['caregivers'] = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return patient

def get_all_caregivers():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM caregivers ORDER BY name')
    result = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return result

def get_all_devices():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT d.*, p.name as patient_name
    FROM devices d LEFT JOIN patients p ON d.patient_id = p.id
    ORDER BY d.created_at DESC
    ''')
    result = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return result

def get_device(device_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM devices WHERE device_id = ?', (device_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def verify_device_api_key(device_id, api_key):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT api_key FROM devices WHERE device_id = ?', (device_id,))
    row = cursor.fetchone()
    if row and row[0] == api_key:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('UPDATE devices SET last_seen = ? WHERE device_id = ?', (now, device_id))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def get_patient_devices(patient_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM devices WHERE patient_id = ?', (patient_id,))
    result = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return result

def generate_api_key():
    return uuid.uuid4().hex

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND is_active = 1', (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user['password_hash'], password):
            session.clear()
            session['user_id']   = user['id']
            session['username']  = user['username']
            session['user_role'] = user['role']
            session['full_name'] = user['full_name']
            flash(f"Welcome back, {user['full_name']}!", 'success')
            return redirect(url_for('index'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username  = request.form.get('username',  '').strip()
        email     = request.form.get('email',     '').strip()
        password  = request.form.get('password',  '')
        confirm   = request.form.get('confirm_password', '')
        role      = request.form.get('role', 'caregiver')

        if not all([full_name, username, email, password]):
            flash('All fields are required.', 'error')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('register.html')
        if role not in ('admin', 'doctor', 'caregiver'):
            role = 'caregiver'

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, email, password_hash, role, full_name, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                (username, email, generate_password_hash(password), role, full_name,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()
            flash('Account created successfully. You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'error')
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')

    return render_template('register.html')

# ═══════════════════════════════════════════════════════════════════════════════
# USER MANAGEMENT (admin only)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/users')
@login_required
@role_required('admin')
def list_users():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
    users = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return render_template('users.html', users=users)

@app.route('/users/add', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def add_user():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username  = request.form.get('username',  '').strip()
        email     = request.form.get('email',     '').strip()
        password  = request.form.get('password',  '')
        role      = request.form.get('role', 'caregiver')
        is_active = 1 if request.form.get('is_active') else 0
        phone     = request.form.get('phone', '').strip()

        if not all([full_name, username, email, password]):
            flash('All fields are required.', 'error')
            return render_template('edit_user.html', user=None)
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('edit_user.html', user=None)
        if role not in ('admin', 'doctor', 'caregiver'):
            role = 'caregiver'

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, email, password_hash, role, full_name, phone, created_at, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (username, email, generate_password_hash(password), role, full_name, phone,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), is_active)
            )
            conn.commit()
            conn.close()
            flash('User created successfully.', 'success')
            return redirect(url_for('list_users'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'error')
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')

    return render_template('edit_user.html', user=None)

@app.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def edit_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        flash('User not found.', 'error')
        return redirect(url_for('list_users'))
    user = dict(row)

    if request.method == 'POST':
        full_name    = request.form.get('full_name', '').strip()
        email        = request.form.get('email',     '').strip()
        role         = request.form.get('role', 'caregiver')
        is_active    = 1 if request.form.get('is_active') else 0
        new_password = request.form.get('new_password', '').strip()
        phone        = request.form.get('phone', '').strip()

        if role not in ('admin', 'doctor', 'caregiver'):
            role = 'caregiver'

        try:
            if new_password:
                if len(new_password) < 6:
                    flash('Password must be at least 6 characters.', 'error')
                    conn.close()
                    return render_template('edit_user.html', user=user)
                cursor.execute(
                    'UPDATE users SET full_name=?, email=?, role=?, is_active=?, phone=?, password_hash=? WHERE id=?',
                    (full_name, email, role, is_active, phone, generate_password_hash(new_password), user_id)
                )
            else:
                cursor.execute(
                    'UPDATE users SET full_name=?, email=?, role=?, is_active=?, phone=? WHERE id=?',
                    (full_name, email, role, is_active, phone, user_id)
                )
            conn.commit()
            conn.close()
            flash('User updated successfully.', 'success')
            return redirect(url_for('list_users'))
        except sqlite3.IntegrityError:
            flash('Email already in use by another account.', 'error')
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')

    conn.close()
    return render_template('edit_user.html', user=user)

@app.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
@role_required('admin')
def delete_user(user_id):
    if user_id == session.get('user_id'):
        flash('You cannot delete your own account.', 'error')
        return redirect(url_for('list_users'))
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        flash('User deleted successfully.', 'success')
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
    return redirect(url_for('list_users'))

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD & READINGS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
@login_required
def index():
    patients = get_all_patients()
    patient_id = request.args.get('patient_id', type=int)
    recent_readings = get_recent_readings(patient_id, 10)
    selected_patient = get_patient(patient_id) if patient_id else None
    return render_template("index.html",
                           readings=recent_readings,
                           patients=patients,
                           selected_patient=selected_patient)

@app.route("/api/data", methods=["GET"])
@login_required
def get_data():
    patient_id = request.args.get('patient_id', type=int)
    return jsonify({"readings": get_recent_readings(patient_id, 10)})

@app.route("/api/add_reading", methods=["POST"])
@login_required
def add_reading():
    try:
        data       = request.json
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
@login_required
def submit_reading():
    try:
        heart_rate = int(request.form["heart_rate"])
        patient_id = int(request.form["patient_id"])
        if patient_id <= 0:
            flash("Please select a patient", "error")
            return redirect(url_for('index'))
        result = calculate_health_metrics(heart_rate, patient_id)
        if result["is_dangerous"]:
            flash("Warning: Dangerous condition detected! Caregivers have been notified.", "warning")
        else:
            flash("Reading added successfully.", "success")
        return redirect(url_for('index', patient_id=patient_id))
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for('index'))

# ═══════════════════════════════════════════════════════════════════════════════
# PATIENT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/patients", methods=["GET"])
@login_required
def list_patients():
    return render_template("patients.html", patients=get_all_patients())

@app.route("/patients/add", methods=["GET", "POST"])
@login_required
@role_required('admin', 'doctor')
def add_patient():
    if request.method == "POST":
        try:
            name            = request.form["name"]
            age             = int(request.form["age"])
            gender          = request.form["gender"]
            medical_history = request.form.get("medical_history", "")
            conn   = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO patients (name, age, gender, medical_history, created_at) VALUES (?, ?, ?, ?, ?)',
                (name, age, gender, medical_history, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()
            flash("Patient added successfully.", "success")
            return redirect(url_for('list_patients'))
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
    return render_template("add_patient.html")

@app.route("/patients/<int:patient_id>", methods=["GET"])
@login_required
def view_patient(patient_id):
    patient = get_patient(patient_id)
    if not patient:
        flash("Patient not found.", "error")
        return redirect(url_for('list_patients'))
    readings       = get_recent_readings(patient_id, 20)
    all_caregivers = get_all_caregivers()
    return render_template("view_patient.html",
                           patient=patient,
                           readings=readings,
                           all_caregivers=all_caregivers)

@app.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@login_required
@role_required('admin', 'doctor')
def edit_patient(patient_id):
    patient = get_patient(patient_id)
    if not patient:
        flash("Patient not found.", "error")
        return redirect(url_for('list_patients'))
    if request.method == "POST":
        try:
            name            = request.form["name"]
            age             = int(request.form["age"])
            gender          = request.form["gender"]
            medical_history = request.form.get("medical_history", "")
            conn   = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE patients SET name=?, age=?, gender=?, medical_history=? WHERE id=?',
                (name, age, gender, medical_history, patient_id)
            )
            conn.commit()
            conn.close()
            flash("Patient updated successfully.", "success")
            return redirect(url_for('view_patient', patient_id=patient_id))
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
    return render_template("edit_patient.html", patient=patient)

# ═══════════════════════════════════════════════════════════════════════════════
# CAREGIVER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/caregivers", methods=["GET"])
@login_required
@role_required('admin')
def list_caregivers():
    return render_template("caregivers.html", caregivers=get_all_caregivers())

@app.route("/caregivers/add", methods=["GET", "POST"])
@login_required
@role_required('admin')
def add_caregiver():
    if request.method == "POST":
        try:
            name  = request.form["name"]
            email = request.form["email"]
            phone = request.form.get("phone", "")
            conn   = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO caregivers (name, email, phone, created_at) VALUES (?, ?, ?, ?)',
                (name, email, phone, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()
            flash("Caregiver added successfully.", "success")
            return redirect(url_for('list_caregivers'))
        except sqlite3.IntegrityError:
            flash("A caregiver with this email already exists.", "error")
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
    return render_template("add_caregiver.html")

@app.route("/caregivers/<int:caregiver_id>/edit", methods=["GET", "POST"])
@login_required
@role_required('admin')
def edit_caregiver(caregiver_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM caregivers WHERE id = ?', (caregiver_id,))
    caregiver = dict(cursor.fetchone() or {})
    if not caregiver:
        conn.close()
        flash("Caregiver not found.", "error")
        return redirect(url_for('list_caregivers'))
    if request.method == "POST":
        try:
            name  = request.form["name"]
            email = request.form["email"]
            phone = request.form.get("phone", "")
            cursor.execute(
                'UPDATE caregivers SET name=?, email=?, phone=? WHERE id=?',
                (name, email, phone, caregiver_id)
            )
            conn.commit()
            conn.close()
            flash("Caregiver updated successfully.", "success")
            return redirect(url_for('list_caregivers'))
        except sqlite3.IntegrityError:
            flash("A caregiver with this email already exists.", "error")
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
    conn.close()
    return render_template("edit_caregiver.html", caregiver=caregiver)

@app.route("/patients/<int:patient_id>/add_caregiver", methods=["POST"])
@login_required
@role_required('admin')
def add_patient_caregiver(patient_id):
    try:
        caregiver_id = int(request.form["caregiver_id"])
        relationship = request.form["relationship"]
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM patients WHERE id = ?', (patient_id,))
        if not cursor.fetchone():
            conn.close()
            flash("Patient not found.", "error")
            return redirect(url_for('list_patients'))
        cursor.execute('SELECT id FROM caregivers WHERE id = ?', (caregiver_id,))
        if not cursor.fetchone():
            conn.close()
            flash("Caregiver not found.", "error")
            return redirect(url_for('view_patient', patient_id=patient_id))
        try:
            cursor.execute(
                'INSERT INTO patient_caregivers (patient_id, caregiver_id, relationship, created_at) VALUES (?, ?, ?, ?)',
                (patient_id, caregiver_id, relationship, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            flash("Caregiver associated successfully.", "success")
        except sqlite3.IntegrityError:
            flash("This caregiver is already associated with this patient.", "error")
        conn.close()
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
    return redirect(url_for('view_patient', patient_id=patient_id))

@app.route("/patients/<int:patient_id>/remove_caregiver/<int:caregiver_id>", methods=["POST"])
@login_required
@role_required('admin')
def remove_patient_caregiver(patient_id, caregiver_id):
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM patient_caregivers WHERE patient_id=? AND caregiver_id=?',
                       (patient_id, caregiver_id))
        conn.commit()
        conn.close()
        flash("Caregiver removed from patient.", "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
    return redirect(url_for('view_patient', patient_id=patient_id))

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/devices", methods=["GET"])
@login_required
def list_devices():
    return render_template("devices.html", devices=get_all_devices())

@app.route("/devices/add", methods=["GET", "POST"])
@login_required
@role_required('admin')
def add_device():
    if request.method == "POST":
        try:
            device_name = request.form["device_name"]
            device_type = request.form["device_type"]
            patient_id  = request.form.get("patient_id") or None
            device_id   = f"DEV-{uuid.uuid4().hex[:8].upper()}"
            api_key     = generate_api_key()
            conn   = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO devices (device_id, device_name, device_type, patient_id, api_key, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                (device_id, device_name, device_type, patient_id, api_key,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()
            flash(f"Device added. ID: {device_id}", "success")
            return redirect(url_for('view_device', device_id=device_id))
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
    return render_template("add_device.html", patients=get_all_patients())

@app.route("/devices/<device_id>", methods=["GET"])
@login_required
def view_device(device_id):
    device = get_device(device_id)
    if not device:
        flash("Device not found.", "error")
        return redirect(url_for('list_devices'))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT r.*, p.name as patient_name
    FROM heart_readings r JOIN patients p ON r.patient_id = p.id
    WHERE r.device_id = ? ORDER BY r.id DESC LIMIT 20
    ''', (device_id,))
    readings = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return render_template("view_device.html",
                           device=device,
                           readings=readings,
                           patients=get_all_patients())

@app.route("/devices/<device_id>/edit", methods=["POST"])
@login_required
@role_required('admin')
def edit_device(device_id):
    if not get_device(device_id):
        flash("Device not found.", "error")
        return redirect(url_for('list_devices'))
    try:
        device_name = request.form["device_name"]
        device_type = request.form["device_type"]
        patient_id  = request.form.get("patient_id") or None
        status      = request.form["status"]
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE devices SET device_name=?, device_type=?, patient_id=?, status=? WHERE device_id=?',
            (device_name, device_type, patient_id, status, device_id)
        )
        conn.commit()
        conn.close()
        flash("Device updated successfully.", "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
    return redirect(url_for('view_device', device_id=device_id))

@app.route("/devices/<device_id>/regenerate_key", methods=["POST"])
@login_required
@role_required('admin')
def regenerate_device_key(device_id):
    if not get_device(device_id):
        flash("Device not found.", "error")
        return redirect(url_for('list_devices'))
    try:
        new_key = generate_api_key()
        conn    = sqlite3.connect(DB_PATH)
        cursor  = conn.cursor()
        cursor.execute('UPDATE devices SET api_key=? WHERE device_id=?', (new_key, device_id))
        conn.commit()
        conn.close()
        flash("API key regenerated.", "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
    return redirect(url_for('view_device', device_id=device_id))

@app.route("/patients/<int:patient_id>/devices", methods=["GET"])
@login_required
def patient_devices(patient_id):
    patient = get_patient(patient_id)
    if not patient:
        flash("Patient not found.", "error")
        return redirect(url_for('list_patients'))
    return render_template("patient_devices.html",
                           patient=patient,
                           devices=get_patient_devices(patient_id))

# ═══════════════════════════════════════════════════════════════════════════════
# IoT DEVICE API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/iot/reading", methods=["POST"])
def iot_add_reading():
    try:
        data = request.json
        for field in ("device_id", "api_key", "heart_rate"):
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        device_id  = data["device_id"]
        api_key    = data["api_key"]
        heart_rate = int(data["heart_rate"])

        if not verify_device_api_key(device_id, api_key):
            return jsonify({"error": "Invalid device ID or API key"}), 401

        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT patient_id, status FROM devices WHERE device_id = ?', (device_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Device not found"}), 404
        patient_id, status = row
        if status != 'active':
            return jsonify({"error": "Device is not active"}), 403
        if not patient_id:
            return jsonify({"error": "Device not assigned to a patient"}), 400

        result = calculate_health_metrics(heart_rate, patient_id)

        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE heart_readings SET device_id=? WHERE id=(SELECT MAX(id) FROM heart_readings WHERE patient_id=?)',
            (device_id, patient_id)
        )
        conn.commit()
        conn.close()

        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────

init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
