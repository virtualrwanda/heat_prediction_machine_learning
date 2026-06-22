from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import sqlite3
import requests
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages

# Database setup
def init_db():
    conn = sqlite3.connect('iot_geolocation.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS device_locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_serial TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        city TEXT,
        region TEXT,
        country TEXT,
        ip_address TEXT,
        timestamp TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

# Helper function to get database connection
def get_db_connection():
    conn = sqlite3.connect('iot_geolocation.db')
    conn.row_factory = sqlite3.Row
    return conn

# Helper function to get geolocation data
def get_geolocation(ip_address=None):
    try:
        if not ip_address:
            # Get the public IP address
            ip_response = requests.get('https://api.ipify.org?format=json')
            ip_data = ip_response.json()
            ip_address = ip_data['ip']
        
        # Use the IP address to get geolocation data
        geo_response = requests.get(f'https://ipapi.co/{ip_address}/json/')
        geo_data = geo_response.json()
        
        # Format latitude and longitude to 6 decimal places
        latitude = float(f"{geo_data.get('latitude', 0):.6f}")
        longitude = float(f"{geo_data.get('longitude', 0):.6f}")
        
        return {
            'ip': geo_data.get('ip', 'N/A'),
            'latitude': latitude,
            'longitude': longitude,
            'city': geo_data.get('city', 'N/A'),
            'region': geo_data.get('region', 'N/A'),
            'country': geo_data.get('country_name', 'N/A')
        }
    except Exception as e:
        print(f"Error getting geolocation data: {e}")
        return None

# Initialize database before first request
@app.before_first_request
def before_first_request():
    init_db()

# Web Routes
@app.route('/')
def index():
    # Get all unique device serials
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT DISTINCT device_serial FROM device_locations
    ''')
    devices = [row['device_serial'] for row in cursor.fetchall()]
    
    # Get total number of records
    cursor.execute('SELECT COUNT(*) as count FROM device_locations')
    total_records = cursor.fetchone()['count']
    
    # Get latest 5 location records
    cursor.execute('''
    SELECT * FROM device_locations
    ORDER BY timestamp DESC
    LIMIT 5
    ''')
    latest_locations = cursor.fetchall()
    conn.close()
    
    return render_template('index.html', 
                          devices=devices, 
                          total_records=total_records,
                          latest_locations=latest_locations)

@app.route('/device/<device_serial>')
def device_detail(device_serial):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get device location history
    cursor.execute('''
    SELECT * FROM device_locations
    WHERE device_serial = ?
    ORDER BY timestamp DESC
    ''', (device_serial,))
    locations = cursor.fetchall()
    
    # Get latest location for map
    cursor.execute('''
    SELECT * FROM device_locations
    WHERE device_serial = ?
    ORDER BY timestamp DESC
    LIMIT 1
    ''', (device_serial,))
    latest = cursor.fetchone()
    
    conn.close()
    
    return render_template('device_detail.html', 
                          device_serial=device_serial, 
                          locations=locations,
                          latest=latest)

@app.route('/map')
def map_view():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all device locations for the map
    cursor.execute('''
    SELECT device_serial, latitude, longitude, city, timestamp
    FROM device_locations
    ORDER BY timestamp DESC
    ''')
    all_locations = cursor.fetchall()
    
    # Group by device (get latest for each)
    devices = {}
    for loc in all_locations:
        if loc['device_serial'] not in devices:
            devices[loc['device_serial']] = {
                'device_serial': loc['device_serial'],
                'latitude': loc['latitude'],
                'longitude': loc['longitude'],
                'city': loc['city'],
                'timestamp': loc['timestamp']
            }
    
    conn.close()
    
    return render_template('map.html', devices=list(devices.values()))

@app.route('/add-location', methods=['GET', 'POST'])
def add_location():
    if request.method == 'POST':
        device_serial = request.form.get('device_serial')
        
        if not device_serial:
            flash('Device serial number is required', 'error')
            return redirect(url_for('add_location'))
        
        # Get geolocation data
        geo_data = get_geolocation()
        
        if not geo_data:
            flash('Failed to get geolocation data', 'error')
            return redirect(url_for('add_location'))
        
        # Store in database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO device_locations 
        (device_serial, latitude, longitude, city, region, country, ip_address, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_serial,
            geo_data['latitude'],
            geo_data['longitude'],
            geo_data['city'],
            geo_data['region'],
            geo_data['country'],
            geo_data['ip'],
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
        
        flash(f'Location data for device {device_serial} stored successfully', 'success')
        return redirect(url_for('device_detail', device_serial=device_serial))
    
    return render_template('add_location.html')

# API Routes
@app.route('/api/location', methods=['POST'])
def store_location():
    data = request.get_json()
    
    # Check if device_serial is provided
    if not data or 'device_serial' not in data:
        return jsonify({'error': 'Device serial number is required'}), 400
    
    device_serial = data['device_serial']
    
    # Get IP address from request or use provided one
    ip_address = data.get('ip_address', request.remote_addr)
    
    # Get geolocation data
    geo_data = get_geolocation(ip_address)
    
    if not geo_data:
        return jsonify({'error': 'Failed to get geolocation data'}), 500
    
    # Store in database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO device_locations 
    (device_serial, latitude, longitude, city, region, country, ip_address, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        device_serial,
        geo_data['latitude'],
        geo_data['longitude'],
        geo_data['city'],
        geo_data['region'],
        geo_data['country'],
        geo_data['ip'],
        datetime.now().isoformat()
    ))
    conn.commit()
    location_id = cursor.lastrowid
    conn.close()
    
    return jsonify({
        'success': True,
        'location_id': location_id,
        'message': f'Location data for device {device_serial} stored successfully',
        'data': geo_data
    })

@app.route('/api/location/<device_serial>', methods=['GET'])
def get_location_history(device_serial):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT * FROM device_locations
    WHERE device_serial = ?
    ORDER BY timestamp DESC
    ''', (device_serial,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert rows to list of dictionaries
    history = []
    for row in rows:
        history.append(dict(row))
    
    return jsonify({
        'device_serial': device_serial,
        'location_count': len(history),
        'locations': history
    })

@app.route('/api/location/<device_serial>/latest', methods=['GET'])
def get_latest_location(device_serial):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT * FROM device_locations
    WHERE device_serial = ?
    ORDER BY timestamp DESC
    LIMIT 1
    ''', (device_serial,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return jsonify({'error': f'No location data found for device {device_serial}'}), 404
    
    return jsonify(dict(row))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
