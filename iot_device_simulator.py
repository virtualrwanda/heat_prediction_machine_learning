#!/usr/bin/env python3
"""
IoT Device Simulator for Heart Monitoring System

This script simulates an IoT heart monitoring device sending data to the server.
It can be used for testing the system without actual hardware.

Usage:
  python iot_device_simulator.py --device_id=DEV-12345678 --api_key=your_api_key --server=http://localhost:5000
"""

import argparse
import requests
import time
import random
import json
from datetime import datetime

def simulate_heart_rate(base_rate=70, variation=10):
    """Simulate a heart rate with some random variation"""
    return max(40, min(200, base_rate + random.randint(-variation, variation)))

def send_reading(device_id, api_key, heart_rate, server_url):
    """Send a reading to the server"""
    endpoint = f"{server_url}/api/iot/reading"
    
    payload = {
        "device_id": device_id,
        "api_key": api_key,
        "heart_rate": heart_rate
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Reading sent successfully: HR={heart_rate} BPM")
            
            # Check if any dangerous conditions were detected
            if data.get('data', {}).get('is_dangerous', False):
                print("⚠️ WARNING: Dangerous condition detected! Caregivers notified.")
                
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Connection error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='IoT Heart Monitor Simulator')
    parser.add_argument('--device_id', required=True, help='Device ID')
    parser.add_argument('--api_key', required=True, help='API Key')
    parser.add_argument('--server', default='http://localhost:5000', help='Server URL')
    parser.add_argument('--interval', type=int, default=30, help='Interval between readings in seconds')
    parser.add_argument('--base_rate', type=int, default=70, help='Base heart rate')
    parser.add_argument('--variation', type=int, default=10, help='Heart rate variation')
    parser.add_argument('--simulate_emergency', action='store_true', help='Simulate emergency condition')
    
    args = parser.parse_args()
    
    print(f"IoT Heart Monitor Simulator")
    print(f"Device ID: {args.device_id}")
    print(f"Server: {args.server}")
    print(f"Sending readings every {args.interval} seconds...")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        while True:
            if args.simulate_emergency and random.random() < 0.2:  # 20% chance of emergency
                # Simulate dangerous heart rate
                heart_rate = random.randint(120, 180)
                print("⚠️ Simulating emergency condition!")
            else:
                heart_rate = simulate_heart_rate(args.base_rate, args.variation)
                
            send_reading(args.device_id, args.api_key, heart_rate, args.server)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nSimulator stopped")

if __name__ == "__main__":
    main()

