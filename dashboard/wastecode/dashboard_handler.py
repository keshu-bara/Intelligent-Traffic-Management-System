import json
import sqlite3
import threading
import queue
import requests
import time
from datetime import datetime
from collections import deque
import numpy as np

class DashboardDataSender:
    def __init__(self, db_name="traffic_simulation"):
        self.db_name = f"{db_name}.db"
        self.data_queue = queue.Queue(maxsize=1000)
        self.running = True
        
        # Initialize database
        self.init_database()
        
        # Start background thread for data processing
        self.data_thread = threading.Thread(target=self._process_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Real-time data storage
        self.real_time_data = {
            'junctions': {},
            'network': {},
            'coordination': {},
            'performance': {}
        }
        
    def init_database(self):
        """Initialize SQLite database for storing simulation data"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Junction data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS junction_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                junction_id TEXT,
                step INTEGER,
                vehicles_north INTEGER,
                vehicles_south INTEGER,
                vehicles_east INTEGER,
                vehicles_west INTEGER,
                congestion_north REAL,
                congestion_south REAL,
                congestion_east REAL,
                congestion_west REAL,
                speed_north REAL,
                speed_south REAL,
                speed_east REAL,
                speed_west REAL,
                current_action INTEGER,
                phase_duration INTEGER,
                reward REAL,
                epsilon REAL
            )
        ''')
        
        # Network performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                step INTEGER,
                total_vehicles INTEGER,
                total_congestion REAL,
                avg_speed REAL,
                coordination_efficiency REAL,
                active_coordinations INTEGER,
                network_throughput REAL
            )
        ''')
        
        # Coordination events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coordination_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sender_junction TEXT,
                receiver_junction TEXT,
                event_type TEXT,
                priority REAL,
                success BOOLEAN,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def send_junction_data(self, junction_id, step, state, action, duration, reward, epsilon):
        """Queue junction data for processing"""
        data = {
            'type': 'junction',
            'junction_id': junction_id,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action,
            'duration': duration,
            'reward': reward,
            'epsilon': epsilon
        }
        
        try:
            self.data_queue.put_nowait(data)
            # Update real-time data
            self.real_time_data['junctions'][junction_id] = data
        except queue.Full:
            print("⚠️ Dashboard data queue is full!")
    
    def send_network_performance(self, step, metrics):
        """Queue network performance data"""
        data = {
            'type': 'network',
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        try:
            self.data_queue.put_nowait(data)
            # Update real-time data
            self.real_time_data['network'] = data
        except queue.Full:
            print("⚠️ Network data queue is full!")
    
    def send_coordination_event(self, sender, receiver, event_type, priority, success, details):
        """Queue coordination event data"""
        data = {
            'type': 'coordination',
            'timestamp': datetime.now().isoformat(),
            'sender': sender,
            'receiver': receiver,
            'event_type': event_type,
            'priority': priority,
            'success': success,
            'details': details
        }
        
        try:
            self.data_queue.put_nowait(data)
            # Update real-time data
            if 'events' not in self.real_time_data['coordination']:
                self.real_time_data['coordination']['events'] = deque(maxlen=50)
            self.real_time_data['coordination']['events'].append(data)
        except queue.Full:
            print("⚠️ Coordination data queue is full!")
    
    def _process_data(self):
        """Background thread to process queued data"""
        while self.running:
            try:
                # Get data from queue (with timeout)
                data = self.data_queue.get(timeout=1.0)
                
                if data['type'] == 'junction':
                    self._store_junction_data(data)
                elif data['type'] == 'network':
                    self._store_network_data(data)
                elif data['type'] == 'coordination':
                    self._store_coordination_data(data)
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing dashboard data: {e}")
    
    def _store_junction_data(self, data):
        """Store junction data in database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        state = data['state']
        cursor.execute('''
            INSERT INTO junction_data (
                junction_id, step, 
                vehicles_north, vehicles_south, vehicles_east, vehicles_west,
                congestion_north, congestion_south, congestion_east, congestion_west,
                speed_north, speed_south, speed_east, speed_west,
                current_action, phase_duration, reward, epsilon
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['junction_id'], data['step'],
            state[0], state[3], state[6], state[9],
            state[1], state[4], state[7], state[10],
            state[2], state[5], state[8], state[11],
            data['action'], data['duration'], data['reward'], data['epsilon']
        ))
        
        conn.commit()
        conn.close()
    
    def _store_network_data(self, data):
        """Store network performance data"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        metrics = data['metrics']
        cursor.execute('''
            INSERT INTO network_performance (
                step, total_vehicles, total_congestion, avg_speed,
                coordination_efficiency, active_coordinations, network_throughput
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['step'], metrics['total_vehicles'], metrics['total_congestion'],
            metrics['avg_speed'], metrics['coordination_efficiency'],
            metrics['active_coordinations'], metrics.get('network_throughput', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def _store_coordination_data(self, data):
        """Store coordination event data"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO coordination_events (
                sender_junction, receiver_junction, event_type,
                priority, success, details
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['sender'], data['receiver'], data['event_type'],
            data['priority'], data['success'], json.dumps(data['details'])
        ))
        
        conn.commit()
        conn.close()
    
    def get_real_time_data(self):
        """Get current real-time data for dashboard"""
        return self.real_time_data.copy()
    
    def get_historical_data(self, table_name, limit=100):
        """Get historical data from database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute(f'''
            SELECT * FROM {table_name} 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.data_thread.is_alive():
            self.data_thread.join(timeout=5.0)

class ESP32DataSender:
    def __init__(self, esp32_ip="192.168.1.100", port=80):
        self.esp32_url = f"http://{esp32_ip}:{port}/data"
        self.enabled = False  # Disable by default, enable when ESP32 is available
        self.retry_count = 3
        
    def send_junction_status(self, junction_id, action, vehicles, congestion):
        """Send junction status to ESP32"""
        if not self.enabled:
            return
        
        data = {
            'junction': junction_id,
            'action': action,
            'vehicles': vehicles,
            'congestion': congestion,
            'timestamp': time.time()
        }
        
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.esp32_url, 
                    json=data, 
                    timeout=2.0
                )
                if response.status_code == 200:
                    return True
            except requests.RequestException as e:
                if attempt == self.retry_count - 1:
                    print(f"Failed to send data to ESP32: {e}")
        
        return False
    
    def enable(self):
        """Enable ESP32 communication"""
        self.enabled = True
        print("📡 ESP32 communication enabled")
    
    def disable(self):
        """Disable ESP32 communication"""
        self.enabled = False
        print("📡 ESP32 communication disabled")