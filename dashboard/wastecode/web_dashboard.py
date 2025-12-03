from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import sqlite3
from datetime import datetime, timedelta
import threading
import time
import os

# Get the directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)
app.config['SECRET_KEY'] = 'traffic_simulation_dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")

class TrafficDashboard:
    def __init__(self, db_name="traffic_simulation.db"):
        self.db_name = db_name
        self.dashboard_data = None
        self.running = False
        
    def set_data_handler(self, dashboard_data_sender):
        """Connect to the dashboard data sender"""
        self.dashboard_data = dashboard_data_sender
        
    def start_real_time_updates(self):
        """Start real-time data broadcasting"""
        self.running = True
        update_thread = threading.Thread(target=self._broadcast_updates)
        update_thread.daemon = True
        update_thread.start()
        
    def _broadcast_updates(self):
        """Broadcast real-time updates to connected clients"""
        while self.running:
            if self.dashboard_data:
                real_time_data = self.dashboard_data.get_real_time_data()
                socketio.emit('real_time_update', real_time_data)
            time.sleep(2)  # Update every 2 seconds

# Global dashboard instance
dashboard = TrafficDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/junction_data/<junction_id>')
def get_junction_data(junction_id):
    """Get historical data for specific junction"""
    if not dashboard.dashboard_data:
        return jsonify({'error': 'No data source available'})
    
    try:
        conn = sqlite3.connect(dashboard.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM junction_data 
            WHERE junction_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''', (junction_id,))
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        data = [dict(zip(columns, row)) for row in rows]
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/network_performance')
def get_network_performance():
    """Get network performance data"""
    if not dashboard.dashboard_data:
        return jsonify({'error': 'No data source available'})
    
    try:
        data = dashboard.dashboard_data.get_historical_data('network_performance', 100)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/coordination_events')
def get_coordination_events():
    """Get coordination events"""
    if not dashboard.dashboard_data:
        return jsonify({'error': 'No data source available'})
    
    try:
        data = dashboard.dashboard_data.get_historical_data('coordination_events', 50)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/real_time_status')
def get_real_time_status():
    """Get current real-time status"""
    if not dashboard.dashboard_data:
        return jsonify({'error': 'No data source available'})
    
    return jsonify(dashboard.dashboard_data.get_real_time_data())

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('message', {'data': 'Connected to Traffic Dashboard'})
    print('🌐 Client connected to dashboard')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('🌐 Client disconnected from dashboard')

@socketio.on('request_junction_update')
def handle_junction_update_request(data):
    """Handle request for specific junction update"""
    junction_id = data.get('junction_id')
    if junction_id and dashboard.dashboard_data:
        real_time_data = dashboard.dashboard_data.get_real_time_data()
        junction_data = real_time_data.get('junctions', {}).get(junction_id, {})
        emit('junction_update', {'junction_id': junction_id, 'data': junction_data})

def run_dashboard_server(dashboard_data_sender, host='127.0.0.1', port=5000):
    """Run the dashboard server"""
    dashboard.set_data_handler(dashboard_data_sender)
    dashboard.start_real_time_updates()
    
    print(f"🚀 Starting Traffic Dashboard at http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)

if __name__ == '__main__':
    run_dashboard_server(None)