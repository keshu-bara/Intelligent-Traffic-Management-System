import numpy as np
import torch
import torch.nn as nn
from collections import deque
import json
from datetime import datetime
import traci

class JunctionCommunicationProtocol:
    def __init__(self, junction_id):
        self.junction_id = junction_id
        self.message_queue = deque(maxlen=100)
        self.broadcast_interval = 5  # seconds
        self.last_broadcast = 0
        
    def create_status_message(self, state, action, duration, phase_end_time):
        """Create status message to broadcast to neighbors"""
        
        message = {
            'sender_id': self.junction_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'status_update',
            
            # Current traffic conditions
            'traffic_conditions': {
                'total_vehicles': float(sum(state[i] for i in [0, 3, 6, 9])),
                'total_congestion': float(sum(state[i] for i in [1, 4, 7, 10])),
                'avg_speed': float(sum(state[i] for i in [2, 5, 8, 11]) / 4),
                'pressure': self.calculate_pressure(state)
            },
            
            # Current action and timing
            'current_action': int(action),
            'phase_duration': int(duration),
            'phase_end_time': float(phase_end_time),
            'phase_remaining': float(phase_end_time - traci.simulation.getTime()) if traci.simulation.getTime() else 0,
            
            # Performance metrics
            'performance': {
                'recent_throughput': self.calculate_recent_throughput(),
                'congestion_trend': self.calculate_congestion_trend(),
                'coordination_requests': self.count_coordination_requests()
            },
            
            # Requests for coordination
            'coordination_request': self.generate_coordination_request(state),
            
            # Priority information
            'emergency_vehicles': self.count_emergency_vehicles(),
            'high_priority_flows': self.identify_priority_flows(state)
        }
        
        return message
    
    def process_incoming_message(self, message):
        """Process message from neighbor junction"""
        
        sender_id = message['sender_id']
        message_type = message['message_type']
        
        if message_type == 'status_update':
            self.process_status_update(sender_id, message)
        elif message_type == 'coordination_request':
            self.process_coordination_request(sender_id, message)
        elif message_type == 'emergency_alert':
            self.process_emergency_alert(sender_id, message)
        elif message_type == 'flow_prediction':
            self.process_flow_prediction(sender_id, message)
        
        # Store message for reference
        self.message_queue.append(message)
    
    def generate_coordination_request(self, state):
        """Generate coordination request based on current conditions"""
        
        # Check if coordination would be beneficial
        if self.needs_coordination(state):
            return {
                'request_type': 'phase_coordination',
                'priority': self.calculate_request_priority(state),
                'preferred_action': self.get_preferred_action(state),
                'flexibility': self.calculate_flexibility(state),
                'reason': self.get_coordination_reason(state)
            }
        
        return None
    
    def broadcast_to_neighbors(self, message, neighbor_list):
        """Broadcast message to neighbor junctions"""
        
        for neighbor_id in neighbor_list:
            try:
                # In real implementation, this would use network communication
                # For simulation, we'll use a message passing system
                self.send_message_to_junction(neighbor_id, message)
                
            except Exception as e:
                print(f"Failed to send message to {neighbor_id}: {e}")

    def calculate_pressure(self, state):
        """Calculate junction pressure"""
        waiting_vehicles = sum(state[i] for i in [1, 4, 7, 10])  # Waiting vehicles
        total_vehicles = sum(state[i] for i in [0, 3, 6, 9])     # Total vehicles
        
        if total_vehicles == 0:
            return 0.0
        
        return waiting_vehicles / total_vehicles

class MessagePassingSystem:
    """Centralized message passing for simulation"""
    
    def __init__(self):
        self.message_queues = {}  # Junction ID -> message queue
        self.delivery_delay = 0.1  # Simulate network delay
        
    def register_junction(self, junction_id):
        """Register a junction for message passing"""
        self.message_queues[junction_id] = deque(maxlen=50)
    
    def send_message(self, sender_id, receiver_id, message):
        """Send message from one junction to another"""
        
        if receiver_id in self.message_queues:
            # Add sender information and timestamp
            message['sender_id'] = sender_id
            message['delivery_time'] = traci.simulation.getTime() + self.delivery_delay
            
            self.message_queues[receiver_id].append(message)
        
    def get_messages(self, junction_id):
        """Get pending messages for a junction"""
        
        if junction_id not in self.message_queues:
            return []
        
        current_time = traci.simulation.getTime()
        ready_messages = []
        
        # Get messages that have reached delivery time
        while self.message_queues[junction_id]:
            message = self.message_queues[junction_id][0]
            if message['delivery_time'] <= current_time:
                ready_messages.append(self.message_queues[junction_id].popleft())
            else:
                break
        
        return ready_messages