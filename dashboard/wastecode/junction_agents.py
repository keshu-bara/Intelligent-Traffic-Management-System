import numpy as np
import torch
import torch.nn as nn
from collections import deque
import json
from datetime import datetime
from AI_Agent.wastecode.base_dqn_agent import EnhancedDQNAgent
from junction_communication import communicatoinion_protocol, MessagePassingSystem

class JunctionAgent:
    def __init__(self, junction_id, state_size, action_size, neighbor_junctions):
        self.junction_id = junction_id
        self.state_size = state_size
        self.action_size = action_size
        self.neighbor_junctions = neighbor_junctions
        
        # Enhanced state includes neighbor information
        self.enhanced_state_size = state_size + len(neighbor_junctions) * 4
        
        # Initialize DQN agent
        self.dqn_agent = EnhancedDQNAgent(self.enhanced_state_size, action_size)
        
        # Communication attributes - ADD THIS
        self.communication_protocol = None  # Will be set by message system
        self.incoming_messages = {}
        self.outgoing_messages = {}
        self.message_history = []
        
        # Performance tracking
        self.rewards_history = []
        self.actions_history = []
        self.coordination_count = 0
        
        # Last action timing
        self.last_action_time = 0
        self.current_phase_duration = 30

    def get_enhanced_state(self, local_state):
        """Combine local state with neighbor information"""
        enhanced_state = local_state.copy()  # This creates numpy array
        
        # Add neighbor states
        for neighbor_id in self.neighbor_junctions:
            if neighbor_id in self.incoming_messages:
                neighbor_info = self.incoming_messages[neighbor_id]
                neighbor_features = [
                    neighbor_info.get('total_vehicles', 0),
                    neighbor_info.get('total_congestion', 0),
                    neighbor_info.get('avg_speed', 0),
                    neighbor_info.get('pressure', 0)
                ]
                # FIX: Convert to list before extending
                enhanced_state = enhanced_state.tolist()
                enhanced_state.extend(neighbor_features)
            else:
                enhanced_state = enhanced_state.tolist() if isinstance(enhanced_state, np.ndarray) else enhanced_state
                enhanced_state.extend([0, 0, 0, 0])
        
        return np.array(enhanced_state)

    def act_with_coordination(self, local_state, current_time):
        """Make action considering both local and neighbor conditions"""
        # Get enhanced state with neighbor information
        enhanced_state = self.get_enhanced_state(local_state)
        
        # Get DQN action
        dqn_action, dqn_duration = self.dqn_agent.act_with_duration(enhanced_state)
        
        # Apply coordination logic
        coordinated_action, coordinated_duration = self.coordinate_with_neighbors(
            dqn_action, dqn_duration, local_state, current_time
        )
        
        return coordinated_action, coordinated_duration

    def coordinate_with_neighbors(self, proposed_action, proposed_duration, local_state, current_time):
        """Coordinate action with neighbors to optimize network flow"""
        
        # Check for conflicts with neighbors
        conflicts = self.detect_conflicts(proposed_action, current_time)
        
        if conflicts:
            # Resolve conflicts using priority and traffic conditions
            resolved_action, resolved_duration = self.resolve_conflicts(
                proposed_action, proposed_duration, conflicts, local_state
            )
            return resolved_action, resolved_duration
        
        # Check for coordination opportunities
        coordination_benefit = self.calculate_coordination_benefit(proposed_action, local_state)
        
        if coordination_benefit > 0.5:  # Significant benefit from coordination
            adjusted_action, adjusted_duration = self.apply_coordination(
                proposed_action, proposed_duration, local_state
            )
            return adjusted_action, adjusted_duration
        
        return proposed_action, proposed_duration

    def detect_conflicts(self, proposed_action, current_time):
        """Detect conflicts with neighbor junction actions"""
        conflicts = []
        
        for neighbor_id in self.neighbor_junctions:
            if neighbor_id in self.incoming_messages:
                neighbor_msg = self.incoming_messages[neighbor_id]
                neighbor_action = neighbor_msg.get('current_action', None)
                neighbor_end_time = neighbor_msg.get('phase_end_time', 0)
                
                # Check for timing conflicts
                if self.creates_conflict(proposed_action, neighbor_action, current_time, neighbor_end_time):
                    conflicts.append({
                        'neighbor_id': neighbor_id,
                        'neighbor_action': neighbor_action,
                        'conflict_type': 'timing',
                        'severity': self.calculate_conflict_severity(neighbor_msg)
                    })
        
        return conflicts

    def creates_conflict(self, my_action, neighbor_action, current_time, neighbor_end_time):
        """Check if actions create traffic flow conflicts"""
        # Define conflicting action pairs (example for 4-way intersections)
        conflict_pairs = {
            0: [2],  # North conflicts with East
            1: [3],  # South conflicts with West  
            2: [0],  # East conflicts with North
            3: [1]   # West conflicts with South
        }
        
        if neighbor_action in conflict_pairs.get(my_action, []):
            # Check timing overlap
            my_start_time = current_time
            neighbor_end_time = neighbor_end_time
            
            if my_start_time < neighbor_end_time:
                return True
        
        return False

    def resolve_conflicts(self, proposed_action, proposed_duration, conflicts, local_state):
        """Resolve conflicts using priority system"""
        
        # Calculate local urgency
        local_urgency = self.calculate_local_urgency(local_state, proposed_action)
        
        # Find highest priority conflict
        highest_priority_conflict = max(conflicts, key=lambda c: c['severity'])
        
        if local_urgency > highest_priority_conflict['severity']:
            # Local traffic has higher priority
            return proposed_action, proposed_duration
        else:
            # Defer to neighbor - choose alternative action
            alternative_action = self.find_alternative_action(proposed_action, local_state)
            alternative_duration = self.calculate_adaptive_duration_for_action(
                alternative_action, local_state
            )
            
            print(f"🤝 Junction {self.junction_id}: Conflict resolved - "
                  f"Action {proposed_action} → {alternative_action}")
            
            return alternative_action, alternative_duration

    def calculate_local_urgency(self, state, action):
        """Calculate urgency for coordination decisions"""
        # Convert to list if numpy array
        if isinstance(state, np.ndarray):
            state = state.tolist()
        
        # Sum of vehicles waiting
        waiting_vehicles = sum(state[i] for i in [0, 3, 6, 9] if i < len(state))
        # Sum of congestion
        congestion = sum(state[i] for i in [1, 4, 7, 10] if i < len(state))
        # Average speed (inverse for urgency)
        speeds = [state[i] for i in [2, 5, 8, 11] if i < len(state)]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.1
        speed_urgency = 1.0 / (avg_speed + 0.1)
        
        urgency = (waiting_vehicles * 0.4 + congestion * 0.4 + speed_urgency * 0.2) / 100.0
        return min(urgency, 1.0)

    def find_alternative_action(self, current_action, state):
        """Find alternative action when current action conflicts"""
        # Simple fallback: return next action in sequence
        return (current_action + 1) % self.action_size

    def calculate_adaptive_duration_for_action(self, action, state):
        """Calculate adaptive duration for given action"""
        base_duration = 15  # Base duration
        
        # Convert to list if numpy array
        if isinstance(state, np.ndarray):
            state = state.tolist()
        
        # Get traffic density for the action direction
        if action * 3 < len(state):
            action_traffic = state[action * 3]  # Vehicles in that direction
            action_congestion = state[action * 3 + 1] if action * 3 + 1 < len(state) else 0
            
            # Extend duration for heavy traffic
            if action_traffic > 10:
                base_duration *= 1.2
            if action_congestion > 0.7:
                base_duration *= 1.3
        
        return int(max(10, min(60, base_duration)))

    def calculate_coordination_benefit(self, proposed_action, local_state):
        """Calculate benefit of coordination with neighbors"""
        try:
            # Convert to list if numpy array
            if isinstance(local_state, np.ndarray):
                local_state = local_state.tolist()
            
            # Simple coordination benefit calculation
            local_urgency = self.calculate_local_urgency(local_state, proposed_action)
            
            # Check if neighbors need help
            neighbor_needs_help = False
            for neighbor_id in self.neighbor_junctions:
                if neighbor_id in self.incoming_messages:
                    neighbor_msg = self.incoming_messages[neighbor_id]
                    neighbor_congestion = neighbor_msg.get('total_congestion', 0)
                    if neighbor_congestion > 0.6:  # High congestion
                        neighbor_needs_help = True
                        break
            
            # Return higher benefit if we can help congested neighbors
            if neighbor_needs_help and local_urgency < 0.5:
                return 0.7
            elif local_urgency > 0.7:
                return 0.3  # Focus on local traffic
            else:
                return 0.5  # Neutral
                
        except Exception as e:
            print(f"Error calculating coordination benefit: {e}")
            return 0.5

    def apply_coordination(self, proposed_action, proposed_duration, local_state):
        """Apply coordination adjustments to action"""
        try:
            # Simple coordination: adjust duration based on neighbor conditions
            adjusted_action = proposed_action
            adjusted_duration = proposed_duration
            
            # Check neighbor messages for coordination needs
            for neighbor_id in self.neighbor_junctions:
                if neighbor_id in self.incoming_messages:
                    neighbor_msg = self.incoming_messages[neighbor_id]
                    neighbor_congestion = neighbor_msg.get('total_congestion', 0)
                    
                    # If neighbor is congested, extend our duration slightly
                    if neighbor_congestion > 0.7:
                        adjusted_duration = min(adjusted_duration * 1.1, 60)
                        break
            
            return adjusted_action, int(adjusted_duration)
            
        except Exception as e:
            print(f"Error applying coordination: {e}")
            return proposed_action, proposed_duration

    def calculate_conflict_severity(self, neighbor_msg):
        """Calculate severity of conflict with neighbor"""
        try:
            congestion = neighbor_msg.get('total_congestion', 0)
            vehicles = neighbor_msg.get('total_vehicles', 0)
            
            # Higher severity for more congested neighbors
            severity = (congestion * 0.6) + (min(vehicles / 30.0, 1.0) * 0.4)
            return min(severity, 1.0)
            
        except Exception as e:
            print(f"Error calculating conflict severity: {e}")
            return 0.5

