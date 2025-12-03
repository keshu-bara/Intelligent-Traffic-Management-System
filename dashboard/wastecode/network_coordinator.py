import numpy as np
from collections import defaultdict, deque
import time

class NetworkCoordinator:
    def __init__(self, junction_agents):
        self.junction_agents = junction_agents
        self.communication_matrix = self.build_communication_matrix()
        self.global_metrics = {
            'total_vehicles': 0,
            'total_congestion': 0,
            'network_throughput': 0,
            'coordination_efficiency': 0
        }
        
        # Performance tracking
        self.coordination_history = deque(maxlen=100)
        self.conflict_resolution_count = 0
        self.successful_coordinations = 0
        
    def build_communication_matrix(self):
        """Build communication relationships between junctions"""
        matrix = {
            'J1': ['J2', 'J3'],
            'J2': ['J1', 'J3'],
            'J3': ['J1', 'J2']
        }
        return matrix
    
    def coordinate_network_actions(self, junction_states, current_time):
        """Coordinate actions across the entire network"""
        
        junction_proposals = {}
        for junction_id, agent in self.junction_agents.items():
            local_state = junction_states[junction_id]
            proposed_action, proposed_duration = agent.act_with_coordination(local_state, current_time)
            
            junction_proposals[junction_id] = {
                'action': proposed_action,
                'duration': proposed_duration,
                'state': local_state,
                'urgency': agent.calculate_local_urgency(local_state, proposed_action)
            }
        
        # Global conflict resolution
        resolved_actions = self.global_conflict_resolution(junction_proposals, current_time)
        
        # Optimize for network flow
        optimized_actions = self.optimize_network_flow(resolved_actions, junction_states)
        
        return optimized_actions
    
    def global_conflict_resolution(self, proposals, current_time):
        """Resolve conflicts at network level"""
        resolved = proposals.copy()
        conflicts = self.identify_global_conflicts(proposals)
        
        for conflict in sorted(conflicts, key=lambda c: c['total_urgency'], reverse=True):
            involved_junctions = conflict['junctions']
            priority_junction = max(involved_junctions, 
                                  key=lambda j: proposals[j]['urgency'])
            
            for junction_id in involved_junctions:
                if junction_id != priority_junction:
                    agent = self.junction_agents[junction_id]
                    alternative = agent.find_alternative_action(
                        proposals[junction_id]['action'],
                        proposals[junction_id]['state']
                    )
                    resolved[junction_id]['action'] = alternative
                    self.conflict_resolution_count += 1
        
        return resolved
    
    def identify_global_conflicts(self, proposals):
        """Identify conflicts between junction actions"""
        conflicts = []
        junction_ids = list(proposals.keys())
        
        for i in range(len(junction_ids)):
            for j in range(i + 1, len(junction_ids)):
                j1_id, j2_id = junction_ids[i], junction_ids[j]
                
                if j2_id in self.communication_matrix.get(j1_id, []):
                    if self.actions_conflict(proposals[j1_id], proposals[j2_id]):
                        conflicts.append({
                            'junctions': [j1_id, j2_id],
                            'total_urgency': proposals[j1_id]['urgency'] + proposals[j2_id]['urgency'],
                            'conflict_type': 'timing'
                        })
        
        return conflicts
    
    def actions_conflict(self, proposal1, proposal2):
        """Check if two junction actions conflict"""
        # Define conflicting directions (opposite flows)
        conflicts = {0: 1, 1: 0, 2: 3, 3: 2}  # N-S, E-W conflicts
        
        action1, action2 = proposal1['action'], proposal2['action']
        return conflicts.get(action1) == action2
    
    def optimize_network_flow(self, actions, states):
        """Optimize actions for network-wide traffic flow"""
        network_pressure = self.calculate_network_pressure(states)
        flow_directions = self.identify_main_flow_directions(states)
        
        # Create green wave opportunities
        optimized = self.create_green_waves(actions, flow_directions)
        
        # Balance load across network
        balanced = self.balance_network_load(optimized, network_pressure)
        
        return balanced
    
    def calculate_network_pressure(self, states):
        """Calculate pressure at each junction"""
        pressures = {}
        
        for junction_id, state in states.items():
            # Calculate pressure as ratio of waiting vehicles to capacity
            waiting_vehicles = sum(state[i] for i in [0, 3, 6, 9])
            capacity = 50  # Assumed capacity per junction
            pressures[junction_id] = min(waiting_vehicles / capacity, 1.0)
        
        return pressures
    
    def identify_main_flow_directions(self, states):
        """Identify main traffic flow patterns"""
        flow_patterns = []
        
        # Analyze flow between connected junctions
        for junction_id, neighbors in self.communication_matrix.items():
            state = states[junction_id]
            
            # Find direction with most traffic
            direction_traffic = [
                sum(state[i:i+3]) for i in [0, 3, 6, 9]
            ]
            main_direction = direction_traffic.index(max(direction_traffic))
            
            flow_patterns.append({
                'junction': junction_id,
                'main_direction': main_direction,
                'traffic_volume': max(direction_traffic)
            })
        
        return flow_patterns
    
    def create_green_waves(self, actions, flow_directions):
        """Create coordinated green waves for main traffic flows"""
        # Find sequences of junctions in same flow direction
        coordinated_sequences = self.find_coordination_sequences(flow_directions)
        
        for sequence in coordinated_sequences:
            if len(sequence) > 1:
                # Calculate optimal timing offsets
                offsets = self.calculate_green_wave_offsets(sequence)
                
                for i, junction_id in enumerate(sequence):
                    if junction_id in actions:
                        actions[junction_id]['start_offset'] = offsets[i]
                        actions[junction_id]['coordinated'] = True
        
        return actions
    
    def find_coordination_sequences(self, flow_directions):
        """Find sequences of junctions that can coordinate"""
        sequences = []
        
        # Group junctions by similar flow direction
        direction_groups = defaultdict(list)
        
        for flow in flow_directions:
            direction_groups[flow['main_direction']].append(flow['junction'])
        
        # Create sequences from connected junctions in same direction
        for direction, junctions in direction_groups.items():
            if len(junctions) > 1:
                connected_sequence = self.find_connected_path(junctions)
                if len(connected_sequence) > 1:
                    sequences.append(connected_sequence)
        
        return sequences
    
    def find_connected_path(self, junction_list):
        """Find connected path through junctions"""
        if not junction_list:
            return []
        
        # Start with first junction
        path = [junction_list[0]]
        remaining = set(junction_list[1:])
        
        while remaining:
            current = path[-1]
            
            # Find next connected junction
            next_junction = None
            for junction in remaining:
                if junction in self.communication_matrix.get(current, []):
                    next_junction = junction
                    break
            
            if next_junction:
                path.append(next_junction)
                remaining.remove(next_junction)
            else:
                break
        
        return path
    
    def calculate_green_wave_offsets(self, sequence):
        """Calculate timing offsets for green wave coordination"""
        offsets = [0]  # First junction starts immediately
        
        # Assume average travel time between junctions is 30 seconds
        travel_time = 30
        
        for i in range(1, len(sequence)):
            # Offset each subsequent junction by travel time
            offsets.append(i * travel_time)
        
        return offsets
    
    def balance_network_load(self, actions, pressures):
        """Balance traffic load across the network"""
        
        # Find junctions with high pressure
        high_pressure_junctions = [
            j_id for j_id, pressure in pressures.items() 
            if pressure > 0.7
        ]
        
        # Extend green time for high-pressure junctions
        for junction_id in high_pressure_junctions:
            if junction_id in actions:
                current_duration = actions[junction_id]['duration']
                actions[junction_id]['duration'] = min(current_duration * 1.2, 60)
                actions[junction_id]['pressure_adjusted'] = True
        
        return actions
    
    def calculate_coordination_efficiency(self):
        """Calculate network coordination efficiency"""
        try:
            if not hasattr(self, 'successful_coordinations'):
                self.successful_coordinations = 0
            if not hasattr(self, 'conflict_resolution_count'):
                self.conflict_resolution_count = 0
                
            total_events = self.successful_coordinations + self.conflict_resolution_count
            
            if total_events == 0:
                return 0.85  # Default good efficiency
            
            efficiency = self.successful_coordinations / total_events
            return max(0.0, min(1.0, efficiency))
            
        except Exception as e:
            print(f"Error in NetworkCoordinator efficiency calculation: {e}")
            return 0.75
    
    def calculate_network_throughput(self):
        """Calculate network-wide traffic throughput"""
        return 0.8  # Placeholder value
    
    def update_coordination_metrics(self, coordination_success=True):
        """Update coordination performance metrics"""
        if coordination_success:
            self.successful_coordinations += 1
        
        self.coordination_history.append({
            'timestamp': time.time(),
            'success': coordination_success,
            'efficiency': self.calculate_coordination_efficiency()
        })