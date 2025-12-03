import os
import sys
import time
import numpy as np
import traci
import sumolib
from collections import deque

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from junction_agents import JunctionAgent
from network_coordinator import NetworkCoordinator
from junction_communication import MessagePassingSystem
from dashboard_handler import DashboardDataSender, ESP32DataSender



class MultiJunctionDQNSimulation:
    def __init__(self):
        # SUMO configuration path
        self.sumo_config = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'simulation', 'cfg', 'simulation.sumocfg'
        )
        
        # Verify config exists
        if not os.path.exists(self.sumo_config):
            raise FileNotFoundError(f"SUMO config not found: {self.sumo_config}")
        
        # Initialize junction agents
        self.junction_agents = {
            'J1': JunctionAgent('J1', 12, 4, ['J2', 'J3']),
            'J2': JunctionAgent('J2', 12, 4, ['J1', 'J3']),
            'J3': JunctionAgent('J3', 12, 4, ['J1', 'J2'])
        }
        
        # Initialize network coordinator
        self.network_coordinator = NetworkCoordinator(self.junction_agents)
        
        # Initialize communication system
        self.message_system = MessagePassingSystem()
        for junction_id in self.junction_agents.keys():
            self.message_system.register_junction(junction_id)
        
        # Initialize data collection (make it optional)
        try:
            self.dashboard = DashboardDataSender("multi_junction_data")
            self.esp32 = ESP32DataSender()
            print("✅ Dashboard initialized")
        except Exception as e:
            print(f"⚠️ Dashboard initialization failed: {e}")
            self.dashboard = None
            self.esp32 = None
        
        # Performance tracking
        self.network_performance = deque(maxlen=200)
        self.coordination_events = deque(maxlen=100)
        
        # Detect actual junction IDs from network file
        self.junction_ids = self.detect_junction_ids()
        
    def run_simulation(self):
        """Run multi-junction coordinated simulation"""
        
        # Start SUMO
        sumo_cmd = [
            "sumo-gui" if os.environ.get('SUMO_GUI', 'True') == 'True' else "sumo",
            "-c", self.sumo_config,
            "--start", "--quit-on-end"
        ]
        
        try:
            traci.start(sumo_cmd)
            print(f"✅ SUMO started with config: {self.sumo_config}")
            
            # Detect actual junction IDs
            actual_junctions = self.detect_junction_ids()
            print(f"Detected traffic light junctions: {actual_junctions}")
            
            # Update junction agents with actual IDs if needed
            if len(actual_junctions) >= 3:
                old_agents = self.junction_agents.copy()
                self.junction_agents = {}
                
                for i, junction_id in enumerate(actual_junctions[:3]):
                    old_key = ['J1', 'J2', 'J3'][i]
                    if old_key in old_agents:
                        self.junction_agents[junction_id] = old_agents[old_key]
                        self.junction_agents[junction_id].junction_id = junction_id
                        # Maintain communication protocol
                        self.junction_agents[junction_id].communication_protocol = self.message_system
        
        # Run simulation loop with improved logic
        step = 0
        max_steps = 5000  # Maximum steps to prevent infinite loop
        min_simulation_time = 300  # Run at least 5 minutes
        no_vehicles_count = 0  # Counter for consecutive steps with no vehicles
        
        junction_timers = {jid: 0 for jid in self.junction_agents.keys()}
        junction_states = {jid: np.zeros(12) for jid in self.junction_agents.keys()}
        
        print("🚀 Starting simulation loop...")
        
        while step < max_steps:
            current_time = traci.simulation.getTime()
            
            # Check if simulation should continue
            min_expected = traci.simulation.getMinExpectedNumber()
            active_vehicles = len(traci.vehicle.getIDList())
            
            # Continue simulation if:
            # 1. Still within minimum time OR
            # 2. There are expected vehicles OR  
            # 3. There are active vehicles
            if (current_time < min_simulation_time or 
                min_expected > 0 or 
                active_vehicles > 0):
                
                # Reset no vehicles counter if we have vehicles
                if min_expected > 0 or active_vehicles > 0:
                    no_vehicles_count = 0
                else:
                    no_vehicles_count += 1
                
                # Only break if no vehicles for 100+ consecutive steps AND past minimum time
                if no_vehicles_count > 100 and current_time > min_simulation_time:
                    print(f"🏁 No vehicles for {no_vehicles_count} steps, ending simulation")
                    break
                    
            else:
                print("🏁 All vehicles completed, ending simulation")
                break
            
            # Advance simulation
            traci.simulationStep()
            step += 1
            
            # Collect states for all junctions
            for junction_id in self.junction_agents.keys():
                try:
                    junction_states[junction_id] = self.get_junction_state(junction_id)
                except Exception as e:
                    print(f"Warning: Could not get state for {junction_id}: {e}")
                    junction_states[junction_id] = np.zeros(12)
            
            # Process communications every 10 steps
            if step % 10 == 0:
                try:
                    self.process_communications(current_time)
                except Exception as e:
                    print(f"Communication error: {e}")
            
            # Check which junctions need action updates
            junctions_to_update = []
            for junction_id, timer in junction_timers.items():
                if current_time - timer >= 15:  # Update every 15 seconds minimum
                    junctions_to_update.append(junction_id)
            
            if junctions_to_update:
                try:
                    # Coordinate actions across network
                    coordinated_actions = self.network_coordinator.coordinate_network_actions(
                        junction_states, current_time
                    )
                    
                    # Apply coordinated actions
                    for junction_id in junctions_to_update:
                        if junction_id in coordinated_actions:
                            success = self.apply_junction_action(
                                junction_id, 
                                coordinated_actions[junction_id],
                                current_time
                            )
                            if success:
                                junction_timers[junction_id] = current_time
                                
                except Exception as e:
                    print(f"Action coordination error: {e}")
            
            # Update network metrics and train
            if step % 100 == 0:  # Every 100 steps
                try:
                    self.update_network_metrics(junction_states, step)
                    self.broadcast_status_updates(junction_states, current_time)
                    
                    # Status report
                    print(f"📊 Step {step}: Time={current_time:.1f}s, "
                          f"Expected={min_expected}, Active={active_vehicles}, "
                          f"NoVehCount={no_vehicles_count}")
                    
                except Exception as e:
                    print(f"Metrics update error: {e}")
            
            # Train agents periodically
            if step % 200 == 0:
                try:
                    self.train_all_agents()
                except Exception as e:
                    print(f"Training error: {e}")
            
            # Small delay for stability
            time.sleep(0.001)
        
        print(f"🏁 Simulation completed after {step} steps ({current_time:.1f} seconds)")
            
    except Exception as e:
        print(f"Multi-junction simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            traci.close()
            self.save_models()
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def get_junction_state(self, junction_id):
        """Get traffic state for specific junction from SUMO"""
        try:
            # Get controlled lanes for this junction
            controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)
            
            if not controlled_lanes:
                return np.zeros(12)
            
            # Group lanes by direction (approximate)
            state = []
            
            # For each direction (North, East, South, West)
            for direction in range(4):
                start_idx = direction * len(controlled_lanes) // 4
                end_idx = (direction + 1) * len(controlled_lanes) // 4
                direction_lanes = controlled_lanes[start_idx:end_idx]
                
                # Get metrics for this direction
                vehicles = sum(traci.lane.getLastStepVehicleNumber(lane) 
                              for lane in direction_lanes if lane)
                
                waiting = sum(traci.lane.getLastStepHaltingNumber(lane) 
                             for lane in direction_lanes if lane)
                
                if vehicles > 0:
                    avg_speed = sum(traci.lane.getLastStepMeanSpeed(lane) 
                                   for lane in direction_lanes if lane) / len(direction_lanes)
                else:
                    avg_speed = 0
                
                state.extend([vehicles, waiting/max(vehicles, 1), avg_speed])
            
            return np.array(state)
            
        except Exception as e:
            print(f"Error getting state for {junction_id}: {e}")
            return np.zeros(12)
    
    def process_communications(self, current_time):
        """Process inter-junction communications"""
        
        for junction_id, agent in self.junction_agents.items():
            # Get pending messages
            messages = self.message_system.get_messages(junction_id)
            
            for message in messages:
                agent.communication_protocol.process_incoming_message(message)
                
                # Log coordination events
                if message.get('message_type') == 'coordination_request':
                    self.coordination_events.append({
                        'time': current_time,
                        'sender': message['sender_id'],
                        'receiver': junction_id,
                        'type': 'coordination',
                        'priority': message.get('priority', 0)
                    })
    
    def broadcast_status_updates(self, junction_states, current_time):
        """Broadcast status updates between junctions"""
        
        for junction_id, agent in self.junction_agents.items():
            state = junction_states[junction_id]
            
            # Create status message
            status_message = agent.communication_protocol.create_status_message(
                state, 
                agent.last_action if hasattr(agent, 'last_action') else 0,
                agent.last_duration if hasattr(agent, 'last_duration') else 15,
                current_time + 15  # Estimated phase end time
            )
            
            # Broadcast to neighbors
            for neighbor_id in agent.neighbor_junctions:
                self.message_system.send_message(junction_id, neighbor_id, status_message)
    
    def update_network_metrics(self, junction_states, step):
        """Update network-wide performance metrics"""
        try:
            # Calculate network metrics
            total_vehicles = sum(sum(state[i] for i in [0, 3, 6, 9] if i < len(state)) 
                           for state in junction_states.values())
            
            total_congestion = sum(sum(state[i] for i in [1, 4, 7, 10] if i < len(state)) 
                                  for state in junction_states.values())
            
            # Calculate average speed safely
            speed_values = []
            for state in junction_states.values():
                for i in [2, 5, 8, 11]:
                    if i < len(state):
                        speed_values.append(state[i])
            
            avg_speed = sum(speed_values) / len(speed_values) if speed_values else 0.0
            
            # Calculate coordination efficiency using the local method
            coordination_efficiency = self.calculate_coordination_efficiency()
            
            # Store metrics
            network_metrics = {
                'step': step,
                'total_vehicles': float(total_vehicles),
                'total_congestion': float(total_congestion), 
                'avg_speed': float(avg_speed),
                'coordination_efficiency': float(coordination_efficiency),
                'active_coordinations': len(self.coordination_events) if hasattr(self, 'coordination_events') else 0
            }
            
            self.network_performance.append(network_metrics)
            
            # Send to dashboard safely
            if hasattr(self, 'dashboard') and self.dashboard:
                try:
                    self.dashboard.send_network_performance(step, network_metrics)
                except Exception as e:
                    print(f"Dashboard error: {e}")
            
            # Print status every 50 steps
            if step % 50 == 0:
                print(f"🌐 Network Step {step}: Vehicles={total_vehicles:.0f}, "
                      f"Congestion={total_congestion:.2f}, "
                      f"AvgSpeed={avg_speed:.1f}, "
                      f"Coordination={coordination_efficiency:.2f}")
            
        except Exception as e:
            print(f"Error updating network metrics: {e}")
    
    def detect_junction_ids(self):
        """Detect actual junction IDs from network file"""
        network_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'simulation', 'net', 'network.net.xml'
        )
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(network_file)
            root = tree.getroot()
            
            junction_ids = []
            for junction in root.findall('.//junction[@type="traffic_light"]'):
                junction_ids.append(junction.get('id'))
            
            print(f"Detected traffic light junctions: {junction_ids}")
            return junction_ids[:3]  # Take first 3 junctions
            
        except Exception as e:
            print(f"Could not detect junctions, using default: {e}")
            return ['J1', 'J2', 'J3']
    
    def apply_junction_action(self, junction_id, action_data, current_time):
        """Apply coordinated action to junction traffic light"""
        try:
            action = action_data['action']
            duration = action_data['duration']
            
            # Map action to traffic light phase
            phase_mapping = {0: 0, 1: 6, 2: 3, 3: 9}  # Adjust based on your tls.tll.xml
            phase = phase_mapping.get(action, 0)
            
            # Apply phase change
            traci.trafficlight.setPhase(junction_id, phase)
            traci.trafficlight.setPhaseDuration(junction_id, duration)
            
            print(f"🚦 {junction_id}: Applied action {action} (phase {phase}) for {duration}s")
            
            return True
            
        except Exception as e:
            print(f"Error applying action to {junction_id}: {e}")
            return False
    
    def train_all_agents(self):
        """Train all junction agents"""
        try:
            for junction_id, agent in self.junction_agents.items():
                if hasattr(agent, 'dqn_agent') and hasattr(agent.dqn_agent, 'replay'):
                    agent.dqn_agent.replay()
        
        except Exception as e:
            print(f"❌ Error training agents: {e}")
    
    def save_models(self):
        """Save all junction agent models"""
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            for junction_id, agent in self.junction_agents.items():
                model_path = os.path.join(models_dir, f"{junction_id}_dqn_model.pth")
                agent.dqn_agent.save_model(model_path)
                print(f"💾 Saved model for {junction_id}")
            
            print("✅ All models saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def debug_simulation_status(self, step):
        """Debug simulation status"""
        try:
            # Get simulation info
            vehicle_count = traci.simulation.getMinExpectedNumber()
            active_vehicles = len(traci.vehicle.getIDList())
            current_time = traci.simulation.getTime()
            
            # Check junction status
            for junction_id in self.junction_agents.keys():
                try:
                    phase = traci.trafficlight.getPhase(junction_id)
                    controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)
                    lane_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane) 
                                      for lane in controlled_lanes[:4])  # First 4 lanes
                    
                    if step % 100 == 0:  # Print every 100 steps
                        print(f"  {junction_id}: Phase={phase}, Vehicles={lane_vehicles}")
                        
                except Exception as e:
                    print(f"  Error checking {junction_id}: {e}")
            
            if step % 100 == 0:
                print(f"📊 Step {step}: Time={current_time:.1f}s, "
                      f"Expected={vehicle_count}, Active={active_vehicles}")
            
            # Check if simulation should continue
            if vehicle_count == 0 and active_vehicles == 0 and current_time > 100:
                print("⚠️ No vehicles remaining, simulation ending")
                return False
                
            return True
            
        except Exception as e:
            print(f"Debug error: {e}")
            return True

    def calculate_coordination_efficiency(self):
        """Calculate coordination efficiency for the network"""
        try:
            if not hasattr(self, 'coordination_events') or len(self.coordination_events) == 0:
                return 0.8  # Default efficiency
            
            # Calculate based on recent coordination events
            recent_events = list(self.coordination_events)[-20:]  # Last 20 events
            if not recent_events:
                return 0.8
            
            successful_coords = sum(1 for event in recent_events 
                                   if event.get('type') == 'coordination' and 
                                   event.get('success', False))
            
            total_coords = len(recent_events)
            efficiency = successful_coords / total_coords if total_coords > 0 else 0.8
            
            return max(0.0, min(1.0, efficiency))
            
        except Exception as e:
            print(f"Error calculating coordination efficiency: {e}")
            return 0.8

if __name__ == "__main__":
    simulation = MultiJunctionDQNSimulation()
    simulation.run_simulation()