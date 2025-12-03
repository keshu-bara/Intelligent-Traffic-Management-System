import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q = self.model(torch.FloatTensor(next_state).unsqueeze(0)).detach()
                target = reward + self.gamma * torch.max(next_q).item()
            current_q = self.model(torch.FloatTensor(state).unsqueeze(0))[0][action]
            loss = self.criterion(current_q, torch.tensor(target))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class NetworkDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, junction_id="J1", neighbors=None):
        super().__init__(state_size, action_size)
        
        self.junction_id = junction_id
        self.neighbors = neighbors or []
        
        # Enhanced parameters
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005
        
        # Experience replay improvements
        self.memory = deque(maxlen=10000)
        self.priority_memory = deque(maxlen=1000)
        
        # Network communication state
        self.neighbor_states = {}  # Store states from neighbor junctions
        self.last_action = 0
        self.last_reward = 0
        self.communication_history = deque(maxlen=50)
        
        # Dynamic timing parameters
        self.min_phase_duration = 5
        self.max_phase_duration = 45
        self.default_duration = 15
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.congestion_history = deque(maxlen=100)
    
    def get_network_state(self, local_state):
        """Combine local state with neighbor information"""
        network_state = list(local_state)
        
        # Add neighbor information (vehicles, congestion, average speed)
        for neighbor_id in self.neighbors:
            if neighbor_id in self.neighbor_states:
                neighbor_info = self.neighbor_states[neighbor_id]
                network_state.extend([
                    neighbor_info.get('total_vehicles', 0),
                    neighbor_info.get('avg_congestion', 0),
                    neighbor_info.get('avg_speed', 0),
                    neighbor_info.get('pressure', 0)
                ])
            else:
                network_state.extend([0, 0, 0, 0])  # Default values
        
        return np.array(network_state)
    
    def update_neighbor_state(self, neighbor_id, state_info):
        """Update state information from neighbor junction"""
        if neighbor_id in self.neighbors:
            self.neighbor_states[neighbor_id] = state_info
            self.communication_history.append({
                'from': neighbor_id,
                'to': self.junction_id,
                'data': state_info,
                'timestamp': len(self.communication_history)
            })
    
    def get_coordination_action(self, local_state):
        """Choose action considering network coordination"""
        network_state = self.get_network_state(local_state)
        action = self.act(network_state)
        
        # Calculate network pressure and adjust if needed
        local_pressure = sum(local_state[1::4])  # Sum congestion values
        neighbor_pressures = []
        
        for neighbor_id in self.neighbors:
            if neighbor_id in self.neighbor_states:
                neighbor_pressures.append(self.neighbor_states[neighbor_id].get('pressure', 0))
        
        # If neighbors are heavily congested, extend our phase duration
        if neighbor_pressures and max(neighbor_pressures) > 0.7:
            duration = min(self.default_duration * 1.5, self.max_phase_duration)
        else:
            duration = self.choose_phase_duration(local_state, action)
        
        return action, int(duration)
        
    def act_with_duration(self, state):
        """Choose both action and phase duration"""
        # Choose action (which phase to activate)
        action = self.act(state)
        
        # Choose duration based on traffic conditions
        duration = self.choose_phase_duration(state, action)
        
        return action, duration
    
    def choose_phase_duration(self, state, action):
        """Dynamically choose phase duration based on traffic conditions"""
        try:
            # Get traffic density for the chosen direction
            direction_index = action
            vehicle_count = state[direction_index * 4]  # Vehicle count for this direction
            halting_count = state[direction_index * 4 + 1]  # Halting vehicles
            speed = state[direction_index * 4 + 2]  # Speed
            
            # Base duration on traffic conditions
            if vehicle_count > 0.7:  # High traffic
                if halting_count > 0.5:  # High congestion
                    duration = self.max_phase_duration  # Long green for clearing
                else:
                    duration = 25  # Medium-long green
            elif vehicle_count > 0.3:  # Medium traffic
                duration = self.default_duration
            else:  # Low traffic
                duration = self.min_phase_duration  # Short green
            
            # Adjust for emergency vehicles
            if len(state) > 18 and state[18] > 0:  # Emergency vehicles present
                duration = min(duration + 10, self.max_phase_duration)
            
            return duration
            
        except Exception as e:
            print(f"Error choosing duration: {e}")
            return self.default_duration
    
    def enhanced_replay(self, batch_size=64):
        """Enhanced experience replay with prioritization"""
        if len(self.memory) < batch_size:
            return
        
        # Sample regular experiences
        regular_batch_size = batch_size // 2
        priority_batch_size = batch_size - regular_batch_size
        
        # Sample from regular memory
        regular_batch = random.sample(self.memory, min(regular_batch_size, len(self.memory)))
        
        # Sample from priority memory (high reward experiences)
        priority_batch = []
        if len(self.priority_memory) > 0:
            priority_batch = random.sample(self.priority_memory, 
                                         min(priority_batch_size, len(self.priority_memory)))
        
        # Combine batches
        minibatch = regular_batch + priority_batch
        
        # Train on batch
        states = torch.FloatTensor([exp[0] for exp in minibatch])
        actions = torch.LongTensor([exp[1] for exp in minibatch])
        rewards = torch.FloatTensor([exp[2] for exp in minibatch])
        next_states = torch.FloatTensor([exp[3] for exp in minibatch])
        dones = torch.BoolTensor([exp[4] for exp in minibatch])
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).detach().max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def remember_priority(self, state, action, reward, next_state, done):
        """Store high-reward experiences in priority memory"""
        self.remember(state, action, reward, next_state, done)
        
        # Store high-reward or high-loss experiences in priority memory
        if abs(reward) > 2.0:  # High absolute reward
            self.priority_memory.append((state, action, reward, next_state, done))
        
        # Track performance
        self.performance_history.append(reward)
        
        # Track congestion reduction
        try:
            current_congestion = sum(state[i] for i in range(1, 16, 4))
            self.congestion_history.append(current_congestion)
        except:
            pass