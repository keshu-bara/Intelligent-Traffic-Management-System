import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class EnhancedDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, junction_id="default"):
        self.state_size = state_size
        self.action_size = action_size
        self.junction_id = junction_id
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = 32
        self.update_freq = 10
        self.target_update_freq = 100
        self.step_count = 0
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Duration prediction (additional network for adaptive timing)
        self.duration_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0-1, scale to duration range
        ).to(self.device)
        
        self.duration_optimizer = optim.Adam(self.duration_network.parameters(), lr=learning_rate)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return int(q_values.argmax().item())
    
    def act_with_duration(self, state):
        """Choose action and predict optimal duration"""
        action = self.act(state)
        
        # Predict duration (scale from 0-1 to 10-60 seconds)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        duration_norm = self.duration_network(state_tensor).item()
        duration = int(10 + duration_norm * 50)  # 10-60 seconds
        
        return action, duration
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.step_count += 1
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'duration_network': self.duration_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.duration_network.load_state_dict(checkpoint['duration_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            print(f"Model loaded for junction {self.junction_id}")

    def calculate_local_urgency(self, state, action):
        """Calculate urgency for coordination decisions"""
        # Sum of vehicles waiting
        waiting_vehicles = sum(state[i] for i in [0, 3, 6, 9])
        # Sum of congestion
        congestion = sum(state[i] for i in [1, 4, 7, 10])
        # Average speed (inverse for urgency)
        avg_speed = sum(state[i] for i in [2, 5, 8, 11]) / 4
        speed_urgency = 1.0 / (avg_speed + 0.1)  # Avoid division by zero
        
        urgency = (waiting_vehicles * 0.4 + congestion * 0.4 + speed_urgency * 0.2) / 100.0
        return min(urgency, 1.0)
    
    def find_alternative_action(self, current_action, state):
        """Find alternative action when current action conflicts"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor).squeeze()
        
        # Sort actions by Q-value, excluding current action
        sorted_actions = torch.argsort(q_values, descending=True)
        
        for action in sorted_actions:
            if action.item() != current_action:
                return action.item()
        
        # Fallback: return next action in sequence
        return (current_action + 1) % self.action_size
    
    def calculate_adaptive_duration_for_action(self, action, state):
        """Calculate adaptive duration for given action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        duration_norm = self.duration_network(state_tensor).item()
        
        # Adjust based on action type and traffic conditions
        base_duration = 10 + duration_norm * 50
        
        # Get traffic density for the action direction
        action_traffic = state[action * 3]  # Vehicles in that direction
        action_congestion = state[action * 3 + 1]  # Congestion in that direction
        
        # Extend duration for heavy traffic
        if action_traffic > 10:
            base_duration *= 1.2
        if action_congestion > 0.7:
            base_duration *= 1.3
            
        return int(max(10, min(60, base_duration)))