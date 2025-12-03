"""
Multi-Junction Traffic Control AI System

This package provides a complete multi-agent DQN system for coordinated
traffic light control across multiple intersections.
"""

from .wastecode.base_dqn_agent import EnhancedDQNAgent, DQNNetwork
from .junction_agents import JunctionAgent
from .junction_communication import JunctionCommunicationProtocol, MessagePassingSystem
from .network_coordinator import NetworkCoordinator
from .dashboard_handler import DashboardDataSender, ESP32DataSender
from .web_dashboard import TrafficDashboard, run_dashboard_server
from .multi_junction_simulation import MultiJunctionDQNSimulation

__version__ = "1.0.0"
__author__ = "Traffic AI System"

__all__ = [
    'EnhancedDQNAgent',
    'DQNNetwork', 
    'JunctionAgent',
    'JunctionCommunicationProtocol',
    'MessagePassingSystem',
    'NetworkCoordinator',
    'DashboardDataSender',
    'ESP32DataSender',
    'TrafficDashboard',
    'run_dashboard_server',
    'MultiJunctionDQNSimulation'
]