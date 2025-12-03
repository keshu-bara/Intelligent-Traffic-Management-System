# test_connection.py
import os
import sys

# Test SUMO_HOME
if 'SUMO_HOME' not in os.environ:
    print("❌ SUMO_HOME not set")
    sys.exit(1)

print(f"✅ SUMO_HOME: {os.environ['SUMO_HOME']}")

# Test file paths
config_path = "../simulation/cfg/simulation.sumocfg"
if os.path.exists(config_path):
    print(f"✅ Config found: {config_path}")
else:
    print(f"❌ Config not found: {config_path}")

# Test imports
try:
    from AI_Agent import MultiJunctionDQNSimulation
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")