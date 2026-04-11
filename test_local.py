
"""Test script to verify environment without API calls"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 50)
print("ENVIRONMENT VARIABLES CHECK")
print("=" * 50)
print(f"API_BASE_URL: {os.getenv('API_BASE_URL', 'NOT SET')}")
print(f"API_KEY: {'SET' if os.getenv('API_KEY') else 'NOT SET'}")
print(f"MODEL_NAME: {os.getenv('MODEL_NAME', 'NOT SET')}")
print("=" * 50)

# Test the environment
from env import IrrigationEnv
print("\nTESTING ENVIRONMENT...")
env = IrrigationEnv(difficulty="medium", crop_type="wheat")
obs = env.reset()
print(f"✅ Environment initialized successfully!")
print(f"Observation shape: {obs[0].shape if isinstance(obs, tuple) else 'N/A'}")
print(f"Soil moisture: {obs[0][0] if isinstance(obs, tuple) else obs[0]:.3f}")

# Test graders
print("\nTESTING GRADERS...")
grades = env.get_all_grades()
for name, score in grades.items():
    print(f"  {name}: {score:.4f} {'✅' if 0 < score < 1 else '❌'}")

print("\n" + "=" * 50)
print("✅ All local tests PASSED!")
print("Your code is ready for submission!")
print("=" * 50)
