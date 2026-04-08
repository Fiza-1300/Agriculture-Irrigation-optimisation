# from env import IrrigationEnv
# from stable_baselines3 import PPO
# import numpy as np

# # Load trained model
# model = PPO.load("model.zip")

# # Create environment (not used much here but good practice)
# env = IrrigationEnv(difficulty="medium")

# # 🔥 Manual test cases (YOU CONTROL INPUT)
# test_states = [
#     [0.30, 0.5, 0.5, 0.0, 0.2],  # Dry soil
#     [0.50, 0.5, 0.5, 0.0, 0.5],  # Perfect soil
#     [0.80, 0.5, 0.5, 0.0, 0.8],  # Too wet
#     [0.45, 0.5, 0.5, 1.0, 0.4],  # Rain situation
# ]

# # Run tests
# for i, s in enumerate(test_states):
#     state = np.array(s, dtype=np.float32)

#     action, _ = model.predict(state)
#     action = int(action)

#     print("\nTest Case", i+1)
#     print("Input State:", s)
#     print("Predicted Action:", action)

# test_model.py
from stable_baselines3 import PPO
try:
    model = PPO.load("model.zip")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model error: {e}")