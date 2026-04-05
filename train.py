from env import IrrigationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

# Wrap environment (IMPORTANT)
env = DummyVecEnv([lambda: IrrigationEnv(difficulty="medium")])

# Train model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("model.zip")

print("Training complete!")