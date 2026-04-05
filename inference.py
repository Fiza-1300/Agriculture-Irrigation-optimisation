from Env import IrrigationEnv
from stable_baselines3 import PPO
import time

env = IrrigationEnv(difficulty="medium")
model = PPO.load("model.zip")

obs, _ = env.reset()
done = False
total_reward = 0
step = 0

print("[START]")

while not done:
    action, _ = model.predict(obs)
    action = int(action)

    obs, reward, done, _, info = env.step(action)
    total_reward += reward

    print(f"[STEP] {step} | action={action} | reward={reward:.2f}")
    step += 1

print(f"[END] total_reward={total_reward:.2f}")

print("[DONE] Keeping container alive...")
while True:
    time.sleep(60)
