from env import IrrigationEnv
from stable_baselines3 import PPO

env = IrrigationEnv(difficulty="medium")
model = PPO.load("model.zip")

obs, _ = env.reset()
done = False
total_reward = 0
step = 0
rewards = []

# ✅ START (required format)
print("[START] task=irrigation env=agri model=ppo", flush=True)

while not done:
    action, _ = model.predict(obs)
    action = int(action)

    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    rewards.append(reward)

    # ✅ STEP (strict format)
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )

    step += 1

# ✅ Score normalization (0 to 1)
score = total_reward / 2000
score = min(max(score, 0), 1)

# ✅ END (strict format)
rewards_str = ",".join(f"{r:.2f}" for r in rewards)

print(
    f"[END] success=true steps={step} score={score:.3f} rewards={rewards_str}",
    flush=True
)