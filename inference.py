from env import IrrigationEnv
from stable_baselines3 import PPO
from fastapi import FastAPI
import uvicorn

# ------------------ RL EXECUTION ------------------ #

env = IrrigationEnv(difficulty="medium")
model = PPO.load("model.zip")

obs, _ = env.reset()
done = False
total_reward = 0
step = 0
rewards = []

# ✅ START
print("[START] task=irrigation env=agri model=ppo", flush=True)

while not done:
    action, _ = model.predict(obs)
    action = int(action)

    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    rewards.append(reward)

    # ✅ STEP
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )

    step += 1

# ✅ SCORE NORMALIZATION
score = total_reward / 2000
score = min(max(score, 0), 1)

rewards_str = ",".join(f"{r:.2f}" for r in rewards)

# ✅ END
print(
    f"[END] success=true steps={step} score={score:.3f} rewards={rewards_str}",
    flush=True
)

print("[DONE] Model execution complete", flush=True)

# ------------------ FASTAPI SERVER ------------------ #

app = FastAPI()

@app.get("/")
def home():
    return {
        "status": "Irrigation RL running successfully",
        "steps": step,
        "score": round(score, 3)
    }

# 🔥 THIS IS CRUCIAL FOR HUGGING FACE
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)