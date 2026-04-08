from fastapi import FastAPI
from env import IrrigationEnv
from stable_baselines3 import PPO

app = FastAPI()

env = IrrigationEnv(difficulty="medium")
model = PPO.load("model.zip")

obs = None

@app.post("/reset")
def reset():
    global obs
    obs, _ = env.reset()
    return {"observation": obs.tolist()}

@app.post("/step")
def step():
    global obs

    action, _ = model.predict(obs)
    action = int(action)

    obs, reward, done, _, _ = env.step(action)

    return {
        "action": action,
        "reward": float(reward),
        "done": done,
        "observation": obs.tolist()
    }