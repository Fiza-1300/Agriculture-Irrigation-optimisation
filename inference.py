# from fastapi import FastAPI
# from env import IrrigationEnv
# from stable_baselines3 import PPO

# app = FastAPI()

# env = IrrigationEnv(difficulty="medium")
# model = PPO.load("model.zip")

# obs = None

# @app.post("/reset")
# def reset():
#     global obs
#     obs, _ = env.reset()
#     return {"observation": obs.tolist()}

# @app.post("/step")
# def step():
#     global obs

#     action, _ = model.predict(obs)
#     action = int(action)

#     obs, reward, done, _, _ = env.step(action)

#     return {
#         "action": action,
#         "reward": float(reward),
#         "done": done,
#         "observation": obs.tolist()
#     }

from fastapi import FastAPI
from env import IrrigationEnv
from stable_baselines3 import PPO
import uvicorn
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.zip")

env = IrrigationEnv(difficulty="medium")
model = PPO.load(MODEL_PATH)

obs = None

@app.get("/")
def root():
    return {"message": "Irrigation AI API is running", "status": "healthy"}

@app.post("/reset")
def reset():
    global obs
    obs, _ = env.reset()
    return {"observation": obs.tolist()}

@app.post("/step")
def step():
    global obs
    if obs is None:
        reset()
    
    action, _ = model.predict(obs)
    action = int(action)
    
    obs, reward, done, _, info = env.step(action)
    
    return {
        "action": action,
        "reward": float(reward),
        "done": done,
        "observation": obs.tolist(),
        "info": info
    }

# THIS IS THE CRITICAL PART - MAIN FUNCTION
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()