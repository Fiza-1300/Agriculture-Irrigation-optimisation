import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from env import IrrigationEnv
from stable_baselines3 import PPO
import uvicorn

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.zip")

env = IrrigationEnv(difficulty="medium")
model = PPO.load(MODEL_PATH)
obs = None

# ========== REQUIRED ENDPOINTS FOR SCALER ==========

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "Irrigation RL",
        "description": "AI-powered irrigation optimization using PPO"
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "discrete",
            "n": 3,
            "description": "0=no water, 1=low water, 2=high water"
        },
        "observation": {
            "type": "box",
            "shape": [5],
            "features": ["soil_moisture", "temperature", "humidity", "rain", "crop_stage"]
        },
        "state": {
            "type": "box",
            "shape": [5]
        }
    }

@app.get("/state")
def get_state():
    global obs
    if obs is None:
        obs, _ = env.reset()
    return {"state": obs.tolist()}

# ========== CORE API ENDPOINTS ==========

@app.get("/")
def root():
    return {"status": "healthy"}

@app.post("/reset")
def reset():
    global obs
    obs, _ = env.reset()
    return {"observation": obs.tolist()}

@app.post("/step")
def step():
    global obs
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(int(action))
    return {"observation": obs.tolist(), "reward": float(reward), "done": done}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()