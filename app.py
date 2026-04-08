
"""HF Space entry point for Agricultural Irrigation Environment"""

from fastapi import FastAPI
from env import IrrigationEnv
import numpy as np

app = FastAPI()
env = IrrigationEnv()

@app.get("/")
def root():
    return {"status": "healthy", "message": "Agricultural Irrigation Environment"}

@app.post("/reset")
def reset():
    global env
    env = IrrigationEnv()
    obs, info = env.reset()
    return {"observation": obs.tolist(), "info": info}

@app.post("/step")
def step(action: int):
    obs, reward, terminated, truncated, info = env.step(action)
    return {
        "observation": obs.tolist(),
        "reward": reward,
        "done": terminated or truncated,
        "info": info
    }

@app.get("/state")
def state():
    return {"status": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
