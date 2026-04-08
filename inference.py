from fastapi import FastAPI
from env import IrrigationEnv
from stable_baselines3 import PPO
import uvicorn
import os
import sys
import requests
import time

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.zip")

env = None
model = None
obs = None

@app.on_event("startup")
async def startup_event():
    global env, model, obs
    
    print("Starting up...")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files: {os.listdir('.')}")
    
    env = IrrigationEnv(difficulty="medium")
    print("✅ Environment loaded")
    
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    print("✅ Model loaded")
    
    obs, _ = env.reset()
    print("✅ Ready!")

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
    action, _ = model.predict(obs)
    action = int(action)
    obs, reward, done, _, info = env.step(action)
    return {
        "action": action,
        "reward": float(reward),
        "done": done,
        "observation": obs.tolist()
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
    
# Wait for server
for i in range(30):
    try:
        requests.get("http://localhost:7860/")
        print("Server ready")
        break
    except:
        print(f"Waiting... {i+1}/30")
        time.sleep(2)

# Test the API
requests.post("http://localhost:7860/reset")
result = requests.post("http://localhost:7860/step")
print(result.json())
