# from fastapi import FastAPI
# from env import IrrigationEnv
# from stable_baselines3 import PPO
# import os

# app = FastAPI()

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, "model.zip")

# env = IrrigationEnv(difficulty="medium")
# model = PPO.load(MODEL_PATH)
# obs = None

# @app.get("/")
# def root():
#     return {"status": "healthy"}

# @app.post("/reset")
# def reset():
#     global obs
#     obs, _ = env.reset()
#     return {"observation": obs.tolist()}

# @app.post("/step")
# def step():
#     global obs
#     action, _ = model.predict(obs)
#     obs, reward, done, _, _ = env.step(int(action))
#     return {"observation": obs.tolist(), "reward": reward, "done": done}

# from fastapi import FastAPI
# from env import IrrigationEnv
# from stable_baselines3 import PPO
# import os
# import uvicorn

# app = FastAPI()

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, "model.zip")

# env = IrrigationEnv(difficulty="medium")
# model = PPO.load(MODEL_PATH)
# obs = None

# @app.get("/")
# def root():
#     return {"status": "healthy"}

# @app.post("/reset")
# def reset():
#     global obs
#     obs, _ = env.reset()
#     return {"observation": obs.tolist()}

# @app.post("/step")
# def step():
#     global obs
#     action, _ = model.predict(obs)
#     obs, reward, done, _, _ = env.step(int(action))
#     return {"observation": obs.tolist(), "reward": reward, "done": done}

# # ⬇️⬇️⬇️ ADD THIS PART (was missing) ⬇️⬇️⬇️
# def main():
#     uvicorn.run(app, host="0.0.0.0", port=7860)

# if __name__ == "__main__":
#     main()

from fastapi import FastAPI
from env import IrrigationEnv
from stable_baselines3 import PPO
import os
import uvicorn

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.zip")

# Initialize globally
env = IrrigationEnv(difficulty="medium")
model = PPO.load(MODEL_PATH)
obs = None

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