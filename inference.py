# from fastapi import FastAPI
# from env import IrrigationEnv
# from stable_baselines3 import PPO
# import uvicorn
# import os
# import sys
# import requests
# import time

# app = FastAPI()

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "model.zip")

# env = None
# model = None
# obs = None

# @app.on_event("startup")
# async def startup_event():
#     global env, model, obs
    
#     print("Starting up...")
#     print(f"Current directory: {os.getcwd()}")
#     print(f"Files: {os.listdir('.')}")
    
#     env = IrrigationEnv(difficulty="medium")
#     print("✅ Environment loaded")
    
#     print(f"Loading model from {MODEL_PATH}...")
#     model = PPO.load(MODEL_PATH)
#     print("✅ Model loaded")
    
#     obs, _ = env.reset()
#     print("✅ Ready!")

# @app.get("/")
# def root():
#     return {"message": "Irrigation AI API is running", "status": "healthy"}

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
#     obs, reward, done, _, info = env.step(action)
#     return {
#         "action": action,
#         "reward": float(reward),
#         "done": done,
#         "observation": obs.tolist()
#     }

# def main():
#     uvicorn.run(app, host="0.0.0.0", port=7860)

# if __name__ == "__main__":
#     main()
    
# # Wait for server
# for i in range(30):
#     try:
#         requests.get("http://localhost:7860/")
#         print("Server ready")
#         break
#     except:
#         print(f"Waiting... {i+1}/30")
#         time.sleep(2)

# # Test the API
# requests.post("http://localhost:7860/reset")
# result = requests.post("http://localhost:7860/step")
# print(result.json())


import requests
import time
import sys

def main():
    # Wait for Docker container to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:7860/", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready")
                break
        except:
            print(f"⏳ Waiting for server... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("❌ Server failed to start")
        sys.exit(1)
    
    # Run a test episode
    print("🔄 Resetting environment...")
    reset_response = requests.post("http://localhost:7860/reset")
    print(f"Reset result: {reset_response.json()}")
    
    print("🚀 Taking 5 test steps...")
    for step in range(5):
        step_response = requests.post("http://localhost:7860/step")
        data = step_response.json()
        print(f"Step {step+1}: reward={data.get('reward', 'N/A')}, done={data.get('done', False)}")
    
    print("✅ inference.py completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())