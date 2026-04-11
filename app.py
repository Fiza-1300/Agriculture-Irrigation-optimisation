"""FastAPI app for Hugging Face Space deployment"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
from env import IrrigationEnv

app = FastAPI(title="Agricultural Irrigation Environment")

# Global environment instance
env = None

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "medium"
    crop_type: Optional[str] = "wheat"

class StepRequest(BaseModel):
    action: int

class ObservationResponse(BaseModel):
    observation: list
    info: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    global env
    env = IrrigationEnv(difficulty="medium", crop_type="wheat")

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Agricultural Irrigation Environment"}

@app.post("/reset")
async def reset_env(request: ResetRequest = ResetRequest()):
    global env
    env = IrrigationEnv(difficulty=request.difficulty, crop_type=request.crop_type)
    obs, info = env.reset()
    return ObservationResponse(observation=obs.tolist(), info=info)

@app.post("/step")
async def step_env(request: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    obs, reward, done, truncated, info = env.step(request.action)
    return {
        "observation": obs.tolist(),
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "info": info
    }

@app.get("/state")
async def get_state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")
    return env.state()