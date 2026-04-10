
"""Baseline inference script for Agricultural Irrigation Environment - FIXED FOR EVALUATOR"""

import asyncio
import os
import re
import sys
from openai import OpenAI

# CRITICAL: Use evaluator's environment variables - DO NOT hardcode!
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")  # Note: It's API_KEY, not OPENAI_API_KEY
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Validate that we're using the evaluator's proxy
if not API_BASE_URL or not API_KEY:
    print("[ERROR] Missing API_BASE_URL or API_KEY environment variables", flush=True)
    print("[ERROR] This script MUST run with evaluator's proxy", flush=True)
    sys.exit(1)

print(f"[INFO] Using API_BASE_URL: {API_BASE_URL}", flush=True)
print(f"[INFO] Using MODEL_NAME: {MODEL_NAME}", flush=True)

from env import IrrigationEnv

TASK_NAME = "agricultural_irrigation"
BENCHMARK = "OpenEnv"
MAX_STEPS = 50
MAX_TOTAL_REWARD = 100
SUCCESS_SCORE_THRESHOLD = 0.7

def log_start(task, env, model):
    """Emit start log in required format"""
    print(f'[START] {{"task": "{task}", "env": "{env}", "model": "{model}"}}', flush=True)

def log_step(step, action, reward, done, error):
    """Emit step log in required format"""
    error_str = f'"{error}"' if error else 'null'
    print(f'[STEP] {{"step": {step}, "action": "{action}", "reward": {reward}, "done": {done}, "error": {error_str}}}', flush=True)

def log_end(success, steps, score, rewards):
    """Emit end log in required format"""
    print(f'[END] {{"success": {success}, "steps": {steps}, "score": {score}, "rewards": {rewards}}}', flush=True)

def get_model_message(client, step, observation, last_reward, history):
    """Get action from evaluator's LLM proxy"""
    try:
        # Extract soil moisture from observation
        if hasattr(observation, '__getitem__'):
            soil_moisture = observation[0] if len(observation) > 0 else 0.5
        else:
            soil_moisture = 0.5
        
        prompt = f"""You are controlling an agricultural irrigation system.
Current step: {step}
Soil moisture: {soil_moisture:.3f} (optimal range: 0.40-0.70)
Last reward: {last_reward:.3f}

Choose an action:
- 0: No irrigation (save water)
- 1: Low irrigation (gentle watering)  
- 2: High irrigation (heavy watering)

Strategy:
- If soil moisture < 0.40: use action 2 (high irrigation)
- If soil moisture > 0.70: use action 0 (no irrigation)
- If soil moisture between 0.40-0.70: use action 1 (low irrigation)

Respond with ONLY the number (0, 1, or 2)."""
        
        # CRITICAL: Use the client with evaluator's base_url and api_key
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.7
        )
        action_str = response.choices[0].message.content.strip()
        
        # Extract first number found
        numbers = re.findall(r'\d', action_str)
        if numbers:
            action = int(numbers[0])
            if action in [0, 1, 2]:
                return action
        
        # Fallback heuristic based on soil moisture
        if soil_moisture < 0.40:
            return 2
        elif soil_moisture > 0.70:
            return 0
        return 1
        
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", flush=True)
        # Fallback heuristic
        if soil_moisture < 0.40:
            return 2
        elif soil_moisture > 0.70:
            return 0
        return 1

async def main():
    """Main inference loop - MUST use evaluator's API proxy"""
    
    # CRITICAL: Initialize OpenAI client with evaluator's environment variables
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )
    
    env = IrrigationEnv(difficulty="medium")
    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Gymnasium v0.26+: reset() returns (obs, info)
        obs, info = env.reset()
        last_reward = 0.0
        
        for step in range(1, MAX_STEPS + 1):
            # Get action from model using evaluator's proxy
            action = get_model_message(client, step, obs, last_reward, history)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Handle reward if it's None
            reward = reward if reward is not None else 0.0
            
            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            
            log_step(step=step, action=str(action), reward=reward, done=done, error=None)
            
            history.append(f"Step {step}: action={action} -> reward {reward:+.3f}")
            
            if done:
                break
        
        # Calculate final score
        total_reward = sum(rewards)
        score = min(total_reward / MAX_TOTAL_REWARD, 1.0) if MAX_TOTAL_REWARD > 0 else 0.0
        score = max(0.0, score)
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        print(f"[DEBUG] Error in main: {e}", flush=True)
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        # Ensure we close any open connections
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
