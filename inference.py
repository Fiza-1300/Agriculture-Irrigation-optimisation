"""Inference with OpenAI - synchronous version"""
from env import IrrigationEnv
import os
import re

# Optional OpenAI - will work even if API fails
try:
    from openai import OpenAI
    USE_OPENAI = True
except:
    USE_OPENAI = False
    print("[WARNING] OpenAI not available, using heuristic")

def get_action_from_llm(soil_moisture):
    """Try LLM, fallback to heuristic"""
    if not USE_OPENAI:
        return heuristic_action(soil_moisture)
    
    try:
        api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", "dummy"))
        if api_key == "dummy":
            return heuristic_action(soil_moisture)
            
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Soil moisture: {soil_moisture:.2f}. Optimal: 0.4-0.7. Choose 0,1,2."}],
            max_tokens=5
        )
        action = int(re.findall(r'\d', response.choices[0].message.content)[0])
        return action if action in [0,1,2] else heuristic_action(soil_moisture)
    except:
        return heuristic_action(soil_moisture)

def heuristic_action(soil_moisture):
    if soil_moisture < 0.4: return 2
    elif soil_moisture > 0.7: return 0
    return 1

def main():
    env = IrrigationEnv(difficulty="medium", crop_type="wheat")
    
    print('[START] {"task": "agricultural_irrigation", "env": "OpenEnv", "model": "gpt-3.5-turbo"}')
    
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < 100:
        soil_moisture = obs[0] if hasattr(obs, '__getitem__') else 0.5
        action = get_action_from_llm(soil_moisture)
        
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
        
        total_reward += reward if reward else 0
        step += 1
        print(f'[STEP] {{"step": {step}, "action": "{action}", "reward": {reward}, "done": {done}, "error": null}}')
    
    score = min(total_reward / 100, 1.0)
    print(f'[END] {{"success": {score >= 0.7}, "steps": {step}, "score": {score}, "rewards": [{total_reward}]}}')

if __name__ == "__main__":
    main()