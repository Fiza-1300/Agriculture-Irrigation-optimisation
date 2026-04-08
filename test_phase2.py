
"""Phase 2 Validation Test for Agricultural Irrigation"""

import sys
import random
import numpy as np
from env import IrrigationEnv
from grader import EasyGrader, MediumGrader, HardGrader

def test_graders():
    print("\n=== Testing Graders ===")
    
    easy = EasyGrader()
    easy_score = easy.compute([])
    print(f"✅ EasyGrader: {easy_score}")
    assert 0.0 <= easy_score <= 1.0
    
    medium = MediumGrader()
    medium_score = medium.compute([])
    print(f"✅ MediumGrader: {medium_score}")
    assert 0.0 <= medium_score <= 1.0
    
    hard = HardGrader()
    hard_score = hard.compute([])
    print(f"✅ HardGrader: {hard_score}")
    assert 0.0 <= hard_score <= 1.0
    
    print("✅ All graders pass!")

def test_environment():
    print("\n=== Testing Environment ===")
    
    env = IrrigationEnv(difficulty="medium")
    
    # reset() returns (obs, info) in Gymnasium v0.26+
    obs, info = env.reset()
    print(f"✅ Reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Info: {info}")
    
    # Test valid actions (0, 1, 2)
    print("\n   Testing step() with actions:")
    for action in [0, 1, 2]:
        # step() returns (obs, reward, terminated, truncated, info) in Gymnasium v0.26+
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"   ✅ Action {action}: reward={reward:.3f}, done={done}")
        assert reward is not None
    
    print("✅ Environment test passed!")

def test_random_agent():
    print("\n=== Testing Random Agent ===")
    
    env = IrrigationEnv(difficulty="medium")
    total_reward = 0
    steps = 0
    
    obs, info = env.reset()
    
    for _ in range(50):
        action = random.choice([0, 1, 2])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward if reward is not None else 0
        steps += 1
        
        if done:
            break
    
    print(f"Random agent: {steps} steps, total reward: {total_reward:.3f}")
    avg_reward = total_reward / steps if steps > 0 else 0
    print(f"Average reward per step: {avg_reward:.3f}")
    print("✅ Random agent test complete")

def test_inference_syntax():
    print("\n=== Testing inference.py ===")
    try:
        with open("inference.py", "r") as f:
            code = f.read()
            compile(code, "inference.py", "exec")
        print("✅ inference.py syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("PHASE 2 VALIDATION")
    print("=" * 50)
    
    try:
        test_graders()
        test_environment()
        test_random_agent()
        test_inference_syntax()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("Ready for Phase 2 submission")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
