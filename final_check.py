
print("=" * 50)
print("FINAL PHASE 2 CHECK")
print("=" * 50)

# Check 1: Imports
try:
    from env import IrrigationEnv
    print("✅ env.py imports OK")
except Exception as e:
    print(f"❌ env.py: {e}")

try:
    from grader import EasyGrader, MediumGrader, HardGrader
    print("✅ grader.py imports OK")
except Exception as e:
    print(f"❌ grader.py: {e}")

# Check 2: Environment initialization
try:
    env = IrrigationEnv()
    obs, info = env.reset()
    print(f"✅ Environment reset OK (obs shape: {obs.shape})")
except Exception as e:
    print(f"❌ Environment: {e}")

# Check 3: Step function
try:
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"✅ Step function OK (reward: {reward})")
except Exception as e:
    print(f"❌ Step: {e}")

# Check 4: Graders
try:
    easy = EasyGrader()
    score = easy.compute([])
    print(f"✅ EasyGrader OK (score: {score})")
    
    medium = MediumGrader()
    score = medium.compute([])
    print(f"✅ MediumGrader OK (score: {score})")
    
    hard = HardGrader()
    score = hard.compute([])
    print(f"✅ HardGrader OK (score: {score})")
except Exception as e:
    print(f"❌ Graders: {e}")

# Check 5: Required files
required = ["inference.py", "openenv.yaml", "Dockerfile", "requirements.txt"]
for f in required:
    import os
    if os.path.exists(f):
        print(f"✅ {f} exists")
    else:
        print(f"❌ {f} MISSING")

print("=" * 50)
print("Check complete!")
print("=" * 50)
