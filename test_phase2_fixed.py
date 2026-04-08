cat > test_phase2_fixed.py << 'EOF'
import sys
import random
import env
import grader

print("=" * 60)
print("PHASE 2 VALIDATION - DIAGNOSTIC")
print("=" * 60)

# Step 1: Find the environment class
print("\n1. Looking for Environment class in env.py...")
env_classes = []
for name in dir(env):
    if "Env" in name and not name.startswith("_"):
        env_classes.append(name)
        print(f"   Found: {name}")

if not env_classes:
    print("   ❌ No class with 'Env' found in env.py")
    print("   Available items:", [x for x in dir(env) if not x.startswith("_")])
    sys.exit(1)

# Use the first found environment class
EnvClass = getattr(env, env_classes[0])
print(f"✅ Using environment class: {env_classes[0]}")

# Step 2: Find grader classes
print("\n2. Looking for Grader classes in grader.py...")
grader_classes = []
for name in dir(grader):
    if "Grader" in name and not name.startswith("_"):
        grader_classes.append(name)
        print(f"   Found: {name}")

if len(grader_classes) < 3:
    print(f"⚠️ Expected 3 graders, found {len(grader_classes)}")
else:
    print(f"✅ Found {len(grader_classes)} graders")

# Step 3: Test environment initialization
print("\n3. Testing environment reset()...")
try:
    env_instance = EnvClass()
    obs = env_instance.reset()
    print(f"✅ reset() successful")
    print(f"   Observation type: {type(obs)}")
except Exception as e:
    print(f"❌ reset() failed: {e}")
    sys.exit(1)

# Step 4: Test random actions
print("\n4. Testing random actions...")
try:
    # Try to get action space if available
    if hasattr(env_instance, 'action_space'):
        print(f"   Action space: {env_instance.action_space}")
    
    # Try some common action types
    test_actions = [0, 1, 2, "irrigate", "wait", "stop", 0.5]
    total_reward = 0
    
    for i, action in enumerate(test_actions[:5]):  # Try first 5 actions
        try:
            obs, reward, done, info = env_instance.step(action)
            total_reward += reward
            print(f"   Step {i+1}: Action={action} -> Reward={reward}")
            if done:
                print(f"   Episode done at step {i+1}")
                break
        except Exception as e:
            print(f"   ⚠️ Action {action} failed: {e}")
    
    print(f"✅ Random test complete. Total reward: {total_reward}")
except Exception as e:
    print(f"❌ Step test failed: {e}")

# Step 5: Test graders
print("\n5. Testing graders...")
for grader_name in grader_classes[:3]:  # Test first 3 graders
    try:
        GraderClass = getattr(grader, grader_name)
        grader_instance = GraderClass()
        
        # Try to find the scoring method
        if hasattr(grader_instance, 'compute'):
            score = grader_instance.compute([])
            print(f"   {grader_name}.compute() -> {score}")
        elif hasattr(grader_instance, 'grade'):
            score = grader_instance.grade([])
            print(f"   {grader_name}.grade() -> {score}")
        elif hasattr(grader_instance, 'score'):
            score = grader_instance.score([])
            print(f"   {grader_name}.score() -> {score}")
        else:
            print(f"   ⚠️ {grader_name} has no scoring method")
            print(f"      Methods: {[x for x in dir(grader_instance) if not x.startswith('_')]}")
    except Exception as e:
        print(f"   ❌ {grader_name} failed: {e}")

# Step 6: Check required files
print("\n6. Checking required files...")
import os
required_files = ['inference.py', 'openenv.yaml', 'Dockerfile']
for file in required_files:
    if os.path.exists(file):
        print(f"   ✅ {file} exists")
    else:
        print(f"   ❌ {file} MISSING")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
EOF

# Run the diagnostic
