from Env import IrrigationEnv
from stable_baselines3 import PPO

def evaluate(difficulty):
    env = IrrigationEnv(difficulty=difficulty)
    model = PPO.load("model.zip")

    obs, _ = env.reset()
    total_reward = 0

    done = False

    while not done:
        action, _ = model.predict(obs)
        action = int(action)

        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    return total_reward


def grade():
    difficulties = ["easy", "medium", "hard"]
    scores = []

    for level in difficulties:
        reward = evaluate(level)

        # Normalize reward → score (0 to 1)
        score = max(0, min(1, reward / 1500))
        scores.append(score)

        print(f"{level} reward:", reward)

    final_score = sum(scores) / len(scores)
    return final_score


if __name__ == "__main__":
    print("\nFinal Score:", grade())