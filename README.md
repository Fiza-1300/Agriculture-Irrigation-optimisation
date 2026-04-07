# 🌱 Smart Agricultural Irrigation using Reinforcement Learning

## 📌 Overview
This project implements a Reinforcement Learning (RL) based system to optimize agricultural irrigation. The agent learns to control water usage intelligently based on environmental conditions, ensuring efficient water use while maintaining optimal crop health.

---

## 🎯 Problem Statement
Traditional irrigation methods often lead to:
- Overwatering ❌ (wasting water, damaging crops)
- Underwatering ❌ (reducing yield)

This project solves this by training an RL agent to:
- Decide **when to irrigate**
- Decide **how much water to use**

---

## 🧠 Solution Approach

We designed a custom RL environment where the agent observes:

### 📊 State Space
- Soil moisture
- Temperature
- Humidity
- Rainfall condition
- Crop growth stage

### 🎮 Action Space
- `0` → No irrigation
- `1` → Low irrigation
- `2` → High irrigation

### 🎯 Reward Function
- Positive reward for optimal moisture
- Penalty for:
  - Overwatering
  - Underwatering
  - Wasting water
  - Irrigating during rain

---

## 🧪 Environment Design

The system includes **3 difficulty levels**:

| Level  | Description |
|--------|------------|
| Easy   | Stable environment, fixed crop stage |
| Medium | Random weather, changing crop stages |
| Hard   | High variability, long-term planning required |

---

## 🤖 Model

- Algorithm: **PPO (Proximal Policy Optimization)**
- Library: Stable-Baselines3
- Trained to maximize long-term reward and efficiency

---

## 📈 Results
Easy Reward : 1885+
Medium Reward : 1708+
Hard Reward : 993+
Final Score : 0.887


✅ Strong performance across all difficulty levels  
✅ Good generalization in dynamic environments  

---

## 🛠️ Project Structure
├── env.py # Custom RL environment
├── train.py # Training script
├── inference.py # Model inference (entry point)
├── grader.py # Evaluation script
├── test_model.py # Test cases
├── model.zip # Trained model
├── Dockerfile # Container setup
├── Openenv.yaml # Hackathon config
├── requirements.txt # Dependencies


---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Run inference
python inference.py
3️⃣ Run evaluation
python grader.py
🐳 Docker Usage
Build image
docker build -t irrigation-ai .
Run container
docker run irrigation-ai
🌍 Real-World Impact

This system can help:

Save water 💧
Improve crop yield 🌾
Enable smart farming 🚜
Support sustainable agriculture 🌱