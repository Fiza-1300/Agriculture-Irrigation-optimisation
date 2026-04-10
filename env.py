# """
# env.py — Agricultural Irrigation Environment
# Person A Deliverable | RL Irrigation Optimization Project

# Defines how the farm world behaves when an action is taken.
# """

# import numpy as np

# np.random.seed(42)
# import gymnasium as gym
# from gymnasium import spaces


# class IrrigationEnv(gym.Env):
#     """
#     Custom Reinforcement Learning Environment for Agricultural Irrigation.

#     State  : [soil_moisture, temperature, humidity, rain, crop_stage]
#     Actions: 0 = no water | 1 = low water | 2 = high water
#     """

#     metadata = {"render.modes": ["human"]}

#     # ─────────────────────────────────────────────
#     # IRRIGATION AMOUNTS per action (in moisture units)
#     # ─────────────────────────────────────────────
#     IRRIGATION_AMOUNT = {
#         0: 0.00,   # No irrigation
#         1: 0.08,   # Low irrigation
#         2: 0.18,   # High irrigation
#     }

#     # ─────────────────────────────────────────────
#     # OPTIMAL SOIL MOISTURE RANGE
#     # ─────────────────────────────────────────────
#     OPTIMAL_LOW  = 0.40
#     OPTIMAL_HIGH = 0.70

#     def __init__(self, difficulty: str = "medium"):
    
#         """
#         Args:
#             difficulty: "easy" | "medium" | "hard"
#         """
#         super(IrrigationEnv, self).__init__()

#         assert difficulty in ("easy", "medium", "hard"), \
#             "difficulty must be 'easy', 'medium', or 'hard'"

#         self.difficulty = difficulty

#         # ── Difficulty Configuration ──────────────────────────────────────
#         self._apply_difficulty_config()

#         # ── Action Space ──────────────────────────────────────────────────
#         # Discrete: 0 (none), 1 (low), 2 (high)
#         self.action_space = spaces.Discrete(3)

#         # ── Observation Space ─────────────────────────────────────────────
#         # [soil_moisture, temperature, humidity, rain, crop_stage]
#         # All values normalised to [0, 1] except temperature (0–1 scaled)
#         low  = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
#         high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
#         self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

#         # Internal counters
#         self.max_steps    = 168   # 1 week in hourly steps
#         self.current_step = 0

#         # State variables (initialised in reset)
#         self.soil_moisture = None
#         self.temperature   = None   # stored as 0-1 scaled value
#         self.humidity      = None
#         self.rain          = None
#         self.crop_stage    = None

#     # ─────────────────────────────────────────────────────────────────────
#     # DIFFICULTY CONFIGURATION
#     # ─────────────────────────────────────────────────────────────────────
#     def _apply_difficulty_config(self):
#         """
#         Sets all difficulty-dependent parameters.
#         Easy   → predictable, low stress
#         Medium → realistic, moderate randomness
#         Hard   → extreme, high stress, noisy sensors
#         """
#         if self.difficulty == "easy":
#             self.rain_probability     = 0.05    # 5% chance of rain each step
#             self.evaporation_rate     = 0.01    # slow drying
#             self.temp_range           = (20, 30) # mild temperature (°C)
#             self.humidity_range       = (40, 60)
#             self.sensor_noise_std     = 0.00    # perfect sensors
#             self.drought_event_prob   = 0.00    # no drought
#             self.rain_intensity_range = (0.05, 0.10)

#         elif self.difficulty == "medium":
#             self.rain_probability     = 0.15
#             self.evaporation_rate     = 0.02
#             self.temp_range           = (25, 40)
#             self.humidity_range       = (30, 70)
#             self.sensor_noise_std     = 0.01    # small noise
#             self.drought_event_prob   = 0.05    # 5% chance of drought spell
#             self.rain_intensity_range = (0.05, 0.20)

#         elif self.difficulty == "hard":
#             self.rain_probability     = 0.25    # frequent but unpredictable
#             self.evaporation_rate     = 0.04    # fast drying
#             self.temp_range           = (35, 45) # extreme heat
#             self.humidity_range       = (10, 90) # wide swings
#             self.sensor_noise_std     = 0.03    # noisy sensors
#             self.drought_event_prob   = 0.15    # frequent droughts
#             self.rain_intensity_range = (0.02, 0.30)

#         # Drought flag — activated stochastically in hard/medium
#         self.in_drought = False

#     # ─────────────────────────────────────────────────────────────────────
#     # RESET
#     # ─────────────────────────────────────────────────────────────────────
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         """
#         Initialise (or reinitialise) environment to a fresh episode start.

#         Returns:
#             state (np.ndarray): initial observation of shape (5,)
#         """
#         self.current_step = 0
#         self.in_drought   = False

#         # Start with slightly sub-optimal moisture so agent must act
#         self.soil_moisture = np.random.uniform(0.30, 0.50)

#         # Temperature: scaled to [0, 1] from temp_range
#         raw_temp           = np.random.uniform(*self.temp_range)
#         self.temperature   = self._scale_temperature(raw_temp)

#         self.humidity      = np.random.uniform(*self.humidity_range) / 100.0
#         self.rain          = 0.0   # no rain at episode start
#         self.crop_stage    = 0.0   # seedling at episode start

#         return self.get_state(), {}

#     # ─────────────────────────────────────────────────────────────────────
#     # STEP
#     # ─────────────────────────────────────────────────────────────────────
#     def step(self, action: int):
#         """
#         Apply one action and advance environment by one timestep (1 hour).

#         Args:
#             action (int): 0 = no water | 1 = low water | 2 = high water

#         Returns:
#             state  (np.ndarray) : new observation  shape (5,)
#             reward (float)      : reward for this timestep
#             done   (bool)       : True if episode is over
#             info   (dict)       : diagnostic information
#         """
#         assert self.action_space.contains(action), f"Invalid action: {action}"

#         # ── 1. Apply Irrigation ───────────────────────────────────────────
#         water_added = self.IRRIGATION_AMOUNT[action]
#         irrigated   = action > 0

#         # ── 2. Simulate Rain ──────────────────────────────────────────────
#         self.rain = self._simulate_rain()

#         # ── 3. Update Soil Moisture ───────────────────────────────────────
#         prev_moisture        = self.soil_moisture
#         self.soil_moisture   = self._update_soil_moisture(water_added)

#         # ── 4. Update Weather ─────────────────────────────────────────────
#         self._update_weather()

#         # ── 5. Advance Crop Stage ─────────────────────────────────────────
#         self.crop_stage = min(1.0, self.crop_stage + (1.0 / self.max_steps))

#         # ── 6. Calculate Reward ───────────────────────────────────────────
#         reward = self._calculate_reward(
#             action      = action,
#             irrigated   = irrigated,
#             water_added = water_added,
#             prev_moisture = prev_moisture
#         )

#         # ── 7. Advance Step Counter ───────────────────────────────────────
#         self.current_step += 1
#         done = self.current_step >= self.max_steps

#         # ── 8. Build Info Dict ────────────────────────────────────────────
#         info = {
#             "step"          : self.current_step,
#             "soil_moisture" : round(self.soil_moisture, 4),
#             "rain"          : round(self.rain, 4),
#             "temperature"   : round(self.temperature, 4),
#             "humidity"      : round(self.humidity, 4),
#             "crop_stage"    : round(self.crop_stage, 4),
#             "water_added"   : water_added,
#             "reward"        : reward,
#             "in_drought"    : self.in_drought,
#             "difficulty"    : self.difficulty,
#         }

#         return self.get_state(), reward, done, False, info

#     # ─────────────────────────────────────────────────────────────────────
#     # GET STATE
#     # ─────────────────────────────────────────────────────────────────────
#     def get_state(self) -> np.ndarray:
#         """
#         Return current environment state as a numpy array.

#         Format (all values in [0, 1]):
#             [soil_moisture, temperature, humidity, rain, crop_stage]

#         Returns:
#             np.ndarray of shape (5,) dtype float32
#         """
#         state = np.array([
#             self.soil_moisture,
#             self.temperature,
#             self.humidity,
#             self.rain,
#             self.crop_stage,
#         ], dtype=np.float32)

#         # Add sensor noise on medium/hard to simulate real-world imperfection
#         if self.sensor_noise_std > 0.0:
#             noise = np.random.normal(0.0, self.sensor_noise_std, size=state.shape)
#             state = state + noise

#         # Clip to valid range after noise
#         state = np.clip(state, 0.0, 1.0)
#         return state

#     # ─────────────────────────────────────────────────────────────────────
#     # REWARD FUNCTION  ← CORE LOGIC (Everything depends on this)
#     # ─────────────────────────────────────────────────────────────────────
#     def _calculate_reward(
#         self,
#         action: int,
#         irrigated: bool,
#         water_added: float,
#         prev_moisture: float
#     ) -> float:
#         """
#         Calculates the reward for one timestep.

#         Design philosophy:
#           ✔ Reward being in optimal moisture band
#           ✔ Penalise extremes (too dry / too wet) harshly
#           ✔ Penalise water waste (irrigating when not needed)
#           ✔ Penalise irrigation during rain (wasteful)
#           ✔ Scale penalties with crop stage (errors more costly near harvest)
#           ✔ Reward efficiency (staying optimal WITHOUT irrigating)

#         Returns:
#             reward (float)
#         """
#         reward = 0.0
#         m      = self.soil_moisture          # current moisture
#         stage  = self.crop_stage             # 0 (seedling) → 1 (harvest)

#         # ── BLOCK 1: Moisture Quality Reward ─────────────────────────────
#         # Core signal: how close is moisture to the optimal band?

#         if self.OPTIMAL_LOW <= m <= self.OPTIMAL_HIGH:
#             # ✅ In optimal range
#             # Bonus if agent achieved this WITHOUT irrigating (efficient)
#             if not irrigated:
#                 reward += 12.0   # Efficiency bonus
#             else:
#                 reward += 10.0   # Still good, but used water

#         elif m < self.OPTIMAL_LOW:
#             # ❌ Too dry — penalty scales with how dry
#             dryness = self.OPTIMAL_LOW - m          # 0 → 0.4
#             penalty = dryness * 30.0                # max ~12 penalty
#             # Extra penalty near harvest (crop loss is catastrophic)
#             harvest_multiplier = 1.0 + stage * 0.5  # 1.0 → 1.5
#             reward -= penalty * harvest_multiplier

#         elif m > self.OPTIMAL_HIGH:
#             # ❌ Too wet — root rot, oxygen deprivation
#             wetness = m - self.OPTIMAL_HIGH          # 0 → 0.3
#             penalty = wetness * 25.0                 # max ~7.5 penalty
#             reward -= penalty

#         # ── BLOCK 2: Critical Threshold Penalty ──────────────────────────
#         # Hard floor / ceiling — severe penalties at extremes

#         if m < 0.15:
#             # 🚨 Crop dying — extreme drought
#             reward -= 15.0
#         elif m > 0.92:
#             # 🚨 Waterlogged — irreversible damage
#             reward -= 12.0

#         # ── BLOCK 3: Irrigation During Rain Penalty ───────────────────────
#         # Irrigating while it's raining = pure waste

#         if irrigated and self.rain > 0.05:
#             # Penalty scales with how much it's raining AND how much water added
#             waste_penalty = self.rain * water_added * 40.0
#             reward -= waste_penalty

#         # ── BLOCK 4: Over-Watering Penalty ───────────────────────────────
#         # Irrigating when soil is already wet enough

#         if irrigated and prev_moisture > self.OPTIMAL_HIGH:
#             # Moisture was already above optimal before this action
#             excess = water_added * 20.0
#             reward -= excess

#         # ── BLOCK 5: Unnecessary Irrigation Penalty ───────────────────────
#         # Irrigating when moisture is comfortably in range wastes resources

#         if irrigated and self.OPTIMAL_LOW <= prev_moisture <= self.OPTIMAL_HIGH:
#             # Small penalty — wasteful but not catastrophic
#             reward -= water_added * 8.0

#         # ── BLOCK 6: High Irrigation Efficiency Check ─────────────────────
#         # Using action=2 (high water) when low would have sufficed

#         if action == 2 and prev_moisture > 0.30:
#             # Soil wasn't critically dry — high water was excessive
#             reward -= 2.0

#         # ── BLOCK 7: Drought Bonus ────────────────────────────────────────
#         # Correctly irrigating during a drought event gets bonus

#         if self.in_drought and irrigated and m < self.OPTIMAL_LOW:
#             reward += 5.0   # Good decision under stress

#         # ── Final clip: keep reward in reasonable range ───────────────────
#         reward = float(np.clip(reward, -30.0, 15.0))

#         return reward

#     # ─────────────────────────────────────────────────────────────────────
#     # INTERNAL HELPERS
#     # ─────────────────────────────────────────────────────────────────────
#     def _simulate_rain(self) -> float:
#         """
#         Returns rainfall intensity for this timestep (0 = no rain).
#         Rain is a random event based on difficulty's rain_probability.
#         """
#         # Stochastic drought: suppresses rain for a stretch
#         if np.random.random() < self.drought_event_prob:
#             self.in_drought = True

#         if self.in_drought:
#             # Drought ends probabilistically
#             if np.random.random() < 0.10:
#                 self.in_drought = False
#             return 0.0  # No rain during drought

#         if np.random.random() < self.rain_probability:
#             intensity = np.random.uniform(*self.rain_intensity_range)
#             return float(np.clip(intensity, 0.0, 1.0))

#         return 0.0

#     def _update_soil_moisture(self, water_added: float) -> float:
#         """
#         Computes new soil moisture after applying:
#           + water_added (irrigation)
#           + rain
#           - evaporation (driven by temperature)
#           - crop absorption (driven by crop stage)
#         """
#         # Evaporation increases with temperature
#         evaporation = self.evaporation_rate * (0.5 + self.temperature)

#         # Crops absorb more water as they grow
#         crop_absorption = 0.005 * (0.5 + self.crop_stage)

#         new_moisture = (
#             self.soil_moisture
#             + water_added
#             + self.rain
#             - evaporation
#             - crop_absorption
#         )

#         return float(np.clip(new_moisture, 0.0, 1.0))

#     def _update_weather(self):
#         """
#         Randomly drift temperature and humidity each step to simulate
#         realistic weather fluctuation.
#         """
#         # Temperature drift (small random walk)
#         temp_delta       = np.random.uniform(-0.02, 0.02)
#         self.temperature = float(np.clip(self.temperature + temp_delta, 0.0, 1.0))

#         # Humidity drift
#         hum_delta     = np.random.uniform(-0.03, 0.03)
#         self.humidity = float(np.clip(self.humidity + hum_delta, 0.0, 1.0))

#     def _scale_temperature(self, raw_temp: float) -> float:
#         """
#         Scales raw temperature (°C) to [0, 1] using global min/max of 0–50°C.
#         """
#         return float(np.clip(raw_temp / 50.0, 0.0, 1.0))

#     # ─────────────────────────────────────────────────────────────────────
#     # RENDER (Optional — for debugging)
#     # ─────────────────────────────────────────────────────────────────────
#     def render(self, mode="human"):
#         m = self.soil_moisture
#         status = (
#             "✅ OPTIMAL"  if self.OPTIMAL_LOW <= m <= self.OPTIMAL_HIGH else
#             "🔴 TOO DRY"  if m < self.OPTIMAL_LOW else
#             "🔵 TOO WET"
#         )
#         print(
#             f"Step {self.current_step:>3} | "
#             f"Moisture: {m:.3f} {status} | "
#             f"Temp: {self.temperature:.2f} | "
#             f"Humidity: {self.humidity:.2f} | "
#             f"Rain: {self.rain:.3f} | "
#             f"Crop: {self.crop_stage:.2f} | "
#             f"Drought: {self.in_drought}"
#         )

#     def close(self):
#         pass
    
#     # Add this to your IrrigationEnv class
# CROP_TYPES = {
#     "wheat": {
#         "optimal_low": 0.35,
#         "optimal_high": 0.55,
#         "water_sensitivity": 0.8,  # How much water affects yield
#         "growth_days": 120,
#         "price_per_kg": 25  # Rupees
#     },
#     "rice": {
#         "optimal_low": 0.50,
#         "optimal_high": 0.80,
#         "water_sensitivity": 1.2,
#         "growth_days": 150,
#         "price_per_kg": 35
#     },
#     "corn": {
#         "optimal_low": 0.40,
#         "optimal_high": 0.65,
#         "water_sensitivity": 1.0,
#         "growth_days": 100,
#         "price_per_kg": 30
#     }
# }

# def __init__(self, difficulty="medium", crop_type="wheat"):
#     super().__init__(difficulty)
#     self.crop_type = crop_type
#     self.crop_data = self.CROP_TYPES[crop_type]
#     # Update optimal ranges based on crop
#     self.OPTIMAL_LOW = self.crop_data["optimal_low"]
#     self.OPTIMAL_HIGH = self.crop_data["optimal_high"]


# def __init__(self, difficulty="medium", crop_type="wheat"):
#     # ... existing code ...
#     self.water_price_per_unit = 5  # Base price in rupees
#     self.time_of_day = 0  # 0-23 hours
#     self.water_usage_total = 0
#     self.water_bill = 0

# def _calculate_water_cost(self, water_amount):
#     """Dynamic pricing based on time of day"""
#     # Peak hours (6-9 AM, 5-8 PM) - 50% more expensive
#     if (6 <= self.time_of_day <= 9) or (17 <= self.time_of_day <= 20):
#         price_multiplier = 1.5
#     # Night hours (10 PM - 5 AM) - 30% cheaper
#     elif self.time_of_day >= 22 or self.time_of_day <= 5:
#         price_multiplier = 0.7
#     else:
#         price_multiplier = 1.0
    
#     cost = water_amount * self.water_price_per_unit * price_multiplier
#     self.water_bill += cost
#     return cost

# def step(self, action):
#     # ... existing code ...
#     water_added = self.IRRIGATION_AMOUNT[action]
    
#     # Calculate water cost
#     water_cost = self._calculate_water_cost(water_added)
#     self.water_usage_total += water_added
    
#     # Update reward to include water cost penalty
#     reward = self._calculate_reward(water_added) - (water_cost * 0.01)
    
#     # Update time of day (each step = 1 hour)
#     self.time_of_day = (self.time_of_day + 1) % 24


# def __init__(self, difficulty="medium", crop_type="wheat"):
#     # ... existing code ...
#     self.weather_forecast_days = 3
#     self.forecast_accuracy = 0.8  # 80% accurate
#     self.actual_weather = []
#     self.forecasted_weather = []

# def _generate_weather_forecast(self):
#     """Generate forecast with uncertainty"""
#     actual_rain = self._simulate_rain()
    
#     # Forecast has uncertainty
#     if np.random.random() < self.forecast_accuracy:
#         forecast_rain = actual_rain
#     else:
#         # Wrong forecast - add error
#         forecast_rain = actual_rain + np.random.normal(0, 0.3)
#         forecast_rain = np.clip(forecast_rain, 0, 1)
    
#     return {
#         "actual": actual_rain,
#         "forecast": forecast_rain,
#         "confidence": self.forecast_accuracy
#     }

# def get_observation(self):
#     """Add forecast to observation"""
#     obs = super().get_observation()
#     forecast = self._generate_weather_forecast()
#     # Add forecast info to observation
#     return np.append(obs, [forecast["forecast"], forecast["confidence"]])

# def __init__(self, difficulty="medium", crop_type="wheat"):
#     # ... existing code ...
#     self.fertilizer_level = 0.5  # 0-1 scale
#     self.fertilizer_cost = 50  # Rupees per unit
#     self.soil_health = 0.7  # 0-1 scale
#     self.action_space = spaces.Discrete(4)  # Added fertilizer action

# def step(self, action):
#     """
#     Action mapping:
#     0: No water, no fertilizer
#     1: Low water, no fertilizer
#     2: High water, no fertilizer
#     3: Apply fertilizer
#     """
#     if action == 3:  # Fertilizer action
#         self.fertilizer_level = min(1.0, self.fertilizer_level + 0.2)
#         reward = -self.fertilizer_cost * 0.01  # Small penalty for cost
#         self.soil_health = min(1.0, self.soil_health + 0.05)
#         return self._get_state(), reward, False, {"action": "fertilizer"}
#     else:
#         # Regular irrigation actions (0, 1, 2)
#         water_amount = self.IRRIGATION_AMOUNT[action]
#         return self._handle_irrigation(water_amount)

# def _calculate_yield_boost(self):
#     """Fertilizer improves crop yield"""
#     if self.fertilizer_level > 0.7:
#         return 1.3  # 30% boost
#     elif self.fertilizer_level > 0.3:
#         return 1.1  # 10% boost
#     return 1.0

"""
env.py — Agricultural Irrigation Environment
Person A Deliverable | RL Irrigation Optimization Project

Defines how the farm world behaves when an action is taken.
"""

import numpy as np

np.random.seed(42)
import gymnasium as gym
from gymnasium import spaces


class IrrigationEnv(gym.Env):
    """
    Custom Reinforcement Learning Environment for Agricultural Irrigation.

    State  : [soil_moisture, temperature, humidity, rain, crop_stage]
    Actions: 0 = no water | 1 = low water | 2 = high water
    """

    metadata = {"render.modes": ["human"]}

    # Crop type configurations
    CROP_TYPES = {
        "wheat": {
            "optimal_low": 0.35,
            "optimal_high": 0.55,
            "water_sensitivity": 0.8,
            "growth_days": 120,
            "price_per_kg": 25
        },
        "rice": {
            "optimal_low": 0.50,
            "optimal_high": 0.80,
            "water_sensitivity": 1.2,
            "growth_days": 150,
            "price_per_kg": 35
        },
        "corn": {
            "optimal_low": 0.40,
            "optimal_high": 0.65,
            "water_sensitivity": 1.0,
            "growth_days": 100,
            "price_per_kg": 30
        }
    }

    # ─────────────────────────────────────────────
    # IRRIGATION AMOUNTS per action (in moisture units)
    # ─────────────────────────────────────────────
    IRRIGATION_AMOUNT = {
        0: 0.00,   # No irrigation
        1: 0.08,   # Low irrigation
        2: 0.18,   # High irrigation
    }

    def __init__(self, difficulty: str = "medium", crop_type: str = "wheat"):
        """
        Args:
            difficulty: "easy" | "medium" | "hard"
            crop_type: "wheat" | "rice" | "corn"
        """
        super(IrrigationEnv, self).__init__()

        assert difficulty in ("easy", "medium", "hard"), \
            "difficulty must be 'easy', 'medium', or 'hard'"
        assert crop_type in self.CROP_TYPES, \
            f"crop_type must be one of {list(self.CROP_TYPES.keys())}"

        self.difficulty = difficulty
        self.crop_type = crop_type
        self.crop_data = self.CROP_TYPES[crop_type]
        
        # Update optimal ranges based on crop
        self.OPTIMAL_LOW = self.crop_data["optimal_low"]
        self.OPTIMAL_HIGH = self.crop_data["optimal_high"]

        # ── Difficulty Configuration ──────────────────────────────────────
        self._apply_difficulty_config()

        # ── Action Space ──────────────────────────────────────────────────
        # Discrete: 0 (none), 1 (low), 2 (high), 3 (fertilizer)
        self.action_space = spaces.Discrete(4)

        # ── Observation Space ─────────────────────────────────────────────
        # [soil_moisture, temperature, humidity, rain, crop_stage, 
        #  forecast_rain, forecast_confidence, fertilizer_level, soil_health]
        low  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal counters
        self.max_steps    = 168   # 1 week in hourly steps
        self.current_step = 0

        # Economic variables
        self.water_price_per_unit = 5
        self.water_usage_total = 0
        self.water_bill = 0
        self.fertilizer_level = 0.5
        self.fertilizer_cost = 50
        self.soil_health = 0.7
        
        # Weather forecast
        self.weather_forecast_days = 3
        self.forecast_accuracy = 0.8
        self.time_of_day = 0

        # Tracking for graders
        self.moisture_history = []
        self.unnecessary_irrigation_count = 0

        # State variables (initialised in reset)
        self.soil_moisture = None
        self.temperature   = None
        self.humidity      = None
        self.rain          = None
        self.crop_stage    = None

    # ─────────────────────────────────────────────────────────────────────
    # DIFFICULTY CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────
    def _apply_difficulty_config(self):
        """
        Sets all difficulty-dependent parameters.
        Easy   → predictable, low stress
        Medium → realistic, moderate randomness
        Hard   → extreme, high stress, noisy sensors
        """
        if self.difficulty == "easy":
            self.rain_probability     = 0.05
            self.evaporation_rate     = 0.01
            self.temp_range           = (20, 30)
            self.humidity_range       = (40, 60)
            self.sensor_noise_std     = 0.00
            self.drought_event_prob   = 0.00
            self.rain_intensity_range = (0.05, 0.10)

        elif self.difficulty == "medium":
            self.rain_probability     = 0.15
            self.evaporation_rate     = 0.02
            self.temp_range           = (25, 40)
            self.humidity_range       = (30, 70)
            self.sensor_noise_std     = 0.01
            self.drought_event_prob   = 0.05
            self.rain_intensity_range = (0.05, 0.20)

        elif self.difficulty == "hard":
            self.rain_probability     = 0.25
            self.evaporation_rate     = 0.04
            self.temp_range           = (35, 45)
            self.humidity_range       = (10, 90)
            self.sensor_noise_std     = 0.03
            self.drought_event_prob   = 0.15
            self.rain_intensity_range = (0.02, 0.30)

        self.in_drought = False

    # ─────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """
        Initialise (or reinitialise) environment to a fresh episode start.

        Returns:
            state (np.ndarray): initial observation
        """
        self.current_step = 0
        self.in_drought   = False
        self.water_usage_total = 0
        self.water_bill = 0
        self.fertilizer_level = 0.5
        self.soil_health = 0.7
        self.time_of_day = 0
        self.moisture_history = []
        self.unnecessary_irrigation_count = 0

        # Start with slightly sub-optimal moisture
        self.soil_moisture = np.random.uniform(0.30, 0.50)

        # Temperature: scaled to [0, 1]
        raw_temp           = np.random.uniform(*self.temp_range)
        self.temperature   = self._scale_temperature(raw_temp)

        self.humidity      = np.random.uniform(*self.humidity_range) / 100.0
        self.rain          = 0.0
        self.crop_stage    = 0.0

        return self.get_state(), {}

    # ─────────────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        """
        Apply one action and advance environment by one timestep (1 hour).

        Args:
            action (int): 0 = no water | 1 = low water | 2 = high water | 3 = fertilizer

        Returns:
            state, reward, done, truncated, info
        """
        # Track moisture for grading
        self.moisture_history.append(self.soil_moisture)
        
        # Track unnecessary irrigation for environmental grader
        if action in [1, 2] and self.OPTIMAL_LOW <= self.soil_moisture <= self.OPTIMAL_HIGH:
            self.unnecessary_irrigation_count += 1
        
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Handle fertilizer action
        if action == 3:
            return self._apply_fertilizer()

        # Regular irrigation actions (0, 1, 2)
        water_added = self.IRRIGATION_AMOUNT[action]
        
        # Calculate water cost
        water_cost = self._calculate_water_cost(water_added)
        self.water_usage_total += water_added

        # Simulate Rain
        self.rain = self._simulate_rain()

        # Update Soil Moisture
        prev_moisture        = self.soil_moisture
        self.soil_moisture   = self._update_soil_moisture(water_added)

        # Update Weather
        self._update_weather()

        # Advance Crop Stage
        self.crop_stage = min(1.0, self.crop_stage + (1.0 / self.max_steps))

        # Calculate Reward
        reward = self._calculate_reward(
            action=action,
            irrigated=(action > 0),
            water_added=water_added,
            prev_moisture=prev_moisture
        )
        
        # Subtract water cost from reward
        reward -= (water_cost * 0.01)

        # Update time of day
        self.time_of_day = (self.time_of_day + 1) % 24

        # Advance Step Counter
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Build Info Dict
        info = {
            "step": self.current_step,
            "soil_moisture": round(self.soil_moisture, 4),
            "rain": round(self.rain, 4),
            "temperature": round(self.temperature, 4),
            "humidity": round(self.humidity, 4),
            "crop_stage": round(self.crop_stage, 4),
            "water_added": water_added,
            "water_cost": water_cost,
            "water_usage_total": self.water_usage_total,
            "water_bill": self.water_bill,
            "in_drought": self.in_drought,
            "difficulty": self.difficulty,
            "fertilizer_level": self.fertilizer_level,
            "soil_health": self.soil_health,
        }

        return self.get_state(), reward, done, False, info

    def _apply_fertilizer(self):
        """Apply fertilizer action"""
        prev_fertilizer = self.fertilizer_level
        self.fertilizer_level = min(1.0, self.fertilizer_level + 0.2)
        self.soil_health = min(1.0, self.soil_health + 0.05)
        
        # Small penalty for cost
        reward = -(self.fertilizer_cost * 0.01)
        
        # Update time and step
        self.time_of_day = (self.time_of_day + 1) % 24
        self.current_step += 1
        self.crop_stage = min(1.0, self.crop_stage + (1.0 / self.max_steps))
        
        done = self.current_step >= self.max_steps
        
        info = {
            "step": self.current_step,
            "action": "fertilizer",
            "fertilizer_level": self.fertilizer_level,
            "soil_health": self.soil_health,
            "fertilizer_increase": self.fertilizer_level - prev_fertilizer
        }
        
        return self.get_state(), reward, done, False, info

    def _calculate_water_cost(self, water_amount):
        """Dynamic pricing based on time of day"""
        # Peak hours (6-9 AM, 5-8 PM) - 50% more expensive
        if (6 <= self.time_of_day <= 9) or (17 <= self.time_of_day <= 20):
            price_multiplier = 1.5
        # Night hours (10 PM - 5 AM) - 30% cheaper
        elif self.time_of_day >= 22 or self.time_of_day <= 5:
            price_multiplier = 0.7
        else:
            price_multiplier = 1.0
        
        cost = water_amount * self.water_price_per_unit * price_multiplier
        self.water_bill += cost
        return cost

    def _generate_weather_forecast(self):
        """Generate forecast with uncertainty"""
        actual_rain = self.rain if hasattr(self, 'rain') else 0
        
        # Forecast has uncertainty
        if np.random.random() < self.forecast_accuracy:
            forecast_rain = actual_rain
        else:
            # Wrong forecast - add error
            forecast_rain = actual_rain + np.random.normal(0, 0.3)
            forecast_rain = np.clip(forecast_rain, 0, 1)
        
        return {
            "actual": actual_rain,
            "forecast": forecast_rain,
            "confidence": self.forecast_accuracy
        }

    def _calculate_yield_boost(self):
        """Fertilizer improves crop yield"""
        if self.fertilizer_level > 0.7:
            return 1.3  # 30% boost
        elif self.fertilizer_level > 0.3:
            return 1.1  # 10% boost
        return 1.0

    # ─────────────────────────────────────────────────────────────────────
    # GET STATE
    # ─────────────────────────────────────────────────────────────────────
    def get_state(self) -> np.ndarray:
        """
        Return current environment state with forecast information.
        """
        forecast = self._generate_weather_forecast()
        
        base_state = np.array([
            self.soil_moisture,
            self.temperature,
            self.humidity,
            self.rain,
            self.crop_stage,
        ], dtype=np.float32)
        
        # Add forecast and fertilizer info
        extended_state = np.append(base_state, [
            forecast["forecast"],
            forecast["confidence"],
            self.fertilizer_level,
            self.soil_health
        ])
        
        # Add sensor noise
        if self.sensor_noise_std > 0.0:
            noise = np.random.normal(0.0, self.sensor_noise_std, size=extended_state.shape)
            extended_state = extended_state + noise

        # Clip to valid range
        extended_state = np.clip(extended_state, 0.0, 1.0)
        return extended_state.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # REWARD FUNCTION
    # ─────────────────────────────────────────────────────────────────────
    def _calculate_reward(
        self,
        action: int,
        irrigated: bool,
        water_added: float,
        prev_moisture: float
    ) -> float:
        """
        Calculates the reward for one timestep including yield boost from fertilizer.
        """
        reward = 0.0
        m      = self.soil_moisture
        stage  = self.crop_stage
        yield_boost = self._calculate_yield_boost()

        # BLOCK 1: Moisture Quality Reward
        if self.OPTIMAL_LOW <= m <= self.OPTIMAL_HIGH:
            if not irrigated:
                reward += 12.0 * yield_boost
            else:
                reward += 10.0 * yield_boost

        elif m < self.OPTIMAL_LOW:
            dryness = self.OPTIMAL_LOW - m
            penalty = dryness * 30.0
            harvest_multiplier = 1.0 + stage * 0.5
            reward -= penalty * harvest_multiplier / yield_boost

        elif m > self.OPTIMAL_HIGH:
            wetness = m - self.OPTIMAL_HIGH
            penalty = wetness * 25.0
            reward -= penalty

        # BLOCK 2: Critical Threshold Penalty
        if m < 0.15:
            reward -= 15.0
        elif m > 0.92:
            reward -= 12.0

        # BLOCK 3: Irrigation During Rain Penalty
        if irrigated and self.rain > 0.05:
            waste_penalty = self.rain * water_added * 40.0
            reward -= waste_penalty

        # BLOCK 4: Over-Watering Penalty
        if irrigated and prev_moisture > self.OPTIMAL_HIGH:
            excess = water_added * 20.0
            reward -= excess

        # BLOCK 5: Unnecessary Irrigation Penalty
        if irrigated and self.OPTIMAL_LOW <= prev_moisture <= self.OPTIMAL_HIGH:
            reward -= water_added * 8.0

        # BLOCK 6: High Irrigation Efficiency Check
        if action == 2 and prev_moisture > 0.30:
            reward -= 2.0

        # BLOCK 7: Drought Bonus
        if self.in_drought and irrigated and m < self.OPTIMAL_LOW:
            reward += 5.0

        # BLOCK 8: Soil Health Bonus
        if self.soil_health > 0.8:
            reward += 2.0

        reward = float(np.clip(reward, -30.0, 20.0))
        return reward

    # ─────────────────────────────────────────────────────────────────────
    # GRADER METHODS (for Phase 2 validation)
    # ─────────────────────────────────────────────────────────────────────
    def grade_water_efficiency(self) -> float:
        """
        Grader 1: Measures water usage efficiency
        Returns score between 0 and 1 (exclusive)
        """
        if self.water_usage_total == 0:
            return 0.5
        
        optimal_water = self.max_steps * 0.05
        efficiency = 1.0 - min(1.0, abs(self.water_usage_total - optimal_water) / optimal_water)
        score = max(0.001, min(0.999, efficiency))
        return float(score)

    def grade_crop_health(self) -> float:
        """
        Grader 2: Measures crop health based on moisture and fertilizer
        Returns score between 0 and 1 (exclusive)
        """
        if len(self.moisture_history) == 0:
            return 0.5
        
        optimal_count = sum(1 for m in self.moisture_history 
                           if self.OPTIMAL_LOW <= m <= self.OPTIMAL_HIGH)
        health_score = optimal_count / len(self.moisture_history)
        
        fertilizer_bonus = self.fertilizer_level * 0.2
        final_score = min(0.999, (health_score * 0.8) + (fertilizer_bonus * 0.2))
        return float(max(0.001, final_score))

    def grade_economic_profit(self) -> float:
        """
        Grader 3: Measures economic efficiency
        Returns score between 0 and 1 (exclusive)
        """
        potential_yield = self.crop_stage * self._calculate_yield_boost()
        total_cost = self.water_bill + (self.fertilizer_level * self.fertilizer_cost)
        
        if total_cost > 0:
            profit_score = (potential_yield * 100) / (total_cost + 1)
            normalized_score = min(0.999, profit_score / 10)
        else:
            normalized_score = 0.5
        
        return float(max(0.001, normalized_score))

    def grade_environmental_impact(self) -> float:
        """
        Grader 4: Measures environmental friendliness
        Returns score between 0 and 1 (exclusive)
        """
        waste_penalty = self.unnecessary_irrigation_count / max(1, self.current_step)
        environmental_score = 1.0 - min(0.999, waste_penalty)
        
        rain_utilization = min(1.0, self.rain * 10)
        final_score = (environmental_score * 0.7) + (rain_utilization * 0.3)
        
        return float(max(0.001, min(0.999, final_score)))

    def get_all_grades(self) -> dict:
        """
        Returns all grader scores for validation
        """
        return {
            "water_efficiency": self.grade_water_efficiency(),
            "crop_health": self.grade_crop_health(),
            "economic_profit": self.grade_economic_profit(),
            "environmental_impact": self.grade_environmental_impact()
        }

    # ─────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────
    def _simulate_rain(self) -> float:
        """Returns rainfall intensity for this timestep."""
        if np.random.random() < self.drought_event_prob:
            self.in_drought = True
        if self.in_drought:
            if np.random.random() < 0.10:
                self.in_drought = False
            return 0.0
        if np.random.random() < self.rain_probability:
            intensity = np.random.uniform(*self.rain_intensity_range)
            return float(np.clip(intensity, 0.0, 1.0))
        return 0.0
    
    def _update_soil_moisture(self, water_added: float) -> float:
        """Computes new soil moisture."""
        evaporation = self.evaporation_rate * (0.5 + self.temperature)
        crop_absorption = 0.005 * (0.5 + self.crop_stage)
        new_moisture = (
            self.soil_moisture + water_added + self.rain - evaporation - crop_absorption
        )
        return float(np.clip(new_moisture, 0.0, 1.0))

    def _update_weather(self):
        """Randomly drift temperature and humidity."""
        temp_delta = np.random.uniform(-0.02, 0.02)
        self.temperature = float(np.clip(self.temperature + temp_delta, 0.0, 1.0))
        hum_delta = np.random.uniform(-0.03, 0.03)
        self.humidity = float(np.clip(self.humidity + hum_delta, 0.0, 1.0))

    def _scale_temperature(self, raw_temp: float) -> float:
        """Scales raw temperature (°C) to [0, 1]."""
        return float(np.clip(raw_temp / 50.0, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────
    # RENDER
    # ─────────────────────────────────────────────────────────────────────
    def render(self, mode="human"):
        m = self.soil_moisture
        status = (
            "✅ OPTIMAL"  if self.OPTIMAL_LOW <= m <= self.OPTIMAL_HIGH else
            "🔴 TOO DRY"  if m < self.OPTIMAL_LOW else
            "🔵 TOO WET"
        )
        print(
            f"Step {self.current_step:>3} | "
            f"Moisture: {m:.3f} {status} | "
            f"Temp: {self.temperature:.2f} | "
            f"Humidity: {self.humidity:.2f} | "
            f"Rain: {self.rain:.3f} | "
            f"Crop: {self.crop_stage:.2f} | "
            f"Drought: {self.in_drought} | "
            f"Fertilizer: {self.fertilizer_level:.2f}"
        )

    def close(self):
        pass