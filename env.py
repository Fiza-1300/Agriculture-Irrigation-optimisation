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

    # ─────────────────────────────────────────────
    # IRRIGATION AMOUNTS per action (in moisture units)
    # ─────────────────────────────────────────────
    IRRIGATION_AMOUNT = {
        0: 0.00,   # No irrigation
        1: 0.08,   # Low irrigation
        2: 0.18,   # High irrigation
    }

    # ─────────────────────────────────────────────
    # OPTIMAL SOIL MOISTURE RANGE
    # ─────────────────────────────────────────────
    OPTIMAL_LOW  = 0.40
    OPTIMAL_HIGH = 0.70

    def __init__(self, difficulty: str = "medium"):
    
        """
        Args:
            difficulty: "easy" | "medium" | "hard"
        """
        super(IrrigationEnv, self).__init__()

        assert difficulty in ("easy", "medium", "hard"), \
            "difficulty must be 'easy', 'medium', or 'hard'"

        self.difficulty = difficulty

        # ── Difficulty Configuration ──────────────────────────────────────
        self._apply_difficulty_config()

        # ── Action Space ──────────────────────────────────────────────────
        # Discrete: 0 (none), 1 (low), 2 (high)
        self.action_space = spaces.Discrete(3)

        # ── Observation Space ─────────────────────────────────────────────
        # [soil_moisture, temperature, humidity, rain, crop_stage]
        # All values normalised to [0, 1] except temperature (0–1 scaled)
        low  = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal counters
        self.max_steps    = 168   # 1 week in hourly steps
        self.current_step = 0

        # State variables (initialised in reset)
        self.soil_moisture = None
        self.temperature   = None   # stored as 0-1 scaled value
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
            self.rain_probability     = 0.05    # 5% chance of rain each step
            self.evaporation_rate     = 0.01    # slow drying
            self.temp_range           = (20, 30) # mild temperature (°C)
            self.humidity_range       = (40, 60)
            self.sensor_noise_std     = 0.00    # perfect sensors
            self.drought_event_prob   = 0.00    # no drought
            self.rain_intensity_range = (0.05, 0.10)

        elif self.difficulty == "medium":
            self.rain_probability     = 0.15
            self.evaporation_rate     = 0.02
            self.temp_range           = (25, 40)
            self.humidity_range       = (30, 70)
            self.sensor_noise_std     = 0.01    # small noise
            self.drought_event_prob   = 0.05    # 5% chance of drought spell
            self.rain_intensity_range = (0.05, 0.20)

        elif self.difficulty == "hard":
            self.rain_probability     = 0.25    # frequent but unpredictable
            self.evaporation_rate     = 0.04    # fast drying
            self.temp_range           = (35, 45) # extreme heat
            self.humidity_range       = (10, 90) # wide swings
            self.sensor_noise_std     = 0.03    # noisy sensors
            self.drought_event_prob   = 0.15    # frequent droughts
            self.rain_intensity_range = (0.02, 0.30)

        # Drought flag — activated stochastically in hard/medium
        self.in_drought = False

    # ─────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """
        Initialise (or reinitialise) environment to a fresh episode start.

        Returns:
            state (np.ndarray): initial observation of shape (5,)
        """
        self.current_step = 0
        self.in_drought   = False

        # Start with slightly sub-optimal moisture so agent must act
        self.soil_moisture = np.random.uniform(0.30, 0.50)

        # Temperature: scaled to [0, 1] from temp_range
        raw_temp           = np.random.uniform(*self.temp_range)
        self.temperature   = self._scale_temperature(raw_temp)

        self.humidity      = np.random.uniform(*self.humidity_range) / 100.0
        self.rain          = 0.0   # no rain at episode start
        self.crop_stage    = 0.0   # seedling at episode start

        return self.get_state(), {}

    # ─────────────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        """
        Apply one action and advance environment by one timestep (1 hour).

        Args:
            action (int): 0 = no water | 1 = low water | 2 = high water

        Returns:
            state  (np.ndarray) : new observation  shape (5,)
            reward (float)      : reward for this timestep
            done   (bool)       : True if episode is over
            info   (dict)       : diagnostic information
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # ── 1. Apply Irrigation ───────────────────────────────────────────
        water_added = self.IRRIGATION_AMOUNT[action]
        irrigated   = action > 0

        # ── 2. Simulate Rain ──────────────────────────────────────────────
        self.rain = self._simulate_rain()

        # ── 3. Update Soil Moisture ───────────────────────────────────────
        prev_moisture        = self.soil_moisture
        self.soil_moisture   = self._update_soil_moisture(water_added)

        # ── 4. Update Weather ─────────────────────────────────────────────
        self._update_weather()

        # ── 5. Advance Crop Stage ─────────────────────────────────────────
        self.crop_stage = min(1.0, self.crop_stage + (1.0 / self.max_steps))

        # ── 6. Calculate Reward ───────────────────────────────────────────
        reward = self._calculate_reward(
            action      = action,
            irrigated   = irrigated,
            water_added = water_added,
            prev_moisture = prev_moisture
        )

        # ── 7. Advance Step Counter ───────────────────────────────────────
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # ── 8. Build Info Dict ────────────────────────────────────────────
        info = {
            "step"          : self.current_step,
            "soil_moisture" : round(self.soil_moisture, 4),
            "rain"          : round(self.rain, 4),
            "temperature"   : round(self.temperature, 4),
            "humidity"      : round(self.humidity, 4),
            "crop_stage"    : round(self.crop_stage, 4),
            "water_added"   : water_added,
            "reward"        : reward,
            "in_drought"    : self.in_drought,
            "difficulty"    : self.difficulty,
        }

        return self.get_state(), reward, done, False, info

    # ─────────────────────────────────────────────────────────────────────
    # GET STATE
    # ─────────────────────────────────────────────────────────────────────
    def get_state(self) -> np.ndarray:
        """
        Return current environment state as a numpy array.

        Format (all values in [0, 1]):
            [soil_moisture, temperature, humidity, rain, crop_stage]

        Returns:
            np.ndarray of shape (5,) dtype float32
        """
        state = np.array([
            self.soil_moisture,
            self.temperature,
            self.humidity,
            self.rain,
            self.crop_stage,
        ], dtype=np.float32)

        # Add sensor noise on medium/hard to simulate real-world imperfection
        if self.sensor_noise_std > 0.0:
            noise = np.random.normal(0.0, self.sensor_noise_std, size=state.shape)
            state = state + noise

        # Clip to valid range after noise
        state = np.clip(state, 0.0, 1.0)
        return state

    # ─────────────────────────────────────────────────────────────────────
    # REWARD FUNCTION  ← CORE LOGIC (Everything depends on this)
    # ─────────────────────────────────────────────────────────────────────
    def _calculate_reward(
        self,
        action: int,
        irrigated: bool,
        water_added: float,
        prev_moisture: float
    ) -> float:
        """
        Calculates the reward for one timestep.

        Design philosophy:
          ✔ Reward being in optimal moisture band
          ✔ Penalise extremes (too dry / too wet) harshly
          ✔ Penalise water waste (irrigating when not needed)
          ✔ Penalise irrigation during rain (wasteful)
          ✔ Scale penalties with crop stage (errors more costly near harvest)
          ✔ Reward efficiency (staying optimal WITHOUT irrigating)

        Returns:
            reward (float)
        """
        reward = 0.0
        m      = self.soil_moisture          # current moisture
        stage  = self.crop_stage             # 0 (seedling) → 1 (harvest)

        # ── BLOCK 1: Moisture Quality Reward ─────────────────────────────
        # Core signal: how close is moisture to the optimal band?

        if self.OPTIMAL_LOW <= m <= self.OPTIMAL_HIGH:
            # ✅ In optimal range
            # Bonus if agent achieved this WITHOUT irrigating (efficient)
            if not irrigated:
                reward += 12.0   # Efficiency bonus
            else:
                reward += 10.0   # Still good, but used water

        elif m < self.OPTIMAL_LOW:
            # ❌ Too dry — penalty scales with how dry
            dryness = self.OPTIMAL_LOW - m          # 0 → 0.4
            penalty = dryness * 30.0                # max ~12 penalty
            # Extra penalty near harvest (crop loss is catastrophic)
            harvest_multiplier = 1.0 + stage * 0.5  # 1.0 → 1.5
            reward -= penalty * harvest_multiplier

        elif m > self.OPTIMAL_HIGH:
            # ❌ Too wet — root rot, oxygen deprivation
            wetness = m - self.OPTIMAL_HIGH          # 0 → 0.3
            penalty = wetness * 25.0                 # max ~7.5 penalty
            reward -= penalty

        # ── BLOCK 2: Critical Threshold Penalty ──────────────────────────
        # Hard floor / ceiling — severe penalties at extremes

        if m < 0.15:
            # 🚨 Crop dying — extreme drought
            reward -= 15.0
        elif m > 0.92:
            # 🚨 Waterlogged — irreversible damage
            reward -= 12.0

        # ── BLOCK 3: Irrigation During Rain Penalty ───────────────────────
        # Irrigating while it's raining = pure waste

        if irrigated and self.rain > 0.05:
            # Penalty scales with how much it's raining AND how much water added
            waste_penalty = self.rain * water_added * 40.0
            reward -= waste_penalty

        # ── BLOCK 4: Over-Watering Penalty ───────────────────────────────
        # Irrigating when soil is already wet enough

        if irrigated and prev_moisture > self.OPTIMAL_HIGH:
            # Moisture was already above optimal before this action
            excess = water_added * 20.0
            reward -= excess

        # ── BLOCK 5: Unnecessary Irrigation Penalty ───────────────────────
        # Irrigating when moisture is comfortably in range wastes resources

        if irrigated and self.OPTIMAL_LOW <= prev_moisture <= self.OPTIMAL_HIGH:
            # Small penalty — wasteful but not catastrophic
            reward -= water_added * 8.0

        # ── BLOCK 6: High Irrigation Efficiency Check ─────────────────────
        # Using action=2 (high water) when low would have sufficed

        if action == 2 and prev_moisture > 0.30:
            # Soil wasn't critically dry — high water was excessive
            reward -= 2.0

        # ── BLOCK 7: Drought Bonus ────────────────────────────────────────
        # Correctly irrigating during a drought event gets bonus

        if self.in_drought and irrigated and m < self.OPTIMAL_LOW:
            reward += 5.0   # Good decision under stress

        # ── Final clip: keep reward in reasonable range ───────────────────
        reward = float(np.clip(reward, -30.0, 15.0))

        return reward

    # ─────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────
    def _simulate_rain(self) -> float:
        """
        Returns rainfall intensity for this timestep (0 = no rain).
        Rain is a random event based on difficulty's rain_probability.
        """
        # Stochastic drought: suppresses rain for a stretch
        if np.random.random() < self.drought_event_prob:
            self.in_drought = True

        if self.in_drought:
            # Drought ends probabilistically
            if np.random.random() < 0.10:
                self.in_drought = False
            return 0.0  # No rain during drought

        if np.random.random() < self.rain_probability:
            intensity = np.random.uniform(*self.rain_intensity_range)
            return float(np.clip(intensity, 0.0, 1.0))

        return 0.0

    def _update_soil_moisture(self, water_added: float) -> float:
        """
        Computes new soil moisture after applying:
          + water_added (irrigation)
          + rain
          - evaporation (driven by temperature)
          - crop absorption (driven by crop stage)
        """
        # Evaporation increases with temperature
        evaporation = self.evaporation_rate * (0.5 + self.temperature)

        # Crops absorb more water as they grow
        crop_absorption = 0.005 * (0.5 + self.crop_stage)

        new_moisture = (
            self.soil_moisture
            + water_added
            + self.rain
            - evaporation
            - crop_absorption
        )

        return float(np.clip(new_moisture, 0.0, 1.0))

    def _update_weather(self):
        """
        Randomly drift temperature and humidity each step to simulate
        realistic weather fluctuation.
        """
        # Temperature drift (small random walk)
        temp_delta       = np.random.uniform(-0.02, 0.02)
        self.temperature = float(np.clip(self.temperature + temp_delta, 0.0, 1.0))

        # Humidity drift
        hum_delta     = np.random.uniform(-0.03, 0.03)
        self.humidity = float(np.clip(self.humidity + hum_delta, 0.0, 1.0))

    def _scale_temperature(self, raw_temp: float) -> float:
        """
        Scales raw temperature (°C) to [0, 1] using global min/max of 0–50°C.
        """
        return float(np.clip(raw_temp / 50.0, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────
    # RENDER (Optional — for debugging)
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
            f"Drought: {self.in_drought}"
        )

    def close(self):
        pass