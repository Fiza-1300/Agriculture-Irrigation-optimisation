
"""Graders for Agricultural Irrigation Optimization"""

class EasyGrader:
    """Easy task: Maintain soil moisture between 40-60%"""
    
    def compute(self, actions, soil_moisture_readings=None):
        if soil_moisture_readings is None:
            return 1.0
        optimal_count = sum(1 for m in soil_moisture_readings if 40 <= m <= 60)
        score = optimal_count / len(soil_moisture_readings) if soil_moisture_readings else 0
        return min(max(score, 0.0), 1.0)


class MediumGrader:
    """Medium task: Optimize water usage while maintaining soil health"""
    
    def compute(self, actions, water_usage=None, soil_moisture=None):
        if water_usage is None:
            return 0.7
        avg_water = sum(water_usage) / len(water_usage) if water_usage else 100
        water_score = max(0, 1 - (avg_water / 100))
        if soil_moisture:
            moisture_score = sum(1 for m in soil_moisture if 30 <= m <= 70) / len(soil_moisture)
        else:
            moisture_score = 0.5
        score = (water_score * 0.6 + moisture_score * 0.4)
        return min(max(score, 0.0), 1.0)


class HardGrader:
    """Hard task: Predict optimal irrigation schedule for 7 days"""
    
    def compute(self, actions, weather_forecast=None, actual_needs=None):
        if actual_needs is None:
            return 0.5
        if len(actions) != len(actual_needs):
            return 0.0
        errors = [abs(a - n) for a, n in zip(actions, actual_needs)]
        max_error = max(actual_needs) if actual_needs else 100
        accuracy = 1 - (sum(errors) / (len(errors) * max_error)) if max_error > 0 else 0
        return min(max(accuracy, 0.0), 1.0)
