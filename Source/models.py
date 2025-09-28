from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class ModelBuilder:
    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=42)
        }

    def get_models(self):
        """Model dictionary qaytaradi"""
        return self.models
