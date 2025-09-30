from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb

class ModelBuilder:
    """
    Bu klass bir nechta regression modellari yaratadi va ularni qaytaradi.
    """

    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
            "SVM": SVR(),
            "XGBoost": xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
        }

    def get_models(self):
        """
        Modellar lug'atini qaytaradi.
        
        Returns:
            dict: model nomlari va obyektlari
        """
        return self.models
