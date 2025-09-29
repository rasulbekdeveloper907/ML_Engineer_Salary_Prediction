from sklearn.model_selection import GridSearchCV

class ModelTuner:
    def __init__(self, model, param_grid, cv=5, scoring="r2"):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def tune(self, X, y):
        """GridSearchCV orqali optimal parametrlarni tanlash"""
        grid = GridSearchCV(self.model, self.param_grid, cv=self.cv, scoring=self.scoring)
        grid.fit(X, y)
        print(f"[INFO] Eng yaxshi parametrlar: {grid.best_params_}")
        return grid.best_estimator_
