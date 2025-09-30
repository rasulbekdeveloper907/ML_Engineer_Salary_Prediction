from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import logging
import os

class ModelTuner:
    def __init__(self, model, param_grid, cv=5, scoring="r2", n_iter=10, random_state=42):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state

        log_path = r"C:\Users\Rasulbek_Ruzmetov\Desktop\SML_R_Project\Log\tuning.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="a"
        )

    def grid_search(self, X, y):
        logging.info(f"{self.model.__class__.__name__} uchun Grid Search boshlandi.")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        logging.info(f"Grid Search yakunlandi. Eng yaxshi parametrlar: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def randomized_search(self, X, y):
        logging.info(f"{self.model.__class__.__name__} uchun Randomized Search boshlandi.")
        randomized_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=-1
        )
        randomized_search.fit(X, y)
        logging.info(f"Randomized Search yakunlandi. Eng yaxshi parametrlar: {randomized_search.best_params_}")
        return randomized_search.best_estimator_, randomized_search.best_params_, randomized_search.best_score_
