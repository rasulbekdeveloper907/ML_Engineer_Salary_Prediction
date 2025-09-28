from data_loader import DataLoader
from preprocessing import DataPreprocessor
from analysis import DataAnalysis
from models import ModelBuilder
from tuning import ModelTuner

def main():
    # 1. Ma’lumotni yuklash
    loader = DataLoader("Data/Raw_Data/global_ai_ml_data_salaries.csv")
    df = loader.load_data()

    # 2. Tahlil
    analysis = DataAnalysis(df)
    analysis.show_info()
    analysis.show_summary()

    # 3. Preprocessing
    preprocessor = DataPreprocessor(df)
    df_clean = preprocessor.handle_missing()
    df_encoded = preprocessor.encode_categorical()
    X, y = preprocessor.get_features_and_target()

    # 4. Model qurish
    builder = ModelBuilder()
    models = builder.get_models()

    # 5. Hyperparameter tuning (masalan RandomForest)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10]
    }
    tuner = ModelTuner(models["RandomForest"], param_grid)
    best_model = tuner.tune(X, y)

    print(f"[RESULT] Eng yaxshi model: {best_model}")

if __name__ == "__main__":
    main()
