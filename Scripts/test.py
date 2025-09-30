from Scripts.data_loader import load_csv
from Scripts.analysis import data_overview, correlation_matrix, plot_histograms
from Scripts.models import ModelBuilder
from Scripts.tuning import ModelTuner
from Scripts.training import train_model
from Scripts.evaluation import evaluate_model
from Scripts.utils import log_message
import os
from sklearn.model_selection import train_test_split

# --- Fayl yo‘llari ---
raw_data_path = r"C:\Users\Rasulbek_Ruzmetov\Desktop\SML_R_Project\Data\Raw_Data\global_ai_ml_data_salaries.csv"
feature_selection_path = r"C:\Users\Rasulbek_Ruzmetov\Desktop\SML_R_Project\Data\Feature_Selection\Feature_Selection.csv"
data_log_path = r"C:\Users\Rasulbek_Ruzmetov\Desktop\SML_R_Project\Log\data_loader.log"
train_log_path = r"C:\Users\Rasulbek_Ruzmetov\Desktop\SML_R_Project\Log\trainer.log"
model_path = r"C:\Users\Rasulbek_Ruzmetov\Desktop\SML_R_Project\Models\final_model.pkl"

# --- 1. Data yuklash ---
df = load_csv(raw_data_path, data_log_path)
if df is None:
    raise Exception("Data yuklanmadi. Log faylni tekshiring.")

# --- 2. Data tahlil ---
print("\n--- Data Overview ---")
data_overview(df)
correlation_matrix(df)
plot_histograms(df)

# --- 3. Feature & Target ajratish ---
X = df.drop("salary", axis=1, errors="ignore")  # 'salary' ustuni mavjud bo‘lsa o‘chiriladi
y = df["salary"] if "salary" in df.columns else None

if y is None:
    raise Exception("'salary' ustuni topilmadi.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Model yaratish ---
models = ModelBuilder().get_models()
model_name = "RandomForest"  # Test uchun
model = models[model_name]

# --- 5. Parametr grid ---
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

# --- 6. Model tuning ---
tuner = ModelTuner(model=model, param_grid=param_grid, cv=3, scoring="r2", n_iter=5)
best_model, best_params, best_score = tuner.grid_search(X_train, y_train)

print(f"\n--- {model_name} Grid Search Natijalari ---")
print("Eng yaxshi parametrlar:", best_params)
print("Eng yaxshi R2 score:", best_score)

# --- 7. Modelni o‘qitish ---
train_model(best_model, X_train, y_train, model_path, train_log_path)

# --- 8. Model baholash ---
evaluation_results = evaluate_model(best_model, X_test, y_test, train_log_path)
print("\n--- Model Evaluation ---")
print(evaluation_results)

# --- 9. Log yozish ---
log_message(f"{model_name} modeli sinovi tugadi. Baholash natijalari: {evaluation_results}", train_log_path)
