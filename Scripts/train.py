import logging
import os
import joblib

def train_model(model, X_train, y_train, model_path, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )

    try:
        logging.info(f"{model.__class__.__name__} modelini o'qitish boshlandi.")
        model.fit(X_train, y_train)
        logging.info(f"{model.__class__.__name__} model muvaffaqiyatli o'qitildi.")

        joblib.dump(model, model_path)
        logging.info(f"Model saqlandi: {model_path}")
    except Exception as e:
        logging.error(f"Model o'qitishda xatolik yuz berdi: {e}")
