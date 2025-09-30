from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging
import os

def evaluate_model(model, X_test, y_test, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )

    try:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        logging.info(f"Model baholandi: R2={r2}, MAE={mae}, MSE={mse}")
        return {"r2": r2, "mae": mae, "mse": mse}
    except Exception as e:
        logging.error(f"Model baholashda xato yuz berdi: {e}")
        return None
