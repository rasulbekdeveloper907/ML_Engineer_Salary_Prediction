import pandas as pd
import logging
import os

def load_csv(file_path, log_path):
    """
    CSV faylini o'qiydi va jarayonni log qiladi.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )

    try:
        logging.info(f"{file_path} faylini o'qish boshlandi.")
        df = pd.read_csv(file_path)
        logging.info(f"{file_path} muvaffaqiyatli o'qildi: {df.shape} o'lchamda.")
        return df
    except FileNotFoundError as e:
        logging.error(f"Fayl topilmadi: {e}")
    except Exception as e:
        logging.error(f"Noma’lum xatolik yuz berdi: {e}")
    return None
