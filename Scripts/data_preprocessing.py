import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import os

def handle_missing_values(df, strategy="mean"):
    """
    Null qiymatlarni to'ldirish.
    strategy: "mean", "median", "mode"
    """
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_categorical(df):
    """
    Kategorik ustunlarni Label Encoding orqali raqamga aylantirish.
    """
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])
    return df

def scale_features(df, feature_columns):
    """
    Belgilangan ustunlarni StandardScaler yordamida skaler qiladi.
    """
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

def remove_outliers(df, column, threshold=3):
    """
    Outlierlarni olib tashlash (Z-score asosida).
    """
    mean_val = df[column].mean()
    std_val = df[column].std()
    z_scores = (df[column] - mean_val) / std_val
    df = df[np.abs(z_scores) < threshold]
    return df

def preprocess_data(df, log_path, fill_strategy="mean", scale=True):
    """
    To'liq data preprocessing jarayoni.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )

    try:
        logging.info("Data preprocessing boshlandi.")
        df = handle_missing_values(df, strategy=fill_strategy)
        logging.info("Null qiymatlar to‘ldirildi.")

        df = encode_categorical(df)
        logging.info("Kategorik ustunlar kodlandi.")

        if scale:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df = scale_features(df, numeric_cols)
            logging.info("Numeric ustunlar skaler qilindi.")

        logging.info("Data preprocessing tugadi.")
        return df

    except Exception as e:
        logging.error(f"Data preprocessingda xato yuz berdi: {e}")
        return df
