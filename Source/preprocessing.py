import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def handle_missing(self):
        """Bo‘sh qiymatlarni to‘ldirish yoki o‘chirish"""
        self.df = self.df.dropna()
        return self.df

    def encode_categorical(self):
        """Categorical ustunlarni LabelEncoder bilan kodlash"""
        le = LabelEncoder()
        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col] = le.fit_transform(self.df[col])
        return self.df

    def get_features_and_target(self, target="salary"):
        """X va y ni ajratib olish"""
        X = self.df.drop(columns=[target])
        y = self.df[target]
        return X, y
