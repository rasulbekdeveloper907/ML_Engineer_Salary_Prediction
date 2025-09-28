import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def show_info(self):
        """Dataset haqida umumiy ma’lumot"""
        print(self.df.info())

    def show_summary(self):
        """Numerik ustunlarning statistikasi"""
        print(self.df.describe())

    def plot_salary_distribution(self, column="salary"):
        """Maosh taqsimotini chizish"""
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[column], kde=True)
        plt.title("Salary Distribution")
        plt.show()

    def correlation_matrix(self):
        """Korrelyatsiya matritsasini chizish"""
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()
