# src/plot_metrics.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_metrics(log_path="results/experiment_log.csv"):
    df = pd.read_csv(log_path)

    # Filter only Ridge and Lasso rows with GridSearchCV
    df = df[(df["model"].isin(["Ridge", "Lasso"])) & (df["training_method"] == "GridSearchCV")]

    metrics = ["r2", "mae", "rmse"]
    df_melted = df.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="value")

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_melted, x="metric", y="value", hue="model")
    plt.title("Ridge vs Lasso: Regression Metrics")
    plt.tight_layout()
    plt.show()
