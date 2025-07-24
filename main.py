from src.data_loader import get_data, load_config
from src.preprocess import preprocess_data
from src.models.logistic_model import train_model
from sklearn.metrics import accuracy_score
from src.evaluator import regression_metrics

from sklearn.linear_model import Ridge, Lasso
import joblib 


# Load and preprocess
df = get_data()
config = load_config()
target = config["target_column"]
X_train, X_test, y_train, y_test = preprocess_data(df, target)

# Train model
model = train_model(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"📊 Logistic Regression Accuracy: {acc:.4f}")

from src.experiment_logger import log_experiment

# Simulated test result
log_experiment({
    "model": "TestModel",
    "loss_function": "MSE",
    "training_method": "manual-test",
    "hyperparameters": "alpha=0.01",
    "r2": 0.90,
    "adj_r2": 0.89,
    "mae": 0.23,
    "mse": 0.15,
    "rmse": 0.39,
    "accuracy": None,
    "f1": None,
    "notes": "First test log"
})

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Ridge
ridge_params = {"alpha": [0.01, 0.1, 1.0, 10.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring="r2")
ridge_grid.fit(X_train, y_train)

best_ridge = ridge_grid.best_estimator_
ridge_preds = best_ridge.predict(X_test)

ridge_results = regression_metrics(y_test, ridge_preds, n_features=X_train.shape[1])
print("📊 Ridge (GridSearch) Metrics:", ridge_results)

log_experiment({
    "model": "Ridge",
    "loss_function": "MSE",
    "training_method": "GridSearchCV",
    "hyperparameters": f"alpha={ridge_grid.best_params_['alpha']}",
    **ridge_results,
    "accuracy": None,
    "f1": None,
    "notes": "GridSearch tuned Ridge"
})


from sklearn.linear_model import Ridge
from src.evaluator import regression_metrics
from src.experiment_logger import log_experiment

# Train Ridge model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Predict
ridge_preds = ridge.predict(X_test)

# Evaluate
ridge_results = regression_metrics(y_test, ridge_preds, n_features=X_train.shape[1])
print("📊 Ridge Regression Metrics:", ridge_results)

# Log to experiment tracker
log_experiment({
    "model": "Ridge",
    "loss_function": "MSE",
    "training_method": "sklearn-Ridge",
    "hyperparameters": "alpha=1.0",
    **ridge_results,
    "accuracy": None,
    "f1": None,
    "notes": "First Ridge run with basic alpha"
})

# Hyperparameter tuning for Lasso
lasso_params = {"alpha": [0.001, 0.01, 0.1, 1.0]}
lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=5, scoring="r2")
lasso_grid.fit(X_train, y_train)

best_lasso = lasso_grid.best_estimator_
lasso_preds = best_lasso.predict(X_test)

lasso_results = regression_metrics(y_test, lasso_preds, n_features=X_train.shape[1])
print("📊 Lasso (GridSearch) Metrics:", lasso_results)

log_experiment({
    "model": "Lasso",
    "loss_function": "MSE",
    "training_method": "GridSearchCV",
    "hyperparameters": f"alpha={lasso_grid.best_params_['alpha']}",
    **lasso_results,
    "accuracy": None,
    "f1": None,
    "notes": "GridSearch tuned Lasso"
})


from sklearn.linear_model import Lasso

# Train Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Predict
lasso_preds = lasso.predict(X_test)

# Evaluate
lasso_results = regression_metrics(y_test, lasso_preds, n_features=X_train.shape[1])
print("📊 Lasso Regression Metrics:", lasso_results)

# Log to experiment tracker
log_experiment({
    "model": "Lasso",
    "loss_function": "MSE",
    "training_method": "sklearn-Lasso",
    "hyperparameters": "alpha=0.1",
    **lasso_results,
    "accuracy": None,
    "f1": None,
    "notes": "Lasso run with alpha=0.1"
})


from src.plot_metrics import plot_model_metrics

plot_model_metrics()


import os
import pandas as pd


if os.path.exists("results/experiment_log.csv"):
    log_df = pd.read_csv("results/experiment_log.csv")
    regression_df = log_df[log_df["r2"].notnull()]
    best_row = regression_df.loc[regression_df["r2"].idxmax()]
    best_model_name = best_row["model"]
    print(f"🏆 Best model: {best_model_name} with R² = {best_row['r2']}")
    os.makedirs("models", exist_ok=True)
    if best_model_name == "LinearRegression":
        joblib.dump(linreg, "models/best_model.pkl")
    elif best_model_name == "Ridge":
        joblib.dump(best_ridge, "models/best_model.pkl")
    elif best_model_name == "Lasso":
        joblib.dump(best_lasso, "models/best_model.pkl")
    else:
        print("⚠️ Model not recognized for saving.")
else:
    print("⚠️ experiment_log.csv not found. Skipping best model selection.")


