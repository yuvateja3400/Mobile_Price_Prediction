# src/models/logistic_model.py

from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model.
    Returns the trained model.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model
