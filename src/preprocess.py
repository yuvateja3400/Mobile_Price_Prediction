# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Prepares the dataset for training:
    - Adds engineered features
    - Drops redundant columns
    - Splits into train/test
    - Scales features
    Returns: X_train_scaled, X_test_scaled, y_train, y_test
    """

    # 🧠 Step 1: Feature Engineering
    df["px_area"] = df["px_height"] * df["px_width"]

    # 🔻 Step 2: Drop redundant columns
    df.drop(columns=["px_height", "px_width"], inplace=True)

    # 🧱 Step 3: Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 🔀 Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔧 Step 5: Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
