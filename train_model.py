import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create models folder
os.makedirs("models", exist_ok=True)

def train_and_save_model(dataset_path, target_column, output_name):
    df = pd.read_csv(dataset_path)

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    df = pd.get_dummies(df)  # one-hot encode

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    # Save model, scaler and feature names
    pickle.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_names": list(X.columns)
        },
        open(f"models/{output_name}.pkl", "wb")
    )

    print(f"✅ Trained and saved: {output_name}.pkl")


# TRAIN MODELS
train_and_save_model("kidney_disease.csv", "classification", "kidney_model")
train_and_save_model("diabetes.csv", "Outcome", "diabetes_model")
train_and_save_model("heart.csv", "target", "heart_model")

print("✅ All models trained successfully!")
