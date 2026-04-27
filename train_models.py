import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -----------------------------
# Parkinson's Model
# -----------------------------
print("Training Parkinson Model...")

parkinsons = pd.read_csv("data/parkinsons.csv")

# Remove string column
parkinsons = parkinsons.drop("name", axis=1)

X = parkinsons.drop("status", axis=1)
y = parkinsons["status"]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Parkinson Accuracy:", accuracy_score(y_test, y_pred))

# Save
pickle.dump(model, open("models/parkinsons_model.pkl", "wb"))
pickle.dump(scaler, open("models/parkinsons_scaler.pkl", "wb"))

print("Parkinson model saved\n")


# -----------------------------
# Kidney Disease Model
# -----------------------------
print("Training Kidney Model...")

kidney = pd.read_csv("data/kidney_disease.csv")

# Remove unwanted column
if "id" in kidney.columns:
    kidney = kidney.drop("id", axis=1)

# Replace '?' with NaN
kidney.replace("?", pd.NA, inplace=True)

# Clean column names
kidney.columns = kidney.columns.str.strip()

# Target column cleaning
kidney["classification"] = kidney["classification"].str.strip()
kidney["classification"] = kidney["classification"].map({"ckd": 1, "notckd": 0})

# Convert to numeric safely
for col in kidney.columns:
    kidney[col] = pd.to_numeric(kidney[col], errors='coerce')

# Fill missing values
kidney = kidney.fillna(kidney.mean(numeric_only=True))

# Split
X = kidney.drop("classification", axis=1)
y = kidney["classification"]

# Encode categorical
X = pd.get_dummies(X)

# Save columns (important for app)
pickle.dump(X.columns, open("models/kidney_columns.pkl", "wb"))

# Final safety
X = X.fillna(0)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Kidney Accuracy:", accuracy_score(y_test, y_pred))

# Save
pickle.dump(model, open("models/kidney_model.pkl", "wb"))
pickle.dump(scaler, open("models/kidney_scaler.pkl", "wb"))

print("Kidney model saved\n")


# -----------------------------
# Liver Disease Model
# -----------------------------
print("Training Liver Model...")

liver = pd.read_csv("data/indian_liver_patient.csv")

# Handle missing values
liver = liver.ffill()

X = liver.drop("Dataset", axis=1)
y = liver["Dataset"]

# Convert target to (binary) 0 and 1
y = y.map({1: 1, 2: 0})

# Encode categorical
X = pd.get_dummies(X)

# Save columns
pickle.dump(X.columns, open("models/liver_columns.pkl", "wb"))

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Liver Accuracy:", accuracy_score(y_test, y_pred))

# Save
pickle.dump(model, open("models/liver_model.pkl", "wb"))
pickle.dump(scaler, open("models/liver_scaler.pkl", "wb"))

print("Liver model saved\n")


print("✅ ALL MODELS TRAINED SUCCESSFULLY")

