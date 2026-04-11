import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# STEP 1: Create Dataset
# -----------------------------
np.random.seed(42)

n = 200

df = pd.DataFrame({
    "age": np.random.randint(20, 70, n),
    "fever": np.random.randint(0, 2, n),
    "tiredness": np.random.randint(0, 2, n),
    "cough": np.random.randint(0, 2, n)
})

# Simple logic-based target
df["risk"] = (
    (df["age"] > 50).astype(int) +
    df["fever"] +
    df["tiredness"] +
    df["cough"]
) >= 2

df["risk"] = df["risk"].astype(int)

# -----------------------------
# STEP 2: Split Data
# -----------------------------
X = df[["age", "fever", "tiredness", "cough"]]
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 3: Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# STEP 4: Accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL TRAINED SUCCESSFULLY")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("==============================\n")

# -----------------------------
# STEP 5: User Input Prediction (FIXED WARNING)
# -----------------------------
print("AI HEALTH PREDICTION SYSTEM")
print("------------------------------")

age = int(input("Enter Age: "))
fever = int(input("Fever (0/1): "))
tiredness = int(input("Tiredness (0/1): "))
cough = int(input("Cough (0/1): "))

input_data = pd.DataFrame([[age, fever, tiredness, cough]],
                          columns=["age", "fever", "tiredness", "cough"])

prediction = model.predict(input_data)

print("\n------------------------------")

if prediction[0] == 1:
    print("⚠️ HIGH RISK DETECTED")
    print("Recommendation: Consult a doctor")
else:
    print("✅ LOW RISK")
    print("Recommendation: Stay healthy")

print("------------------------------\n")
