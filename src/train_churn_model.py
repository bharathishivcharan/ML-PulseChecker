import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)



df = pd.read_csv("data/analytics/user_with_churn.csv")

FEATURES = [
    "total_events",
    "total_sessions",
    "avg_events_per_session",
    "revenue_per_session",
    "active_days"
]

X = df[FEATURES]
y = df["churned"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)



y_proba = model.predict_proba(X_test)[:, 1]



precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Optimize for F1 score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\nOptimal Threshold based on F1: {best_threshold:.4f}")

# Apply optimized threshold
y_pred = (y_proba >= best_threshold).astype(int)



print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== ROC-AUC Score ===")
print(roc_auc_score(y_test, y_proba))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))



print("\n=== Feature Importances ===")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in indices:
    print(f"{FEATURES[i]}: {importances[i]:.4f}")



os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/churn_model.pkl")
print("\nModel saved to models/churn_model.pkl")

# Save test predictions
test_output = X_test.copy()
test_output["churned_true"] = y_test.values
test_output["churn_proba"] = y_proba
test_output["churn_pred"] = y_pred

test_output.to_csv("models/test_predictions.csv", index=False)
print("Test predictions saved to models/test_predictions.csv")