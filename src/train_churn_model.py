
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/analytics/user_with_churn.csv")

# Features 
FEATURES = ["total_events", "total_sessions", "avg_events_per_session", "revenue_per_session", "active_days"]

X = df[FEATURES]
y = df["churned"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/churn_model.pkl")
print("Model saved to models/churn_model.pkl")

X_test["churned_true"] = y_test
X_test["churn_proba"] = y_proba
X_test.to_csv("models/test_predictions.csv", index=False)