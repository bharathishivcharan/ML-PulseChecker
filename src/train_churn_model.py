import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.linear_model import LogisticRegression
from scipy import stats
import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from evidently import Report
from evidently.presets import DataDriftPreset

from sklearn.metrics import precision_recall_curve
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




# Simulate production data drift
production_data = X_test.copy()
production_data["total_sessions"] *= 1.2  # artificial shift

# Combine into dataframes
train_df = pd.DataFrame(X_train, columns=X.columns)
prod_df = pd.DataFrame(production_data, columns=X.columns)

# create and run drift report
drift_report = Report(metrics=[DataDriftPreset()])
drift_result = drift_report.run(
    reference_data=train_df,
    current_data=prod_df
)

#Save metrics

os.makedirs("assets", exist_ok=True)
with open("assets/data_drift_report.json", "w") as f:
    f.write(drift_result.json())

print("Data drift results saved to assets/data_drift_report.json")

#Random forest
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

#XG Boost
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

#Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, xgb_proba, n_bins=10)
plt.plot(prob_pred, prob_true)
plt.plot([0,1],[0,1])
plt.title("Calibration Curve (XGBoost)")
plt.savefig("assets/calibration_curve.png")
plt.close()


#SHAP
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("assets/shap_summary.png")
plt.close()

#SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_res, y_res)

print("SMOTE ROC:", roc_auc_score(y_test, rf_smote.predict_proba(X_test)[:,1]))

control = np.random.normal(loc=5.0, scale=1.0, size=1000)
treatment = np.random.normal(loc=5.3, scale=1.0, size=1000)

t_stat, p_value = stats.ttest_ind(control, treatment)

print("\n=== A/B Test Simulation ===")
print("T-statistic:", t_stat)
print("P-value:", p_value)
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

os.makedirs("assets", exist_ok=True)

#Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

lr_proba = log_reg.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_proba)

print("\nLogistic Regression ROC-AUC:", lr_auc)

# Confusion Matrix Plot 
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("assets/confusion_matrix.png")
plt.close()

#  ROC Curve 
rf_fpr, rf_tpr, _ = roc_curve(y_test, y_proba)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)

plt.plot(rf_fpr, rf_tpr, label="Random Forest")
plt.plot(lr_fpr, lr_tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("assets/roc_curve.png")
plt.close()

# --- Feature Importance ---
plt.bar(FEATURES, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("assets/feature_importance.png")
plt.close()

print("Evaluation plots saved in /assets directory.")