import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

df = pd.read_csv("data/caregiver_timesheets.csv")

feature_cols = [
    "caregiver_tenure_months", "scheduled_duration_min", "duration_deviation_min",
    "arrival_deviation_min", "tasks_assigned", "task_completion_rate",
    "has_signature", "gps_mismatch_km", "is_weekend",
]
df["entry_method_paper"] = (df["entry_method"] == "paper").astype(int)
feature_cols.append("entry_method_paper")

X = df[feature_cols]
y = df["needs_manual_review"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# --- Logistic Regression ---
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_s, y_train)
pred_lr = logreg.predict(X_test_s)
proba_lr = logreg.predict_proba(X_test_s)[:, 1]

results["logistic_regression"] = {
    "accuracy": accuracy_score(y_test, pred_lr),
    "precision": precision_score(y_test, pred_lr),
    "recall": recall_score(y_test, pred_lr),
    "f1": f1_score(y_test, pred_lr),
    "roc_auc": roc_auc_score(y_test, proba_lr),
}

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]

results["random_forest"] = {
    "accuracy": accuracy_score(y_test, pred_rf),
    "precision": precision_score(y_test, pred_rf),
    "recall": recall_score(y_test, pred_rf),
    "f1": f1_score(y_test, pred_rf),
    "roc_auc": roc_auc_score(y_test, proba_rf),
}

print(json.dumps(results, indent=2))

with open("reports/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# --- Feature importance (Random Forest) ---
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(7, 5))
importances.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=150)
plt.close()

# --- ROC curves ---
plt.figure(figsize=(6, 5))
for name, proba in [("Logistic Regression", proba_lr), ("Random Forest", proba_rf)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150)
plt.close()

# --- Confusion matrix (Logistic Regression, chosen as production model) ---
cm = confusion_matrix(y_test, pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["OK", "Needs Review"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Logistic Regression")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()

# --- Save logistic regression coefficients for the interactive dashboard (JS reimplementation) ---
coef_export = {
    "features": feature_cols,
    "coefficients": logreg.coef_[0].tolist(),
    "intercept": float(logreg.intercept_[0]),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
}
with open("models/logreg_coefficients.json", "w") as f:
    json.dump(coef_export, f, indent=2)

print("\nSaved metrics, plots, and exported coefficients.")
