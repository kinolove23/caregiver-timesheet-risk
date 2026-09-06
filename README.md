# Caregiver Timesheet Review-Risk Prediction

Predicting which home-care visit timesheets are likely to need **manual review**
before payroll/billing, so review effort can be focused where it's actually needed.

## Background

Home-care agencies collect a timesheet after every caregiver visit (time in/out, tasks
completed, signature). Not every timesheet passes automatic verification — some need a
staff member to check them by hand for missing signatures, incomplete task lists, or
inconsistent timing. This project explores whether a simple ML model can predict *in
advance* which visits are likely to need that manual check.

**Data note:** this project uses a synthetically generated dataset (`generate_data.py`)
that mirrors realistic relationships between visit features and review outcomes. No real
company, caregiver, or patient data is used or represented.

## What's in this repo

| Path | Description |
|---|---|
| `generate_data.py` | Generates the synthetic visit dataset (`data/caregiver_timesheets.csv`) |
| `train_model.py` | Trains & compares Logistic Regression and Random Forest models, saves metrics/plots |
| `notebooks/caregiver_timesheet_review_risk.ipynb` | Full EDA + modeling walkthrough with results |
| `reports/` | Saved metrics, feature importance, ROC curve, confusion matrix |
| `models/logreg_coefficients.json` | Exported model weights (used by the dashboard) |
| `dashboard/timesheet_risk_dashboard.jsx` | Interactive React dashboard — enter visit details, see live risk score |

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.747 | 0.658 | 0.575 | 0.614 | 0.835 |
| Random Forest | 0.791 | 0.744 | 0.615 | 0.674 | 0.876 |

Logistic Regression was selected as the production model despite slightly lower accuracy,
because its coefficients are fully transparent and can be re-implemented outside Python
(e.g., in the interactive dashboard) without needing to serve the sklearn model itself.

**Top predictors:** task completion rate, number of tasks assigned, and missing signature —
all signals available at the moment a timesheet is submitted.

## Running it yourself

```bash
pip install pandas numpy scikit-learn matplotlib
python generate_data.py
python train_model.py
```

Then open `notebooks/caregiver_timesheet_review_risk.ipynb` in Jupyter to see the full
walkthrough with plots.

## Dashboard

`dashboard/timesheet_risk_dashboard.jsx` is a small React app that re-implements the
trained logistic regression scoring function client-side, so you can adjust visit details
with sliders/toggles and see the predicted review risk update live.
