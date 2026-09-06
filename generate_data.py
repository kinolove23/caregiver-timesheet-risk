"""
Generates a synthetic dataset of home-care caregiver timesheets.
This is SIMULATED data inspired by real-world home care operations
(no real company or patient data is used).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 6000
N_CAREGIVERS = 120

caregiver_ids = rng.integers(1, N_CAREGIVERS + 1, size=N)
caregiver_tenure_months = rng.integers(1, 60, size=N)

scheduled_duration = rng.choice([30, 60, 90, 120], size=N, p=[0.15, 0.45, 0.25, 0.15])

# Actual duration deviates more for less-tenured caregivers
tenure_factor = np.clip(1.2 - caregiver_tenure_months / 60, 0.3, 1.2)
duration_noise = rng.normal(0, 12, size=N) * tenure_factor
actual_duration = np.clip(scheduled_duration + duration_noise, 10, None).round()
duration_deviation = actual_duration - scheduled_duration

# Arrival deviation (minutes late/early)
arrival_deviation = rng.normal(0, 10, size=N) * tenure_factor
arrival_deviation = arrival_deviation.round()

# Tasks assigned vs completed
tasks_assigned = rng.integers(2, 7, size=N)
completion_rate_base = np.clip(rng.normal(0.9, 0.12, size=N), 0.3, 1.0)
# Less tenured caregivers complete slightly fewer tasks on average
completion_rate = np.clip(completion_rate_base - (tenure_factor - 0.5) * 0.05, 0.2, 1.0)
tasks_completed = np.floor(tasks_assigned * completion_rate).astype(int)
tasks_completed = np.minimum(tasks_completed, tasks_assigned)

# Signature present
signature_prob = np.clip(0.97 - (tenure_factor - 0.5) * 0.1, 0.6, 0.99)
has_signature = rng.binomial(1, signature_prob)

# Entry method: mobile app vs paper (paper is more error-prone / older workflow)
entry_method = rng.choice(["mobile_app", "paper"], size=N, p=[0.75, 0.25])

# GPS distance mismatch (km) between check-in location and client address
gps_mismatch = np.abs(rng.normal(0, 0.6, size=N))
gps_mismatch = np.where(entry_method == "paper", 0.0, gps_mismatch)  # only tracked for app

day_of_week = rng.integers(0, 7, size=N)  # 0=Mon ... 6=Sun
is_weekend = (day_of_week >= 5).astype(int)

df = pd.DataFrame({
    "caregiver_id": caregiver_ids,
    "caregiver_tenure_months": caregiver_tenure_months,
    "scheduled_duration_min": scheduled_duration,
    "actual_duration_min": actual_duration,
    "duration_deviation_min": duration_deviation,
    "arrival_deviation_min": arrival_deviation,
    "tasks_assigned": tasks_assigned,
    "tasks_completed": tasks_completed,
    "task_completion_rate": (tasks_completed / tasks_assigned).round(3),
    "has_signature": has_signature,
    "entry_method": entry_method,
    "gps_mismatch_km": gps_mismatch.round(2),
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
})

# ---- Construct target label: needs_manual_review ----
# Business logic (mirrors real verification rules), plus random noise
# to keep the problem realistic (not perfectly separable).

risk_score = (
    (df["has_signature"] == 0).astype(int) * 3.0
    + (df["task_completion_rate"] < 0.7).astype(int) * 2.0
    + (df["duration_deviation_min"].abs() > 25).astype(int) * 1.5
    + (df["arrival_deviation_min"].abs() > 20).astype(int) * 1.0
    + (df["entry_method"] == "paper").astype(int) * 0.8
    + (df["gps_mismatch_km"] > 1.2).astype(int) * 1.2
    + (df["caregiver_tenure_months"] < 6).astype(int) * 0.5
)

noise = rng.normal(0, 1.0, size=N)
final_score = risk_score + noise
threshold = np.quantile(final_score, 0.65)  # ~35% flagged, matching real ~35% manual-review rate
df["needs_manual_review"] = (final_score > threshold).astype(int)

df.to_csv("data/caregiver_timesheets.csv", index=False)
print(df["needs_manual_review"].value_counts(normalize=True))
print(df.head())
print(f"\nSaved {len(df)} rows to data/caregiver_timesheets.csv")
