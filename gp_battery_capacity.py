"""
Gaussian Process Regression for Battery Capacity Fade Prediction
-----------------------------------------------------------------
Predicts the capacity (Ah) of a Li-ion battery over its discharge lifetime
using a Gaussian Process, with a focus on predictive uncertainty.

Data: NASA B0005 battery dataset (discharge cycles only).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
CSV_PATH = "data/B005_discharge.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 0
TRAIN_FRACTION = 0.75

# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
d = pd.read_csv(CSV_PATH)
print(f"Loaded {d.shape[0]} rows, {d['cycle_index'].nunique()} unique cycles")

# ----------------------------------------------------------------------
# 2. Clean
# ----------------------------------------------------------------------
def first_valid(s):
    """Return the first non-NaN value in a series, or NaN if none."""
    s = s.dropna()
    return s.iloc[0] if len(s) > 0 else np.nan

# One capacity value per cycle
cap = d.groupby("cycle_index")["Capacity"].apply(first_valid)

# Drop cycles with no capacity value
cap = cap.dropna()

# Drop aborted cycles (< 100 samples)
lengths = d.groupby("cycle_index").size()
cap = cap[cap.index.isin(lengths[lengths >= 100].index)]

# Drop non-physical values
cap = cap[cap > 0]

# Re-index to a clean sequential counter
cap = cap.reset_index(drop=True)

print(f"After cleaning: {len(cap)} cycles")
print(f"Capacity range: {cap.min():.3f} – {cap.max():.3f} Ah")

# ----------------------------------------------------------------------
# 3. Plot cleaned raw data
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(cap.index, cap.values, "k.-", markersize=4)
plt.xlabel("Discharge cycle (sequential)")
plt.ylabel("Capacity (Ah)")
plt.title("B0005 capacity fade — cleaned")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "capacity_clean.png"), dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 4. Time-aware split
# ----------------------------------------------------------------------
X = cap.index.values.reshape(-1, 1).astype(float)
y = cap.values

n_train = int(TRAIN_FRACTION * len(X))
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

print(f"Train: {len(X_train)} cycles | Test: {len(X_test)} cycles")

# ----------------------------------------------------------------------
# 5. Gaussian Process
# ----------------------------------------------------------------------
kernel = (
    ConstantKernel(1.0, (1e-2, 1e2))
    * Matern(length_scale=50.0, length_scale_bounds=(5.0, 500.0), nu=1.5)
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=RANDOM_STATE,
)
gp.fit(X_train, y_train)

print(f"Optimised kernel: {gp.kernel_}")

# ----------------------------------------------------------------------
# 6. Predict + evaluate
# ----------------------------------------------------------------------
y_mean, y_std = gp.predict(X_test, return_std=True)
y_true = np.asarray(y_test).ravel()
y_gp = np.asarray(y_mean).ravel()

z = 1.96  # 95% Gaussian interval

rmse_gp = np.sqrt(np.mean((y_true - y_gp) ** 2))
inside = (y_true >= y_gp - z * y_std) & (y_true <= y_gp + z * y_std)
coverage_gp = inside.mean()
width_gp = np.mean(2 * z * y_std)

print(f"GP RMSE: {rmse_gp:.4f} Ah")
print(f"95% interval coverage: {coverage_gp:.1%}")
print(f"Mean 95% interval width: {width_gp:.4f} Ah")

# ----------------------------------------------------------------------
# 7. Random Forest baseline
# ----------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train.ravel())
y_rf = rf.predict(X_test)

rmse_rf = np.sqrt(np.mean((y_true - y_rf) ** 2))
print(f"Random Forest RMSE: {rmse_rf:.4f} Ah")

# ----------------------------------------------------------------------
# 8. Main result figure
# ----------------------------------------------------------------------
plt.figure(figsize=(11, 4.5))
plt.plot(X_train.ravel(), y_train, "k.", markersize=6, label="Train")
plt.plot(X_test.ravel(), y_true, "r.", markersize=6, label="Test (true)")
plt.plot(X_test.ravel(), y_gp, "b-", linewidth=2, label="GP mean")
plt.fill_between(
    X_test.ravel(),
    y_gp - z * y_std,
    y_gp + z * y_std,
    alpha=0.25,
    color="blue",
    label="95% credible interval",
)
plt.xlabel("Discharge cycle (sequential)")
plt.ylabel("Capacity (Ah)")
plt.title(
    f"GP regression on B0005 capacity fade   |   "
    f"RMSE = {rmse_gp:.4f} Ah   |   Coverage = {coverage_gp:.0%}"
)
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "gp_capacity_fade.png"), dpi=150)
plt.close()

print(f"Figures saved to {FIG_DIR}/")
