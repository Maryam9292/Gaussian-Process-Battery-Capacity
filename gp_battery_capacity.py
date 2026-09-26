"""
Gaussian Process Regression for Battery Capacity Fade Prediction
----------------------------------------------------------------
Predicts capacity (Ah) over discharge cycles for NASA B0005 using a
Gaussian Process with a Matern 3/2 kernel, and benchmarks against a
Random Forest baseline under a time-aware train/test split.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor

np.random.seed(0)
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

MAT_PATH = "data/B0005.mat"
DISCHARGE_CSV = "data/B005_discharge.csv"
FIG_DIR = "figures"

# ----------------------------------------------------------------------
# 1. Convert .mat -> B005.csv (all cycles)
# ----------------------------------------------------------------------
data = loadmat(MAT_PATH)
cycles = data["B0005"]["cycle"][0, 0]
n_cycles = cycles.shape[1]
print(f"Found {n_cycles} cycles")

rows = []
for i in range(n_cycles):
    c = cycles[0, i]
    ctype = str(c["type"][0])
    ambient = float(np.asarray(c["ambient_temperature"]).ravel()[0])
    d = c["data"][0]

    cols = {}
    for f in d.dtype.names:
        val = d[f]
        if isinstance(val, np.ndarray) and val.dtype == object:
            val = val[0] if len(val) > 0 else np.array([])
        cols[f] = np.asarray(val).ravel()

    non_empty = {k: v for k, v in cols.items() if v.size > 0}
    if not non_empty:
        print(f"Skipped cycle {i} ({ctype}): all fields empty")
        continue

    max_len = max(len(v) for v in non_empty.values())
    padded = {}
    for k, v in non_empty.items():
        if len(v) < max_len:
            v = np.concatenate([v, np.full(max_len - len(v), np.nan)])
        padded[k] = v

    df = pd.DataFrame(padded)
    df.insert(0, "cycle_index", i)
    df.insert(1, "cycle_type", ctype)
    df.insert(2, "ambient_temperature", ambient)
    rows.append(df)

full = pd.concat(rows, ignore_index=True)
full.to_csv("data/B005.csv", index=False)
print("Saved B005.csv:", full.shape)

# ----------------------------------------------------------------------
# 2. Split into cycle-type CSVs
# ----------------------------------------------------------------------
for ctype in ["charge", "discharge", "impedance"]:
    sub = full[full["cycle_type"] == ctype].dropna(axis=1, how="all")
    sub.to_csv(f"data/B005_{ctype}.csv", index=False)
    print(f"B005_{ctype}.csv: {sub.shape}")

# ----------------------------------------------------------------------
# 3. Load discharge cycles and clean
# ----------------------------------------------------------------------
d = pd.read_csv(DISCHARGE_CSV)
print("Columns:", d.columns.tolist())
print("Shape:", d.shape)

def first_valid(s):
    s = s.dropna()
    return s.iloc[0] if len(s) > 0 else np.nan

cap_raw = d.groupby("cycle_index")["Capacity"].apply(first_valid)
print("Total discharge cycles:", len(cap_raw))
print("Cycles with capacity:", cap_raw.notna().sum())
print("Cycles with NaN:", cap_raw.isna().sum())

cap = cap_raw.dropna()
print("After dropping NaN:", len(cap))

lengths = d.groupby("cycle_index").size()
long_enough = lengths[lengths >= 100].index
cap = cap[cap.index.isin(long_enough)]
print("After dropping short cycles:", len(cap))

print("Non-positive capacity:", (cap <= 0).sum())
cap = cap[cap > 0]

cap = cap.reset_index(drop=True)
print("Final number of cycles:", len(cap))
print(cap.describe())

# ----------------------------------------------------------------------
# 4. Plot cleaned data
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(cap.index, cap.values, "k.-", markersize=4)
plt.xlabel("Discharge cycle (sequential)")
plt.ylabel("Capacity (Ah)")
plt.title("B0005 capacity fade - cleaned")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/capacity_clean.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# 5. GP regression
# ----------------------------------------------------------------------
kernel = (
    ConstantKernel(1.0, (1e-2, 1e2))
    * Matern(length_scale=50.0, length_scale_bounds=(5.0, 500), nu=1.5)
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
)

X = cap.index.values.reshape(-1, 1).astype(float)
y = cap.values

n_train = int(0.75 * len(X))
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

print(f"Train: {len(X_train)} cycles, Test: {len(X_test)} cycles")

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=0,
)
gp.fit(X_train, y_train)

print("Optimised kernel:", gp.kernel_)
print("Log marginal likelihood:", gp.log_marginal_likelihood_value_)

y_mean, y_std = gp.predict(X_test, return_std=True)
rmse = np.sqrt(np.mean((y_test - y_mean) ** 2))
print(f"GP RMSE: {rmse:.4f} Ah")

z = 1.96
inside = (y_test >= y_mean - z * y_std) & (y_test <= y_mean + z * y_std)
coverage = inside.mean()
print(f"95% interval coverage: {coverage:.1%}")

mean_width = np.mean(2 * z * y_std)
print(f"Mean 95% interval width: {mean_width:.4f} Ah")

# ----------------------------------------------------------------------
# 6. Random Forest baseline
# ----------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=0)
rf.fit(X_train, y_train.ravel())
y_rf = rf.predict(X_test)

rmse_rf = np.sqrt(np.mean((y_test.ravel() - y_rf) ** 2))
print(f"Random Forest RMSE: {rmse_rf:.4f} Ah")
print(f"GP RMSE:            {rmse:.4f} Ah")

# ----------------------------------------------------------------------
# 7. Main result figure
# ----------------------------------------------------------------------
plt.figure(figsize=(11, 4.5))
plt.plot(X_train.ravel(), y_train, "k.", markersize=6, label="Train")
plt.plot(X_test.ravel(), y_test, "r.", markersize=6, label="Test (true)")
plt.plot(X_test.ravel(), y_mean, "b-", linewidth=2, label="GP mean")
plt.fill_between(
    X_test.ravel(),
    y_mean - 1.96 * y_std,
    y_mean + 1.96 * y_std,
    alpha=0.25,
    color="blue",
    label="95% credible interval",
)
plt.xlabel("Discharge cycle (sequential)")
plt.ylabel("Capacity (Ah)")
plt.title(
    f"GP regression on B0005 capacity fade   |   "
    f"RMSE = {rmse:.4f} Ah   |   Coverage = {coverage:.0%}"
)
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/gp_capacity_fade.png", dpi=150)
plt.show()
