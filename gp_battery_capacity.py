import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# Set seed for reproducibility
np.random.seed(0)


def first_valid(s):
    """Return the first non-NaN value in the series, or NaN if none."""
    s = s.dropna()
    return s.iloc[0] if len(s) > 0 else np.nan


# ==========================================
# 1. Load MAT file and extract cycle data
# ==========================================
data = loadmat("/content/B0005.mat")
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

    # Drop empty columns so they don't force a length of 0
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
full.to_csv("B005.csv", index=False)
print("Saved:", full.shape)
print(full.groupby("cycle_type").size())

# ==========================================
# 2. Split dataset by cycle type
# ==========================================
full = pd.read_csv("B005.csv")

for ctype in ["charge", "discharge", "impedance"]:
    sub = full[full["cycle_type"] == ctype].dropna(axis=1, how="all")
    sub.to_csv(f"B005_{ctype}.csv", index=False)
    print(f"B005_{ctype}.csv: {sub.shape}")

# ==========================================
# 3. Process discharge capacity sequence
# ==========================================
d = pd.read_csv("B005_discharge.csv")

# One capacity value per discharge cycle
cap = d.groupby("cycle_index")["Capacity"].last().dropna()

# Re-index to a clean 0, 1, 2, 3, ... sequence
cap = cap.reset_index(drop=True)

print(cap.head())
print("Number of discharge cycles:", len(cap))
print(cap.describe())
print("Any NaN?", cap.isna().any())
print("Any zeros or negatives?", (cap <= 0).any())

# Reload discharge dataset for detailed validation
d = pd.read_csv("/content/B005_discharge.csv")
print(d.columns.tolist())
print(d.shape)
print(d.head(3))

# ==========================================
# 4. Data cleaning and filtering
# ==========================================
cap_raw = d.groupby("cycle_index")["Capacity"].apply(first_valid)
print("Total discharge cycles:", len(cap_raw))
print("Cycles with a capacity value:", cap_raw.notna().sum())
print("Cycles with NaN capacity:", cap_raw.isna().sum())

# Drop cycle with missing capacity
cap = cap_raw.dropna()
print("After dropping NaN:", len(cap))

# Drop aborted or short cycles (fewer than 100 rows)
lengths = d.groupby("cycle_index").size()
long_enough = lengths[lengths >= 100].index
cap = cap[cap.index.isin(long_enough)]
print("After dropping short cycles:", len(cap))
print("Non-positive capacity values:", (cap <= 0).sum())
cap = cap[cap > 0]

# Sanity check
print("\nFinal capacity stats:")
print(cap.describe())

# Re-index in sequence
cap = cap.reset_index(drop=True)
print("Final number of cycles:", len(cap))
print(cap.head())

# ==========================================
# 5. Data visualization
# ==========================================
plt.figure(figsize=(10, 4))
plt.plot(cap.index, cap.values, "k.-", markersize=4)
plt.xlabel("Discharge cycle (sequential)")
plt.ylabel("Capacity (Ah)")
plt.title("B0005 capacity fade — cleaned")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("capacity_clean.png", dpi=150)
plt.show()

# ==========================================
# 6. Gaussian Process & Random Forest Models
# ==========================================
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
    length_scale=50.0, length_scale_bounds=(5.0, 500), nu=1.5
) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

X = cap.index.values.reshape(-1, 1).astype(float)
y = cap.values

# Time-aware split: train on early cycles, test on later cycles
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

# Point-prediction accuracy
rmse = np.sqrt(np.mean((y_test - y_mean) ** 2))
print(f"GP RMSE: {rmse:.4f} Ah")

# Calibration: how often does the true value fall inside the 95% interval?
z = 1.96
inside = (y_test >= y_mean - z * y_std) & (y_test <= y_mean + z * y_std)
coverage = inside.mean()
print(f"95% interval coverage: {coverage:.1%}")

# Average interval width — how informative are the intervals?
mean_width = np.mean(2 * z * y_std)
print(f"Mean 95% interval width: {mean_width:.4f} Ah")

# Random Forest comparison
rf = RandomForestRegressor(n_estimators=200, random_state=0)
rf.fit(X_train, y_train.ravel())
y_rf = rf.predict(X_test)

rmse_rf = np.sqrt(np.mean((y_test.ravel() - y_rf) ** 2))
print(f"Random Forest RMSE: {rmse_rf:.4f} Ah")
print(f"GP RMSE:            {rmse:.4f} Ah")

# ==========================================
# 7. Final Model Evaluation Plot
# ==========================================
plt.figure(figsize=(11, 4.5))

# Training data
plt.plot(X_train.ravel(), y_train, "k.", markersize=6, label="Train")

# Test data
plt.plot(X_test.ravel(), y_test, "r.", markersize=6, label="Test (true)")

# GP mean
plt.plot(X_test.ravel(), y_mean, "b-", linewidth=2, label="GP mean")

# 95% interval
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
    f"GP regression on B0005 capacity fade   |   RMSE = {rmse:.4f} Ah   |   Coverage = {coverage:.0%}"
)
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("gp_capacity_fade.png", dpi=150)
plt.show()
