# =====================================================================
# Repeated K-fold cross-validation
# Does Central U.S. Weather Explain National Crop Yields? (1981-2023)
#
# Run from the repository root:  python code/03_kfold_cross_validation.py
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
df = pd.read_csv("data/processed/us_crop_yield_climate_1981_2023.csv")

# ---------------- Settings ----------------
K = 5            # number of folds (about 8-9 test seasons per fold)
REPEATS = 20     # repeat with different random splits for stable estimates
CROPS = ["Corn", "Soybeans", "Rice", "Cotton", "Wheat"]
SEASON = {"Corn": "MaySep", "Soybeans": "MaySep", "Rice": "MaySep",
          "Cotton": "MaySep", "Wheat": "OctJun"}
VARS = ["Hottest_C", "Coldest_C", "Precip_total_mm", "RH_pct"]
MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                                   learning_rate=0.05, random_state=42),
}


def crop_data(crop):
    """Climate features and detrended yield anomaly (%), same as the main analysis."""
    cols = [f"{SEASON[crop]}_{v}" for v in VARS]
    d = df[["Year", f"{crop}_yield_kg_ha"] + cols].dropna()
    yr, yld = d["Year"].values, d[f"{crop}_yield_kg_ha"].values
    trend = np.polyval(np.polyfit(yr, yld, 1), yr)
    return d[cols].values, (yld - trend) / trend * 100


# ---------------- Repeated K-fold ----------------
# In each repeat, every season is predicted exactly once by a model that did not
# see it. Metrics are computed on these pooled out-of-fold predictions, which is
# more stable than averaging R² over tiny folds of 8-9 seasons.
rows = []
for crop in CROPS:
    X, y = crop_data(crop)
    for name, model in MODELS.items():
        for rep in range(REPEATS):
            cv = KFold(n_splits=K, shuffle=True, random_state=rep)
            pred = cross_val_predict(model, X, y, cv=cv)
            rows.append({"Crop": crop, "Model": name, "Repeat": rep,
                         "R2": r2_score(y, pred),
                         "RMSE": np.sqrt(mean_squared_error(y, pred)),
                         "MAE": mean_absolute_error(y, pred)})
        print(f"{crop:9s} {name:18s} done")

res = pd.DataFrame(rows)
res.to_csv("results/kfold_all_repeats.csv", index=False)

# ---------------- Summary table (mean ± SD across repeats) ----------------
summary = res.groupby(["Crop", "Model"]).agg(
    R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
    RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
    MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std")).round(3)
summary = summary.reindex(CROPS, level="Crop")
print(f"\n=== {REPEATS}x repeated {K}-fold CV (target = % yield anomaly) ===")
print(summary)
summary.to_csv("results/kfold_summary.csv")

# Paper-ready R² table: "mean ± SD"
table = res.groupby(["Crop", "Model"])["R2"].agg(["mean", "std"])
table = (table["mean"].map("{:.3f}".format) + " ± " + table["std"].map("{:.3f}".format)).unstack()
table = table.loc[CROPS, list(MODELS)]
print(f"\n=== Paper table: cross-validated R² (mean ± SD, {REPEATS}x{K}-fold) ===")
print(table)
table.to_csv("results/kfold_r2_table.csv")

# ---------------- Figure: R² distribution across repeats ----------------
plt.figure(figsize=(10, 4.5))
sns.boxplot(data=res, x="Crop", y="R2", hue="Model", order=CROPS)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.ylabel("Cross-validated R²")
plt.title(f"Model performance across {REPEATS} repeats of {K}-fold cross-validation")
plt.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig07_kfold_r2_boxplot.png", dpi=200)
plt.show()
