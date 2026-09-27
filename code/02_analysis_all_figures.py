# =====================================================================
# Does Central U.S. Weather Explain National Crop Yields?
# A Correlation and Machine Learning Analysis (1981-2023)
#
# ONE script: all tables + all figures.
# Run from the repository root:  python code/02_analysis_all_figures.py
# (Colab: upload the repo or the processed CSV; run  !pip install shap -q  first)
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
DATA = "data/processed/us_crop_yield_climate_1981_2023.csv"
FIG_DIR, RES_DIR = "figures", "results"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)
df = pd.read_csv(DATA)

# ---------------- Settings ----------------
CROPS = ["Corn", "Soybeans", "Rice", "Cotton", "Wheat"]
SEASON = {"Corn": "MaySep", "Soybeans": "MaySep", "Rice": "MaySep",
          "Cotton": "MaySep", "Wheat": "OctJun"}     # winter wheat = Oct-Jun
VARS = ["Hottest_C", "Coldest_C", "Precip_total_mm", "RH_pct"]
LABELS = ["Hottest temp (°C)", "Coldest temp (°C)", "Rainfall (mm)", "Humidity (%)"]
SHORT = ["Hottest temp", "Coldest temp", "Rainfall", "Humidity"]
COLORS = {"Corn": "#E1A200", "Soybeans": "#4C9A2A", "Rice": "#2A7AB0",
          "Cotton": "#B04A8C", "Wheat": "#A0522D"}
WEATHER_POINT = (39.83, -98.58)   # lat, lon (Lebanon, Kansas)
MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                                   learning_rate=0.05, random_state=42),
}


def crop_data(crop):
    """Years, climate features (nice labels), and detrended yield anomaly (%)."""
    cols = [f"{SEASON[crop]}_{v}" for v in VARS]
    d = df[["Year", f"{crop}_yield_kg_ha"] + cols].dropna()
    yr, yld = d["Year"].values, d[f"{crop}_yield_kg_ha"].values
    trend = np.polyval(np.polyfit(yr, yld, 1), yr)
    X = d[cols].copy()
    X.columns = LABELS
    return yr, X, (yld - trend) / trend * 100


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.show()


# =====================================================================
# PART 1: ANALYSIS (tables)
# =====================================================================
results, corrs, perm_imp, loo_preds = [], [], {}, {}

for crop in CROPS:
    _, X, y = crop_data(crop)

    # Models with leave-one-out cross-validation
    for name, model in MODELS.items():
        pred = cross_val_predict(model, X.values, y, cv=LeaveOneOut())
        if name == "Random Forest":
            loo_preds[crop] = (y, pred)
        results.append({"Crop": crop, "Model": name, "R2": r2_score(y, pred),
                        "RMSE_%": np.sqrt(mean_squared_error(y, pred)),
                        "MAE_%": mean_absolute_error(y, pred)})

    # Pearson correlations
    for lab, s in zip(LABELS, SHORT):
        r, p = pearsonr(X[lab], y)
        corrs.append({"Crop": crop, "Variable": s, "r": r, "p_value": p})

    # Permutation importance (Random Forest)
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42).fit(X, y)
    pi = permutation_importance(rf, X, y, n_repeats=30, random_state=42)
    perm_imp[crop] = pd.Series(pi.importances_mean, index=SHORT)

# --- Table 1: model performance
res = pd.DataFrame(results).round(3)
print("=== Leave-one-out R² (paper Table 2) (target = % yield anomaly) ===")
print(res.pivot(index="Crop", columns="Model", values="R2"))
res.to_csv(os.path.join(RES_DIR, "loocv_model_results.csv"), index=False)

# --- Table 2: correlations with FDR (Benjamini-Hochberg) correction
corr = pd.DataFrame(corrs)
p = corr["p_value"].values
order = np.argsort(p)
q = np.empty(len(p))
q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
corr["q_FDR"] = np.minimum(q, 1)
corr = corr.round(3)
print("\n=== Significant correlations (paper Table 1) (p < 0.05), with FDR-corrected q ===")
print(corr[corr["p_value"] < 0.05].to_string(index=False))
corr.to_csv(os.path.join(RES_DIR, "correlations_fdr.csv"), index=False)

# --- Table 3: permutation importance
imp = pd.DataFrame(perm_imp).T
print("\n=== Random Forest permutation importance (Fig. S1) ===")
print(imp.round(3))
imp.to_csv(os.path.join(RES_DIR, "permutation_importance.csv"))

# --- Table 4: collinearity among climate factors
print("\n=== Collinearity among climate factors ===")
for crop in ["Corn", "Wheat"]:
    _, X, _ = crop_data(crop)
    print(f"\n{crop} ({SEASON[crop]} season):")
    print(X.corr().round(2))

# --- Table 5: driest growing seasons
print("\n=== Driest May-Sep seasons ===")
print(df[["Year", "Corn_yield_kg_ha", "MaySep_Hottest_C", "MaySep_Precip_total_mm"]]
      .sort_values("MaySep_Precip_total_mm").head(5).to_string(index=False))


# =====================================================================
# PART 2: FIGURES
# =====================================================================

# ---- Fig 1: Study area map ----
REGIONS = {
    "Corn & Soybean Belt": (["IA", "IL", "NE", "MN", "IN", "OH", "SD"], "#F2D16B"),
    "Wheat Belt": (["KS", "ND", "OK", "MT", "WA", "CO"], "#D9A48C"),
    "Southern Rice & Cotton": (["AR", "LA", "MS", "TX", "GA", "AL", "CA"], "#9CC3E0"),
}
try:
    import geopandas as gpd
    states = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip")
    states = states[~states["STUSPS"].isin(["AK", "HI", "PR"])]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    states.plot(ax=ax, color="#EEEEEE", edgecolor="white", linewidth=0.8)
    for name, (codes, color) in REGIONS.items():
        states[states["STUSPS"].isin(codes)].plot(ax=ax, color=color,
                                                  edgecolor="white", linewidth=0.8)
        ax.scatter([], [], color=color, s=120, marker="s", label=name)
    ax.scatter(WEATHER_POINT[1], WEATHER_POINT[0], marker="*", s=450, color="red",
               edgecolor="black", zorder=5, label="NASA POWER weather point")
    ax.set_title("Study area: weather point vs. major crop production regions")
    ax.legend(loc="lower left"); ax.set_axis_off()
except Exception as e:
    print("Map download failed, drawing simple version:", e)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(WEATHER_POINT[1], WEATHER_POINT[0], marker="*", s=450, color="red")
    ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("NASA POWER weather point (39.83°N, 98.58°W)")
save("fig01_study_area_map.png")

# ---- Fig 2: Raw yield trends ----
fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
for ax, crop in zip(axes, CROPS):
    ax.plot(df["Year"], df[f"{crop}_yield_kg_ha"], marker="o", ms=3, color=COLORS[crop])
    ax.set_title(crop); ax.set_xlabel("Year"); ax.set_ylabel("kg/ha")
save("fig02_yield_trends.png")

# ---- Fig 3: Detrended yield anomalies ----
fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
for ax, crop in zip(axes, CROPS):
    yr, _, anom = crop_data(crop)
    ax.bar(yr, anom, color=np.where(anom >= 0, COLORS[crop], "#999999"))
    ax.axhline(0, color="black", linewidth=0.7)
    for drought in [1988, 2012]:
        ax.axvspan(drought - 0.5, drought + 0.5, color="red", alpha=0.15)
    ax.set_ylabel(f"{crop}\n(% from trend)")
axes[0].set_title("Detrended yield anomalies, 1981–2023 (red = major drought years 1988, 2012)")
axes[-1].set_xlabel("Year")
save("fig03_yield_anomalies.png")

# ---- Fig 4: Correlation heatmap ----
r_mat = corr.pivot(index="Crop", columns="Variable", values="r").loc[CROPS, SHORT]
p_mat = corr.pivot(index="Crop", columns="Variable", values="p_value").loc[CROPS, SHORT]
annot = r_mat.round(2).astype(str) + p_mat.map(
    lambda v: "**" if v < 0.01 else "*" if v < 0.05 else "")
plt.figure(figsize=(7.5, 4.5))
sns.heatmap(r_mat.astype(float), annot=annot, fmt="", cmap="RdBu", vmin=-0.6, vmax=0.6,
            linewidths=0.5, cbar_kws={"label": "Pearson r"})
plt.title("Correlation of climate factors with yield anomaly\n(* p < 0.05, ** p < 0.01)")
save("fig04_correlation_heatmap.png")

# ---- Fig 5: Wheat scatter plots ----
_, Xw, anom_w = crop_data("Wheat")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, lab in zip(axes, ["Rainfall (mm)", "Hottest temp (°C)"]):
    x = Xw[lab].values
    r, pv = pearsonr(x, anom_w)
    slope, intercept = np.polyfit(x, anom_w, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.scatter(x, anom_w, color=COLORS["Wheat"], alpha=0.8, edgecolor="black")
    ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel(f"Oct–Jun {lab}"); ax.set_ylabel("Wheat yield anomaly (%)")
    ax.set_title(f"r = {r:.2f}, p = {pv:.3f}")
fig.suptitle("Winter wheat yield anomaly vs. growing-season climate")
save("fig05_wheat_scatter.png")

# ---- Fig 6: Observed vs predicted (Random Forest, LOOCV) ----
fig, axes = plt.subplots(1, 5, figsize=(20, 4.3))
for ax, crop in zip(axes, CROPS):
    y, pred = loo_preds[crop]
    lim = max(abs(y).max(), abs(pred).max()) * 1.1
    ax.scatter(y, pred, color=COLORS[crop], edgecolor="black", alpha=0.8)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(f"{crop} (R² = {r2_score(y, pred):.2f})")
    ax.set_xlabel("Observed anomaly (%)")
axes[0].set_ylabel("Predicted anomaly (%)"); axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Observed vs. predicted yield anomalies (Random Forest, leave-one-out CV)")
save("fig06_observed_vs_predicted.png")

# ---- Fig S1: Permutation importance (supplementary) ----
imp.plot(kind="bar", figsize=(9, 4))
plt.ylabel("Importance"); plt.title("Climate factor importance by crop (permutation)")
plt.xticks(rotation=0)
save("figS1_permutation_importance.png")

# ---- Figs 8-10: SHAP ----
try:
    import shap
    shap_mean, shap_saved = {}, {}
    for crop in CROPS:
        _, X, y = crop_data(crop)
        rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42).fit(X, y)
        sv = shap.TreeExplainer(rf).shap_values(X)
        shap_mean[crop] = np.abs(sv).mean(axis=0)
        shap_saved[crop] = (sv, X)
        shap.summary_plot(sv, X, show=False)                       # Fig 9 panels
        plt.title(f"{crop}: SHAP values (effect on yield anomaly, %)")
        save(f"fig09_shap_beeswarm_{crop.lower()}.png")

    shap_imp = pd.DataFrame(shap_mean, index=LABELS).T
    print("\n=== Mean |SHAP| (paper Fig. 8) (average effect on yield anomaly, %) ===")
    print(shap_imp.round(2))
    shap_imp.to_csv(os.path.join(RES_DIR, "shap_importance.csv"))

    shap_imp.plot(kind="bar", figsize=(9, 4))                     # Fig 8
    plt.ylabel("Mean |SHAP| (% yield anomaly)")
    plt.title("Average climate factor impact by crop (SHAP)"); plt.xticks(rotation=0)
    save("fig08_shap_bar_all_crops.png")

    sv, X = shap_saved["Wheat"]                                    # Fig 10a-b
    for feat, tag in [("Rainfall (mm)", "rainfall"), ("Hottest temp (°C)", "heat")]:
        shap.dependence_plot(feat, sv, X, interaction_index=None, show=False)
        plt.title(f"Wheat: effect of {feat} on yield anomaly")
        save(f"fig10_shap_dependence_wheat_{tag}.png")
except ImportError:
    print("\nSHAP not installed. Run  pip install shap  and re-run for Figs 8-10.")

print("\nDone. All tables and figures are saved in results/ and figures/.")
