# =====================================================================
# 01_data_preparation.py
# Builds the analysis dataset from the two raw files:
#   data/raw/nasa_power_monthly_39.83N_98.58W_1981_2023.csv   (NASA POWER)
#   data/raw/faostat_usa_yield_5crops_1961_2024.csv            (FAOSTAT)
# Output:
#   data/processed/climate_growing_season_1981_2023.csv
#   data/processed/us_crop_yield_climate_1981_2023.csv
# Run from the repository root:  python code/01_data_preparation.py
# =====================================================================

import calendar
import numpy as np
import pandas as pd

RAW_POWER = "data/raw/nasa_power_monthly_39.83N_98.58W_1981_2023.csv"
RAW_FAO = "data/raw/faostat_usa_yield_5crops_1961_2024.csv"
OUT_CLIMATE = "data/processed/climate_growing_season_1981_2023.csv"
OUT_MERGED = "data/processed/us_crop_yield_climate_1981_2023.csv"
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# ---------- 1. NASA POWER monthly data -> long format ----------
# Skip the text header block (ends with the line "-END HEADER-")
with open(RAW_POWER) as f:
    skip = next(i for i, line in enumerate(f) if line.startswith("-END HEADER-")) + 1
power = pd.read_csv(RAW_POWER, skiprows=skip).replace(-999.0, np.nan)

long = power.melt(id_vars=["PARAMETER", "YEAR"], value_vars=MONTHS,
                  var_name="MON", value_name="value")
long["month"] = long["MON"].map({m: i + 1 for i, m in enumerate(MONTHS)})
long["days"] = [calendar.monthrange(y, m)[1] for y, m in zip(long["YEAR"], long["month"])]
w = long.pivot_table(index=["YEAR", "month", "days"], columns="PARAMETER",
                     values="value").reset_index()
w["PREC_MM"] = w["PRECTOTCORR"] * w["days"]          # mm/day -> mm per month


def season_stats(sub):
    """Growing-season summary. Solar stays NaN if any month is missing."""
    return pd.Series({
        "Hottest_C": sub["T2M_MAX"].mean(),         # mean of monthly maxima
        "Coldest_C": sub["T2M_MIN"].mean(),         # mean of monthly minima
        "Precip_total_mm": sub["PREC_MM"].sum(min_count=len(sub)),
        "RH_pct": sub["RH2M"].mean(),
        "Solar_MJ_m2_day": sub["ALLSKY_SFC_SW_DWN"].mean(skipna=False),
    })


# ---------- 2. Growing seasons ----------
# May-Sep: corn, soybeans, rice, cotton
maysep = (w[w["month"].between(5, 9)].groupby("YEAR").apply(season_stats)
          .add_prefix("MaySep_"))

# Oct (previous year) - Jun: winter wheat
ww = w[(w["month"] >= 10) | (w["month"] <= 6)].copy()
ww["SeasonYear"] = np.where(ww["month"] >= 10, ww["YEAR"] + 1, ww["YEAR"])
n_months = ww.groupby("SeasonYear").size()
octjun = ww.groupby("SeasonYear").apply(season_stats)
octjun = octjun[n_months == 9].add_prefix("OctJun_")      # complete seasons only

climate = maysep.join(octjun).reset_index().rename(columns={"YEAR": "Year"})
climate = climate[climate["Year"] <= 2023].round(2)
# Monthly T2M_MAX/T2M_MIN are monthly extremes, so name them Hottest/Coldest
climate.to_csv(OUT_CLIMATE, index=False)
print(f"Saved {OUT_CLIMATE}: {climate.shape}")

# ---------- 3. FAOSTAT yields ----------
fao = pd.read_csv(RAW_FAO)
names = {"Maize (corn)": "Corn", "Soya beans": "Soybeans", "Wheat": "Wheat",
         "Rice": "Rice", "Seed cotton, unginned": "Cotton"}
fao = fao[(fao["Element"] == "Yield") & fao["Item"].isin(names)]
yields = (fao.assign(Crop=fao["Item"].map(names))
          .pivot(index="Year", columns="Crop", values="Value"))
yields.columns = [f"{c}_yield_kg_ha" for c in yields.columns]

# ---------- 4. Merge ----------
merged = climate.merge(yields.reset_index(), on="Year", how="left")
merged.to_csv(OUT_MERGED, index=False)
print(f"Saved {OUT_MERGED}: {merged.shape}")
print("Missing values per column:")
print(merged.isna().sum()[merged.isna().sum() > 0])
