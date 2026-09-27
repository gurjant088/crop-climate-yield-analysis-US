# Does Central U.S. Weather Explain National Crop Yields?

**A Correlation and Machine Learning Analysis (1981–2023)**

This repository contains the data and code for a study testing whether growing-season weather at a single point, the geographic center of the contiguous United States (39.83°N, 98.58°W), explains national yield anomalies of five major U.S. crops: corn, soybeans, rice, cotton and winter wheat.

## Key findings

- After false discovery rate (FDR) correction, only **winter wheat** showed significant climate associations (q = 0.045): yield anomalies rose with October–June rainfall (r = 0.43) and humidity (r = 0.45) and fell with heat (r = −0.41).
- SHAP analysis revealed a **nonlinear rainfall threshold** for winter wheat, with losses below about 400 mm and a plateau above about 480 mm.
- Corn and soybean associations were suggestive but not significant after correction; rice and cotton showed none.
- **No model achieved predictive skill** under leave-one-out or repeated 5-fold cross-validation (R² from −0.77 to 0.08).
- Temperature, precipitation and humidity were strongly intercorrelated (|r| = 0.81–0.86), so factor attributions reflect a combined heat–moisture signal.

## Repository structure

```
├── code/
│   ├── 01_data_preparation.py        # raw data -> growing-season features -> merged dataset
│   ├── 02_analysis_all_figures.py    # correlations, FDR, LOOCV models, SHAP, all figures
│   └── 03_kfold_cross_validation.py  # repeated 5-fold cross-validation (robustness check)
├── data/
│   ├── raw/
│   │   ├── nasa_power_monthly_39.83N_98.58W_1981_2023.csv
│   │   └── faostat_usa_yield_5crops_1961_2024.csv
│   └── processed/
│       ├── climate_growing_season_1981_2023.csv
│       └── us_crop_yield_climate_1981_2023.csv   # analysis dataset (43 rows)
├── figures/                          # figures as numbered in the paper
├── results/                          # output tables
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## How to reproduce

Requires Python 3.9 or later. Run all commands from the repository root.

```bash
pip install -r requirements.txt
python code/01_data_preparation.py
python code/02_analysis_all_figures.py
python code/03_kfold_cross_validation.py
```

The study-area map downloads U.S. state boundaries from the U.S. Census Bureau, so it needs an internet connection; without one, a simplified map is drawn. Small numerical differences (typically in the third decimal place) can occur between library versions.

To run in Google Colab, upload the repository folder (or clone it with `!git clone <repo-url>`), change into it with `%cd <repo-name>`, and run each script with `!python code/<script>.py`.

## Data

| Dataset | Source | Details |
| --- | --- | --- |
| Crop yields | [FAOSTAT](https://www.fao.org/faostat) – Crops and livestock products | U.S. national yield (kg/ha) for maize (corn), soya beans, rice, seed cotton and wheat. Seed cotton values for 2015–2023 are FAO estimates (flag E); all others are official. |
| Climate | [NASA POWER](https://power.larc.nasa.gov) – Agroclimatology, monthly | Point 39.83°N, 98.58°W, 1981–2023. MERRA-2 temperature, humidity and precipitation; CERES solar radiation. |

### Processed dataset columns

| Column | Description | Unit |
| --- | --- | --- |
| `Year` | Harvest year | – |
| `MaySep_*` | May–September season (corn, soybeans, rice, cotton) | – |
| `OctJun_*` | October (previous year) to June season (winter wheat) | – |
| `*_Hottest_C` | Mean of monthly maximum 2 m temperature (monthly extremes) | °C |
| `*_Coldest_C` | Mean of monthly minimum 2 m temperature (monthly extremes) | °C |
| `*_Precip_total_mm` | Total corrected precipitation | mm |
| `*_RH_pct` | Mean 2 m relative humidity | % |
| `*_Solar_MJ_m2_day` | Mean all-sky surface shortwave radiation (not used in models) | MJ/m²/day |
| `<Crop>_yield_kg_ha` | National yield | kg/ha |

Missing values are expected: solar radiation is unavailable before 1984, and the October–June season has no 1981 value because it requires October–December 1980.

## Methods summary

1. Yields were detrended with a crop-specific linear trend and expressed as percentage anomalies.
2. Pearson correlations between four climate features and yield anomalies were tested, with Benjamini–Hochberg FDR correction across 20 tests.
3. Linear Regression, Random Forest and Gradient Boosting models were evaluated by leave-one-out cross-validation and by 20 repeats of 5-fold cross-validation.
4. Models were interpreted with permutation importance and SHAP (TreeExplainer).

## Citation

If you use this code or data, please cite the associated paper (citation to be added on publication) or this repository using the `CITATION.cff` file. Please also cite the original data sources, FAOSTAT and NASA POWER, according to their terms of use.

## License

Code is released under the MIT License. Raw data remain subject to the terms of use of FAO (FAOSTAT) and NASA (POWER).

## Contact

[Your Full Name] · [Email address]
