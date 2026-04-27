# Estimating measles reintroduction risk in São Paulo using machine learning

Authors: Denise Cammarota, MSc, Danilo Pereira Mori, MSc, Flávia Cristina da Silva Sales, PhD, Isabela Galvão Fernandes Alves, MSc, Alícia Tavares da Silva Gomes, PhD, Frederico Prado, PhD, Ramon Wilk da Silva, PhD, Raquel Gardini Sanches Palasio, PhD, Pamella Cristina de Carvalho Lucas, PhD, Telma Regina Marques Pinto Carvalhanas, MSc, Ana Lúcia Frugis Yu, PhD. 

# Folder structure and scripts

## Folder structure
```
*    ├── data/  
*    ├── figs/
*    ├── models/
*    └── README.md
```

## Scripts
```
*    ├── 01_feature_selection.py: performs feature selection (S1).
*    ├── 02_training_full.py: trains and evaluates ML models, plots ROC curve (S4).
*    ├── 03_get_score_muns.py: calculates continuous risk and categories of high/low risk for 04_plot_map.R.
*    ├── 04_plot_map.R: plots map of continuous risk and categories of high/low risk (Figure 1).
*    └── 05_calculate_shap_values.py: calculates beeswarm SHAP plot (Figure 2).
*    └── 06_compare_risk_maps.py: compares continuous risks and categories of high/low risk (S5).
```

# Software versions

## Python (3.12.7)
- numpy: 1.26.4
- pandas: 2.2.2
- seaborn: 0.13.2
- geobr: 0.2.2
- matplotlib: 3.9.2
- shap: 0.47.1
- scikit-learn: 1.5.1
- catboost: 1.2.8
- xgboost: 3.0.0
- joblib: 1.4.2

## R (4.5.0)
- sf: 1.0.20
- spdep: 1.3.11
- ggspatial: 1.1.9
- magick: 2.8.6
- cowplot: 1.1.3
- geobr: 1.9.1
- ggthemes: 5.1.0
- tidyverse: 2.0.0
