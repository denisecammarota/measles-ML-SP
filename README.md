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
