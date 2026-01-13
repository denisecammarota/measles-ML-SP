---
title: "Estimating measles reintroduction risk in São Paulo using machine learning"
output: html_document
date: "2026-01-13"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

Authors: xxx

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

