# Script that plots the results of the best model (risk map) and classifies 
# municipalities into low and high risk of reintroduction.
# Also performs SHAP value analysis on test data.
# Input: - data/dataset_ML.csv, the best model and its threshold. 
# Careful!: - For the first part of the script, analysis is done to get the 
#           predictions for the year of interest. So, update that year accordingly
#           - For the second part, we exclude that year. So, also update accordingly
#           - Update the model and threshold that you wish to use, based on results from the previous script!!!
# Output: - Plots of probability of reintroduction and high/low risk. 
#         - SHAP value plots on test data. 


# Script in order to plot the results of the best model (risk map) ############
# and for classifying municipalities into low and high risk of reintroduction

import numpy as np
import pandas as pd
import seaborn as sns
import geobr
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.metrics import auc, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import plot_importance
from xgboost import XGBClassifier
import joblib


# Loading the best model (xgboost) ###########################################
# This needs to be modified if models were retrained on new data!!!
model = joblib.load('models/XGBoost.pkl')

# Doing SHAP value explainability #############################################

# Loading, cleaning, separating out 2024 ######################################
df = pd.read_csv('data/dataset_ML.csv')
df = df[df['ANO'] != 2024] # excluding 2024 (no data on 2025 yet, so we will not use) update here!!!
df = df.drop(columns = ['ID_MUN', 'ANO', 'Unnamed: 0']) # removing irrelevant columns
df['target'] = (df['casos'] >= 1).astype(int)
df = df.drop(columns = ['casos', 'n_imig_LAG', 'IA_LAG', 'nv_gru_LAG', 'nv_vcp_LAG', 'dist_campinas_LAG', 'dist_santos_LAG', 'perc_pop_meio_LAG'])
X = df.drop(columns = 'target')
y = pd.Series(df['target'].to_numpy())

# Train, test and validation splits ###########################################
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.4, stratify = y, random_state=42)
X_val, X_test, y_val, y_test =  train_test_split(X_val, y_val, test_size = 0.5, stratify = y_val, random_state=42)

# SHAP values on test data ####################################################

## Simple SHAP values barplot ###########################################
explainer = shap.TreeExplainer(model.best_estimator_, X_test)
shap_values = explainer(X_test)
#shap.plots.bar(shap_values, max_display=19)

## Beeswarm SHAP values plot ############################################
shap.plots.beeswarm(shap_values, max_display = 19, show = False)
plt.tight_layout()
plt.savefig('figs/fig2.pdf')
plt.show()
