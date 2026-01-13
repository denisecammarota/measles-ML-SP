# Script that plots and saves the results of reintroduction risk per municipality ###########
# for all municipalities in the state of SP, which will then be used in R for calculating the
# risk indices (categorical and continuous).
# Inputs: - data/dataset_ML.csv
# Careful!: - check the epidemiological year of interest, as well as the model that you wish to use,
#           which should also have its corresponding decision threshold specified. 
# Outputs: - data/risk_score.csv

# Script in order to plot the results of the best model (risk map) ############
# for all municipalities of the state of SP, to then be treated in R

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

 
# Loading data for epidemiological year of interest #########################
df = pd.read_csv('data/dataset_ML.csv')
df = df[df['ANO'] == 2024] # update year here !!!!!!!
df_muns = df['ID_MUN']
df = df.drop(columns = ['ID_MUN', 'ANO', 'Unnamed: 0']) # removing irrelevant columns
df['target'] = (df['casos'] >= 1).astype(int)
df = df.drop(columns = ['casos', 'n_imig_LAG', 'IA_LAG', 'nv_gru_LAG', 'nv_vcp_LAG', 'dist_campinas_LAG', 'dist_santos_LAG', 'perc_pop_meio_LAG'])
X = df.drop(columns = 'target')
y = pd.Series(df['target'].to_numpy())


# Loading the best model (catboost) ###########################################
#model = joblib.load('models/CatBoost.pkl')
model = joblib.load('models/XGBoost.pkl') # update best model here!!!!!!!
#model = joblib.load('models/Random Forest.pkl')
#model = joblib.load('models/AdaBoost.pkl')
res = model.predict_proba(X)[:, 1]
res = pd.DataFrame({'ID_MUN': df_muns, 'prob': res})
res = res.reset_index()

# Turning this into also a categorical risk score #############################
thr_risk = 0.0809919238090515 # update decision threshold here!!!
res['risk_category'] = 0
res.loc[res['prob'] >= thr_risk, 'risk_category'] = 1


res.to_csv('data/risk_score.csv')







