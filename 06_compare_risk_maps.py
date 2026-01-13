# Script that compares risk maps across all ML models (except LR) ##############
# Inputs: - Models and their thresholds.
# Careful!: modify epidemiological year of interest for predictions, as well as
#           models and their respective thresholds. 
# Outputs: - Maps in console. 

# Script for comparing the risk maps obtained from all trained models #########
# Except for LR, supplementary material S3 ####################################

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


# Loading data for 2024 #######################################################
df = pd.read_csv('data/dataset_ML.csv')
df = df[df['ANO'] == 2024] # modify here!!!
df_muns = df['ID_MUN']
df = df.drop(columns = ['ID_MUN', 'ANO', 'Unnamed: 0']) # removing irrelevant columns
df['target'] = (df['casos'] >= 1).astype(int)
df = df.drop(columns = ['casos', 'n_imig_LAG', 'IA_LAG', 'nv_gru_LAG', 'nv_vcp_LAG', 'dist_campinas_LAG', 'dist_santos_LAG', 'perc_pop_meio_LAG'])
X = df.drop(columns = 'target')
y = pd.Series(df['target'].to_numpy())

# Doing maps for all models (together, new)

list_models = ['CatBoost', 'XGBoost', 'Random Forest', 'AdaBoost']
list_title = ['a', 'b', 'c', 'd']
list_thr = [0.0555651653675258, 0.0809919238090515, 0.0933717801931074, 0.424480463673856]
i = 0

fig_cont, axs_cont = plt.subplots(2, 2, figsize=(20, 20))
axs_cont = axs_cont.flatten()

for name_model in list_models:
    thr_opt = list_thr[i]
    ax = axs_cont[i]

    model = joblib.load('models/' + name_model + '.pkl')
    res = model.predict_proba(X)[:, 1]
    res = pd.DataFrame({'ID_MUN': df_muns, 'prob': res})

    map_sp = geobr.read_municipality(year=2020)
    map_sp = map_sp[map_sp['code_state'] == 35]
    map_sp['code_muni_6'] = (
        map_sp['code_muni']
        .astype(str)
        .str.slice(0, 6, 1)
        .astype(int)
    )
    map_sp = map_sp.merge(res, how='left',
                          left_on='code_muni_6',
                          right_on='ID_MUN')

    map_sp['logres'] = np.log(map_sp['prob'])

    map_sp.plot(
        ax=ax,
        column='logres',
        cmap='viridis',
        edgecolor='black',
        linewidth=0.4,
        missing_kwds=dict(color='grey')
    )

    sm = plt.cm.ScalarMappable(
        cmap='viridis',
        norm=plt.Normalize(
            vmin=map_sp['logres'].min(),
            vmax=map_sp['logres'].max()
        )
    )
    cbar = fig_cont.colorbar(sm, ax=ax, fraction=0.046, pad=0.01)
    cbar.set_label('Risk score (Log)', fontsize=12)

    ax.set_title(name_model, fontsize=18)
    ax.text(
        0.02, 0.98,
        f"({list_title[i]})",
        transform=ax.transAxes,
        fontsize=18,
        fontweight='bold',
        va='top'
    )

    ax.axis("off")
    i += 1

plt.tight_layout()
plt.savefig("figs/figS5.pdf", bbox_inches="tight")
plt.close()

fig_cat, axs_cat = plt.subplots(2, 2, figsize=(20, 20))
axs_cat = axs_cat.flatten()

i = 0
for name_model in list_models:
    thr_opt = list_thr[i]
    ax = axs_cat[i]

    model = joblib.load('models/' + name_model + '.pkl')
    res = model.predict_proba(X)[:, 1]
    res = pd.DataFrame({'ID_MUN': df_muns, 'prob': res})

    map_sp = geobr.read_municipality(year=2020)
    map_sp = map_sp[map_sp['code_state'] == 35]
    map_sp['code_muni_6'] = (
        map_sp['code_muni']
        .astype(str)
        .str.slice(0, 6, 1)
        .astype(int)
    )
    map_sp = map_sp.merge(res, how='left',
                          left_on='code_muni_6',
                          right_on='ID_MUN')

    map_sp['risco'] = 'Low risk'
    map_sp.loc[map_sp['prob'] >= thr_opt, 'risco'] = 'High risk'

    map_sp.plot(
        ax=ax,
        column='risco',
        categorical=True,
        cmap='viridis',
        edgecolor='black',
        linewidth=0.4,
        legend=True,
        legend_kwds={'fontsize':12, 'frameon':False},
        missing_kwds=dict(color='grey')
    )

    ax.set_title(name_model, fontsize=18)

    ax.text(
        0.02, 0.98,
        f"({list_title[i]})",
        transform=ax.transAxes,
        fontsize=18,
        fontweight='bold',
        va='top'
    )

    ax.axis("off")
    i += 1

plt.tight_layout()
plt.savefig("figs/figS6.pdf", bbox_inches="tight")
plt.close()



