# Script to train, test algorithms for predicting measles cases ##############
# Evaluates performance as well, plots AUC-ROC ###############################

import numpy as np
import pandas as pd
import seaborn as sns
import geobr
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, brier_score_loss # added brier score
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
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
import random
import os

np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
    
# Loading, cleaning, separating out 2024 ######################################
df = pd.read_csv('data/dataset_ML.csv')
df = df[df['ANO'] != 2024] # excluding 2024 (no data on 2025 yet, so we will not use)
years = df['ANO'].values # keeping track of the year of each observation
df = df.drop(columns = ['ID_MUN', 'ANO', 'Unnamed: 0']) # removing irrelevant columns
df['target'] = (df['casos'] >= 1).astype(int)
df = df.drop(columns = ['casos', 'n_imig_LAG', 'IA_LAG', 'nv_gru_LAG', 'nv_vcp_LAG', 'dist_campinas_LAG', 'dist_santos_LAG', 'perc_pop_meio_LAG'])
X = df.drop(columns = 'target')
y = df['target'].values

# Defining temporal folds #####################################################
folds = [
    {'train': list(range(2007, 2020)), 'test': [2020]},
    {'train': list(range(2007, 2021)), 'test': [2021]},
    {'train': list(range(2007, 2022)), 'test': [2022]},
    {'train': list(range(2007, 2023)), 'test': [2023]},
]

# Defining pipelines and grids for all of our algorithms ######################

models_config = {
    "Logistic Regression": {
        "model": Pipeline([('scl', StandardScaler()), ('lr', LogisticRegression(random_state=42))]),
        "grid": ParameterGrid({'lr__penalty': ['l1', 'l2'], 'lr__C': [0.1, 0.5, 1], 'lr__solver': ['liblinear']})
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "grid": ParameterGrid({'n_estimators': [50, 100, 500, 1000], 'max_depth': [2, 3, 4, 5, 6], 'min_samples_split': [2, 5, 10]})
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "grid": ParameterGrid({'n_estimators': [50, 100, 500, 1000], 'learning_rate': [0.01, 0.1, 1]})
    },
    "CatBoost": {
        "model": CatBoostClassifier(random_state=42, thread_count=-1, verbose=False),
        "grid": ParameterGrid({'iterations': [100, 500, 1000, 1500], 'learning_rate': [0.01, 0.03, 0.1, 1], 'l2_leaf_reg': [0.1, 1, 3, 5, 10], 'depth': [2, 3, 4, 5, 6, 7, 8]})
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, n_jobs=-1),
        "grid": ParameterGrid({'n_estimators': [50, 100, 500, 1000], 'max_depth': [2, 3, 4, 5, 6, 7, 8], 'learning_rate': [0.01, 0.1, 0.3, 1], 'min_child_weight': [1, 2, 5]})
    }
}

# Custom loop for training through temporal BCV ###################

def temporal_cv(model, params, X, y, years, folds):
    
    fold_aucs = []
    
    for fold in folds:
        train_mask = np.isin(years, fold['train'])
        test_mask = np.isin(years, fold['test'])
        
        X_train, y_train = X.iloc[train_mask], y[train_mask]
        X_test, y_test = X.iloc[test_mask], y[test_mask]
        
        if len(np.unique(y_test)) < 2:
            print(f"Skipping fold {fold['test']} - only one class present.")
            continue
        
        fresh_model = clone(model) 
        fresh_model.set_params(**params)
        fresh_model.fit(X_train, y_train)
        
        y_proba = fresh_model.predict_proba(X_test)[:, 1]
        fold_aucs.append(roc_auc_score(y_test, y_proba))
        
        
    return np.mean(fold_aucs)

# Training and testing all models #################################


results_oof = [] # saving model performance results
data_plot = {} # saving data for plotting (auc-roc for train and test + calibration plot on test)

for name, config in models_config.items():
    print(f'\n Training model: {name}')
    
    # doing temporal BCV to get the best hyperparameters
    best_mean_auc = -1
    best_params = None
    
    for params in config['grid']:
        curr_mean_auc = temporal_cv(config['model'], params, X, y, years, folds)
        if curr_mean_auc > best_mean_auc:
            best_mean_auc = curr_mean_auc
            best_params = params
    
    # get oof predictions for the best model
    
    y_oof_test = []
    proba_oof_test = []
    y_oof_train = []
    proba_oof_train = []
    
    for fold in folds:
        train_mask = np.isin(years, fold['train'])
        test_mask = np.isin(years, fold['test'])
        
        X_train, y_train = X.iloc[train_mask], y[train_mask]
        X_test, y_test = X.iloc[test_mask], y[test_mask]
        
        tmp_model = clone(config['model'])
        tmp_model.set_params(**best_params)
        tmp_model.fit(X_train, y_train)
        
        y_proba_test = tmp_model.predict_proba(X_test)[:, 1]
        y_oof_test.extend(y_test)
        proba_oof_test.extend(y_proba_test)
        
        y_proba_train = tmp_model.predict_proba(X_train)[:, 1]
        y_oof_train.extend(y_train)
        proba_oof_train.extend(y_proba_train)
    
    y_oof_test = np.array(y_oof_test)
    proba_oof_test = np.array(proba_oof_test)
    
    y_oof_train = np.array(y_oof_train)
    proba_oof_train = np.array(proba_oof_train)
    
    # calculating threshold
    fpr_test, tpr_test, thrs_test = roc_curve(y_oof_test, proba_oof_test)
    fpr_train, tpr_train, thrs_train = roc_curve(y_oof_train, proba_oof_train)
    thr_opt = thrs_test[np.argmin(abs(tpr_test - (1 - fpr_test)))]
    y_pred_test = (proba_oof_test >= thr_opt).astype(int)
    y_pred_train =  (proba_oof_train >= thr_opt).astype(int)
    
    # calculating calibration curve on test data
    prob_true, prob_pred = calibration_curve(y_oof_test, proba_oof_test, n_bins = 10)
    
    # saving data for plotting later
    data_plot[name] = {
        'fpr_test': fpr_test, 'tpr_test': tpr_test, 'auc_test': roc_auc_score(y_oof_test, proba_oof_test),
        'fpr_train': fpr_train, 'tpr_train': tpr_train, 'auc_train': roc_auc_score(y_oof_train, proba_oof_train),
        'p_true': prob_true, 'p_pred': prob_pred
    }
    
    # calculating metrics on test and train data
    
    # Metrics on test data
    roc_test = roc_auc_score(y_oof_test, proba_oof_test) # ok
    precision, recall, thresholds = precision_recall_curve(y_oof_test, proba_oof_test) # ok
    prc_test = auc(recall, precision) # ok
    tn, fp, fn, tp = confusion_matrix(y_oof_test, y_pred_test).ravel() # ok
    precision_test = precision_score(y_oof_test, y_pred_test) # ok
    recall_test = recall_score(y_oof_test, y_pred_test) # ok
    sp_test = tn / (tn + fp) 
    ss_test = tp/ (tp + fn)
    f1_test = (2*tp)/(2*tp + fp + fn)
    brier_test = brier_score_loss(y_oof_test, proba_oof_test) # ok
    
    # Metrics on train data
    roc_train = roc_auc_score(y_oof_train, proba_oof_train) # ok
    precision, recall, thresholds = precision_recall_curve(y_oof_train, proba_oof_train) # ok
    prc_train = auc(recall, precision) # ok
    tn, fp, fn, tp = confusion_matrix(y_oof_train, y_pred_train).ravel() # ok
    precision_train = precision_score(y_oof_train, y_pred_train) # ok
    recall_train = recall_score(y_oof_train, y_pred_train) # ok
    sp_train = tn / (tn + fp) 
    ss_train = tp/ (tp + fn) 
    f1_train = (2*tp)/(2*tp + fp + fn)
    brier_train = brier_score_loss(y_oof_train, proba_oof_train) # ok
        
    results_oof.append({'Model': name,
                    'Threshold': thr_opt,
                    'AUC-ROC Train': roc_train,
                    'AUC-PRC Train': prc_train,
                    'Precision Train': precision_train,
                    'Recall Train': recall_train,
                    'Specificity Train': sp_train,
                    'Sensibility Train': ss_train,
                    'F1-score Train': f1_train,
                    'Brier Train': brier_train,
                    'AUC-ROC Test': roc_test,
                    'AUC-PRC Test': prc_test,
                    'Precision Test': precision_test,
                    'Recall Test': recall_test,
                    'Specificity Test': sp_test,
                    'Sensibility Test': ss_test,
                    'F1-score Test': f1_test,
                    'Brier Test': brier_test})
    
    # re-training the best model on full data and saving 
    name_model = 'models/'+str(name)+'_CV.pkl'
    final_model = clone(config['model'])
    final_model = final_model.set_params(**best_params)
    final_model.fit(X, y)
    joblib.dump(final_model, name_model)
    

pd.DataFrame(results_oof).to_csv("temporal_cv_results.csv", index=False)
    



        
