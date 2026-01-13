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


# Basic scheme:
    # Loading data, cleaning dataset, separating out 2024
    # Train, test and validation splits
    # Testing several algorithms with some gridsearchCV, probability calibration and thr tuning
    # Evaluating all algorithms for final classification
    # For the best one, we will then calculate shap values
    # Plotting results for the 2024 epidemiological year 
    # Models: LogisticRegression, DT, RF, AdaBoost, CatBoost, XGBoost
    
# Loading, cleaning, separating out 2024 ######################################
df = pd.read_csv('data/dataset_ML.csv')
df = df[df['ANO'] != 2024] # excluding 2024 (no data on 2025 yet, so we will not use)
df = df.drop(columns = ['ID_MUN', 'ANO', 'Unnamed: 0']) # removing irrelevant columns
df['target'] = (df['casos'] >= 1).astype(int)
df = df.drop(columns = ['casos', 'n_imig_LAG', 'IA_LAG', 'nv_gru_LAG', 'nv_vcp_LAG', 'dist_campinas_LAG', 'dist_santos_LAG', 'perc_pop_meio_LAG'])
X = df.drop(columns = 'target')
y = pd.Series(df['target'].to_numpy())

# Train, test and validation splits ###########################################
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.4, stratify = y, random_state=42)
X_val, X_test, y_val, y_test =  train_test_split(X_val, y_val, test_size = 0.5, stratify = y_val, random_state=42)


# Defining pipelines and grids for all of our algorithms ######################

## Logistic Regression ########################################################
pipeline_lr = Pipeline([('scl', StandardScaler()),
                        ('lr', LogisticRegression(random_state = 42))]) # LR pipeline
lr_params = [{'lr__penalty': ['l1', 'l2'],
              'lr__C': [0.1, 0.5, 1],
              'lr__solver': ['liblinear']}]# parameter grid
grid_lr = GridSearchCV(pipeline_lr, lr_params, n_jobs = -1, cv = 5, scoring="roc_auc")

## Random Forest ##############################################################
rf_params = [{'n_estimators': [50, 100, 500, 1000],
              'max_depth': [2, 3, 4, 5, 6],
              'min_samples_split': [2, 5, 10]}]
grid_rf = GridSearchCV(RandomForestClassifier(random_state = 42, n_jobs = -1), rf_params, n_jobs = -1, cv = 5, scoring="roc_auc")

## AdaBoost ###################################################################
ada_params = [{'n_estimators': [50, 100, 500, 1000],
               'learning_rate': [0.01, 0.1, 1]}]
grid_ada = GridSearchCV(AdaBoostClassifier(random_state = 42), ada_params, n_jobs = -1, cv = 5, scoring="roc_auc")

## CatBoost ###################################################################
cat_params = [{'iterations': [100, 500, 1000, 1500],
               'learning_rate': [0.01, 0.03, 0.1, 1],
               'l2_leaf_reg': [0.1, 1, 3, 5, 10],
               'depth': [2, 3, 4, 5, 6, 7, 8]}]
grid_cat =  GridSearchCV(CatBoostClassifier(random_state = 42, thread_count=-1, verbose = False), cat_params, n_jobs = -1, cv = 5, scoring="roc_auc")

## XGBoost ####################################################################
xg_params = [{'n_estimators': [50, 100, 500, 1000],
              'max_depth': [2, 3, 4, 5, 6, 7, 8],
              'learning_rate': [0.01, 0.1, 0.3, 1],
              'min_child_weight': [1, 2, 5]}]
grid_xg = GridSearchCV(XGBClassifier(random_state = 42, n_jobs = -1), xg_params, n_jobs = -1, cv = 5, scoring="roc_auc")

# Evaluating all pipelines and grids, performing hyperparameter optimization #######
grid_list = [grid_lr, grid_rf, grid_ada, grid_cat, grid_xg]
text_list = ['Logistic Regression', 'Random Forest', 'AdaBoost', 'CatBoost', 'XGBoost']
df = pd.DataFrame()
i = 0

for grid_element in grid_list:
    df_aux = pd.DataFrame()
    print(text_list[i])
    grid_element.fit(X_train, y_train)
    
    # Saving the best model 
    name_model = 'models/'+str(text_list[i])+'.pkl'
    joblib.dump(grid_element, name_model)
    
    # Printing the best parameters
    print(grid_element.best_params_)
    
    # Increasing counter variable
    i = i + 1
    
    # Choosing threshold on validation data
    y_val_proba = grid_element.predict_proba(X_val)[:, 1]
    y_train_proba = grid_element.predict_proba(X_train)[:, 1]
    y_test_proba = grid_element.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba, pos_label = 1)
    n_arg = np.argmin(abs(tpr-(1-fpr)))
    thr_opt = thresholds[n_arg]
    
    y_val_pred = y_val_proba.copy()
    filt_pos = (y_val_pred >= thr_opt)
    filt_neg = (y_val_pred < thr_opt)
    y_val_pred[filt_pos] = 1
    y_val_pred[filt_neg] = 0
    
    y_train_pred = y_train_proba.copy()
    filt_pos = (y_train_pred >= thr_opt)
    filt_neg = (y_train_pred < thr_opt)
    y_train_pred[filt_pos] = 1
    y_train_pred[filt_neg] = 0
    
    y_test_pred = y_test_proba.copy()
    filt_pos = (y_test_pred >= thr_opt)
    filt_neg = (y_test_pred < thr_opt)
    y_test_pred[filt_pos] = 1
    y_test_pred[filt_neg] = 0
    
    # Metrics on train data
    roc_train = roc_auc_score(y_train, y_train_proba)
    precision, recall, thresholds = precision_recall_curve(y_train, y_train_proba)
    prc_train = auc(recall, precision)
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    sp_train = tn / (tn + fp) 
    ss_train = tp/ (tp + fn) 
    f1_train = (2*tp)/(2*tp + fp + fn)
    
    # Metrics on validation data
    roc_val = roc_auc_score(y_val, y_val_proba)
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
    prc_val = auc(recall, precision)
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    precision_val = precision_score(y_val, y_val_pred)
    recall_val = recall_score(y_val, y_val_pred)
    sp_val = tn / (tn + fp) 
    ss_val = tp/ (tp + fn) 
    f1_val = (2*tp)/(2*tp + fp + fn)
    
    
    # Metrics on test data
    roc_test = roc_auc_score(y_test, y_test_proba)
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
    prc_test = auc(recall, precision)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    sp_test = tn / (tn + fp) 
    ss_test = tp/ (tp + fn)
    f1_test = (2*tp)/(2*tp + fp + fn)
    
    # Creating df_aux pandas dataframe
    df_aux = pd.DataFrame([{'AUC-ROC Train': roc_train,
                           'AUC-PRC Train': prc_train,
                           'Precision Train': precision_train,
                           'Recall Train': recall_train,
                           'Specificity Train': sp_train,
                           'Sensibility Train': ss_train,
                           'F1-score Train': f1_train,
                           'Threshold': thr_opt,
                           'AUC-ROC Validation': roc_val,
                           'AUC-PRC Validation': prc_val,
                           'Precision Validation': precision_val,
                           'Recall Validation': recall_val,
                           'Specificity Validation': sp_val,
                           'Sensibility Validation': ss_val,
                           'F1-score Validation': f1_val,
                           'AUC-ROC Test': roc_test,
                           'AUC-PRC Test': prc_test,
                           'Precision Test': precision_test,
                           'Recall Test': recall_test,
                           'Specificity Test': sp_test,
                           'Sensibility Test': ss_test,
                           'F1-score Test': f1_test}])

    # Concatenating results
    df = pd.concat([df, df_aux])

df.to_csv('models/res_total.csv')
    
# Plotting AUCROC of all models on train, test and validation data ############
data_type = ['train', 'test']

for dataset in data_type:
    if(dataset == 'train'):
        y_dataset = y_train.copy()
        X_dataset = X_train.copy()
    #elif(dataset == 'validation'):
    #    y_dataset = y_val.copy()
    #    X_dataset = X_val.copy()
    else:
        y_dataset = y_test.copy()
        X_dataset = X_test.copy()
    i = 0
    for grid_element in grid_list:
        y_proba = grid_element.predict_proba(X_dataset)[:, 1]    
        fpr, tpr, thresholds = roc_curve(y_dataset, y_proba)
        plt.plot(fpr, tpr, label = text_list[i])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        i = i + 1
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/figS4_'+str(dataset)+'.pdf')
    plt.show()
    




