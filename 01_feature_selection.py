# Script that performs feature selection, using correlation matrix to detect
# collinearity and mutual information to remove less informative variables.
# Inputs: - data/dataset_ML.csv
# Careful!: - see if it is necessary to remove 2024 or not (it was removed 
#           originally because all predictors were not available for 2024)
# Ideally, the features to be removed will be the same as when first run, so 
# it is an optional step to run this script. 


# Feature selection and collinearity ##########################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

# Loading file in question ####################################################
df = pd.read_csv('data/dataset_ML.csv')
df = df[df['ANO'] != 2024] # excluding 2024 (no data on 2025 yet, so we will not use)
df = df.drop(columns = ['ID_MUN', 'ANO', 'Unnamed: 0']) # removing irrelevant columns
df['target'] = (df['casos'] >= 1).astype(int)
df = df.drop(columns = ['casos'])

# Correlation matrix between predictors #######################################
plt.figure(figsize = (12, 10))
sns.heatmap(df.drop(columns = 'target').corr(), vmin=-1, vmax=1, annot=True, fmt='.2f', annot_kws={"fontsize": 8}, cbar_kws={'label': 'Pearson correlation'})
plt.tight_layout()
plt.savefig('figs/figS1.pdf')
plt.show()

# Mutual information between predictors and result ############################
X = df.drop(columns = 'target').to_numpy()
y = df['target'].to_numpy()
mi_features = mutual_info_classif(X, y)
name_features = df.columns[:-1]

fig, ax = plt.subplots(figsize = (12, 8))
order = np.argsort(-mi_features)
ax.bar(name_features[order], mi_features[order])
plt.xticks(rotation=90)
plt.xlabel('Feature', fontsize = 18)
plt.ylabel('Mutual Information', fontsize = 18)
plt.tight_layout()
plt.savefig('figs/figS2.pdf')
plt.show()
