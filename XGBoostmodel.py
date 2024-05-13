# Importing necessary libraries
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv('cirrhosis.csv', index_col='ID')

# Handling missing values for numerical columns by replacing them with median
df_num_col = df.select_dtypes(include=(['int64', 'float64'])).columns
for c in df_num_col:
    df[c].fillna(df[c].median(), inplace=True)

# Handling missing values for categorical columns by replacing them with mode
df_cat_col = df.select_dtypes(include=('object')).columns
for c in df_cat_col:
    df[c].fillna(df[c].mode().values[0], inplace=True)

# Converting 'Stage' column into binary classification (0 or 1)
df['Stage'] = np.where(df['Stage'] == 4, 1, 0)

# Replacing categorical data with integers
df['Sex'] = df['Sex'].replace({'M': 0, 'F': 1})  # Male : 0 , Female :1
df['Ascites'] = df['Ascites'].replace({'N': 0, 'Y': 1})  # N : 0, Y : 1
df['Drug'] = df['Drug'].replace({'D-penicillamine': 0, 'Placebo': 1})  # D-penicillamine : 0, Placebo : 1
df['Hepatomegaly'] = df['Hepatomegaly'].replace({'N': 0, 'Y': 1})  # N : 0, Y : 1
df['Spiders'] = df['Spiders'].replace({'N': 0, 'Y': 1})  # N : 0, Y : 1
df['Edema'] = df['Edema'].replace({'N': 0, 'Y': 1, 'S': -1})  # N : 0, Y : 1, S : -1
df['Status'] = df['Status'].replace({'C': 0, 'CL': 1, 'D': -1})  # 'C':0, 'CL':1, 'D':-1

# Setting up Features and Target
X = df.drop(['Status', 'N_Days', 'Stage'], axis=1)
y = df.pop('Stage')

# Initializing Stratified K-Fold cross-validator
skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

# Initializing XGBoost classifier model
model = XGBClassifier(learning_rate=0.75, max_depth=3, random_state=1, gamma=0, eval_metric='error')

# Empty list to store accuracy scores for each fold
acc = []

# Function to perform training on each fold
def training(train, test, fold_no):
    X_train = train
    y_train = y.iloc[train_index]
    X_test = test
    y_test = y.iloc[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    acc.append(score)

# Looping over each fold and performing training
fold_no = 1
for train_index, test_index in skf.split(X, y):
    train = X.iloc[train_index, :]
    test = X.iloc[test_index, :]
    training(train, test, fold_no)
    fold_no += 1

# Printing mean accuracy score
print('XGboost model Mean Accuracy = ', np.mean(acc))

# Predicting using the XGBoost model
XGB_model_predict = model.predict(test)
XGB_model_predict_proba = model.predict_proba(test)
XGB_model_predict = model.predict(test)
XGB_model_predict_proba = model.predict_proba(test)

# Printing classification report
print(classification_report(y.iloc[test_index], XGB_model_predict))

# Function to predict using the model
def predict(user):
    return model.predict(user)
