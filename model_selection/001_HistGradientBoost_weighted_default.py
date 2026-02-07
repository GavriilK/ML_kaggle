from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score
import pandas as pd
import optuna
import datetime
import numpy as np

train = pd.read_parquet('processing/train_features.parquet')
submission = pd.read_parquet('processing/test_features.parquet')
sample_submission = pd.read_csv('knn_sample_submission.csv', index_col=0)
y = pd.read_parquet('processing/y_train.parquet').values.ravel()

# train/test/val split :
X_intermediate, X_val, y_intermediate, y_val = train_test_split(train, y, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_intermediate, y_intermediate, test_size=0.1, random_state=42)

def objective(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'max_iter': 500,
        'max_depth': None,
        #'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
        'random_state': 42,
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 50),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30)
    }
    
    model = HistGradientBoostingClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    return f1 

study = optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(n_startup_trials=5),sampler=optuna.samplers.TPESampler(n_startup_trials=10,multivariate=True))
study.optimize(objective, n_trials=30,n_jobs=-1)
print('Best trial:')
trial = study.best_trial
print(f'  F1 Score: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# train on full train and estimate value on validation set
best_params = trial.params
model = HistGradientBoostingClassifier(**best_params)
model.fit(X_intermediate, y_intermediate)
y_val_pred = model.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred, average='macro')
val2_f1 = f1_score(y_val, y_val_pred, average=None)
print(f'Validation F1 Score: {val_f1}, \n Validation F1 Score (non-weighted): {val2_f1}')
print(classification_report(y_val, y_val_pred))

model = HistGradientBoostingClassifier(**trial.params)
model.fit(train, y)
print(classification_report(y, model.predict(train)))
# Predict on submission set
submission_pred = model.predict(submission)
date = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
pd.DataFrame(submission_pred, columns=['change_type'], index=sample_submission.index).to_csv(f'submissions/histgb_submission_{date}.csv', index=True, index_label='Id')