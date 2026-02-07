from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score
import pandas as pd
import optuna
import datetime
import numpy as np
from sklearn.multiclass import OneVsRestClassifier

train = pd.read_parquet('processing/train_features.parquet')
submission = pd.read_parquet('processing/test_features.parquet')
sample_submission = pd.read_csv('knn_sample_submission.csv', index_col=0)
y = pd.read_parquet('processing/y_train.parquet').values.ravel()

# train/test/val split :
print("Splitting data into train, validation, and test sets...")
X_intermediate, X_val, y_intermediate, y_val = train_test_split(train, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_intermediate, y_intermediate, test_size=0.1, random_state=42, stratify=y_intermediate)

def objective(trial):
    
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'max_depth': trial.suggest_categorical('max_depth', [None] + list(range(3, 21))),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy','log_loss']),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'random_state': 42,
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        "max_samples": 0.7,
    }
    
    model = OneVsRestClassifier(RandomForestClassifier(**param),verbose=10,n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')

    trial.report(f1, 0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return f1

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=3,n_warmup_steps=0))
study.optimize(objective, n_trials=10,n_jobs=1,show_progress_bar=True)
print('Best trial:')
trial = study.best_trial
print(f'  F1 Score: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# train on full train and estimate value on validation set
best_params = trial.params
model = OneVsRestClassifier(RandomForestClassifier(**best_params))
model.fit(X_intermediate, y_intermediate)
y_val_pred = model.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred, average='macro')
val2_f1 = f1_score(y_val, y_val_pred, average=None)
print(f'Validation F1 Score: {val_f1}, \n Validation F1 Score (non-weighted): {val2_f1}')
print(classification_report(y_val, y_val_pred))

# Predict on submission set
submission_pred = model.predict(submission)
date = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
pd.DataFrame(submission_pred, columns=['change_type'], index=sample_submission.index).to_csv(f'submissions/OneVsRestRf_{date}.csv', index=True, index_label='Id')