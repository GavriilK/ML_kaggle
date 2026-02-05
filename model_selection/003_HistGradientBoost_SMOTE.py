from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import pandas as pd
import optuna
import datetime
from imblearn.over_sampling import SMOTE

train = pd.read_parquet('processing/train_features.parquet')
submission = pd.read_parquet('processing/test_features.parquet')
sample_submission = pd.read_csv('knn_sample_submission.csv', index_col=0)
y = pd.read_parquet('processing/y_train.parquet').values.ravel()

# Check for NaN in original data
print(f"NaN in train before SMOTE: {train.isna().sum().sum()}")
print(f"NaN in y: {pd.isna(y).sum()}")

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42, sampling_strategy="not majority")
X_resampled, y_resampled = smote.fit_resample(train, y)

# train/test/val split :
X_intermediate, X_val, y_intermediate, y_val = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42, stratify=y_resampled)
X_train, X_test, y_train, y_test = train_test_split(X_intermediate, y_intermediate, test_size=0.1, random_state=42, stratify=y_intermediate)

def objective(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),
        'max_iter': 500,
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
        'random_state': 42,
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
    }
    
    model = HistGradientBoostingClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
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
val_f1 = f1_score(y_val, y_val_pred, average='weighted')
val2_f1 = f1_score(y_val, y_val_pred, average=None)
print(f'Validation F1 Score: {val_f1}, \n Validation F1 Score (non-weighted): {val2_f1}')


# Predict on submission set
submission_pred = model.predict(submission)
date = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
pd.DataFrame(submission_pred, columns=['change_type'], index=sample_submission.index).to_csv(f'submissions/histgb_SMOTENC_submission_{date}.csv', index=True, index_label='Id')