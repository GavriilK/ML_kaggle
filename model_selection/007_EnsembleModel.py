from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import datetime
import optuna

train = pd.read_parquet('processing/train_features.parquet')
submission = pd.read_parquet('processing/test_features.parquet')
sample_submission = pd.read_csv('knn_sample_submission.csv', index_col=0)
y = pd.read_parquet('processing/y_train.parquet').values.ravel()

print(train.shape, y.shape)
print(train.head(1))
# train/test/val split :
X_intermediate, X_val, y_intermediate, y_val = train_test_split(train, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_intermediate, y_intermediate, test_size=0.1, random_state=42, stratify=y_intermediate)

binary_models = {}

for i in range(6):
    print(f"Training model for category {i}...")
    # Create binary labels for the current category
    y_train_binary = (y_train == i).astype(int)
    y_test_binary = (y_test == i).astype(int)
    
    def objective(trial):
        param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'max_iter': 200,
        'max_depth': None,
        #'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
        'random_state': 42,
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 1000),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30)
        }
    
        model = HistGradientBoostingClassifier(**param)
        model.fit(X_train, y_train_binary)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test_binary, y_pred, average='binary')
        return f1 

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage='sqlite:///model_selection/models/optuna_study.db',
        load_if_exists=True,
        study_name=f'class_{i}_optimization'
    )
    study.optimize(objective, n_trials=15, show_progress_bar=True)
    print('Best trial:')
    trial = study.best_trial
    print(f'  F1 Score: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # train on full train and estimate value on validation set
    best_params = trial.params
    model = HistGradientBoostingClassifier(**best_params)
    model.fit(X_intermediate, (y_intermediate == i).astype(int))
    y_val_pred = model.predict(X_val)
    val_f1 = f1_score((y_val == i).astype(int), y_val_pred, average='binary')
    val2_f1 = f1_score((y_val == i).astype(int), y_val_pred, average=None)
    print(f'Validation F1 Score: {val_f1}, \n Validation F1 Score (non-weighted): {val2_f1}')
    print(classification_report((y_val == i).astype(int), y_val_pred))
    
    # Store the trained model
    binary_models[i] = model
    
    # Evaluate the model on the test set
    y_pred_binary = model.predict(X_test)
    f1 = f1_score(y_test_binary, y_pred_binary, average='binary')
    print(f"F1 Score for category {i}: {f1}")

general_model_rf = RandomForestClassifier(random_state=42, class_weight='balanced',verbose=3,n_jobs=-1,n_estimators=29,
    max_depth=19,
    criterion='log_loss',
    min_samples_split=3,
    min_samples_leaf=7,
    )
general_model_rf.fit(X_train, y_train)
f1_rf = f1_score(y_test, general_model_rf.predict(X_test), average=None)
print(f"Random Forest F1 Score Per Class: {f1_rf}")

general_model_hist = HistGradientBoostingClassifier(random_state=42, class_weight='balanced',verbose=3,n_iter_no_change=5,max_depth=15,max_iter=500)
general_model_hist.fit(X_train, y_train)
f1_hist = f1_score(y_test, general_model_hist.predict(X_test), average=None)
print(f"HistGradientBoosting F1 Score Per Class: {f1_hist}")

#combining 
probas_binaries = np.zeros((len(X_val), 6))
for i in range(6):
    probas_binaries[:, i] = binary_models[i].predict_proba(X_val)[:, 1]
probas_general_rf = general_model_rf.predict_proba(X_val)
probas_general_hist = general_model_hist.predict_proba(X_val)

#logistic regression ensemble
logit = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=50, n_jobs=-1)
logit.fit(np.concatenate([probas_binaries, probas_general_rf, probas_general_hist], axis=1), y_val)
probas_ensemble = logit.predict_proba(np.concatenate([probas_binaries, probas_general_rf, probas_general_hist], axis=1))
y_pred_ensemble = np.argmax(probas_ensemble, axis=1)
f1_ensemble = f1_score(y_val, y_pred_ensemble, average=None)
print(f"Ensemble F1 Score Per Class: {f1_ensemble}")
print(classification_report(y_val, y_pred_ensemble))

# Simple average ensemble
def combine(*probas):
    return np.mean(probas, axis=0)
probas_ensemble = combine(probas_binaries, probas_general_rf, probas_general_hist)
y_pred_ensemble = np.argmax(probas_ensemble, axis=1)
f1_ensemble = f1_score(y_val, y_pred_ensemble, average=None)
print(f"Ensemble F1 Score Per Class: {f1_ensemble}")
print(classification_report(y_val, y_pred_ensemble))

# make submission
submission_pred_rf = general_model_rf.predict(submission)
submission_pred_hist = general_model_hist.predict(submission)
probas_binaries_submission = np.zeros((len(submission), 6))
for i in range(6):
    probas_binaries_submission[:, i] = binary_models[i].predict_proba(submission)[:, 1]

probas_ensemble_submission = combine(probas_binaries_submission, general_model_rf.predict_proba(submission), general_model_hist.predict_proba(submission))
submission_pred_ensemble = np.argmax(probas_ensemble_submission, axis=1)

date = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
pd.DataFrame(submission_pred_ensemble, columns=['change_type'], index=sample_submission.index).to_csv(f'submissions/ensemble_{date}.csv', index=True, index_label='Id')
