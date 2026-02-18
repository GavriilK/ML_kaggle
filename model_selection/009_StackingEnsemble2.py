from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import datetime
import optuna
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
import lightgbm as lgb
lgb.set_log_level('error')

PATH_TO_DB = 'sqlite:///model_selection/models/optuna_study2.db'
train = pd.read_parquet('processing/train_features.parquet')
submission = pd.read_parquet('processing/test_features.parquet')
sample_submission = pd.read_csv('knn_sample_submission.csv', index_col=0)
y = pd.read_parquet('processing/y_train.parquet').values.ravel()

print(train.shape, y.shape)
print(train.head(1))
# train/test/val split :
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.1, random_state=42, stratify=y)

binary_models = {}

for i in range(6):
    print(f"Training model for category {i}...")
    # Create binary labels for the current category
    y_train_binary = (y_train == i).astype(int)
    y_test_binary = (y_test == i).astype(int)
    
    def objective(trial):
        param = {
        "objective": "binary",
        "random_state": 42,
        "n_jobs": -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': -1,
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
        'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
        }
    
        model = HistGradientBoostingClassifier(**param)
        model.fit(X_train, y_train_binary)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test_binary, y_pred, average='binary')

        trial.set_user_attr('y_pred', y_pred.tolist())
        return f1 

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=PATH_TO_DB,
        load_if_exists=True,
        study_name=f'class_{i}_2_optimization'
    )
    study.optimize(objective, n_trials=15, show_progress_bar=True)
    print('Best trial:')
    trial = study.best_trial
    print(f'  F1 Score: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    print(classification_report(y_test_binary,trial.user_attrs['y_pred']))

    # train on full train
    best_params = trial.params
    model = HistGradientBoostingClassifier(**best_params)
    model.fit(train, (y == i).astype(int))
    
    # Store the trained model
    binary_models[i] = model
print("Training general Random Forest...")
general_model_rf = RandomForestClassifier(random_state=42, class_weight='balanced',verbose=3,n_jobs=-1,n_estimators=29,
    max_depth=19,
    criterion='log_loss',
    min_samples_split=3,
    min_samples_leaf=7,
    )
general_model_rf.fit(X_train, y_train)
f1_rf = f1_score(y_test, general_model_rf.predict(X_test), average=None)
print(f"Random Forest F1 Score Per Class: {f1_rf}")
general_model_rf.fit(train, y)

print("Training general HistGradientBoosting...")
general_model_hist = HistGradientBoostingClassifier(random_state=42, class_weight='balanced',verbose=3,n_iter_no_change=5,max_depth=15,max_iter=500)
general_model_hist.fit(X_train, y_train)
f1_hist = f1_score(y_test, general_model_hist.predict(X_test), average=None)
print(f"HistGradientBoosting F1 Score Per Class: {f1_hist}")
general_model_hist.fit(train, y)

#combining 
probas_binaries = np.zeros((len(train), 6))
for i in range(6):
    probas_binaries[:, i] = binary_models[i].predict_proba(train)[:, 1]
probas_general_rf = general_model_rf.predict_proba(train)
probas_general_hist = general_model_hist.predict_proba(train)

#Stacking lightGBM ensemble
meta_rf = LGBMClassifier(num_leaves=75, min_data_in_leaf=2000, random_state=42,num_threads=-1,verbosity=-1)

# F1 score for meta RF ensemble
cv_scores_weighted = cross_val_score(meta_rf, np.concatenate([probas_binaries, probas_general_rf, probas_general_hist], axis=1), y, cv=5, scoring='f1_weighted')
cv_scores_macro = cross_val_score(meta_rf, np.concatenate([probas_binaries, probas_general_rf, probas_general_hist], axis=1), y, cv=5, scoring='f1_macro')
print(f"CV Macro F1: {cv_scores_macro.mean():.4f} ± {cv_scores_macro.std():.4f}")
print(f"CV Weighted F1: {cv_scores_weighted.mean():.4f} ± {cv_scores_weighted.std():.4f}")

# fit on whole data and make report (biased but gives an idea)
meta_rf.fit(np.concatenate([probas_binaries, probas_general_rf, probas_general_hist], axis=1), y)
probas_ensemble = meta_rf.predict_proba(np.concatenate([probas_binaries, probas_general_rf, probas_general_hist], axis=1))
y_pred_ensemble = np.argmax(probas_ensemble, axis=1)
f1_ensemble = f1_score(y, y_pred_ensemble, average=None)
print(f"Ensemble F1 Score Per Class: {f1_ensemble}")
print(classification_report(y, y_pred_ensemble))


# make submission
submission_pred_rf = general_model_rf.predict(submission)
submission_pred_hist = general_model_hist.predict(submission)
probas_binaries_submission = np.zeros((len(submission), 6))
for i in range(6):
    probas_binaries_submission[:, i] = binary_models[i].predict_proba(submission)[:, 1]

probas_ensemble_submission = meta_rf.predict_proba(np.concatenate([probas_binaries_submission, general_model_rf.predict_proba(submission), general_model_hist.predict_proba(submission)], axis=1))
submission_pred_ensemble = np.argmax(probas_ensemble_submission, axis=1)

date = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
pd.DataFrame(submission_pred_ensemble, columns=['change_type'], index=sample_submission.index).to_csv(f'submissions/ensemble_{date}.csv', index=True, index_label='Id')
