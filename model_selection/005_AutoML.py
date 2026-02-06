from flaml import AutoML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

train = pd.read_parquet('processing/train_features.parquet')
submission = pd.read_parquet('processing/test_features.parquet')
sample_submission = pd.read_csv('knn_sample_submission.csv', index_col=0)
y = pd.read_parquet('processing/y_train.parquet').values.ravel()

# Convert column names to regular strings
train.columns = train.columns.astype(str)
submission.columns = submission.columns.astype(str)

# train/test :
#X_train, X_test, y_train, y_test = train_test_split(train,y,test_size=0.15,random_state=42,stratify=y)

# automl run
automl_settings = {
    "time_budget": 300,  # total time in seconds
    "metric": "macro_f1",
    "task": "classification",
    "verbose": 5,
    "seed": 42,
    }
automl = AutoML()
automl.fit(X_train=train.to_numpy(), y_train=y, **automl_settings)

y_pred = automl.predict(train.to_numpy())
print(classification_report(y, y_pred))

#make submission
submission_pred = automl.predict(submission.to_numpy())
date = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
pd.DataFrame(submission_pred, columns=['change_type'], index=sample_submission.index).to_csv(f'submissions/bova_submission_{date}.csv', index=True, index_label='Id')