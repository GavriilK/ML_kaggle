from flaml import AutoML
import pandas as pd
from sklearn.model_selection import train_test_split


train = pd.read_parquet('processing/train_features.parquet')
submission = pd.read_parquet('processing/test_features.parquet')
sample_submission = pd.read_csv('knn_sample_submission.csv', index_col=0)
y = pd.read_parquet('processing/y_train.parquet').values.ravel()

# train/test :
X_train, X_test, y_train, y_test = train_test_split(train,y,test_size=0.15,random_state=42,stratify=y)

# automl run
automl = AutoML()
