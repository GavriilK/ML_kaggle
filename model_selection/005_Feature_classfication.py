

train = pd.read_parquet('processing/train_features.parquet')
submission = pd.read_parquet('processing/test_features.parquet')
sample_submission = pd.read_csv('knn_sample_submission.csv', index_col=0)
y = pd.read_parquet('processing/y_train.parquet').values.ravel()

# train/test/val split :
X_intermediate, X_val, y_intermediate, y_val = train_test_split(train, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_intermediate, y_intermediate, test_size=0.1, random_state=42, stratify=y_intermediate)
