import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
import warnings
warnings.filterwarnings('ignore') #because we don't project into a specific coordinate reference system, we get warnings about distance calculations. 
# We don't really care as we just need the polygon as shapes not as geospatial objects.

#import data
train_df = gpd.read_file('data/train.geojson')
test_df = gpd.read_file('data/test.geojson')

# Encode y labels
print("Encoding target variable...")
change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5}
y = train_df['change_type'].apply(lambda x: change_type_map[x])

# One hot encode urban_type and geography_type (split comma-separated values)
def split_categories(df, column):
    return df[column].apply(lambda x: [item.strip() for item in x.split(',') if item.strip() not in ['N', 'A']])

# Encode urban_type
print("Encoding urban_type...")
mlb_urban = MultiLabelBinarizer()
train_urban = mlb_urban.fit_transform(split_categories(train_df, 'urban_type'))
test_urban = mlb_urban.transform(split_categories(test_df, 'urban_type'))
train_urban_df = pd.DataFrame(train_urban, columns=[f'urban_{c}' for c in mlb_urban.classes_], index=train_df.index)
test_urban_df = pd.DataFrame(test_urban, columns=[f'urban_{c}' for c in mlb_urban.classes_], index=test_df.index)

# Encode geography_type
print("Encoding geography_type...")
mlb_geo = MultiLabelBinarizer()
train_geo = mlb_geo.fit_transform(split_categories(train_df, 'geography_type'))
test_geo = mlb_geo.transform(split_categories(test_df, 'geography_type'))
train_geo_df = pd.DataFrame(train_geo, columns=[f'geo_{c}' for c in mlb_geo.classes_], index=train_df.index)
test_geo_df = pd.DataFrame(test_geo, columns=[f'geo_{c}' for c in mlb_geo.classes_], index=test_df.index)

# Time intervals between dates - TRAIN
print("Calculating time intervals between dates...")
date1_train = pd.to_datetime(train_df['date1'], format='%d-%m-%Y')
date2_train = pd.to_datetime(train_df['date2'], format='%d-%m-%Y')
date3_train = pd.to_datetime(train_df['date3'], format='%d-%m-%Y')
date4_train = pd.to_datetime(train_df['date4'], format='%d-%m-%Y')
train_intervals = pd.DataFrame({
    'interval_1_2': np.abs((date2_train - date1_train).dt.days),
    'interval_2_3': np.abs((date3_train - date2_train).dt.days),
    'interval_3_4': np.abs((date4_train - date3_train).dt.days)
}, index=train_df.index)

# Time intervals between dates - TEST
date1_test = pd.to_datetime(test_df['date1'], format='%d-%m-%Y')
date2_test = pd.to_datetime(test_df['date2'], format='%d-%m-%Y')
date3_test = pd.to_datetime(test_df['date3'], format='%d-%m-%Y')
date4_test = pd.to_datetime(test_df['date4'], format='%d-%m-%Y')
test_intervals = pd.DataFrame({
    'interval_1_2': np.abs((date2_test - date1_test).dt.days),
    'interval_2_3': np.abs((date3_test - date2_test).dt.days),
    'interval_3_4': np.abs((date4_test - date3_test).dt.days)
}, index=test_df.index)

# Select only image features to standardize and keep
print("Standardizing image features...")
img_features = [col for col in train_df.columns if col.startswith('img_')]
train_img = train_df[img_features].copy()
test_img = test_df[img_features].copy()

# create variation features for the images
train_diff_img = train_img.diff(axis=1).add_suffix('_diff')
test_diff_img = test_img.diff(axis=1).add_suffix('_diff')
train_img = pd.concat([train_img, train_diff_img], axis=1)
test_img = pd.concat([test_img, test_diff_img], axis=1)

# create mean/std features 
std_features = [name for name in train_img.columns if 'std' in name]
mean_features = [name for name in train_img.columns if 'mean' in name]
for names in zip(std_features, mean_features):
    std_name = names[0]
    mean_name = names[1]
    train_img[f'{mean_name}_over_{std_name}'] = train_img[mean_name] / (train_img[std_name] + 1e-5) # add small value to avoid division by zero
    test_img[f'{mean_name}_over_{std_name}'] = test_img[mean_name] / (test_img[std_name] + 1e-5)

# Standardize std and mean features
scaler = StandardScaler()
train_img_scaled = pd.DataFrame(
    scaler.fit_transform(train_img),
    columns=train_img.columns,
    index=train_df.index
)
test_img_scaled = pd.DataFrame(
    scaler.transform(test_img),
    columns=test_img.columns,
    index=test_df.index
)

# Creating label for categorical features
print("Encoding change_status_date features...")
change_status_cols = [f'change_status_date{i}' for i in range(1, 5)]
change_status_train = train_df[change_status_cols].apply(LabelEncoder().fit_transform)
change_status_test = test_df[change_status_cols].apply(LabelEncoder().fit_transform)

# Polygon features
print("area and perimeter (log and raw)...")
area_train = train_df.geometry.area.values
area_test = test_df.geometry.area.values
perimeter_train = train_df.geometry.length.values
perimeter_test = test_df.geometry.length.values

min_perimeter = min(perimeter_train.min(), perimeter_test.min())*0.01
polygon_features_train = pd.DataFrame({
    'area_log': np.log1p(area_train),
    'area': area_train,
    'perimeter_log': np.log1p(perimeter_train),
    'perimeter': perimeter_train,
    'x_centroid': train_df.geometry.centroid.x.values,
    'y_centroid': train_df.geometry.centroid.y.values,
    'product': area_train * perimeter_train,
    'ratio': area_train / (perimeter_train + min_perimeter)
}, index=train_df.index)

polygon_features_test = pd.DataFrame({
    'area_log': np.log1p(area_test),
    'area': area_test,
    'perimeter_log': np.log1p(perimeter_test),
    'perimeter': perimeter_test,
    'x_centroid': test_df.geometry.centroid.x.values,
    'y_centroid': test_df.geometry.centroid.y.values,
    'product': area_test * perimeter_test,
    'ratio': area_test / (perimeter_test + min_perimeter)
}, index=test_df.index)

# Standardize polygon features
print("Standardizing polygon features...")
scaler_polygon = StandardScaler()
polygon_features_train_scaled = pd.DataFrame(scaler_polygon.fit_transform(polygon_features_train), columns=polygon_features_train.columns, index=train_df.index)
polygon_features_test_scaled = pd.DataFrame(scaler_polygon.transform(polygon_features_test), columns=polygon_features_test.columns, index=test_df.index)

# Try a classifier that predicts the two minority classes and adds it as a feature:
print("Training a classifier to predict the minority classes and adding it as a feature...")
minority_class = [4, 5] # industrial and mega projects
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced',n_jobs=-1)
rf.fit(train_img_scaled, y.isin(minority_class))
train_minority_proba = rf.predict_proba(train_img_scaled)[:, 1] # probability of being in the minority class
test_minority_proba = rf.predict_proba(test_img_scaled)[:, 1]
train_minority_proba_df = pd.DataFrame(train_minority_proba, columns=['minority_proba'], index=train_df.index)
test_minority_proba_df = pd.DataFrame(test_minority_proba, columns=['minority_proba'], index=test_df.index)

#same with medium frequency classes
print("Training a classifier to predict the medium frequency classes and adding it as a feature...")
medium_high_class = [0, 1]
rf_medium_high = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_medium_high.fit(train_img_scaled, y.isin(medium_high_class))
train_medium_high_proba = rf_medium_high.predict_proba(train_img_scaled)[:, 1] # probability of being in the medium/high frequency class
test_medium_high_proba = rf_medium_high.predict_proba(test_img_scaled)[:, 1]
train_medium_high_proba_df = pd.DataFrame(train_medium_high_proba, columns=['medium_high_proba'], index=train_df.index) 
test_medium_high_proba_df = pd.DataFrame(test_medium_high_proba, columns=['medium_high_proba'], index=test_df.index)    

# same with the most frequent class
"""print("Training a classifier to predict the most frequent class and adding it as a feature...")
most_freq_class = [2, 3]
rf_most_freq = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_most_freq.fit(train_img_scaled, y.isin(most_freq_class))
train_most_freq_proba = rf_most_freq.predict_proba(train_img_scaled)[:, 1] # probability of being in the most frequent class
test_most_freq_proba = rf_most_freq.predict_proba(test_img_scaled)[:, 1]
train_most_freq_proba_df = pd.DataFrame(train_most_freq_proba, columns=['most_freq_proba'], index=train_df.index) 
test_most_freq_proba_df = pd.DataFrame(test_most_freq_proba, columns=['most_freq_proba'], index=test_df.index)
""" #--- IGNORE --- 


# Concatenate all features
print("Concatenating all features...")
train_features = pd.concat([
    train_img_scaled,
    train_urban_df,
    change_status_train,
    train_geo_df,
    train_intervals,
    polygon_features_train_scaled,
    train_minority_proba_df,
    train_medium_high_proba_df,
    #train_most_freq_proba_df
], axis=1)

test_features = pd.concat([
    test_img_scaled,
    test_urban_df,
    change_status_test,
    test_geo_df,
    test_intervals,
    polygon_features_test_scaled,
    test_minority_proba_df,
    test_medium_high_proba_df,
    #test_most_freq_proba_df
], axis=1)
# Check for missing values
print(f"Missing values in train: {train_features.isna().sum().sum()}")
print(f"Columns with all NaN: {train_features.columns[train_features.isna().all()].tolist()}")

# Drop columns that are entirely NaN
train_features = train_features.dropna(axis=1, how='all')
test_features = test_features[train_features.columns]  # Keep same columns in test

# Use an imputer to fill in missing values in the features
print("Imputing missing values with the imputer...")
#imputer = KNNImputer(n_neighbors=3) #was too slow 
imputer = SimpleImputer(strategy='median')
train_features_imputed = pd.DataFrame(imputer.fit_transform(train_features), columns=train_features.columns, index=train_features.index)
test_features_imputed = pd.DataFrame(imputer.transform(test_features), columns=test_features.columns, index=test_features.index)

# Convert column names to regular strings (not StringDtype)
train_features_imputed.columns = train_features_imputed.columns.astype(str)
test_features_imputed.columns = test_features_imputed.columns.astype(str)

# export
print("Exporting features to parquet files...")
train_features_imputed.to_parquet('processing/train_features.parquet')
test_features_imputed.to_parquet('processing/test_features.parquet')
pd.DataFrame(y).to_parquet('processing/y_train.parquet')
