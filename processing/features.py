import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
import numpy as np
from sklearn.impute import KNNImputer
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

# Standardize std and mean features
scaler = StandardScaler()
train_img_scaled = pd.DataFrame(
    scaler.fit_transform(train_img),
    columns=img_features,
    index=train_df.index
)
test_img_scaled = pd.DataFrame(
    scaler.transform(test_img),
    columns=img_features,
    index=test_df.index
)

# Creating label for categorical features
print("Encoding change_status_date features...")
change_status_cols = [f'change_status_date{i}' for i in range(1, 5)]
change_status_train = train_df[change_status_cols].apply(LabelEncoder().fit_transform)
change_status_test = test_df[change_status_cols].apply(LabelEncoder().fit_transform)

# Polygon features
print("Calculating polygon features...")
area_train = pd.DataFrame({'area': np.log1p(train_df.geometry.area)}, index=train_df.index)
area_test = pd.DataFrame({'area': np.log1p(test_df.geometry.area)}, index=test_df.index)

perimeter_train = pd.DataFrame({'perimeter': np.log1p(train_df.geometry.length)}, index=train_df.index)
perimeter_test = pd.DataFrame({'perimeter': np.log1p(test_df.geometry.length)}, index=test_df.index)

x_centroid_train = pd.DataFrame({'x_centroid': train_df.geometry.centroid.x}, index=train_df.index)
y_centroid_train = pd.DataFrame({'y_centroid': train_df.geometry.centroid.y}, index=train_df.index)
x_centroid_test = pd.DataFrame({'x_centroid': test_df.geometry.centroid.x}, index=test_df.index)
y_centroid_test = pd.DataFrame({'y_centroid': test_df.geometry.centroid.y}, index=test_df.index)

# Standardize polygon features
polygon_features_train = pd.concat([area_train, perimeter_train, x_centroid_train, y_centroid_train], axis=1)
polygon_features_test = pd.concat([area_test, perimeter_test, x_centroid_test, y_centroid_test], axis=1)
scaler_polygon = StandardScaler()
polygon_features_train_scaled = pd.DataFrame(scaler_polygon.fit_transform(polygon_features_train), columns=polygon_features_train.columns, index=train_df.index)
polygon_features_test_scaled = pd.DataFrame(scaler_polygon.transform(polygon_features_test), columns=polygon_features_test.columns, index=test_df.index)

# Concatenate all features
train_features = pd.concat([
    train_img_scaled,
    train_urban_df,
    change_status_train,
    train_geo_df,
    train_intervals,
    polygon_features_train_scaled
], axis=1)

test_features = pd.concat([
    test_img_scaled,
    test_urban_df,
    change_status_test,
    test_geo_df,
    test_intervals,
    polygon_features_test_scaled
], axis=1)

# Use an knn imputer to fill in missing values in the features
imputer = KNNImputer(n_neighbors=5)
train_features_imputed = pd.DataFrame(imputer.fit_transform(train_features), columns=train_features.columns, index=train_features.index)
test_features_imputed = pd.DataFrame(imputer.transform(test_features), columns=test_features.columns, index=test_features.index)

train_features_imputed.to_parquet('processing/train_features.parquet')
test_features_imputed.to_parquet('processing/test_features.parquet')
pd.DataFrame(y).to_parquet('processing/y_train.parquet')