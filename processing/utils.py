import pandas as pd
from scikit-learn.preprocessing import StandardScaler, Yule, OneHotEncoder

dict_params = {
    'remove_nulls': {'threshold': 0.5},
    'encode_categorical': {'method': 'onehot'},
    'scale_features': {'method': 'standard'},
    'normalize_data': {'method': ''},
}

def process_data(dataframe,dict_params):
    # Function to process data that uses all the functions specified in the dict_params
    for func, params in dict_params.items():
        dataframe = pd.apply(func, **params, inplace=True)
    return dataframe

# Non linear features:
def power_features(dataframe,power):