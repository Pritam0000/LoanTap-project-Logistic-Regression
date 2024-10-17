import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher

def convert_term(term):
    return int(term.split()[0])

def ensure_data_types(df):
    type_dict = {
        'loan_amnt': 'float32',
        'term': 'object',
        'int_rate': 'float32',
        'emp_length': 'object',
        'annual_inc': 'float32'
    }
    for col, dtype in type_dict.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df

def preprocess_data(df, is_training=True):
    df = ensure_data_types(df)

    if 'term' in df.columns:
        df['term'] = df['term'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) else x)

    categorical_cols = ['grade', 'sub_grade', 'emp_title', 'home_ownership', 
                        'verification_status', 'purpose', 'title', 
                        'initial_list_status', 'application_type']

    for col in ['pub_rec', 'mort_acc', 'pub_rec_bankruptcies']:
        if col in df.columns:
            df[f'{col}_flag'] = (df[col] > 1).astype(int)

    hashing_cols = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'purpose']
    
    if is_training:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        joblib.dump(numeric_columns, 'models/numeric_columns.pkl')
    else:
        numeric_columns = joblib.load('models/numeric_columns.pkl')

    # Ensure all expected columns are present
    for col in numeric_columns:
        if col not in df.columns:
            df[col] = 0

    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    hasher = FeatureHasher(n_features=100, input_type='string')
    hashed_features = hasher.fit_transform(df[hashing_cols].astype(str).values)

    X = np.hstack([df[numeric_columns].values, hashed_features.toarray()])

    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(hashing_cols, 'models/hashing_cols.pkl')
        joblib.dump(X.shape[1], 'models/n_features.pkl')

        y = df['loan_status'].values if 'loan_status' in df.columns else None
        return X_scaled, y
    else:
        scaler = joblib.load('models/scaler.pkl')
        n_features = joblib.load('models/n_features.pkl')
        
        # Ensure the number of features matches the training data
        if X.shape[1] < n_features:
            X = np.pad(X, ((0, 0), (0, n_features - X.shape[1])), mode='constant')
        elif X.shape[1] > n_features:
            X = X[:, :n_features]
        
        X_scaled = scaler.transform(X)
        return X_scaled