from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

def clean_data(df):
    # Convert any potential string dates to datetime (without using deprecated 'errors' argument)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError) as e:
                # Ignore columns that can't be converted to datetime
                continue
    
    # Handle numerical columns
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        df[num_cols] = imputer_num.fit_transform(df[num_cols])
    
    # Identify datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime']).columns
    
    # Fill missing values in datetime columns (use ffill directly instead of fillna)
    if len(datetime_cols) > 0:
        df[datetime_cols] = df[datetime_cols].ffill()  # You can use .bfill() as needed
    
    # Handle categorical columns (after ensuring no datetime columns are included)
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    
    return df






