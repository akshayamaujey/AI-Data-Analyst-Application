import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression for trend analysis
def linear_regression_analysis(X, y):
    """
    Perform linear regression to identify trends in the data.
    
    Args:
        X (DataFrame): The features for prediction.
        y (Series): The target variable to predict.

    Returns:
        model: Trained linear regression model.
        mse (float): Mean Squared Error of the model.
        r2 (float): R-squared value of the model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

# K-Means for clustering/pattern recognition
def kmeans_clustering(df, n_clusters=3):
    """
    Perform K-Means clustering to group data points based on their similarities.
    
    Args:
        df (DataFrame): The cleaned dataset to perform clustering on.
        n_clusters (int): The number of clusters to form.

    Returns:
        df (DataFrame): DataFrame with cluster labels added.
        model: Trained KMeans model.
    """
    model = KMeans(n_clusters=n_clusters)
    df['Cluster'] = model.fit_predict(df.select_dtypes(include=np.number))
    return df, model

# Random Forest for predictions and feature importance
def random_forest_analysis(X, y):
    """
    Perform Random Forest regression to predict values and identify important features.
    
    Args:
        X (DataFrame): The features for prediction.
        y (Series): The target variable to predict.

    Returns:
        model: Trained random forest model.
        mse (float): Mean Squared Error of the model.
        r2 (float): R-squared value of the model.
        feature_importances (array): Feature importance scores.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    feature_importances = model.feature_importances_
    
    return model, mse, r2, feature_importances

