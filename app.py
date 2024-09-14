import streamlit as st
from dataingestion import load_data
from datacleaning import clean_data
from analysis_engine import linear_regression_analysis,random_forest_analysis,kmeans_clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("AI EMPLOYEE ANALYSIS & REPORT")

# Upload data file
uploaded_file = st.file_uploader("Upload your data file (csv, json, excel)", type=["csv", "json", "xlsx"])

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        st.write(df.head())  # Display the data to ensure it's loaded correctly
        st.success("Data Loaded Successfully!")
        # Clean data
        if st.button('Clean Data'):
            cleaned_data = clean_data(df)
            st.write("Cleaned Data:", cleaned_data.head()) 
            # Select analysis method
            analysis_option = st.selectbox("Select Analysis Method", ("Linear Regression", "K-Means Clustering", "Random Forest"))
            # Linear Regression
            if analysis_option == "Linear Regression":
                target_column = st.selectbox("Select Target Column for Linear Regression",tuple(cleaned_data.columns))
                if st.button('Run Linear Regression'):
                    st.write("inside regression ")
                    X = cleaned_data.drop(columns=[target_column])
                    y = cleaned_data[target_column]
                    st.write("data_Split")
                    model, mse, r2 =linear_regression_analysis(X, y)
                    st.write("model accessed")
                    st.write(f"Mean Squared Error: {mse}")
                    st.write(f"R-squared: {r2}")

                    # Plot Regression Results
                    st.subheader('Regression Results')
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y, model.predict(X), alpha=0.5)
                    plt.xlabel('True Values')
                    plt.ylabel('Predictions')
                    plt.title('True Values vs Predictions')
                    st.pyplot(plt)

                # K-Means Clustering
                elif analysis_option == "K-Means Clustering" and 'cleaned_data' in locals():
                    num_clusters = st.slider("Select number of clusters", 2, 10, 3)
                    if st.button('Run K-Means Clustering'):
                        results, model = kmeans_clustering(cleaned_data, num_clusters)
                        st.write("Clustered Data:", results.head())  # Display the clustered results
                        st.write("Cluster Labels: ", np.unique(results['Cluster']))

                        # Plot Clusters
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(data=results, x=results.columns[0], y=results.columns[1], hue='Cluster', palette='viridis')
                        plt.title('K-Means Clustering')
                        st.pyplot(plt)
                        plt.clf()  # Clear the figure to prevent overlap


                # Random Forest
                elif analysis_option == "Random Forest" and 'cleaned_data' in locals():
                    target_column = st.selectbox("Select Target Column for Random Forest", cleaned_data.columns)
                    if st.button('Run Random Forest'):
                        X = cleaned_data.drop(columns=[target_column])
                        y = cleaned_data[target_column]
                        model, mse, r2, feature_importances = random_forest_analysis(X, y)
                        st.write(f"Mean Squared Error: {mse}")
                        st.write(f"R-squared: {r2}")
                        st.write("Feature Importances:", feature_importances)

                        # Plot Feature Importances
                        st.subheader('Feature Importances')
                        plt.figure(figsize=(10, 6))
                        features = X.columns
                        importance = feature_importances
                        indices = np.argsort(importance)
                        plt.barh(range(len(indices)), importance[indices], align='center')
                        plt.yticks(range(len(indices)), features[indices])
                        plt.xlabel('Feature Importance')
                        plt.title('Random Forest Feature Importances')
                        st.pyplot(plt)




    except Exception as e:
        st.error(f"Error loading data: {e}")




