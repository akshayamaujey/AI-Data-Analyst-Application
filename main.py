import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import sqlite3
import google.generativeai as genai
import numpy as np

# Configure the Google Generative AI API key
GOOGLE_API_KEY="AIzaSyCYp9phlVaUCxiqS_Y50xpI90f9UC24J30"
genai.configure(api_key=GOOGLE_API_KEY)

def load_data(file):
    file_type = file.type
    if file_type == "text/csv":
        df = pd.read_csv(file)
    elif file_type == "application/json":
        df = pd.read_json(file)
    elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None
    return df

def clean_data(df):
    df = df.dropna()
    df['Rank'] = df['Rank'].astype(int)
    df['Gold'] = df['Gold'].astype(int)
    df['Silver'] = df['Silver'].astype(int)
    df['Bronze'] = df['Bronze'].astype(int)
    df['Total'] = df['Total'].astype(int)
    return df

def plot_regression_results(y_true, y_pred, algo):
    fig = px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
                     title=f'{algo} - Actual vs Predicted')
    st.plotly_chart(fig)

def plot_feature_importance(model, df):
    features = ['Gold', 'Silver', 'Bronze']  # Ensure these match your input columns
    feature_importance = model.feature_importances_
    
    # Check lengths of both features and feature_importance
    if len(features) == len(feature_importance):
        fig = px.bar(x=features, y=feature_importance, title='Feature Importance')
        st.plotly_chart(fig)
    else:
        st.error("The length of feature importance does not match the number of features.")
        

def perform_analysis(df, algo):
    X = df[['Gold', 'Silver', 'Bronze']]  # We use only Gold, Silver, and Bronze as features
    y = df['Rank']  # Target variable
    
    if algo == 'Linear Regression':
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        evs = explained_variance_score(y, preds)
        
        # Plotting regression results
        plot_regression_results(y, preds, algo)
        
        return {'MSE': mse, 'R2': r2, 'MAE': mae, 'Explained Variance Score': evs}

    elif algo == 'Random Forest':
        model = RandomForestRegressor()
        model.fit(X, y)
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        evs = explained_variance_score(y, preds)
        
        # Plot feature importance
        try:
            plot_feature_importance(model, df)
        except ValueError as e:
            st.write(f"An error occurred while plotting feature importance: {e}")
        
        return {'MSE': mse, 'R2': r2, 'MAE': mae, 'Explained Variance Score': evs}

    elif algo == 'K-Means Clustering':
        model = KMeans(n_clusters=3)
        df['Cluster'] = model.fit_predict(X)

        # Plot clusters
        fig = px.scatter(df, x='Total', y='Rank', color='Cluster', title='K-Means Clustering')
        st.plotly_chart(fig)

        # Calculate K-Means metrics
        inertia = model.inertia_
        silhouette = silhouette_score(X, model.labels_)
        calinski_harabasz = calinski_harabasz_score(X, model.labels_)
        davies_bouldin = davies_bouldin_score(X, model.labels_)

        return {
            'Inertia': inertia,
            'Silhouette Coefficient': silhouette,
            'Calinski-Harabasz Index': calinski_harabasz,
            'Davies-Bouldin Index': davies_bouldin
        }

def generate_model_summary(graph_img, metrics, algo):
    prompt = f'''
    Act as a professional data analyst. Analyze the following graph and metrics for {algo}:
    Metrics: {metrics}
    Generate a point-wise summary of key insights.
    '''
    model = genai.GenerativeModel('gemini-pro')
    try:
        response = model.generate_content(prompt)
        return response.text.strip("```").strip()
    except ValueError as e:
        if "response was blocked" in str(e):
            return "Summary blocked due to legal reasons."
        else:
            return "An unexpected error occurred. Please try again."

def predict_values(df, model, algo, input_data):
    X = pd.DataFrame([input_data], columns=['Gold', 'Silver', 'Bronze'])
    
    if algo == 'Linear Regression':
        prediction = model.predict(X)
        st.sidebar.write(f"Predicted Rank (Linear Regression): {round(prediction[0])}")
        
    elif algo == 'Random Forest':
        prediction = model.predict(X)
        st.sidebar.write(f"Predicted Rank (Random Forest): {round(prediction[0])}")
        
    elif algo == 'K-Means Clustering':
        cluster = model.predict(X)
        st.sidebar.write(f"Predicted Cluster (K-Means): {cluster[0]}")
        
def generate_response(user_query, df):
    prompt = f'''
    Act as a professional data analyst. Given the following data columns and types:
    {list(df.columns)} with their types.
    The user query is: "{user_query}"
    Generate an SQL query to answer this question. Use the table name 'data'. Do not include the word 'sql' at the beginning of the query.
    '''
    model = genai.GenerativeModel('gemini-pro')
    try:
        response = model.generate_content(prompt)
        sql_query = response.text.strip("```").strip()
        # Ensure no 'sql' prefix in the query
        if sql_query.lower().startswith("sql"):
            sql_query = sql_query[3:].strip()
        return sql_query
    except ValueError as e:
        if "response was blocked" in str(e):
            return "Answer blocked due to legal reasons."
        else:
            return "An unexpected error occurred. Please try again."

def main():
    st.title('AI Data Analyst App')
    uploaded_file = st.sidebar.file_uploader("Upload Data (CSV, JSON, Excel)", type=['csv', 'json', 'xlsx'])
    selected_algo = st.sidebar.selectbox("Select Machine Learning Algorithm", ['Linear Regression', 'Random Forest', 'K-Means Clustering'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            df = clean_data(df)
            st.write("Cleaned Data Preview:", df.head())
            
            # Input section for predictions
            st.sidebar.subheader("Input for Prediction")
            input_gold = st.sidebar.number_input("Gold Medals", min_value=0, step=1)
            input_silver = st.sidebar.number_input("Silver Medals", min_value=0, step=1)
            input_bronze = st.sidebar.number_input("Bronze Medals", min_value=0, step=1)
            
            analysis_results = perform_analysis(df, selected_algo)
            st.subheader("Perform analysis")
            st.write(f"{selected_algo} Results:", analysis_results)

            if selected_algo != 'K-Means Clustering':
                # Generate a summary for linear regression or random forest
                summary = generate_model_summary("graph_placeholder", analysis_results, selected_algo)
                st.subheader("Generated Summary")
                st.write(summary)
            elif selected_algo == 'K-Means Clustering':
                # Generate a summary for K-Means Clustering
                summary = generate_model_summary("graph_placeholder", analysis_results, selected_algo)
                st.subheader("Generated Summary")
                st.write(summary)

            # Prediction section
            if st.sidebar.button("Predict"):
                input_data = [input_gold, input_silver, input_bronze]
                
                # Get the trained model and perform prediction
                if selected_algo == 'Linear Regression':
                    model = LinearRegression()
                elif selected_algo == 'Random Forest':
                    model = RandomForestRegressor()
                elif selected_algo == 'K-Means Clustering':
                    model = KMeans(n_clusters=3)

                # Fit the model on the data
                X = df[['Gold', 'Silver', 'Bronze']]
                y = df['Rank']
                model.fit(X, y)
                
                # Predict values
                predict_values(df, model, selected_algo, input_data)

            # Query section
            st.subheader("Talk with Data")
            user_query = st.text_input("Ask a question to data")
            if user_query:
                sql_query = generate_response(user_query, df)
                if sql_query:
                    try:
                        conn = sqlite3.connect(":memory:")
                        df.to_sql('data', conn, index=False, if_exists='replace')
                        query_result = pd.read_sql_query(sql_query, conn)
                        st.write("Query Result:", query_result)
                    except Exception as e:
                        st.write(f"An error occurred while executing the query: {e}")

if __name__ == "__main__":
    main()
