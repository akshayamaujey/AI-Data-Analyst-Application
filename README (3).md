## AI Data Analyst Application
## Table of Contents
1. Overview
2. Key Features
3. Application Workflow
4. Challenges and Solutions
5. Future Enhancements
6. How to Use
7. Contact
# 1. Project Overview

Overview

The AI Data Analyst Application is a versatile tool designed to simplify data analysis and prediction tasks using machine learning algorithms. It enables users to easily upload datasets, clean the data, apply machine learning models, and receive AI-generated insights, all through a user-friendly interface.

## 2.Project Components
Data Upload: 
Supports CSV, JSON, and Excel file uploads for analysis.

Data Cleaning: 
Automatically handles missing values and adjusts data types to ensure compatibility with machine learning models.

Machine Learning Models:
Linear Regression: Predicts continuous variables based on input features.

Random Forest: 
A robust ensemble model for handling non-linear relationships and visualizing feature importance.

K-Means Clustering:
 Groups data into clusters with performance metrics like Inertia and Silhouette Coefficient.

AI Integration:
 Uses Google Gemini AI to generate SQL queries and provide insights based on user input.

Interactive Visualizations:
 Includes scatter plots, bar charts, and cluster plots to make sense of model performance and predictions
## 3.Application Workflow
***Data Ingestion and Cleaning:***
 Users upload datasets (CSV, JSON, or Excel), and the app cleans the data using the Pandas library by addressing missing values and ensuring proper data types.

***Model Selection and Prediction:*** Choose between Linear Regression, Random Forest, or K-Means Clustering to predict outcomes or group data into clusters. Users can input parameters to run predictions or clustering.

***AI-Generated Insights:*** Google Gemini AI provides detailed summaries of model performance and auto-generates SQL queries from user-provided natural language prompts.

***Visualizations:*** Interactive plots that provide an intuitive understanding of model predictions, feature importance, and clustering results.

## 4.Challenges and Solutions
***Model Performance:*** Improved feature mapping for Random Forest to resolve visualization errors.

***AI Integration:*** Overcame response blocking and crafted better prompts for accurate AI output.

***User Interaction:*** Simplified UI for seamless data input and output visualizations.
## 5. Future Enhancements
Support for more advanced algorithms (e.g., SVM, Gradient Boosting).

Improved AI integration for contextual suggestions and automated SQL execution.

Enhanced user interaction with dynamic input methods like sliders.
Additional unit testing and model hyperparameter tuning for more control over analysis.
## How to Use
1. Clone this repository
git clone https://github.com/akshayamaujey/AI-Data-Analyst-Application.git

2. Install the required dependencies:
pip install -r requirements.txt

3. Run the application locally:
streamlit run app.py



## Contact 
For any questions or further information, please feel free to reach out:

GitHub: https://github.com/akshayamaujey

Email: akshayamaujey@gmail.com