# Linear-Regression-Model
Linear Regression App – A Streamlit-powered web app for performing linear regression on user-uploaded CSV datasets. Supports data preprocessing, model training, evaluation, and visualization.
--
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats

def load_dataset(file):
    """Load dataset from a CSV file and clean data."""
    df = pd.read_csv(file)
    st.write("### Raw Data Preview:")
    st.write(df.head())
    
    df.replace("?", np.nan, inplace=True)
    
    # Convert only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values with column mean only for numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Handle categorical columns efficiently
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if df[col].nunique() <= 10:  # Use One-Hot Encoding only for limited categories
            df = pd.get_dummies(df, columns=[col], drop_first=True)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    st.write("### Cleaned Data Preview:")
    st.write(df.head())
    
    if df.empty or df.isnull().all().all():
        st.error("Dataset is empty or contains only missing values after cleaning. Please upload a valid dataset.")
        return None
    
    return df

def perform_regression(df, target_col):
    """Perform linear regression on the dataset."""
    if df is None or df.empty:
        st.error("Cannot perform regression on an empty dataset.")
        return None, None, None, None, None, None, None, None
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if X.empty or y.empty:
        st.error("No valid features or target column found. Please check the dataset.")
        return None, None, None, None, None, None, None, None
    
    # Split data
    if len(df) < 2:
        st.error("Not enough data points for regression. Please upload a larger dataset.")
        return None, None, None, None, None, None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if X_train.empty or X_test.empty:
        st.error("Train or test set is empty. Try adjusting the dataset size.")
        return None, None, None, None, None, None, None, None
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, X_train, y_train, X_test, y_test, y_pred

def plot_regression(X_test, y_test, y_pred):
    """Plot the regression results (only for single feature)."""
    if X_test is None or y_test is None or y_pred is None:
        return
    
    if X_test.shape[1] == 1:
        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, color='blue', label='Actual')
        ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Target')
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("Plotting is only available for datasets with one feature.")

def analyze_regression_results(y_test, y_pred):
    """Perform additional analysis on regression results."""
    residuals = y_test - y_pred
    
    st.write("### Residual Analysis")
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residual Distribution")
    st.pyplot(fig)
    
    # Q-Q Plot for normality check
    fig, ax = plt.subplots()
    stats.probplot(residuals, dist="norm", plot=ax)
    st.pyplot(fig)
    
    # Scatter plot of residuals
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Predicted Values")
    st.pyplot(fig)

st.title("Linear Regression App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = load_dataset(uploaded_file)
    
    if df is not None:
        st.write("Dataset Preview:", df.head())
        
        target_col = st.selectbox("Select the target column", df.columns)
        
        if st.button("Run Regression"):
            model, mse, r2, X_train, y_train, X_test, y_test, y_pred = perform_regression(df, target_col)
            
            if model is not None:
                st.write(f"Mean Squared Error: {mse}")
                st.write(f"R-squared Score: {r2}")
                plot_regression(X_test, y_test, y_pred)
                analyze_regression_results(y_test, y_pred)
