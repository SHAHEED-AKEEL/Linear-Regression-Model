import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_dataset(file):
    """Load dataset from a CSV file."""
    return pd.read_csv(file)

def perform_regression(df, target_col):
    """Perform linear regression on the dataset."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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

st.title("Linear Regression App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = load_dataset(uploaded_file)
    st.write("Dataset Preview:", df.head())
    
    target_col = st.selectbox("Select the target column", df.columns)
    
    if st.button("Run Regression"):
        model, mse, r2, X_train, y_train, X_test, y_test, y_pred = perform_regression(df, target_col)
        
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared Score: {r2}")
        
        plot_regression(X_test, y_test, y_pred)
