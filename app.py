"""
Heart Disease Classification - Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                            recall_score, f1_score, matthews_corrcoef,
                            confusion_matrix, classification_report)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide"
)

# Title and description
st.title("Heart Disease Classification System")
st.markdown("""
This interactive application demonstrates multiple machine learning models for predicting heart disease.
Upload test data, select a model, and view detailed performance metrics and predictions.
""")

# Sidebar for model selection
st.sidebar.header("Configuration")

# Model selection
model_options = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree': 'model/decision_tree.pkl',
    'K-Nearest Neighbors (KNN)': 'model/knn.pkl',
    'Naive Bayes': 'model/naive_bayes.pkl',
    'Random Forest': 'model/random_forest.pkl',
    'Gradient Boosting': 'model/gradient_boosting.pkl'
}

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(model_options.keys())
)

# Load scaler
@st.cache_resource
def load_scaler():
    try:
        with open('model/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        st.error("Scaler file not found. Please run train_models.py first.")
        return None

# Load single model
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except:
        st.error(f"Model file not found: {model_path}")
        return None

# Load all models for comparison
@st.cache_resource
def load_all_models():
    """Load all models at once for comparison"""
    models = {}
    for name, path in model_options.items():
        try:
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        except:
            st.warning(f"Could not load {name}")
    return models

# Calculate metrics for a single model
def calculate_model_metrics(model, X_test_scaled, y_test):
    """Calculate all metrics for a given model"""
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    return metrics, y_pred, y_pred_proba

# Calculate metrics for all models
def calculate_all_metrics(models, X_test_scaled, y_test):
    """Calculate metrics for all models on the current test data"""
    all_metrics = {}
    for model_name, model in models.items():
        metrics, _, _ = calculate_model_metrics(model, X_test_scaled, y_test)
        all_metrics[model_name] = metrics
    
    return pd.DataFrame(all_metrics).T

scaler = load_scaler()
model = load_model(model_options[selected_model_name])
all_models = load_all_models()  # Load all models for comparison

# File upload section
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=['csv'],
    help="Upload a CSV file with the same features as the training data"
)

# Use default test data if no file uploaded
if uploaded_file is None:
    if os.path.exists('model/test_data.csv'):
        st.sidebar.info("Using default test dataset")
        test_data = pd.read_csv('model/test_data.csv')
        use_default = True
    else:
        st.warning("Please upload a test dataset or run train_models.py to generate default test data.")
        st.stop()
else:
    test_data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {len(test_data)} samples from uploaded file")
    use_default = False


# Make predictions if model and scaler are loaded
if model is not None and scaler is not None and 'target' in test_data.columns:
    
    # Separate features and target
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate metrics for selected model
    metrics, y_pred, y_pred_proba = calculate_model_metrics(model, X_test_scaled, y_test)
    
    # Display model performance
    st.header(f"{selected_model_name} Performance")
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        st.metric("Precision", f"{metrics['Precision']:.4f}")
    with col2:
        st.metric("AUC Score", f"{metrics['AUC']:.4f}")
        st.metric("Recall", f"{metrics['Recall']:.4f}")
    with col3:
        st.metric("F1 Score", f"{metrics['F1']:.4f}")
        st.metric("MCC Score", f"{metrics['MCC']:.4f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Disease', 'Predicted Disease'],
        y=['Actual No Disease', 'Actual Disease'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader(" Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"))
    
    # Model comparison section - DYNAMICALLY CALCULATED
    st.header(" All Models Comparison")
    
    # Calculate metrics for ALL models using current test data
    metrics_df = calculate_all_metrics(all_models, X_test_scaled, y_test)
    
    # Display metrics table with highlighting
    st.dataframe(
        metrics_df.style.format("{:.4f}").background_gradient(cmap='Greens', axis=0),
        use_container_width=True
    )
    
    # Visualize comparison
    st.subheader("Visual Comparison of Models")
    
    # Create bar chart for all metrics
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(metrics_df.columns)
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    colors = ['#1f77b4', '#87ceeb', '#ff6b6b', '#90ee90', '#ffa07a', '#dda0dd']
    
    for idx, (col, pos) in enumerate(zip(metrics_df.columns, positions)):
        fig.add_trace(
            go.Bar(
                x=metrics_df.index,
                y=metrics_df[col],
                name=col,
                marker_color=colors[idx],
                showlegend=False
            ),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(
        height=600, 
        title_text="Model Performance Metrics Comparison (Current Test Data)"
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)
    

elif model is not None and scaler is not None and 'target' not in test_data.columns:
    st.warning("No 'target' column found in test data. Cannot calculate metrics.")
    st.info("Please upload a CSV file with a 'target' column to see performance metrics.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Heart Disease Classification System | Built with Streamlit</p>
    <p>Dataset: UCI Heart Disease Dataset from Kaggle</p>
</div>
""", unsafe_allow_html=True)