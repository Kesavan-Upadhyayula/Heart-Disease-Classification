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


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Heart Disease Classification System | Built with Streamlit</p>
    <p>Dataset: UCI Heart Disease Dataset from Kaggle</p>
</div>
""", unsafe_allow_html=True)
