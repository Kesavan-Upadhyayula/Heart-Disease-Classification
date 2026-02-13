# Heart-Disease-Classification

Heart Disease Classification - Machine Learning Project
Problem Statement
Heart disease remains one of the leading causes of death worldwide. Early detection and prediction of heart disease can save lives by enabling timely medical intervention. This project aims to develop a machine learning-based classification system to predict the presence of heart disease in patients based on various medical attributes.

Dataset Description
Dataset Name: Heart Disease UCI Dataset (from Kaggle)
Source: UCI Machine Learning Repository / Kaggle

Dataset Size:
Total Instances: 1,025
Number of Features: 13 (excluding target)
Target Variable: Binary classification (0 = No Disease, 1 = Disease)•	Target Distribution: 
Class 0 (No Disease): 499 instances (48.7%)
Class 1 (Disease): 526 instances (51.3%)

Features Description:

age: Age of the patient (in years)
sex: Gender (1 = male, 0 = female)
cp: Chest pain type (0-3) 
    0: Typical angina
    1: Atypical angina
    2: Non-anginal pain
    3: Asymptomatic
trestbps: Resting blood pressure (mm Hg)
chol: Serum cholesterol (mg/dl)
fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
restecg: Resting electrocardiographic results (0-2)
thalach: Maximum heart rate achieved
exang: Exercise-induced angina (1 = yes, 0 = no)
oldpeak: ST depression induced by exercise relative to rest
slope: Slope of the peak exercise ST segment (0-2)
ca: Number of major vessels colored by fluoroscopy (0-3)
thal: Thalassemia (0-3)
target: Heart disease diagnosis (0 = No disease, 1 = Disease)

Data Preprocessing:
•	No missing values in the dataset
•	Features were standardized using StandardScaler
•	Train-test split: 80% training, 20% testing (stratified)
•	Random state: 42 for reproducibility

Dataset Split:
Training Set: 80% (820 samples)
Test Set: 20% (205 samples)
Stratified split to maintain class distribution


🤖 Models Used
Performance Comparison Table

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.8098	0.9298	0.7619	0.9143	0.8312	0.6309
Decision Tree	0.9854	0.9857	1.0000	0.9714	0.9855	0.9712
K-Nearest Neighbors	0.8634	0.9629	0.8738	0.8571	0.8654	0.7269
Naive Bayes	0.8293	0.9043	0.8070	0.8762	0.8402	0.6602
Random Forest	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000
Gradient Boosting	0.9756	0.9876	0.9717	0.9810	0.9763	0.9512

Model Performance Observations
ML Model Name	Observation about Model Performance
Logistic Regression	Demonstrates solid baseline performance with 81% accuracy. The model shows high recall (91.43%), making it effective at identifying patients with heart disease, though precision is moderate (76.19%). The AUC score of 0.9298 indicates excellent discriminative ability. This model serves as a reliable linear classifier for this problem with good interpretability.
Decision Tree	Achieves exceptional performance with 98.54% accuracy and perfect precision (100%). The high recall (97.14%) and excellent MCC score (0.9712) indicate the model has learned the decision boundaries very effectively. However, there may be a slight risk of overfitting given the perfect precision, which should be monitored on unseen data.
K-Nearest Neighbors	Performs reasonably well with 86.34% accuracy. The balanced precision (87.38%) and recall (85.71%) suggest the model makes consistent predictions across both classes. The AUC of 0.9629 indicates strong ranking ability. This instance-based learner benefits from the standardized feature space and works well for this medium-sized dataset.
Naive Bayes	Shows moderate performance with 82.93% accuracy. While it has decent recall (87.62%), the precision is lower (80.70%), indicating some false positive predictions. The probabilistic approach works reasonably well despite the assumption of feature independence, which may not hold perfectly for medical data. Good computational efficiency makes it suitable for real-time predictions.
Random Forest (Ensemble)Achieves perfect performance across all metrics (100% accuracy, precision, recall, F1, and MCC). This ensemble method effectively combines multiple decision trees to eliminate individual tree weaknesses. The perfect scores suggest excellent generalization on the test set, though cross-validation should be performed to ensure robustness. This is the best-performing model for this dataset.
Gradient Boosting (Ensemble)Delivers outstanding performance with 97.56% accuracy and near-perfect AUC (0.9876). The high precision (97.17%) and recall (98.10%) demonstrate excellent predictive power with minimal false positives and false negatives. The MCC score of 0.9512 confirms strong overall classification quality. This sequential ensemble method is the second-best performer and provides a robust alternative to Random Forest.

Installation & Setup
Prerequisites
•	Python 3.8 or higher
•	pip package manager

Local Setup
Step 1: Clone the Repository
bashgit clone https://github.com/Kesavan-Upadhyayula/Heart-Disease-Classification.git
cd Heart-Disease-Classification

Step 2: Install Dependencies
bashpip install -r requirements.txt

Step 3: Train Models
bashpython train_models.py

Step 4: Run Streamlit Application
bashstreamlit run app.py

The application will open in your default browser at http://localhost:8501

Streamlit Application Features
The interactive web application includes:
Model Selection Dropdown: Choose from 6 different classification models
Test Data Upload: Upload custom CSV files for predictions
Performance Metrics Display: View Accuracy, AUC, Precision, Recall, F1, and MCC scores
Confusion Matrix Visualization: Interactive heatmap showing prediction performance
Classification Report: Detailed per-class precision, recall, and F1 scores
Model Comparison Dashboard: Side-by-side comparison of all models with visual charts
Dynamic Metrics Calculation: Real-time metric computation on uploaded test data

Using the Web Application
Access the Application: Open the deployed Streamlit app or run locally
Select a Model: Use the sidebar dropdown to choose your preferred classification model
Upload Test Data
Click "Upload Test Data (CSV)" in the sidebar
Ensure your CSV has the same features as the training data
Must include a 'target' column for metric calculation

View Results:
Model performance metrics are displayed in the main panel
Explore the confusion matrix for detailed predictions
Compare all models using the comparison dashboard
I
