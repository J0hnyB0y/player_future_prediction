import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import shap
import re

# Load the dataset
file_path = "D:\\MS project\\ATB project\\Final\\final_table.xlsx"  # Updated with new file path
data = pd.read_excel(file_path)

# Step 1: Data Preparation
# ----------------------------------------------
def convert_currency(value):
    if isinstance(value, str):
        value = re.sub(r'[^\d.]', '', value.split(' ')[1])  # Extract the number part and remove non-numeric characters
        return float(value) if value else np.nan
    return value

# Apply currency conversion to relevant columns
currency_columns = ['Weekly Wages', 'Annual Wages']  # Replace with your actual column names
for col in currency_columns:
    data[col] = data[col].apply(convert_currency)

# Check for non-numeric values and handle them
def convert_to_numeric(value):
    """
    Converts values to numeric if possible, or returns NaN for non-numeric values.
    """
    try:
        return pd.to_numeric(value)
    except ValueError:
        # Return NaN for any non-numeric value
        return np.nan

# Apply the function to all columns that should be numeric
for col in data.columns:
    if data[col].dtype == 'object':  # Check only object type columns
        # Check if '90s' or similar non-numeric values are present
        data[col] = data[col].replace('90s', np.nan)  # Replace '90s' with NaN
        # Convert column to numeric, coercing errors to NaN
        data[col] = data[col].apply(convert_to_numeric)

# Fill all NaN values with 0
data.fillna(0, inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
data['Pos'] = label_encoder.fit_transform(data['Pos'])
data['Squad'] = label_encoder.fit_transform(data['Squad'])

# Update numerical features list to only include columns that exist in the data
numerical_features = ['Age', 'SCA', 'SCA90', 'GCA90', 'Blocks', 'Interceptions', 'clearance', 'pass_completion_%', 'Weekly Wages', 'Annual Wages']
numerical_features = [feature for feature in numerical_features if feature in data.columns]  # Ensure columns exist

# Feature Scaling
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Feature Engineering
if 'Blocks' in data.columns and 'Interceptions' in data.columns and 'clearance' in data.columns:
    data['defensive_score'] = data['Blocks'] + data['Interceptions'] + data['clearance']
if 'SCA90' in data.columns and 'GCA90' in data.columns:
    data['offensive_score'] = data['SCA90'] + data['GCA90']

# Step 2: Assign New 'Future' Based on Broader Criteria
# ----------------------------------------------
# Only use available columns
if 'offensive_score' in data.columns and 'defensive_score' in data.columns and 'Weekly Wages' in data.columns and 'Age' in data.columns:
    # Use percentiles to set thresholds
    offensive_threshold = np.percentile(data['offensive_score'], 70)  # Top 30% players
    defensive_threshold = np.percentile(data['defensive_score'], 70)  # Top 30% players
    wage_threshold = np.percentile(data['Weekly Wages'], 40)  # Bottom 40% wages
    age_threshold = np.percentile(data['Age'], 60)  # Top 40% younger players

    # Assign 'look for better option' if a player is above thresholds for performance and below for wages
    data['future'] = np.where(
        ((data['offensive_score'] > offensive_threshold) | 
         (data['defensive_score'] > defensive_threshold)) & 
        (data['Weekly Wages'] < wage_threshold) & 
        (data['Age'] < age_threshold), 
        'look for better option', 
        'stay in the club'
    )
elif 'Age' in data.columns and 'Weekly Wages' in data.columns:
    print("Relevant columns not found or insufficient data, using broader alternative criteria.")
    
    # Broader criteria using median values and other conditions
    median_age = data['Age'].median() if 'Age' in data.columns else None
    median_wages = data['Weekly Wages'].median() if 'Weekly Wages' in data.columns else None

    # Assign 'look for better option' based on available features
    data['future'] = np.where(
        ((data['Age'] < median_age) & (data['Weekly Wages'] < median_wages)) |
        ('offensive_score' in data.columns and (data['offensive_score'] > data['offensive_score'].median())), 
        'look for better option', 
        'stay in the club'
    )
else:
    print("Insufficient data to create 'future' predictions.")

# Ensure the 'future' column has both categories
if 'future' in data.columns and data['future'].nunique() <= 1:
    raise ValueError("Target variable has only one class after assigning criteria. Please revise the 'future' assignment criteria to create a balanced dataset.")

# Check new target distribution
print("New target variable distribution after revised criteria:")
if 'future' in data.columns:
    print(data['future'].value_counts())

# Encode target variable after reassignment
if 'future' in data.columns:
    y = data['future'].apply(lambda x: 1 if x == 'look for better option' else 0)  # Encode target variable


# Step 3: Check Target Distribution
# ----------------------------------------------
# Check for the distribution of target variable
print("Target variable distribution (before):")
print(data['future'].value_counts())

# Check if there is more than one unique value in the target variable
unique_values = data['future'].nunique()
if unique_values <= 1:
    print("Warning: The target variable has only one class. Consider revising the criteria for generating 'future'.")

# Step 4: Visualizations (Histograms and Heatmap)
# ----------------------------------------------
# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Age Distribution of Players')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Weekly Wages Distribution
if 'Weekly Wages' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Weekly Wages'], bins=20, kde=True)
    plt.title('Weekly Wages Distribution of Players')
    plt.xlabel('Weekly Wages')
    plt.ylabel('Frequency')
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Player Features')
plt.show()

# Step 5: Feature Selection using Mutual Information
# ----------------------------------------------
# Check if 'Player' and 'future' columns exist before dropping them
columns_to_drop = [col for col in ['Player', 'future'] if col in data.columns]

# Selecting top features using mutual information
X = data.drop(columns=columns_to_drop)  # Drop columns only if they exist

print("\nTarget variable distribution (after encoding):")
print(pd.Series(y).value_counts())  # Check distribution again

if y.nunique() <= 1:
    raise ValueError("Target variable has only one class after encoding. Please revise the 'future' assignment criteria.")

best_features = SelectKBest(score_func=mutual_info_classif, k='all')  # Use mutual information for feature selection
fit = best_features.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# Concat two dataframes for better visualization
feature_scores = pd.concat([dfcolumns, dfscores], axis=1)
feature_scores.columns = ['Specs', 'Score']  # Naming the dataframe columns
print(feature_scores.nlargest(10, 'Score'))  # Print 10 best features

# Step 6: Handling Missing Values and Model Building
# ----------------------------------------------
# Impute missing values for numeric features using the mean
numeric_features = X.select_dtypes(include=[np.number]).columns
numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

# Impute missing values for categorical features using the most frequent value
categorical_features = X.select_dtypes(exclude=[np.number]).columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

# Check if there are any remaining NaN values in the dataset after imputation
if X.isna().sum().sum() > 0:
    raise ValueError("There are still NaN values in the dataset after imputation. Please check the data preprocessing steps.")

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: XGBoost Classifier
model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_xgb.fit(X_train, y_train)

# Model 2: RandomForest Classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Custom Model: Threshold-Based Classifier
class CustomThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        self.mean_ = X.mean(axis=0)  # Calculate mean of each feature
        self.thresholds_ = X.mean(axis=0) * self.threshold  # Define thresholds based on the mean
        return self

    def predict(self, X):
        # Simple prediction logic: If feature value is above a threshold, classify as 1, else 0
        predictions = np.where(X > self.thresholds_, 1, 0)
        # Aggregate by sum and threshold to decide class
        return (predictions.sum(axis=1) > (X.shape[1] / 2)).astype(int)

# Initialize and train the custom model
custom_model = CustomThresholdClassifier(threshold=0.5)
custom_model.fit(X_train, y_train)

# Evaluate Models
def evaluate_model(model, X_test, y_test, model_name="Model"):
    print(f"\nEvaluation for {model_name}:")
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# Evaluate XGBoost Model
evaluate_model(model_xgb, X_test, y_test, model_name="XGBoost Classifier")

# Evaluate RandomForest Model
evaluate_model(model_rf, X_test, y_test, model_name="RandomForest Classifier")

# Evaluate Custom Model
evaluate_model(custom_model, X_test, y_test, model_name="Custom Threshold Classifier")

# Step 7: Model Interpretation using SHAP (for XGBoost only)
# ----------------------------------------------
explainer = shap.Explainer(model_xgb)
shap_values = explainer(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# SHAP dependence plot for the top feature
top_feature = X.columns[np.argmax(np.abs(shap_values.values).mean(0))]
shap.dependence_plot(top_feature, shap_values.values, X_test)

# Generate Feature Importance Heatmap
# ----------------------------------------------
# Get feature importance from XGBoost model
feature_importances = model_xgb.feature_importances_
features = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(importance_df.set_index('Feature').T, annot=True, cmap='YlGnBu', cbar=False)
plt.title('Feature Importance Heatmap - XGBoost Classifier')
plt.show()

# Step 8: Adding Predictions to the Data
# ----------------------------------------------
# Predict future for the entire dataset using each model
data['future_xgb'] = model_xgb.predict(X)
data['future_rf'] = model_rf.predict(X)
data['future_custom'] = custom_model.predict(X)

# Map the predictions back to "stay in the club" or "look for better options"
data['future_xgb'] = data['future_xgb'].apply(lambda x: 'look for better option' if x == 1 else 'stay in the club')
data['future_rf'] = data['future_rf'].apply(lambda x: 'look for better option' if x == 1 else 'stay in the club')
data['future_custom'] = data['future_custom'].apply(lambda x: 'look for better option' if x == 1 else 'stay in the club')

# Step 9: Save the Updated Data to a New Excel File
# ----------------------------------------------
# Ensure the 'Player' column is included in the output and placed first
if 'Player' not in data.columns:
    raise ValueError("The 'Player' column is not available in the dataset.")

# Reorder columns to have 'Player' as the first column
output_columns = ['Player'] + [col for col in data.columns if col != 'Player']

# Save to Excel
output_path = r"C:\Users\allen\Downloads\player_future_predictions_with_names.xlsx"  # Change path to your desired location
data[output_columns].to_excel(output_path, index=False)

print(f"Predictions with player names have been saved to {output_path}")