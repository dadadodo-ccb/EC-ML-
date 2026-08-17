# -*- coding: utf-8 -*-
"""
SVR Regression Model Training Script (Kaggle Version, with Train/Test Split)
Includes: Z-score Standardization, Bayesian Optimization Tuning, Evaluation Metrics, Prediction Output
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from bayes_opt import BayesianOptimization

# ==================== 0. Global Settings & Warning Suppression ====================
# Suppress non-critical warnings (e.g., convergence, deprecation) to keep output clean
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ==================== 1. User Configuration Area (Modify as needed) ====================
# [Required] Path to Excel file (usually /kaggle/input/xxx/xxx.xlsx on Kaggle)
FILE_PATH = ''

# [Required] Column name of the target variable (output)
TARGET_COL = 'log_I'

# [Required] Folder path for saving images and models
SAVE_DIR = ''

# Number of iterations for Bayesian Optimization (30-50 recommended; adjust based on Kaggle resources)
N_ITER = 100

# Test set ratio
TEST_SIZE = 0.1

# ==================== 2. Create Output Directory ====================
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"[INFO] Results save path: {SAVE_DIR}")

# ==================== 3. Data Loading & Preprocessing ====================
print("[INFO] Reading Excel data...")
try:
    df = pd.read_excel(FILE_PATH, engine='openpyxl')
except Exception as e:
    raise ValueError(f"Failed to read Excel, please check path and format: {e}")

# Check if target column exists
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in data. Available columns: {list(df.columns)}")

# Separate features and target variable
y = df[TARGET_COL].values.reshape(-1, 1)
X = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).values

print(f"[INFO] Data shape: X={X.shape}, y={y.shape}")
print(f"[INFO] Number of features: {X.shape[1]}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=64
)
print(f"[INFO] Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Z-score Standardization (fit on training set, transform both)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train).ravel()

X_test_scaled = scaler_X.transform(X_test) # Use mean/std from training set
y_test_scaled = scaler_y.transform(y_test).ravel()

print("[INFO] Z-score standardization completed")

# ==================== 4. Bayesian Optimization for Optimal SVR Parameters ====================
# Uses Gaussian Process (GP) as surrogate model and EI as acquisition function
# RBF kernel is Gaussian kernel; gamma controls kernel width, C is regularization, epsilon is insensitive loss band width

def svr_eval(C, gamma, epsilon):
    """
    Objective function for Bayesian Optimization
    Trains on training set, evaluates on test set
    Returns R² score on test set (higher is better)
    """
    model = SVR(
        kernel='rbf',       # Gaussian Kernel (RBF)
        C=C,                # Regularization parameter
        gamma=gamma,        # Kernel coefficient
        epsilon=epsilon,    # Epsilon-tube width
        cache_size=1024     # Increase cache to avoid memory warnings
    )
    
    model.fit(X_train_scaled, y_train_scaled)
    y_pred_scaled = model.predict(X_test_scaled)
    score = r2_score(y_test_scaled, y_pred_scaled)
    return score

# Define search space (logarithmic scale is often better for SVR parameters)
pbounds = {
    'C': (1e-4, 1),          # Regularization parameter range
    'gamma': (1e-4, 1),      # Gaussian kernel gamma range
    'epsilon': (1e-4, 1)     # Epsilon range
}

print("\n[INFO] Starting Bayesian Optimization...")
optimizer = BayesianOptimization(
    f=svr_eval,
    pbounds=pbounds,
    random_state=42,
    verbose=0  # Disable redundant output
)

optimizer.maximize(
    init_points=10,   # Number of initial random exploration points
    n_iter=N_ITER,    # Number of Bayesian optimization iterations
)

# Get best parameters
best_params = optimizer.max['params']
best_score = optimizer.max['target']
print(f"\n[INFO] Bayesian Optimization Completed!")
print(f"[INFO] Best Test Set R² (Scaled) = {best_score:.4f}")
print(f"[INFO] Best Parameters: C={best_params['C']:.4f}, "
      f"gamma={best_params['gamma']:.6f}, epsilon={best_params['epsilon']:.6f}")

# ==================== 5. Train Final Model with Best Parameters ====================
print("\n[INFO] Training final SVR model with best parameters...")
final_model = SVR(
    kernel='rbf',
    C=best_params['C'],
    gamma=best_params['gamma'],
    epsilon=best_params['epsilon'],
    cache_size=1024
)
final_model.fit(X_train_scaled, y_train_scaled)

# Predictions (in scaled space)
y_train_pred_scaled = final_model.predict(X_train_scaled)
y_test_pred_scaled = final_model.predict(X_test_scaled)

# Inverse transform to original scale
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
y_train_actual = y_train.ravel()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
y_test_actual = y_test.ravel()

# ==================== 6. Calculate Evaluation Metrics ====================
# Training set metrics
train_r2 = r2_score(y_train_actual, y_train_pred)
train_mae = mean_absolute_error(y_train_actual, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))

# Test set metrics
test_r2 = r2_score(y_test_actual, y_test_pred)
test_mae = mean_absolute_error(y_test_actual, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))

print("\n" + "=" * 50)
print("          Model Evaluation Results (Original Scale)")
print("=" * 50)
print("--- Training Set ---")
print(f"  R²  = {train_r2:.4f}")
print(f"  MAE = {train_mae:.4f}")
print(f"  RMSE= {train_rmse:.4f}")
print("--- Test Set ---")
print(f"  R²  = {test_r2:.4f}")
print(f"  MAE = {test_mae:.4f}")
print(f"  RMSE= {test_rmse:.4f}")
print("=" * 50)

# Save evaluation metrics to CSV
metrics_df = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'R2': [train_r2, test_r2],
    'MAE': [train_mae, test_mae],
    'RMSE': [train_rmse, test_rmse]
})
metrics_path = os.path.join(SAVE_DIR, 'evaluation_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"[INFO] Evaluation metrics saved: {metrics_path}")

# ==================== 7. Save Prediction Results ====================
# Create DataFrame containing predictions for both training and test sets
train_results_df = pd.DataFrame({
    'Dataset': ['Train'] * len(y_train_actual),
    'Actual': y_train_actual,
    'Predicted': y_train_pred,
    'Residual': y_train_actual - y_train_pred
})

test_results_df = pd.DataFrame({
    'Dataset': ['Test'] * len(y_test_actual),
    'Actual': y_test_actual,
    'Predicted': y_test_pred,
    'Residual': y_test_actual - y_test_pred
})

all_results_df = pd.concat([train_results_df, test_results_df], ignore_index=True)
results_path = os.path.join(SAVE_DIR, 'SVR_predictions_and_residuals.csv')
all_results_df.to_csv(results_path, index=False)
print(f"[INFO] Predictions, actual values, and residuals saved: {results_path}")

# ==================== 8. Save Model ====================
import joblib

model_path = os.path.join(SAVE_DIR, 'svr_model.pkl')
joblib.dump(final_model, model_path)
print(f"[INFO] SVR model saved: {model_path}")

# Also save scalers (needed for predicting new data)
scaler_X_path = os.path.join(SAVE_DIR, 'scaler_X.pkl')
scaler_y_path = os.path.join(SAVE_DIR, 'scaler_y.pkl')
joblib.dump(scaler_X, scaler_X_path)
joblib.dump(scaler_y, scaler_y_path)
print(f"[INFO] Scalers saved: {scaler_X_path}, {scaler_y_path}")

print("\n✅ All processes completed! Please check the output folder for results.")