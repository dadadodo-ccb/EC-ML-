# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer
import warnings
import os
import joblib

# Ignore meaningless warning messages
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path, target_column):
    """
    Load Excel data and perform preprocessing.
    
    Parameters:
    file_path: Path to the Excel file
    target_column: Name of the target column
    
    Returns:
    X: Feature data
    y: Target data
    """
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Separate features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

def standardize_data(X_train, X_test):
    """
    Standardize training and testing set data.
    
    Parameters:
    X_train: Training set features
    X_test: Test set features
    
    Returns:
    Standardized X_train, X_test, and the scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def bayesian_optimization(X_train, y_train):
    """
    Use Bayesian optimization to find the best hyperparameters.
    
    Parameters:
    X_train: Training set features
    y_train: Training set target values
    
    Returns:
    Best hyperparameters
    """
    def objective(params):
        n_estimators, max_depth, min_samples_split, min_samples_leaf = params
        
        # Create Random Forest model
        rf = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if max_depth > 0 else None,
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        rf.fit(X_train, y_train)
        
        # Calculate validation score (use negative R2 because optimizer minimizes the objective)
        y_pred = rf.predict(X_train)
        score = -r2_score(y_train, y_pred)
        
        return score
    
    # Define search space
    space = [
        Integer(100, 1000, name='n_estimators'),          # Number of decision trees
        Integer(1, 20, name='max_depth'),               # Maximum depth
        Integer(2, 20, name='min_samples_split'),       # Minimum samples required to split an internal node
        Integer(1, 10, name='min_samples_leaf')         # Minimum samples required at a leaf node
    ]
    
    # Run Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=50,  # Number of calls, adjust as needed
        random_state=42,
        n_jobs=-1
    )
    
    return result.x

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance.
    
    Parameters:
    model: Trained model
    X_train, y_train: Training set
    X_test, y_test: Test set
    
    Returns:
    Dictionary containing various metrics
    """
    # Training set prediction
    y_train_pred = model.predict(X_train)
    
    # Test set prediction
    y_test_pred = model.predict(X_test)
    
    # Calculate training set metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Calculate test set metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Calculate residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    return {
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_residuals': train_residuals,
        'test_residuals': test_residuals
    }

def main():
    # User configuration
    file_path = ""
    target_column = "Ecorr (VSCE)"
    model_save_folder = ""
    RANDOM_STATE = 89  # Random seed
    
    # Create save directory
    os.makedirs(model_save_folder, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X, y = load_and_preprocess_data(file_path, target_column)
    
    # Split dataset with fixed seed
    print(f"Splitting data with random state: {RANDOM_STATE}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE
    )
    
    # Standardize data
    X_train_scaled, X_test_scaled, scaler = standardize_data(X_train, X_test)
    
    # Use Bayesian optimization to find best hyperparameters
    print("Starting Bayesian optimization...")
    best_params = bayesian_optimization(X_train_scaled, y_train)
    
    # Train final model using best hyperparameters
    final_model = RandomForestRegressor(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]) if best_params[1] > 0 else None,
        min_samples_split=int(best_params[2]),
        min_samples_leaf=int(best_params[3]),
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    final_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    results = evaluate_model(final_model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Output metrics
    print("\n=== Model Performance Metrics ===")
    print(f"Training Set R²: {results['train_r2']:.4f}")
    print(f"Training Set MAE: {results['train_mae']:.4f}")
    print(f"Training Set RMSE: {results['train_rmse']:.4f}")
    print(f"Test Set R²: {results['test_r2']:.4f}")
    print(f"Test Set MAE: {results['test_mae']:.4f}")
    print(f"Test Set RMSE: {results['test_rmse']:.4f}")
    
    # Create DataFrames containing predicted values, actual values, and residuals
    train_df = pd.DataFrame({
        'Actual': y_train.values,
        'Predicted': results['y_train_pred'],
        'Residual': results['train_residuals']
    })
    
    test_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': results['y_test_pred'],
        'Residual': results['test_residuals']
    })
    
    # Save to Excel file
    with pd.ExcelWriter(os.path.join(model_save_folder, 'RF.xlsx')) as writer:
        train_df.to_excel(writer, sheet_name='Train_Data', index=False)
        test_df.to_excel(writer, sheet_name='Test_Data', index=False)
    
    print(f"\nPredictions, actual values, and residuals saved to {os.path.join(model_save_folder, 'RF.xlsx')}")
    
    # Save model and scaler
    joblib.dump(final_model, os.path.join(model_save_folder, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(model_save_folder, 'scaler.pkl'))
    
    print(f"Model saved to {os.path.join(model_save_folder, 'random_forest_model.pkl')}")
    print(f"Scaler saved to {os.path.join(model_save_folder, 'scaler.pkl')}")
    
    print(f"\nDetailed results with random state {RANDOM_STATE}:")
    print(f"Training Set R²: {results['train_r2']:.4f}")
    print(f"Training Set MAE: {results['train_mae']:.4f}")
    print(f"Training Set RMSE: {results['train_rmse']:.4f}")
    print(f"Test Set R²: {results['test_r2']:.4f}")
    print(f"Test Set MAE: {results['test_mae']:.4f}")
    print(f"Test Set RMSE: {results['test_rmse']:.4f}")

if __name__ == "__main__":
    main()