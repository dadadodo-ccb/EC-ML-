import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import os
import warnings
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Ignore non-critical warnings
warnings.filterwarnings('ignore')

class GradientBoostingTrainer:
    def __init__(self, file_path, target_column, save_folder):
        """
        Initialize the Gradient Boosting Trainer.
        Parameters:
        file_path: Path to the Excel file
        target_column: Name of the target variable column
        save_folder: Path to the folder for saving results
        """
        self.file_path = file_path
        self.target_column = target_column
        self.save_folder = save_folder
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        # Create output directory if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)

    def load_data(self):
        """Load data from Excel file."""
        print("Loading data...")
        try:
            self.data = pd.read_excel(self.file_path)
            print(f"Data loaded successfully, shape: {self.data.shape}")
        except Exception as e:
            print(f"Failed to load data: {e}")
            raise

    def prepare_data(self):
        """Prepare training and testing data."""
        print("Preparing data...")
        # Separate features and target variable
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Check for missing values
        if X.isnull().any().any():
            print("Missing values found in features, filling with mean")
            X = X.fillna(X.mean())
        if y.isnull().any():
            print("Missing values found in target variable, filling with mean")
            y = y.fillna(y.mean())

        # Split into training and testing sets (9:1 ratio)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1, random_state=76
        )

        # Standardize features
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)

        # Standardize target variable (for training)
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)).ravel()
        self.y_test_scaled = self.scaler_y.transform(self.y_test.values.reshape(-1, 1)).ravel()

        print(f"Training set shape: {self.X_train_scaled.shape}, Test set shape: {self.X_test_scaled.shape}")

    def objective_function(self, params):
        """
        Objective function for Bayesian optimization.
        """
        # Unpack parameters
        n_estimators = params[0]
        learning_rate = params[1]
        max_depth = params[2]
        min_samples_split = params[3]
        min_samples_leaf = params[4]
        subsample = params[5]

        # Create model
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=42
        )

        # Train model
        model.fit(self.X_train_scaled, self.y_train_scaled)

        # Predict
        y_pred_scaled = model.predict(self.X_test_scaled)

        # Inverse transform predictions to original scale
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Calculate negative R² score (since optimizer minimizes the objective)
        score = -r2_score(self.y_test, y_pred)
        return score

    def bayesian_optimization(self):
        """Use Bayesian optimization to find optimal hyperparameters."""
        print("Starting Bayesian optimization...")
        # Define search space
        space = [
            Integer(1000, 1500, name='n_estimators'), # Number of decision trees
            Real(0.005, 0.008, name='learning_rate'), # Learning rate
            Integer(3, 8, name='max_depth'), # Maximum depth
            Integer(2, 10, name='min_samples_split'), # Minimum samples required to split an internal node
            Integer(1, 10, name='min_samples_leaf'), # Minimum samples required at a leaf node
            Real(0.8, 1.0, name='subsample') # Subsampling ratio
        ]

        # Execute optimization
        result = gp_minimize(
            func=self.objective_function,
            dimensions=space,
            n_calls=50, # Number of calls
            random_state=42,
            n_random_starts=10 # Number of random starting points
        )

        # Get best parameters
        self.best_params = {
            'n_estimators': result.x[0],
            'learning_rate': result.x[1],
            'max_depth': result.x[2],
            'min_samples_split': result.x[3],
            'min_samples_leaf': result.x[4],
            'subsample': result.x[5]
        }
        print(f"Best parameters: {self.best_params}")
        return self.best_params

    def train_model(self):
        """Train the final model."""
        print("Training model...")
        self.model = GradientBoostingRegressor(**self.best_params, random_state=42)
        self.model.fit(self.X_train_scaled, self.y_train_scaled)
        print("Model training completed")

    def evaluate_model(self):
        """Evaluate model performance."""
        print("Evaluating model...")
        
        # --- Test Set Prediction & Evaluation ---
        y_pred_scaled = self.model.predict(self.X_test_scaled)
        self.y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        self.r2_test = r2_score(self.y_test, self.y_pred)
        self.mae_test = mean_absolute_error(self.y_test, self.y_pred)
        self.rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        
        # --- Training Set Prediction & Evaluation (Added MAE and RMSE) ---
        y_train_pred_scaled = self.model.predict(self.X_train_scaled)
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        
        self.r2_train = r2_score(self.y_train, y_train_pred)
        self.mae_train = mean_absolute_error(self.y_train, y_train_pred)      # Added
        self.rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))  # Added
        
        # --- Output metrics for both sets separately ---
        print(f"Training Set R²: {self.r2_train:.4f}")
        print(f"Training Set MAE: {self.mae_train:.4f}")
        print(f"Training Set RMSE: {self.rmse_train:.4f}")
        print("-" * 40)
        print(f"Test Set R²: {self.r2_test:.4f}")
        print(f"Test Set MAE: {self.mae_test:.4f}")
        print(f"Test Set RMSE: {self.rmse_test:.4f}")
        
        return self.r2_test, self.mae_test, self.rmse_test

    def save_predictions_and_residuals(self):
        """Save predicted values and residuals to CSV files."""
        print("Saving predictions and residuals...")

        # 1. Process Training Set Data
        y_train_pred_scaled = self.model.predict(self.X_train_scaled)
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        residuals_train = self.y_train.values - y_train_pred

        df_train = pd.DataFrame({
            'actual_values': self.y_train.values,
            'predicted_values': y_train_pred,
            'residuals': residuals_train
        })
        train_file_path = os.path.join(self.save_folder, "RF_train_predictions_residuals.csv")
        df_train.to_csv(train_file_path, index=False)
        print(f"Training set predictions and residuals saved to: {train_file_path}")

        # 2. Process Test Set Data
        residuals_test = self.y_test.values - self.y_pred
        df_test = pd.DataFrame({
            'actual_values': self.y_test.values,
            'predicted_values': self.y_pred,
            'residuals': residuals_test
        })
        test_file_path = os.path.join(self.save_folder, "RF_test_predictions_residuals.csv")
        df_test.to_csv(test_file_path, index=False)
        print(f"Test set predictions and residuals saved to: {test_file_path}")

    def save_model_and_scalers(self):
        """Save the model and scalers."""
        import joblib
        # Save model
        joblib.dump(self.model, os.path.join(self.save_folder, 'gradient_boosting_model.pkl'))
        joblib.dump(self.scaler_X, os.path.join(self.save_folder, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(self.save_folder, 'scaler_y.pkl'))
        print("Model and scalers saved")

    def run(self):
        """Run the complete workflow."""
        # Load data
        self.load_data()
        # Prepare data
        self.prepare_data()
        # Bayesian optimization
        self.bayesian_optimization()
        # Train model
        self.train_model()
        # Evaluate model
        self.evaluate_model()
        # Save predictions and residuals
        self.save_predictions_and_residuals()
        # Save model
        self.save_model_and_scalers()

# Main program
if __name__ == "__main__":
    # Get user input
    file_path = ""
    target_column = "log_I"
    save_folder = ""

    # Create trainer instance and run
    trainer = GradientBoostingTrainer(file_path, target_column, save_folder)
    trainer.run()
    print("Training completed! Results saved to the specified folder.")