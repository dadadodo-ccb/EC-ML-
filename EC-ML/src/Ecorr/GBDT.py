# =============================================================================
# Gradient Boosting Decision Tree (GBDT) + Bayesian Optimization Training Script
# Platform: Kaggle | Data Format: Excel | Includes Standardization & Evaluation
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ========================= 🔇 Suppress Non-critical Warnings =========================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_estimators.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ========================= 👇 User Configuration Area 👇 =========================
EXCEL_FILE_PATH = ""   # ① Path to Excel file
TARGET_COLUMN_NAME = "Ecorr (VSCE)"                          # ② Target column name
OUTPUT_FOLDER = ""             # ③ Output folder for models and results
RANDOM_STATE = 89                                          # ④ Fixed random seed for reproducibility
N_OPTUNA_TRIALS = 100                                       # Number of trials for Bayesian optimization
TEST_SIZE = 0.1                                            # Test set ratio (0.1 = 10%)
# ========================= 👆 End of User Configuration 👆 =========================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_data(excel_path, target_col):
    """Load Excel data and remove meaningless columns."""
    print(f"[INFO] Loading file: {excel_path}")
    df = pd.read_excel(excel_path)
    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found! Available columns: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Remove empty or constant columns
    drop_cols = [c for c in X.columns if X[c].isnull().all() or X[c].nunique() <= 1]
    if drop_cols:
        print(f"[WARN] Removed {len(drop_cols)} meaningless columns: {drop_cols}")
        X = X.drop(columns=drop_cols)
    return X, y


def standardize(X_train, X_test):
    """Fit scaler on training set and transform both sets to prevent data leakage."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def bayesian_optimization(X_train, y_train, n_trials, seed):
    """Bayesian optimization to find optimal GBDT hyperparameters."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.004, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": seed,
        }
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        return r2_score(y_train, model.predict(X_train))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_params["random_state"] = seed
    return best_params, study.best_value


def evaluate_model(y_true, y_pred, dataset_name):
    """Calculate and print R², MAE, and RMSE."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"   📊 {dataset_name} | R²: {r2:.6f} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
    return {"R2": r2, "MAE": mae, "RMSE": rmse}


def export_predictions(y_train, y_test, pred_train, pred_test, output_folder):
    """Export actual values, predicted values, and residuals to DGBT.xlsx."""
    df_result = pd.DataFrame({
        "Dataset": ["Train"] * len(y_train) + ["Test"] * len(y_test),
        "Actual_Value": np.concatenate([y_train.values, y_test.values]),
        "Predicted_Value": np.concatenate([pred_train, pred_test]),
        "Residual": np.concatenate([y_train.values - pred_train, y_test.values - pred_test]),
    })
    save_path = os.path.join(output_folder, "DGBT.xlsx")
    df_result.to_excel(save_path, index=False)
    print(f"[EXPORT] Predictions saved to: {save_path}")


# =============================================================================
#                         🚀 Main Execution Flow
# =============================================================================
if __name__ == "__main__":
    # Step 1: Load raw data
    X_raw, y_raw = load_data(EXCEL_FILE_PATH, TARGET_COLUMN_NAME)

  

    # Step 2: Split data with fixed seed
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Step 3: Standardize features
    X_train, X_test, scaler = standardize(X_train_raw, X_test_raw)

    # Step 4: Bayesian Optimization
    print(f"   [OPTUNA] Starting Bayesian Optimization ({N_OPTUNA_TRIALS} trials)...")
    best_params, best_train_r2 = bayesian_optimization(
        X_train, y_train, N_OPTUNA_TRIALS, RANDOM_STATE
    )
    print(f"   [OPTUNA] Best Training R²: {best_train_r2:.4f}")

    # Step 5: Train final model with optimal parameters
    model = GradientBoostingRegressor(**best_params)
    model.fit(X_train, y_train)
    
    # Step 6: Prediction and Evaluation
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    print(f"\n[FINAL] Final Model Evaluation:")
    train_metrics = evaluate_model(y_train, pred_train, "Training Set")
    test_metrics = evaluate_model(y_test, pred_test, "Test Set")

    # Step 7: Export results and save model
    export_predictions(y_train, y_test, pred_train, pred_test, OUTPUT_FOLDER)

    model_path = os.path.join(OUTPUT_FOLDER, "gbdt_best_model.pkl")
    scaler_path = os.path.join(OUTPUT_FOLDER, "standard_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[EXPORT] Model saved to: {model_path}")
    print(f"[EXPORT] Scaler saved to: {scaler_path}")

    print(f"\n✅ Process Completed! Seed={RANDOM_STATE}, Test R²={test_metrics['R2']:.6f}") 