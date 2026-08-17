# ============================================================
# XGBoost Complete Training Code (Fixed Seed + Bayesian Optimization)
# Suitable for Kaggle environment, input Excel file
# ============================================================

# ---------- 1. Install necessary libraries (uncomment for first run on Kaggle) ----------
# !pip install optuna openpyxl xgboost scikit-learn -q

import os
import warnings
import logging

# ---------- 2. Suppress meaningless warnings and non-fatal errors ----------
warnings.filterwarnings("ignore")
logging.getLogger("optuna").setLevel(logging.ERROR)       # Keep only ERROR level
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import joblib
import optuna
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# Set optuna to silent (placed after import)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# ★★★ User Configuration Area — Please modify your parameters here ★★★
# ============================================================

# [File Path] Path to your Excel file (single file)
FILE_PATH = ""

# [Output Folder Path] All results and models are saved to this folder
OUTPUT_DIR = ""

# [Target Variable Column Name] The name of your output variable (dependent variable)
TARGET_COL = "log_I"

# [Fixed Random Seed] Used for data splitting and model initialization to ensure reproducibility
RANDOM_STATE = 76  

# [Number of Bayesian Optimization Trials] Number of hyperparameter search iterations
N_BAYESIAN_TRIALS = 100

# [Data Split Ratio] Test set proportion 0.1 means 9:1 split
TEST_SIZE = 0.1

# ============================================================
# Main code body below, generally no modification needed
# ============================================================


def create_output_dir(output_dir):
    """Create output folder if it does not exist."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output folder ready: {output_dir}")


def load_data(file_path, target_col):
    """
    Load Excel file, separate features and labels.
    Automatically remove meaningless columns (single value/empty/non-numeric).
    """
    print(f"📂 Loading file: {file_path}")
    df = pd.read_excel(file_path)
    print(f"   Original data shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(
            f"❌ Target column '{target_col}' not found in data!\n"
            f"   Available columns: {list(df.columns)}"
        )

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Remove single-value or empty columns
    nunique = X.nunique()
    drop_single = nunique[nunique <= 1].index.tolist()
    if drop_single:
        print(f"   ⚠️ Removed {len(drop_single)} single-value/empty columns: {drop_single}")
        X = X.drop(columns=drop_single)

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    dropped_non_num = [c for c in X.columns if c not in numeric_cols]
    if dropped_non_num:
        print(f"   ⚠️ Removed {len(dropped_non_num)} non-numeric columns: {dropped_non_num}")
    X = X[numeric_cols]

    print(f"   Final number of features: {X.shape[1]}\n")
    return X, y


def split_and_scale(X, y, test_size, random_seed):
    """
    Split dataset (9:1) first, then fit StandardScaler on training set,
    transform training and test sets separately to prevent data leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def make_objective(X_train, y_train, X_test, y_test):
    """
    Return the objective function closure for Bayesian optimization.
    Sample hyperparameters → Train XGB → Return test set R² (maximize).
    """
    def objective(trial):
        param = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 1000),
            "max_depth":         trial.suggest_int("max_depth", 2, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state":      42,
            "tree_method":       "hist",
            "device":            "cpu",   # Recommended 'cpu' for compatibility; change to 'cuda' if GPU available
        }

        model = xgb.XGBRegressor(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        return r2_score(y_test, y_pred)

    return objective


def bayesian_search(X_train, y_train, X_test, y_test, n_trials):
    """
    Bayesian optimization to search for optimal hyperparameters, returns (best_params, best_r2).
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    obj_fn = make_objective(X_train, y_train, X_test, y_test)
    study.optimize(obj_fn, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


def train_final_model(X_train, y_train, best_params):
    """Train final model with optimal parameters (no eval_set passed to avoid redundant calculation)."""
    final_params = {
        **best_params,
        "random_state": 42,
        "tree_method":  "hist",
        "device":       "cpu",  # Recommended 'cpu' for compatibility
    }
    model = xgb.XGBRegressor(**final_params)
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate(model, X, y):
    """Calculate R², MAE, RMSE."""
    y_pred = model.predict(X)
    return {
        "R²":   r2_score(y, y_pred),
        "MAE":  mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
    }


def run_training(X, y, test_size, n_bayesian_trials, random_state):
    """
    ★ Single complete training flow ★
    Split → Scale → Bayesian Optimization → Train → Evaluate
    Returns result dictionary.
    """

    
    # 1) Split + Scale
    X_tr, X_te, y_tr, y_te, scaler = split_and_scale(X, y, test_size, random_state)

    # 2) Bayesian Optimization
    best_params, best_val_r2 = bayesian_search(
        X_tr, y_tr, X_te, y_te, n_bayesian_trials
    )

    # 3) Train final model
    model = train_final_model(X_tr, y_tr, best_params)

    # 4) Evaluate
    train_metrics = evaluate(model, X_tr, y_tr)
    test_metrics  = evaluate(model, X_te, y_te)

    # 5) Predictions & Residuals
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)

    print(f"   ✅ Training complete | Train R²={train_metrics['R²']:.4f} | Test R²={test_metrics['R²']:.4f}")

    return {
        "best_params":   best_params,
        "best_val_r2":   best_val_r2,
        "train_metrics": train_metrics,
        "test_metrics":  test_metrics,
        "scaler":        scaler,
        "model":         model,
        "y_tr":          y_tr,
        "y_te":          y_te,
        "y_tr_pred":     y_tr_pred,
        "y_te_pred":     y_te_pred,
    }


def save_results(result, output_dir):
    """
    Save three types of files:
      1. model_metrics.xlsx        — Summary of train/test metrics for current model
      2. predictions_Train.xlsx    — Training set prediction details
      3. predictions_Test.xlsx     — Test set prediction details
      4. xgboost_model.joblib      — Model
      5. scaler.joblib             — Scaler
    """
    print(f"\n🏆 Final Test Set R²: {result['test_metrics']['R²']:.4f}")

    # ---- 1. Summary Metrics Excel ----
    rows = []
    for dataset, m in [("Train", result["train_metrics"]), ("Test", result["test_metrics"])]:
        rows.append({
            "Dataset": dataset,
            "R²":   m["R²"],
            "MAE":  m["MAE"],
            "RMSE": m["RMSE"],
        })
    df_metrics = pd.DataFrame(rows)
    metrics_path = os.path.join(output_dir, "model_metrics.xlsx")
    df_metrics.to_excel(metrics_path, index=False, engine="openpyxl")
    print(f"   📁 Metrics summary saved: {metrics_path}")

    # ---- 2. Prediction Details ----
    for name, y_true, y_pred in [
        ("Train", result["y_tr"], result["y_tr_pred"]),
        ("Test", result["y_te"], result["y_te_pred"]),
    ]:
        df_pred = pd.DataFrame({
            "Actual": y_true.values if hasattr(y_true, "values") else y_true,
            "Predicted": y_pred,
            "Residual":   (y_true.values if hasattr(y_true, "values") else y_true) - y_pred,
        })
        fpath = os.path.join(output_dir, f"predictions_{name}.xlsx")
        df_pred.to_excel(fpath, index=False, engine="openpyxl")
        print(f"   📁 {name} prediction details saved: {fpath}")

    # ---- 3. Save Model & Scaler ----
    joblib.dump(result["model"],  os.path.join(output_dir, "xgboost_model.joblib"))
    joblib.dump(result["scaler"], os.path.join(output_dir, "scaler.joblib"))
    print(f"   📁 Model & Scaler saved")


# ============================================================
# Main Process
# ============================================================

def main():
    print("=" * 60)
    print("  XGBoost Training (Fixed Seed + Bayesian Optimization)")
    print("=" * 60)

    # Step 0: Create output directory
    create_output_dir(OUTPUT_DIR)

    # Step 1: Load data
    X, y = load_data(FILE_PATH, TARGET_COL)

    # Step 2: Execute single training run
    print(f"🚀 Starting training...\n")
    result = run_training(X, y, TEST_SIZE, N_BAYESIAN_TRIALS, RANDOM_STATE)

    # Step 3: Save results
    print("\n" + "=" * 60)
    print("  💾 Saving result files")
    print("=" * 60)
    save_results(result, OUTPUT_DIR)

    # Final Summary
    print("\n" + "=" * 60)
    print("  🎉 All done! Output file summary:")
    print("=" * 60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"   📄 {f}  ({size_kb:.1f} KB)")
    print(f"\n   📂 All files saved in: {OUTPUT_DIR}")

    # Print detailed parameters
    print(f"\n🏆 Optimal Hyperparameters:")
    for k, v in result["best_params"].items():
        print(f"   {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main() 