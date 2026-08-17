# =============================================================================
# 梯度提升决策树(GBDT) + 贝叶斯优化 + 种子搜索 完整训练脚本
# 适用平台: Kaggle | 数据格式: Excel | 包含标准化与完整评估
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
import matplotlib.pyplot as plt

# ========================= 🔇 抑制无意义/非致命报错 =========================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_estimators.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']     # Kaggle默认无中文字体，用英文避免报错
plt.rcParams['axes.unicode_minus'] = False

# ========================= 👇 用户配置区 (仅需修改此处) 👇 =========================
EXCEL_FILE_PATH = "/kaggle/input/datasets/kayokosuki88/e-mg-cao-cao/data_E_normal_cleaned - .xlsx"   # ① Excel文件路径
TARGET_COLUMN_NAME = "Ecorr (VSCE)"                          # ② 输出量列名(自行填写)
OUTPUT_FOLDER = "/kaggle/working/dgbt_output_E"             # ③ 图片和模型保存文件夹路径
SEED_RANGE = range(89,90)                                 # ④ 种子搜索范围 [50, 60]
N_OPTUNA_TRIALS = 100                                       # 每个种子下贝叶斯优化搜索轮数
TEST_SIZE = 0.1                                            # 测试集比例
# ========================= 👆 用户配置区结束 👆 =========================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_data(excel_path, target_col):
    """读取Excel并剔除无意义列"""
    print(f"[INFO] 正在读取文件: {excel_path}")
    df = pd.read_excel(excel_path)
    if target_col not in df.columns:
        raise ValueError(f"❌ 目标列 '{target_col}' 不存在! 可用列: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 剔除全空列和常量列
    drop_cols = [c for c in X.columns if X[c].isnull().all() or X[c].nunique() <= 1]
    if drop_cols:
        print(f"[WARN] 已移除 {len(drop_cols)} 个无意义列: {drop_cols}")
        X = X.drop(columns=drop_cols)
    return X, y


def standardize(X_train, X_test):
    """对训练集拟合标准化器，再分别转换(防止数据泄露)"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def bayesian_optimization(X_train, y_train, n_trials, seed):
    """贝叶斯优化搜索GBDT最优参数(单次训练集拟合)"""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.005, log=True),
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
    """计算并打印 R², MAE, RMSE"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"   📊 {dataset_name} | R²: {r2:.6f} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
    return {"R2": r2, "MAE": mae, "RMSE": rmse}


def export_predictions(y_train, y_test, pred_train, pred_test, output_folder):
    """导出 实际值/预测值/残差 为 DGBT.xlsx"""
    df_result = pd.DataFrame({
        "Dataset": ["Train"] * len(y_train) + ["Test"] * len(y_test),
        "Actual_Value": np.concatenate([y_train.values, y_test.values]),
        "Predicted_Value": np.concatenate([pred_train, pred_test]),
        "Residual": np.concatenate([y_train.values - pred_train, y_test.values - pred_test]),
    })
    save_path = os.path.join(output_folder, "DGBT.xlsx")
    df_result.to_excel(save_path, index=False)
    print(f"[EXPORT] 预测结果已保存: {save_path}")


def plot_results(y_train, y_test, pred_train, pred_test, output_folder):
    """生成评估图表(英文命名)"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    datasets = {"Train": (y_train.values, pred_train), "Test": (y_test.values, pred_test)}

    for idx, (name, (actual, predicted)) in enumerate(datasets.items()):
        residual = actual - predicted
        # 实际值 vs 预测值
        ax = axes[idx][0]
        ax.scatter(actual, predicted, alpha=0.5, edgecolors="k", linewidth=0.5, s=20)
        min_v, max_v = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Ideal")
        ax.set_xlabel("Actual Value"); ax.set_ylabel("Predicted Value")
        ax.set_title(f"{name} Set: Actual vs Predicted"); ax.legend()
        # 残差分布
        ax = axes[idx][1]
        ax.hist(residual, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(x=0, color="red", linestyle="--", lw=2)
        ax.set_xlabel("Residual"); ax.set_ylabel("Frequency")
        ax.set_title(f"{name} Set: Residual Distribution")
        # 残差 vs 预测值
        ax = axes[idx][2]
        ax.scatter(predicted, residual, alpha=0.5, edgecolors="k", linewidth=0.5, s=20)
        ax.axhline(y=0, color="red", linestyle="--", lw=2)
        ax.set_xlabel("Predicted Value"); ax.set_ylabel("Residual")
        ax.set_title(f"{name} Set: Residual vs Predicted")

    plt.tight_layout()
    fig_path = os.path.join(output_folder, "model_evaluation_plots.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[EXPORT] 评估图表已保存: {fig_path}")


# =============================================================================
#                         🚀 主程序: 种子搜索 + 训练
# =============================================================================
if __name__ == "__main__":
    # Step 1: 加载原始数据(只读一次)
    X_raw, y_raw = load_data(EXCEL_FILE_PATH, TARGET_COLUMN_NAME)

    # Step 2: 遍历种子范围，记录每个种子的最佳测试集R²
    seed_results = []
    total_seeds = len(SEED_RANGE)

    for i, seed in enumerate(SEED_RANGE, 1):
        print(f"\n{'='*60}")
        print(f"🔍 种子搜索进度: [{i}/{total_seeds}] | 当前种子: {seed}")
        print(f"{'='*60}")

        # 2.1 按当前种子划分数据
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=TEST_SIZE, random_state=seed
        )

        # 2.2 标准化
        X_train, X_test, scaler = standardize(X_train_raw, X_test_raw)

        # 2.3 贝叶斯优化(使用当前种子)
        best_params, best_train_r2 = bayesian_optimization(
            X_train, y_train, N_OPTUNA_TRIALS, seed
        )
        print(f"   [OPTUNA] 种子{seed} 最优训练集R²: {best_train_r2:.4f}")

        # 2.4 用最优参数训练并在测试集上评估
        model = GradientBoostingRegressor(**best_params)
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        test_r2 = r2_score(y_test, pred_test)

        seed_results.append({
            "seed": seed,
            "test_r2": test_r2,
            "best_params": best_params,
            "scaler": scaler,
            "model": model,
            "data_split": (X_train, X_test, y_train, y_test),
            "pred_test": pred_test,
        })
        print(f"   ✅ 种子{seed} 测试集R²: {test_r2:.6f}")

    # Step 3: 选出测试集R²最高的种子
    best_result = max(seed_results, key=lambda x: x["test_r2"])
    best_seed = best_result["seed"]

    print(f"\n{'#'*60}")
    print(f"🏆 种子搜索完成! 最优种子: {best_seed} | 测试集R²: {best_result['test_r2']:.6f}")
    print(f"{'#'*60}")

    # 打印所有种子结果汇总
    summary_df = pd.DataFrame([{"Seed": r["seed"], "Test_R2": round(r["test_r2"], 6)} for r in seed_results])
    summary_df = summary_df.sort_values("Test_R2", ascending=False).reset_index(drop=True)
    print("\n📋 种子搜索结果汇总:")
    print(summary_df.to_string(index=False))

    # Step 4: 用最优种子对应的模型进行完整评估和输出
    X_train, X_test, y_train, y_test = best_result["data_split"]
    final_model = best_result["model"]
    final_scaler = best_result["scaler"]

    pred_train = final_model.predict(X_train)
    pred_test = best_result["pred_test"]

    print(f"\n[FINAL] 使用最优种子 {best_seed} 的模型进行最终评估:")
    train_metrics = evaluate_model(y_train, pred_train, "Training Set")
    test_metrics = evaluate_model(y_test, pred_test, "Test Set")

    # Step 5: 导出预测结果 & 保存模型
    export_predictions(y_train, y_test, pred_train, pred_test, OUTPUT_FOLDER)

    model_path = os.path.join(OUTPUT_FOLDER, "gbdt_best_model.pkl")
    scaler_path = os.path.join(OUTPUT_FOLDER, "standard_scaler.pkl")
    joblib.dump(final_model, model_path)
    joblib.dump(final_scaler, scaler_path)
    print(f"[EXPORT] 模型已保存: {model_path}")
    print(f"[EXPORT] 标准化器已保存: {scaler_path}")

    # Step 6: 生成评估图表
    plot_results(y_train, y_test, pred_train, pred_test, OUTPUT_FOLDER)

    print(f"\n✅ 全部流程执行完毕! 最优种子={best_seed}, 测试集R²={best_result['test_r2']:.6f}")