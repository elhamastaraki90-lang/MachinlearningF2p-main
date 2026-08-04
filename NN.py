# ==========================================================
#  FULL PROFESSIONAL NEURAL NETWORK REGRESSION PIPELINE (MLP)
#  Replaced Gaussian Process with an ensemble of MLPRegressors
#  to provide predictive mean and uncertainty (std).
#  All other outputs, metrics, and plots remain unchanged.
# ==========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv("F2BCMS.csv")

df["logQ2"] = np.log10(df["Q^2"])
df["logx"] = np.log10(df["x"])

X = df[["logx", "logQ2"]]
y = df["F2_exp"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_all_s = scaler.transform(X)

# ==========================================================
# 2. ADVANCED MLP ENSEMBLE MODEL (for mean & std)
# ==========================================================

class MLPEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble of MLPRegressors to provide predictive mean and
    estimate of uncertainty (standard deviation across ensemble).
    """
    def __init__(self, n_estimators=10, hidden_layer_sizes=(120, 120, 50),
                 activation='relu', max_iter=1000, early_stopping=True,
                 validation_fraction=0.1):
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.models = []

    def fit(self, X, y):
        self.models = []
        for i in range(self.n_estimators):
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                validation_fraction=self.validation_fraction,
                random_state=i
            )
            mlp.fit(X, y)
            self.models.append(mlp)
        return self

    def predict(self, X, return_std=False):
        preds = np.array([model.predict(X) for model in self.models])
        mean = np.mean(preds, axis=0)
        if return_std:
            std = np.std(preds, axis=0)
            return mean, std
        return mean

gpr = MLPEnsemble(
    n_estimators=10,
    hidden_layer_sizes=(100, 100, 50),
    activation='relu',
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1
)

print("Training MLP Ensemble...")
gpr.fit(X_train_s, y_train)

# Predictions
y_train_pred, y_train_std = gpr.predict(X_train_s, return_std=True)
y_test_pred, y_test_std = gpr.predict(X_test_s, return_std=True)
y_all_pred, y_all_std = gpr.predict(X_all_s, return_std=True)

# ==========================================================
# 3. REGRESSION METRICS (unchanged)
# ==========================================================

def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE(%)": mean_absolute_percentage_error(y_true, y_pred) * 100
    }

train_metrics = regression_metrics(y_train, y_train_pred)
test_metrics = regression_metrics(y_test, y_test_pred)

# ==========================================================
# 4. CLASSIFICATION METRICS
# ==========================================================

median_val = np.median(y)
y_test_bin = (y_test > median_val).astype(int)
y_pred_bin = (y_test_pred > median_val).astype(int)

binary_metrics = {
    "Accuracy": accuracy_score(y_test_bin, y_pred_bin),
    "Precision": precision_score(y_test_bin, y_pred_bin),
    "Recall": recall_score(y_test_bin, y_pred_bin),
    "F1": f1_score(y_test_bin, y_pred_bin)
}

num_bins = 5
y_test_multi = pd.qcut(y_test, num_bins, labels=False)
y_pred_multi = pd.qcut(y_test_pred, num_bins, labels=False)
multi_cm = confusion_matrix(y_test_multi, y_pred_multi)

# ==========================================================
# 5. CROSS VALIDATION
# ==========================================================

cv_scores = cross_val_score(gpr, X_all_s, y, cv=5, scoring="r2")

# ==========================================================
# 6. SAVE PREDICTIONS (CSV)
# ==========================================================

results = df.copy()
results["F2_pred"] = y_all_pred
results["uncertainty_std"] = y_all_std
results["error"] = y - y_all_pred

results.to_csv("GPR_predictions_full.csv", index=False)

# ==========================================================
# 7. SAVE SUMMARY (EXCEL)
# ==========================================================

summary = pd.DataFrame({
    "Metric": list(train_metrics.keys()),
    "Train": list(train_metrics.values()),
    "Test": list(test_metrics.values())
})
summary.to_excel("GPR_metrics_summary.xlsx", index=False)

# ==========================================================
# 8. FULL REPORT (PDF)
# ==========================================================
#  FULL 4-PLOT EVALUATION (Train/Test + Residuals)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Test Actual vs Predicted
axes[0,0].scatter(y_test, y_test_pred, alpha=0.7, color="blue")
axes[0,0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--', linewidth=2
)
axes[0,0].set_title("Test Set: Actual vs Predicted (MLP)")
axes[0,0].set_xlabel("Actual F2")
axes[0,0].set_ylabel("Predicted F2")
axes[0,0].grid(True)

# 2) Train Actual vs Predicted
axes[0,1].scatter(y_train, y_train_pred, alpha=0.7, color="green")
axes[0,1].plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    'k--', linewidth=2
)
axes[0,1].set_title("Train Set: Actual vs Predicted (MLP)")
axes[0,1].set_xlabel("Actual F2")
axes[0,1].set_ylabel("Predicted F2")
axes[0,1].grid(True)

# 3) Test Residual Plot
axes[1,0].scatter(y_test_pred, y_test - y_test_pred, alpha=0.7, color="red")
axes[1,0].axhline(0, color='black', linestyle='--', linewidth=2)
axes[1,0].set_title("Residuals Plot (Test Set)(MLP)")
axes[1,0].set_xlabel("Predicted F2")
axes[1,0].set_ylabel("Residual (Actual - Predicted)")
axes[1,0].grid(True)

    # 4) Distribution of Residuals
axes[1,1].hist(y_test - y_test_pred, bins=25, color='steelblue', edgecolor="black", alpha=0.8)
axes[1,1].set_title("Distribution of Residuals (Test Set)(MLP)")
axes[1,1].set_xlabel("Residuals")
axes[1,1].set_ylabel("Count")
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig("MLP_4_Evaluation_Plots.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ==========================================================
# 4-Panel Error Analysis Plot (Test Set)
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error vs x (test set)
test_indices = y_test.index
x_test = df.loc[test_indices, 'x'].values
Q2_test = df.loc[test_indices, 'Q^2'].values
errors = np.abs(y_test.values - y_test_pred)

# Absolute error vs x
axes[0, 0].scatter(x_test, errors, alpha=0.6)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xlabel('x (log scale)')
axes[0, 0].set_ylabel('Absolute Error')
axes[0, 0].set_title('Absolute Error vs x (Test Set) (MLP)')
axes[0, 0].grid(True, alpha=0.3)

# Absolute error vs Q²
axes[0, 1].scatter(Q2_test, errors, alpha=0.6)
axes[0, 1].set_xscale('log')
axes[0, 1].set_xlabel('Q² (log scale)')
axes[0, 1].set_ylabel('Absolute Error')
axes[0, 1].set_title('Absolute Error vs Q² (Test Set) (MLP)')
axes[0, 1].grid(True, alpha=0.3)

# Relative error vs x
relative_errors = errors / (y_test.values + 1e-10)
axes[1, 0].scatter(x_test, relative_errors * 100, alpha=0.6)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel('x (log scale)')
axes[1, 0].set_ylabel('Relative Error (%)')
axes[1, 0].set_title('Relative Error vs x (Test Set) (MLP)')
axes[1, 0].grid(True, alpha=0.3)

# Distribution of errors
axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Absolute Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Absolute Errors (Test Set) (MLP)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("0.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# --------------------------
# Actual vs Predicted
# --------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted (MLP)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True, alpha=0.3)
plt.savefig('1.pdf', format='pdf', bbox_inches='tight')
plt.show();
plt.close();

# --------------------------
# Residual Plot
# --------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.6)
plt.axhline(0, linestyle="--", color="red")
plt.title("Residual Plot (MLP)")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.grid(True, alpha=0.3)
plt.savefig('2.pdf', format='pdf', bbox_inches='tight')
plt.show();
plt.close();

# --------------------------
# Error Distribution
# --------------------------
plt.figure(figsize=(8,6))
plt.hist(y_test - y_test_pred, bins=30, edgecolor="black")
plt.title("Error Distribution (MLP)")
plt.xlabel("Error")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.savefig('3.pdf', format='pdf', bbox_inches='tight')
plt.show();
plt.close();

# --------------------------
# First 20 Points
# --------------------------
idx20 = df.index[:20]
X20 = scaler.transform(X.loc[idx20][["logx", "logQ2"]])
y20_pred, y20_std = gpr.predict(X20, return_std=True)

plt.figure(figsize=(10,6))
plt.scatter(df.loc[idx20,"x"], df.loc[idx20,"F2_exp"],
            label="Actual", s=100)
plt.scatter(df.loc[idx20,"x"], y20_pred,
            label="Predicted", s=100, marker="s")

for i in range(20):
    plt.plot([df.loc[idx20,"x"].iloc[i]]*2,
             [df.loc[idx20,"F2_exp"].iloc[i], y20_pred[i]],
             "k--", alpha=0.4)

plt.xscale("log")
plt.xlabel("x")
plt.ylabel("F2")
plt.title("First 20: Actual vs Predicted (MLP)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('4.pdf', format='pdf', bbox_inches='tight')
plt.show();
plt.close();

# --------------------------
# All Data with 95% CI
# --------------------------
sorted_idx = np.argsort(df["x"])
x_sorted = df["x"].values[sorted_idx]
y_sorted = y_all_pred[sorted_idx]
std_sorted = y_all_std[sorted_idx]

plt.figure(figsize=(10,6))
plt.scatter(df["x"], y, alpha=0.3, label="Actual")
plt.plot(x_sorted, y_sorted, color="red", label="Predicted")
plt.fill_between(
    x_sorted,
    y_sorted - 1.96 * std_sorted,
    y_sorted + 1.96 * std_sorted,
    color="red", alpha=0.25, label="95% CI"
)
plt.xscale("log")
plt.xlabel("x")
plt.ylabel("F2")
plt.title("All Data: Prediction with 95% CI (MLP)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('5.pdf', format='pdf', bbox_inches='tight')
plt.show();
plt.close();

# --------------------------
# Multiclass Confusion Matrix
# --------------------------
plt.figure(figsize=(8,6))
sns.heatmap(multi_cm, annot=True, fmt="d",
            cmap="Blues", cbar=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Multiclass Confusion Matrix (MLP)")
plt.grid(True, alpha=0.3)
plt.savefig('6.pdf', format='pdf', bbox_inches='tight')
plt.show();
plt.close();

# --------------------------
# Binary Confusion Matrix
# --------------------------
cm_bin = confusion_matrix(y_test_bin, y_pred_bin)

plt.figure(figsize=(6,5))
sns.heatmap(cm_bin, annot=True, fmt="d",
            cmap="Blues", cbar=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Binary Confusion Matrix (MLP)")
plt.grid(True, alpha=0.3)
plt.savefig('7.pdf', format='pdf', bbox_inches='tight')
plt.show();
plt.close();

# --------------------------
# Feature Importance (Permutation)
# --------------------------
perm = permutation_importance(gpr, X_test_s, y_test,
                              n_repeats=10, random_state=42)
fi = pd.Series(perm.importances_mean, index=X.columns)

plt.figure(figsize=(8,6))
fi.sort_values().plot(kind="barh")
plt.title("Permutation Feature Importance (MLP)")
plt.xlabel("Importance")
plt.grid(True, alpha=0.3)
plt.savefig('8.pdf', format='pdf', bbox_inches='tight')
plt.show();
plt.close();

# --------------------------
# 2D Error Heatmap (x, Q2)
# --------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df["x"], y=df["Q^2"],
    hue=results["error"],
    palette="coolwarm", s=70
)
plt.xscale("log")
plt.yscale("log")
plt.title("2D Error Heatmap (x vs Q²) (MLP)")
plt.xlabel("x")
plt.ylabel("Q²")
plt.grid(True, alpha=0.3)
plt.savefig('9.pdf', format='pdf', bbox_inches='tight')
plt.show();
plt.close();

# ==========================================================
# 9. FINAL SUMMARY OUTPUT
# ==========================================================

print("\n" + "="*60)
print("       FINAL MLP MODEL SUMMARY (FULL PROFESSIONAL)")
print("="*60)

print("\n--- Regression Test Metrics ---")
for k,v in test_metrics.items():
    print(f"{k}: {v:.6f}")

print("\n--- Binary Classification ---")
for k,v in binary_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nMean CV R2:", cv_scores.mean())
print("Overfitting (Train R2 - Test R2):",
      train_metrics["R2"] - test_metrics["R2"])

print("\nFull Report Saved as: MLP_Full_Report.pdf")
print("CSV Saved: MLP_predictions_full.csv")
print("Excel Summary Saved: MLP_metrics_summary.xlsx")
print("="*60)

# ==========================================================
# PREDICT NEW POINT FUNCTION (MLP VERSION)
# ==========================================================

def predict_new_point(x_value, Q2_value):
    logx = np.log10(x_value)
    logQ2 = np.log10(Q2_value)

    new_features = np.array([[logx[0], logQ2[0]]])
    new_features_scaled = scaler.transform(new_features)

    prediction, std = gpr.predict(new_features_scaled, return_std=True)

    print(f"\n=== New Prediction (MLP Ensemble) ===")
    print(f"Input: x = {x_value[0]:.6f}, Q² = {Q2_value[0]:.6f}")
    print(f"Predicted F2: {prediction[0]:.6f}")
    print(f"Uncertainty (std): {std[0]:.6f}")

    return prediction[0]

print("\n=== Example Predictions (MLP Ensemble) ===")

# First sample
sample_1_x = df["x"].iloc[0:1].values
sample_1_Q2 = df["Q^2"].iloc[0:1].values
predict_new_point(sample_1_x, sample_1_Q2)
print(f"Actual F2: {df['F2_exp'].iloc[0]:.6f}")

# 500th sample
sample_500_x = df["x"].iloc[500:501].values
sample_500_Q2 = df["Q^2"].iloc[500:501].values
predict_new_point(sample_500_x, sample_500_Q2)
print(f"Actual F2: {df['F2_exp'].iloc[500]:.6f}")

# Custom point
custom_x = np.array([0.1])
custom_Q2 = np.array([10.0])
predict_new_point(custom_x, custom_Q2)

results_df = pd.DataFrame({
    'x': df['x'],
    'Q^2': df['Q^2'],
    'logQ2': df['logQ2'],
    'F2_actual': df['F2_exp'],
    'F2_predicted': y_all_pred,
    'uncertainty_std': y_all_std,
    'error': df['F2_exp'] - y_all_pred,
    'abs_error': np.abs(df['F2_exp'] - y_all_pred),
    'relative_error': (df['F2_exp'] - y_all_pred) / (df['F2_exp'] + 1e-10),
    'F2_category': pd.qcut(df['F2_exp'], q=5, labels=False),
    'F2_binary': (df['F2_exp'] > np.median(df['F2_exp'])).astype(int)
})

results_df.to_csv('mlp_predictions_detailed.csv', index=False)

print("✅ Predictions saved to 'mlp_predictions_detailed.csv'")

# ======================================================
# FINAL SUMMARY TABLES (MLP VERSION)
# ======================================================

# -------------------------
# Regression Summary
# -------------------------

regression_summary = pd.DataFrame({
    "Metric": ["Train R2", "Test R2", "Train MAE", "Test MAE",
               "Train RMSE", "Test RMSE"],
    "Value": [
        train_metrics["R2"],
        test_metrics["R2"],
        train_metrics["MAE"],
        test_metrics["MAE"],
        train_metrics["RMSE"],
        test_metrics["RMSE"]
    ]
})

regression_summary.to_excel(
    "regression_metrics_mlp.xlsx",
    index=False
)

# Save to PDF
with PdfPages("regression_metrics_mlp.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    ax.table(
        cellText=regression_summary.values,
        colLabels=regression_summary.columns,
        cellLoc='center',
        loc='center'
    )
    ax.set_title("Regression Metrics – MLP Ensemble")
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print("\n✅ MLP SUMMARY TABLES SAVED")
print("📂 Directory:", os.getcwd())
print("\n=== Overfitting Analysis (MLP) ===")
print(f"Training R²: {train_metrics['R2']:.6f}")
print(f"Test R²: {test_metrics['R2']:.6f}")
print(f"Difference: {train_metrics['R2'] - test_metrics['R2']:.6f}")

# ======================================================
# ✅ FINAL AUTO SUMMARY (FULL METRICS + PDF)
# ======================================================

def final_auto_report():

    # -----------------------------
    # 1️⃣ Regression Metrics
    # -----------------------------
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2  = r2_score(y_test, y_test_pred)
    diff_r2  = train_r2 - test_r2

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae  = mean_absolute_error(y_test, y_test_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse  = mean_squared_error(y_test, y_test_pred)

    train_rmse = np.sqrt(train_mse)
    test_rmse  = np.sqrt(test_mse)

    # -----------------------------
    # 2️⃣ Binary Classification
    # -----------------------------
    threshold = np.median(y)

    y_test_bin  = (y_test > threshold).astype(int)
    y_pred_bin  = (y_test_pred > threshold).astype(int)

    accuracy  = accuracy_score(y_test_bin, y_pred_bin)
    precision = precision_score(y_test_bin, y_pred_bin)
    recall    = recall_score(y_test_bin, y_pred_bin)
    f1 = f1_score(y_test_bin, y_pred_bin)
    roc_auc = roc_auc_score(y_test_bin, y_test_pred)

    # Multi‑class
    y_test_multi = pd.qcut(y_test, q=5, labels=False)
    y_pred_multi = pd.qcut(y_test_pred, q=5, labels=False)
    cm_multi = confusion_matrix(y_test_multi, y_pred_multi)

    # =============================================
    print("\n" + "="*70)
    print("           MLP ENSEMBLE FINAL AUTOMATED REPORT")
    print("="*70)

    print("\n[REGRESSION]")
    print(f"  Train R²: {train_r2:.6f}")
    print(f"  Test  R²: {test_r2:.6f}")
    print(f"  Overfit (ΔR²): {diff_r2:.6f}")
    print(f"  Test MAE:  {test_mae:.6f}")
    print(f"  Train MAE: {test_mae:.6f}")
    print(f"  Test RMSE: {test_rmse:.6f}")
    print(f"  Train RMSE: {test_rmse:.6f}")
    print("\n[BINARY CLASSIFICATION]")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  roc_auc:  {roc_auc:.4f}")
    print("\n[MULTI‑CLASS CONFUSION MATRIX (5 bins)]")
    print(cm_multi)

    print("\n[CROSS‑VALIDATION]")
    print(f"  Mean R² (5‑fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save a clean summary as PDF
    with PdfPages("MLP_Full_AutoReport.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        report_text = (
            "MLP Ensemble Final Summary\n\n"
            f"Regression Test R²: {test_r2:.4f}\n"
            f"Test MAE: {test_mae:.4f}\n"
            f"Test RMSE: {test_rmse:.4f}\n"
            f"Overfit (TRAIN R² – TEST R²): {diff_r2:.4f}\n\n"
            f"Binary Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}\n\n"
            f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )
        ax.text(0.5, 0.5, report_text, transform=ax.transAxes,
                fontsize=12, ha='center', va='center', family='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print("\n📄 Full auto report saved: MLP_Full_AutoReport.pdf")
    print("="*70)

# ==============================
final_auto_report()

print("\n✅ All analyses completed successfully.")
