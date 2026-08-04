import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import *
from sklearn.inspection import permutation_importance
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# SVR Ensemble wrapper to get uncertainty
# ==========================================================
class SVREnsemble:
    def __init__(self, n_estimators=10, **svr_kwargs):
        base_svr = SVR(**svr_kwargs)
        self.model = BaggingRegressor(
            estimator=base_svr,
            n_estimators=n_estimators,
            bootstrap=True,
            n_jobs=1,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        if return_std:
            preds = np.array([est.predict(X) for est in self.model.estimators_])
            mean = preds.mean(axis=0)
            std = preds.std(axis=0)
            return mean, std
        else:
            return self.model.predict(X)

# ========== 1. Load data ==========
df = pd.read_csv("F2BCMS.csv")
df = df.dropna()
df["logx"] = np.log10(df["x"])
df["logQ2"] = np.log10(df["Q^2"])

# ========== 2. Features and target ==========
X = df[["logx", "logQ2"]]
y = df["F2_exp"]

# ========== 3. Train / Test split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ========== 4. Define SVR Ensemble Model ==========
svr = SVREnsemble(
    n_estimators=10,
    kernel='rbf',
    C=10.0,
    epsilon=0.1,
    gamma='scale'
)

# ========== 5. Fit model ==========
svr.fit(X_train_s, y_train)

# ========== 6. Predictions ==========
y_train_pred = svr.predict(X_train_s, return_std=False)
y_test_pred, y_test_std = svr.predict(X_test_s, return_std=True)

# ========== 7. Regression Metrics ==========
train_metrics = {
    "R2": r2_score(y_train, y_train_pred),
    "MAE": mean_absolute_error(y_train, y_train_pred),
    "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred))
}
test_metrics = {
    "R2": r2_score(y_test, y_test_pred),
    "MAE": mean_absolute_error(y_test, y_test_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred))
}

print("\n--- Regression Metrics (SVR Ensemble) ---")
print(f"Train R2: {train_metrics['R2']:.6f}")
print(f"Test R2 : {test_metrics['R2']:.6f}")
print(f"Train MAE: {train_metrics['MAE']:.6f}")
print(f"Test MAE : {test_metrics['MAE']:.6f}")

# ========== 8. Binary Classification ==========
threshold = np.median(y)

y_test_bin = (y_test > threshold).astype(int)
y_pred_bin = (y_test_pred > threshold).astype(int)

roc_auc = roc_auc_score(y_test_bin, y_test_pred)

binary_metrics = {
    "Accuracy": accuracy_score(y_test_bin, y_pred_bin),
    "Precision": precision_score(y_test_bin, y_pred_bin),
    "Recall": recall_score(y_test_bin, y_pred_bin),
    "F1": f1_score(y_test_bin, y_pred_bin),
    "ROC-AUC": roc_auc
}

print("\nBinary Classification Metrics:")
for k, v in binary_metrics.items():
    print(f"  {k}: {v:.4f}")

# Multi‑class (5 bins)
y_test_multi = pd.qcut(y_test, q=5, labels=False)
y_pred_multi = pd.qcut(y_test_pred, q=5, labels=False)
multi_cm = confusion_matrix(y_test_multi, y_pred_multi)

# ========== 9. Cross‑Validation ==========
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# For CV we need a single SVR (ensemble inside CV would be heavy; use base SVR for speed)
base_svr = SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')
cv_scores = cross_val_score(base_svr, scaler.fit_transform(X), y,
                            cv=cv, scoring='r2')
print(f"\nCross‑Validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ========== 10. Full dataset predictions (for plots) ==========
X_all_s = scaler.transform(X)
y_all_pred, y_all_std = svr.predict(X_all_s, return_std=True)

# ========== 11. Plots (all titles updated to SVR) ==========

# 4‑Panel Evaluation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].scatter(y_test, y_test_pred, alpha=0.7)
axes[0,0].plot([y_test.min(), y_test.max()],
               [y_test.min(), y_test.max()], 'r--')
axes[0,0].set_title("Test Set: Actual vs Predicted (SVR)")
axes[0,0].set_xlabel("Actual F2")
axes[0,0].set_ylabel("Predicted F2")
axes[0,0].grid(True)

axes[0,1].scatter(y_test_pred, y_test - y_test_pred, alpha=0.6)
axes[0,1].axhline(0, linestyle='--', color='red')
axes[0,1].set_title("Residual Plot (Test Set) (SVR)")
axes[0,1].set_xlabel("Predicted F2")
axes[0,1].set_ylabel("Residual (Actual - Predicted)")
axes[0,1].grid(True)

axes[1,0].scatter(df.loc[y_test.index, 'x'].values, y_test - y_test_pred, alpha=0.6)
axes[1,0].set_xscale('log')
axes[1,0].set_title("Residuals vs x (Test Set) (SVR)")
axes[1,0].set_xlabel("x (log scale)")
axes[1,0].set_ylabel("Residual")
axes[1,0].grid(True)

axes[1,1].hist(y_test - y_test_pred, bins=25, color='steelblue', edgecolor="black", alpha=0.8)
axes[1,1].set_title("Distribution of Residuals (Test Set) (SVR)")
axes[1,1].set_xlabel("Residual")
axes[1,1].set_ylabel("Count")
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig("SVR_4_Evaluation_Plots.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Error analysis plots (4‑panel)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
test_indices = y_test.index
x_test = df.loc[test_indices, 'x'].values
Q2_test = df.loc[test_indices, 'Q^2'].values
errors = np.abs(y_test.values - y_test_pred)

axes[0,0].scatter(x_test, errors, alpha=0.6)
axes[0,0].set_xscale('log')
axes[0,0].set_xlabel('x (log scale)')
axes[0,0].set_ylabel('Absolute Error')
axes[0,0].set_title('Absolute Error vs x (Test Set) (SVR)')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].scatter(Q2_test, errors, alpha=0.6)
axes[0,1].set_xscale('log')
axes[0,1].set_xlabel('Q² (log scale)')
axes[0,1].set_ylabel('Absolute Error')
axes[0,1].set_title('Absolute Error vs Q² (Test Set) (SVR)')
axes[0,1].grid(True, alpha=0.3)

relative_errors = errors / (y_test.values + 1e-10)
axes[1,0].scatter(x_test, relative_errors * 100, alpha=0.6)
axes[1,0].set_xscale('log')
axes[1,0].set_xlabel('x (log scale)')
axes[1,0].set_ylabel('Relative Error (%)')
axes[1,0].set_title('Relative Error vs x (Test Set) (SVR)')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
axes[1,1].set_xlabel('Absolute Error')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Absolute Errors (Test Set) (SVR)')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("0.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Actual vs Predicted (simple)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted (SVR)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True, alpha=0.3)
plt.savefig('1.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# Residual plot
plt.figure(figsize=(8,6))
plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.6)
plt.axhline(0, linestyle="--", color="red")
plt.title("Residual Plot (SVR)")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.grid(True, alpha=0.3)
plt.savefig('2.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# Error distribution
plt.figure(figsize=(8,6))
plt.hist(y_test - y_test_pred, bins=30, edgecolor="black")
plt.title("Error Distribution (SVR)")
plt.xlabel("Error")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.savefig('3.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# First 20 points
idx20 = df.index[:20]
X20 = scaler.transform(X.loc[idx20][["logx", "logQ2"]])
y20_pred, y20_std = svr.predict(X20, return_std=True)

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
plt.title("First 20: Actual vs Predicted (SVR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('4.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# All data with 95% CI
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
plt.title("All Data: Prediction with 95% CI (SVR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('5.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# Multiclass confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(multi_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Multiclass Confusion Matrix (SVR)")
plt.grid(True, alpha=0.3)
plt.savefig('6.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# Binary confusion matrix
cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
plt.figure(figsize=(6,5))
sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Binary Confusion Matrix (SVR)")
plt.grid(True, alpha=0.3)
plt.savefig('7.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# Permutation feature importance
perm = permutation_importance(svr.model, X_test_s, y_test,
                              n_repeats=10, random_state=42)
fi = pd.Series(perm.importances_mean, index=X.columns)
plt.figure(figsize=(8,6))
fi.sort_values().plot(kind="barh")
plt.title("Permutation Feature Importance (SVR)")
plt.xlabel("Importance")
plt.grid(True, alpha=0.3)
plt.savefig('8.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# 2D Error heatmap
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df["x"], y=df["Q^2"],
    hue=np.abs(df["F2_exp"] - y_all_pred),
    palette="coolwarm", s=70
)
plt.xscale("log")
plt.yscale("log")
plt.title("2D Error Heatmap (x vs Q²) (SVR)")
plt.xlabel("x")
plt.ylabel("Q²")
plt.grid(True, alpha=0.3)
plt.savefig('9.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# ========== 12. Final Summary Output ==========
print("\n" + "="*60)
print("       FINAL SVR MODEL SUMMARY (FULL PROFESSIONAL)")
print("="*60)

print("\n--- Regression Test Metrics ---")
for k, v in test_metrics.items():
   print(f"  {k}: {v:.6f}")

print("\n--- Regression Train Metrics ---")
for k, v in train_metrics.items():
   print(f"  {k}: {v:.6f}")

print("\n--- Binary Classification Test Metrics ---")
for k, v in binary_metrics.items():
   print(f"  {k}: {v:.4f}")

# Multiclass accuracy (from confusion matrix)
multi_acc = np.sum(np.diag(multi_cm)) / np.sum(multi_cm)
print(f"\n--- Multiclass (5 bins) Accuracy ---")
print(f"  Accuracy: {multi_acc:.4f}")

print(f"\n--- Cross‑Validation R² ---")
print(f"  Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance ranking
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm.importances_mean,
    'Std': perm.importances_std
}).sort_values(by='Importance', ascending=False)
print("\n--- Permutation Feature Importance ---")
print(importance_df.to_string(index=False))

# ========== 13. Save predictions to CSV and XLSX ==========
# Test set predictions
test_results = pd.DataFrame({
    'logx': X_test['logx'],
    'logQ2': X_test['logQ2'],
    'Actual_F2': y_test,
    'Predicted_F2': y_test_pred,
    'Prediction_Std': y_test_std,
    'Absolute_Error': np.abs(y_test - y_test_pred),
    'Relative_Error_pct': np.abs(y_test - y_test_pred) / (y_test + 1e-10) * 100
})

# Full dataset predictions
all_results = pd.DataFrame({
    'logx': df['logx'],
    'logQ2': df['logQ2'],
    'x': df['x'],
    'Q2': df['Q^2'],
    'Actual_F2': df['F2_exp'],
    'Predicted_F2': y_all_pred,
    'Prediction_Std': y_all_std,
    'Absolute_Error': np.abs(df['F2_exp'] - y_all_pred),
    'Relative_Error_pct': np.abs(df['F2_exp'] - y_all_pred) / (df['F2_exp'] + 1e-10) * 100
})

# Save CSV
test_results.to_csv('SVR_Test_Predictions.csv', index=False)
all_results.to_csv('SVR_Full_Predictions.csv', index=False)

# Save XLSX (Excel)
with pd.ExcelWriter('SVR_Predictions.xlsx', engine='openpyxl') as writer:
    test_results.to_excel(writer, sheet_name='Test_Set', index=False)
    all_results.to_excel(writer, sheet_name='Full_Data', index=False)

print("\nPredictions saved to:")
print("   - SVR_Test_Predictions.csv")
print("   - SVR_Full_Predictions.csv")
print("   - SVR_Predictions.xlsx")

# ========== 14. Save summary report to PDF ==========
with PdfPages('SVR_Report.pdf') as pdf:
    # Page 1: Text metrics
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    text = f"""
SVR Model Summary Report
=========================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Model Hyperparameters:
  kernel: rbf
  C: 10.0
  epsilon: 0.1
  gamma: scale
  Ensemble estimators: 10

Regression Metrics (Test Set):
  R²  : {test_metrics['R2']:.6f}
  MAE : {test_metrics['MAE']:.6f}
  RMSE: {test_metrics['RMSE']:.6f}

Regression Metrics (Train Set):
  R²  : {train_metrics['R2']:.6f}
  MAE : {train_metrics['MAE']:.6f}
  RMSE: {train_metrics['RMSE']:.6f}

Binary Classification (Test, threshold={threshold:.3f}):
  Accuracy : {binary_metrics['Accuracy']:.4f}
  Precision: {binary_metrics['Precision']:.4f}
  Recall   : {binary_metrics['Recall']:.4f}
  F1 Score : {binary_metrics['F1']:.4f}
  
Multiclass (5 bins) Accuracy: {multi_acc:.4f}

5‑Fold Cross‑Validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

Feature Importances (Permutation):
{importance_df.to_string(index=False)}

Total data points: {len(df)}
Test data points: {len(X_test)}
    """
    ax.text(0.05, 0.5, text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close()

# ========== 15. Function to predict new points ==========
def predict_new_point(logx, logQ2):
    """
    Predict F2 for a new point using the trained SVR ensemble.
    Parameters:
        logx (float): log10(x)
        logQ2 (float): log10(Q^2)
    Returns:
        predicted F2, uncertainty (std)
    """
    point = pd.DataFrame({'logx': [logx], 'logQ2': [logQ2]})
    point_s = scaler.transform(point[['logx', 'logQ2']])
    mean, std = svr.predict(point_s, return_std=True)
    return mean[0], std[0]

# Example of using the function
ex_logx, ex_logQ2 = -1.5, 1.2
pred_ex, std_ex = predict_new_point(ex_logx, ex_logQ2)
print(f"\nExample prediction (logx={ex_logx}, logQ2={ex_logQ2}): "
      f"F2 = {pred_ex:.4f} ± {std_ex:.4f}")
print("\nCode execution completed successfully.")
