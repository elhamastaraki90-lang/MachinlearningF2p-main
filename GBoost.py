import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
import os
import warnings
warnings.filterwarnings('ignore')

# ========== 1. Load Data ==========
df = pd.read_csv('F2BCMS.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

# ========== 2. Preprocessing ==========
df['logx'] = np.log10(df['x'])
df['logQ2'] = np.log10(df['Q^2'])
X = df[['logx', 'logQ2']]
y = df['F2_exp']

# ========== 3. Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ========== 4. Feature Scaling ==========
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_train_s = pd.DataFrame(X_train_s, columns=X.columns, index=X_train.index)
X_test_s = pd.DataFrame(X_test_s, columns=X.columns, index=X_test.index)

# ========== 5. GBoost Ensemble Model ==========
class GBoostEnsemble:
    def __init__(self, n_estimators=10, **gb_kwargs):
        """
        Ensemble of Gradient Boosting models for uncertainty estimation.
        
        Parameters:
        -----------
        n_estimators : int, default=10
            Number of GBoost models in ensemble
        **gb_kwargs : dict
            Additional parameters for GradientBoostingRegressor
        """
        self.models = []
        for i in range(n_estimators):
            model = GradientBoostingRegressor(
                random_state=42 + i,  # Different seed for each model
                **gb_kwargs
            )
            self.models.append(model)
        self.n_estimators = n_estimators
    
    def fit(self, X, y):
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X, return_std=False):
        """
        Predict using the ensemble.
        
        Parameters:
        -----------
        X : array-like
            Input features
        return_std : bool, default=False
            If True, return mean and std of predictions
            
        Returns:
        --------
        if return_std=False: array of mean predictions
        if return_std=True: tuple of (mean_predictions, std_predictions)
        """
        all_preds = []
        for model in self.models:
            pred = model.predict(X)
            all_preds.append(pred)
        
        all_preds = np.array(all_preds)  # shape: (n_estimators, n_samples)
        mean_pred = np.mean(all_preds, axis=0)
        
        if return_std:
            std_pred = np.std(all_preds, axis=0)
            return mean_pred, std_pred
        else:
            return mean_pred

# Create and train the GBoost ensemble
gb_params = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8
}

print("\nTraining GBoost Ensemble...")
gboost = GBoostEnsemble(n_estimators=10, **gb_params)
gboost.fit(X_train_s, y_train)

# Predictions
y_train_pred, y_train_std = gboost.predict(X_train_s, return_std=True)
y_test_pred, y_test_std = gboost.predict(X_test_s, return_std=True)

# ========== 6. Regression Metrics ==========
def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

test_metrics = calculate_metrics(y_test, y_test_pred)
train_metrics = calculate_metrics(y_train, y_train_pred)

print("\n=== Regression Metrics ===")
print("Test Set:")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.6f}")

print("\nTrain Set:")
for k, v in train_metrics.items():
    print(f"  {k}: {v:.6f}")

# ========== 7. Binary Classification ==========
threshold = np.median(y)
y_test_binary = (y_test > threshold).astype(int)
y_pred_binary = (y_test_pred > threshold).astype(int)

binary_cm = confusion_matrix(y_test_binary, y_pred_binary)
binary_report = classification_report(y_test_binary, y_pred_binary, output_dict=True)


roc_auc = roc_auc_score(y_test_binary, y_test_pred)

binary_metrics = {
    'Accuracy': binary_report['accuracy'],
    'Precision': binary_report['weighted avg']['precision'],
    'Recall': binary_report['weighted avg']['recall'],
    'F1': binary_report['weighted avg']['f1-score'],
    'ROC_AUC': roc_auc
}

print("\n=== Binary Classification (threshold = median) ===")
print(f"Confusion Matrix:\n{binary_cm}")
print(f"Accuracy : {binary_metrics['Accuracy']:.4f}")
print(f"Precision: {binary_metrics['Precision']:.4f}")
print(f"Recall   : {binary_metrics['Recall']:.4f}")
print(f"F1       : {binary_metrics['F1']:.4f}")
print(f"ROC AUC  : {binary_metrics['ROC_AUC']:.4f}")


# ========== 8. Multiclass Classification ==========
n_bins = 5
quantiles = np.percentile(y, np.linspace(0, 100, n_bins + 1))
y_test_multi = pd.cut(y_test, bins=quantiles, labels=False, include_lowest=True)
y_pred_multi = pd.cut(y_test_pred, bins=quantiles, labels=False, include_lowest=True)

multi_cm = confusion_matrix(y_test_multi, y_pred_multi)
print("\n=== Multiclass Classification (5 bins) ===")
print(f"Confusion Matrix:\n{multi_cm}")

# ========== 9. Cross-Validation ==========
cv_scores = cross_val_score(
    GradientBoostingRegressor(**gb_params),
    X_train_s, y_train,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2'
)
print(f"\n=== 5-Fold Cross-Validation R² ===")
print(f"Scores: {cv_scores}")
print(f"Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ========== 10. Feature Importance ==========
# Use the first model for permutation importance
perm = permutation_importance(
    gboost.models[0], X_test_s, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=1
)

print("\n=== Feature Importance (Permutation) ===")
for i, col in enumerate(X.columns):
    print(f"{col}: {perm.importances_mean[i]:.6f}")
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm.importances_mean,
    'Std': perm.importances_std
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))

# ========== 11. Save Results to CSV/Excel ==========
# Prepare results dataframe
results_df = pd.DataFrame({
    'x': df.loc[X_test.index, 'x'],
    'Q^2': df.loc[X_test.index, 'Q^2'],
    'logx': X_test['logx'],
    'logQ2': X_test['logQ2'],
    'F2_actual': y_test,
    'F2_pred': y_test_pred,
    'F2_std': y_test_std,
    'Residual': y_test - y_test_pred,
    'Binary_Actual': y_test_binary,
    'Binary_Pred': y_pred_binary,
    'Multiclass_Actual': y_test_multi,
    'Multiclass_Pred': y_pred_multi
})

# Save to CSV and Excel
results_df.to_csv('GBoost_Predictions.csv', index=False)
results_df.to_excel('GBoost_Predictions.xlsx', index=False)
print("\n=== Predictions saved to CSV and Excel ===")

# ========== 12. Generate Individual Plots ==========
# Plot 1: 4 Evaluation Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('GBoost: 4 Evaluation Plots', fontsize=16)

# 1.1 Actual vs Predicted (scatter)
axs[0,0].scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k')
axs[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axs[0,0].set_xlabel('Actual F2')
axs[0,0].set_ylabel('Predicted F2')
axs[0,0].set_title('Actual vs Predicted (Test(GBoost))')
axs[0,0].grid(True, alpha=0.3)

# 1.2 Residuals vs Predicted
residuals = y_test - y_test_pred
axs[0,1].scatter(y_test_pred, residuals, alpha=0.5, edgecolors='k')
axs[0,1].axhline(y=0, color='r', linestyle='--', lw=2)
axs[0,1].set_xlabel('Predicted F2')
axs[0,1].set_ylabel('Residuals')
axs[0,1].set_title('Residuals vs Predicted(GBoost)')
axs[0,1].grid(True, alpha=0.3)

# 1.3 Histogram of residuals
axs[1,0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axs[1,0].axvline(x=0, color='r', linestyle='--', lw=2)
axs[1,0].set_xlabel('Residuals')
axs[1,0].set_ylabel('Frequency')
axs[1,0].set_title('Distribution of Residuals(GBoost)')
axs[1,0].grid(True, alpha=0.3)

# 1.4 Q-Q plot of residuals
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axs[1,1])
axs[1,1].set_title('Q-Q Plot of Residuals(GBoost)')
axs[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('GBoost_4_Evaluation_Plots.png', dpi=300, bbox_inches='tight')
plt.savefig('GBoost_4_Evaluation_Plots.pdf', bbox_inches='tight')
plt.show();
plt.close()

# Plot 2: Simple Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual F2')
ax.set_ylabel('Predicted F2')
ax.set_title('GBoost: Actual vs Predicted (Test Set)(GBoost)')
ax.grid(True, alpha=0.3)
plt.savefig('GBoost_Actual_vs_Predicted.png', dpi=300, bbox_inches='tight')
plt.savefig('GBoost_Actual_vs_Predicted.pdf', bbox_inches='tight')
plt.show()
plt.close()

# Plot 3: Simple Residuals
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test_pred, residuals, alpha=0.5, edgecolors='k')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted F2')
ax.set_ylabel('Residuals')
ax.set_title('GBoost: Residuals vs Predicted(GBoost)')
ax.grid(True, alpha=0.3)
plt.savefig('GBoost_Residuals.png', dpi=300, bbox_inches='tight')
plt.savefig('GBoost_Residuals.pdf', bbox_inches='tight')
plt.show
plt.close()
# Plot 4: Histogram of errors (absolute errors)
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(np.abs(residuals), bins=30, edgecolor='black', alpha=0.7, label='Absolute Residuals')
ax.axvline(np.mean(np.abs(residuals)), color='red', linestyle='--', lw=2, label=f'Mean = {np.mean(np.abs(residuals)):.4f}')
ax.set_xlabel('Absolute Residual')
ax.set_ylabel('Frequency')
ax.set_title('GBoost: Histogram of Absolute Residuals (Test Set(GBoost))')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('GBoost_Absolute_Residuals_Hist.png', dpi=300, bbox_inches='tight')
plt.savefig('GBoost_Absolute_Residuals_Hist.pdf', bbox_inches='tight')
plt.show()
plt.close()

# Plot 5: 4 Error Plots (Absolute/Relative Error vs x and Q^2)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('GBoost: 2D Error Analysis', fontsize=16)

abs_error = np.abs(residuals)
rel_error = np.abs(residuals / (y_test.values + 1e-10)) * 100

# 1. Absolute Error vs x
axs[0,0].scatter(results_df['x'], abs_error, alpha=0.6, edgecolors='k')
axs[0,0].set_xlabel('x')
axs[0,0].set_ylabel('Absolute Error')
axs[0,0].set_title('Absolute Error vs x(GBoost)')
axs[0,0].grid(True, alpha=0.3)

# 2. Absolute Error vs Q^2
axs[0,1].scatter(results_df['Q^2'], abs_error, alpha=0.6, edgecolors='k')
axs[0,1].set_xlabel('Q^2')
axs[0,1].set_ylabel('Absolute Error')
axs[0,1].set_title('Absolute Error vs Q^2(GBoost)')
axs[0,1].grid(True, alpha=0.3)

# 3. Relative Error vs x
axs[1,0].scatter(results_df['x'], rel_error, alpha=0.6, edgecolors='k')
axs[1,0].set_xlabel('x')
axs[1,0].set_ylabel('Relative Error (%)')
axs[1,0].set_title('Relative Error vs x(GBoost)')
axs[1,0].grid(True, alpha=0.3)

# 4. Relative Error vs Q^2
axs[1,1].scatter(results_df['Q^2'], rel_error, alpha=0.6, edgecolors='k')
axs[1,1].set_xlabel('Q^2')
axs[1,1].set_ylabel('Relative Error (%)')
axs[1,1].set_title('Relative Error vs Q^2(GBoost)')
axs[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('GBoost_2D_Error_Plots.png', dpi=300, bbox_inches='tight')
plt.savefig('GBoost_2D_Error_Plots.pdf', bbox_inches='tight')
plt.show()
plt.close()

# Plot 6: First 20 Points (actual vs predicted) with error bars from std
fig, ax = plt.subplots(figsize=(12, 6))
n_points = min(20, len(y_test))
indices = range(n_points)
y_actual_20 = y_test.iloc[:n_points]
y_pred_20 = y_test_pred[:n_points]
y_std_20 = y_test_std[:n_points]

ax.errorbar(indices, y_actual_20, yerr=None, fmt='bo-', label='Actual', capsize=5, markersize=5)
ax.errorbar(indices, y_pred_20, yerr=y_std_20, fmt='rs--', label='Predicted', capsize=5, markersize=5)
ax.fill_between(indices, y_pred_20 - y_std_20, y_pred_20 + y_std_20, alpha=0.2, color='gray')
ax.set_xlabel('Test Sample Index')
ax.set_ylabel('F2')
ax.set_title('GBoost: First 20 Test Points (Actual vs Predicted with 95% CI)(GBoost)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('GBoost_First20_Points.png', dpi=300, bbox_inches='tight')
plt.savefig('GBoost_First20_Points.pdf', bbox_inches='tight')
plt.show()
plt.close()

# Plot 7: All Test Data with 95% CI (sorted by actual for better visualization)
sorted_idx = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_idx]
y_pred_sorted = y_test_pred[sorted_idx]
y_std_sorted = y_test_std[sorted_idx]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(y_test)), y_test_sorted, 'b.', markersize=4, label='Actual')
ax.plot(range(len(y_test)), y_pred_sorted, 'r.', markersize=3, label='Predicted')
ax.fill_between(range(len(y_test)), y_pred_sorted - y_std_sorted, y_pred_sorted + y_std_sorted, alpha=0.3, color='gray', label='95% CI')
ax.set_xlabel('Sample (sorted by actual)')
ax.set_ylabel('F2')
ax.set_title('GBoost: All Test Data with 95% Confidence Interval (Sorted)(GBoost)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('GBoost_All_Data_CI.png', dpi=300, bbox_inches='tight')
plt.savefig('GBoost_All_Data_CI.pdf', bbox_inches='tight')
plt.show
plt.close()

# Plot 8: Confusion Matrices (Binary and Multiclass)
plt.figure(figsize=(6, 5))
sns.heatmap(
    binary_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Low (≤ median)', 'High (> median)'],
    yticklabels=['Low (≤ median)', 'High (> median)']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('GBoost: Binary Confusion Matrix (GBoost)')

plt.tight_layout()
plt.savefig('GBoost_Binary_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('GBoost_Binary_Confusion_Matrix.pdf', bbox_inches='tight')
plt.show()
plt.close()
class_names = [f'Bin {i+1}' for i in range(n_bins)]

plt.figure(figsize=(6, 5))
sns.heatmap(
    multi_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'GBoost: Multiclass ({n_bins}-bins) Confusion Matrix (GBoost)')

plt.tight_layout()
plt.savefig('GBoost_Multiclass_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('GBoost_Multiclass_Confusion_Matrix.pdf', bbox_inches='tight')
plt.show()
plt.close()

# Plot 9: Feature Importance (Permutation)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'], capsize=5, color='skyblue')
ax.set_xlabel('Mean Permutation Importance')
ax.set_ylabel('Feature')
ax.set_title('GBoost: Feature Importance (Permutation)(GBoost)')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('GBoost_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.savefig('GBoost_Feature_Importance.pdf', bbox_inches='tight')
plt.show()
plt.close()

# Plot 10: 2D Error Heatmap (x vs Q^2, colored by absolute error)
import seaborn as sns

# ساخت دیتافریم برای رسم
plot_df = results_df.copy()
plot_df["error"] = abs_error

# تبدیل خطا به بازه‌ها (levels)
plot_df["error_bin"] = pd.cut(plot_df["error"], bins=6)

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    data=plot_df,
    x="x",
    y="Q^2",
    hue="error_bin",
    palette="coolwarm",
    edgecolor="k",
    linewidth=0.3,
    ax=ax
)

ax.set_xlabel("x")
ax.set_ylabel("Q^2")
ax.set_title("2D Error Heatmap (x vs Q²) (GBoost)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("GBoost_2D_Error_Heatmap.png", dpi=300)
plt.savefig("GBoost_2D_Error_Heatmap.pdf")
plt.show()

print("\n=== All plots generated and saved ===")

# ========== 13. Predict on Entire Dataset ==========
# Scale full dataset
X_full = X  # it's already a DataFrame
X_full_s = scaler.transform(X_full)
X_full_s = pd.DataFrame(X_full_s, columns=X.columns, index=X_full.index)

# Predict
y_full_pred, y_full_std = gboost.predict(X_full_s, return_std=True)

# Full results dataframe
full_results = pd.DataFrame({
    'x': df['x'],
    'Q^2': df['Q^2'],
    'logx': df['logx'],
    'logQ2': df['logQ2'],
    'F2_actual': y,
    'F2_pred': y_full_pred,
    'F2_std': y_full_std,
    'Residual': y - y_full_pred,
    'Absolute_Error': np.abs(y - y_full_pred),
    'Relative_Error_%': np.abs((y - y_full_pred) / (y + 1e-10)) * 100
})

full_results.to_csv('GBoost_predictions_full.csv', index=False)
full_results.to_excel('GBoost_predictions_full.xlsx', index=False)
print("\nFull predictions saved to CSV and Excel.")

# ========== 14. Save Metrics Summary to Excel ==========
metrics_data = {
    'Metric': ['R²', 'MAE', 'RMSE', 'MAPE'],
    'Train': [train_metrics['R2'], train_metrics['MAE'], train_metrics['RMSE'], train_metrics['MAPE']],
    'Test': [test_metrics['R2'], test_metrics['MAE'], test_metrics['RMSE'], test_metrics['MAPE']]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_excel('GBoost_metrics_summary.xlsx', index=False)

# Also add binary classification metrics
binary_metrics_df = pd.DataFrame({
    'Metric': list(binary_metrics.keys()),
    'Value': list(binary_metrics.values())
})
with pd.ExcelWriter('GBoost_metrics_summary.xlsx', mode='a', engine='openpyxl') as writer:
    binary_metrics_df.to_excel(writer, sheet_name='Binary_Classification', index=False)
    cv_df = pd.DataFrame({
        'Fold': list(range(1, 6)),
        'R²': cv_scores
    })
    cv_df.to_excel(writer, sheet_name='Cross_Validation', index=False)

print("Metrics summary saved to GBoost_metrics_summary.xlsx")

# ========== 15. Predict New Point Function ==========
def predict_new_point(x_value, Q2_value):
    """
    Predict F2 for a new point using the trained GBoost ensemble.
    
    Parameters:
    -----------
    x_value : float
        Value of x (linear scale)
    Q2_value : float
        Value of Q² (linear scale)
    
    Returns:
    --------
    dict with 'prediction', 'std'
    """
    # Preprocess
    logx = np.log10(x_value)
    logQ2 = np.log10(Q2_value)
    # Create feature array and scale
    point = pd.DataFrame([[logx, logQ2]], columns=['logx', 'logQ2'])
    point_s = scaler.transform(point)
    # Predict
    pred, std = gboost.predict(point_s, return_std=True)
    return {'prediction': pred[0], 'std': std[0]}

# Example usage
example_x = 1e-3
example_Q2 = 5.0
example_pred = predict_new_point(example_x, example_Q2)
print(f"\nExample prediction for x={example_x}, Q²={example_Q2}:")
print(f"  F2 predicted = {example_pred['prediction']:.6f} ± {example_pred['std']:.6f}")

# ========== 16. Generate Comprehensive PDF Report ==========
print("\nGenerating PDF report...")
pdf_filename = 'GBoost_Full_AutoReport.pdf'
with PdfPages(pdf_filename) as pdf:
    # --- Page 1: Title and Summary ---
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
    ax.axis('off')
    text = f"""
    Gradient Boosting Regression (GBoost) Auto-Report
    ================================================
    
    Dataset: F2BCMS.csv
    Target: F2 (Experimental)
    Features: log10(x) and log10(Q^2)
    
    Model: Ensemble of 10 GradientBoostingRegressor
    Base estimator parameters:
      - n_estimators: 100
      - learning_rate: 0.1
      - max_depth: 3
      - subsample: 0.8
    
    Train/Test Split: 80%/20% (random_state=42)
    Standardization: StandardScaler
    
    ===== Regression Metrics (Test Set) =====
    R²   : {test_metrics['R2']:.6f}
    MAE  : {test_metrics['MAE']:.6f}
    RMSE : {test_metrics['RMSE']:.6f}
    MAPE : {test_metrics['MAPE']:.4f}%ّ
    
    ===== Binary Classification (median threshold) =====
    Accuracy  : {binary_metrics['Accuracy']:.4f}
    Precision : {binary_metrics['Precision']:.4f}
    Recall    : {binary_metrics['Recall']:.4f}
    F1 Score  : {binary_metrics['F1']:.4f}
    
    ===== Cross-Validation (5-Fold) =====
    R² Mean ± Std : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
    
    ===== Feature Importance (Permutation) =====
       logx   : {importance_df[importance_df['Feature']=='logx']['Importance'].values[0]:.6f}
       logQ2  : {importance_df[importance_df['Feature']=='logQ2']['Importance'].values[0]:.6f}
    """
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    pdf.savefig(fig)
    plt.close()

    # --- Subsequent pages: all the saved plots ---
    plot_filenames = [
        'GBoost_4_Evaluation_Plots.png',
        'GBoost_Actual_vs_Predicted.png',
        'GBoost_Residuals.png',
        'GBoost_Absolute_Residuals_Hist.png',
        'GBoost_2D_Error_Plots.png',
        'GBoost_First20_Points.png',
        'GBoost_All_Data_CI.png',
        'GBoost_Confusion_Matrices.png',
        'GBoost_Feature_Importance.png',
        'GBoost_2D_Error_Heatmap.png'
    ]
    
    for fname in plot_filenames:
        if os.path.exists(fname):
            img = plt.imread(fname)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        else:
            print(f"Warning: {fname} not found. Skipping in PDF.")

print(f"PDF report saved to {pdf_filename}")

# ========== 17. Final Message ==========
print("\n" + "="*60)
print("GBoost Regression Pipeline Completed Successfully!")
print("="*60)
print("Generated files:")
print("  - GBoost_Predictions.csv / .xlsx (test set)")
print("  - GBoost_predictions_full.csv / .xlsx (full dataset)")
print("  - GBoost_metrics_summary.xlsx")
print("  - GBoost_Full_AutoReport.pdf")
print("  - Various PNG and PDF plots")
print("="*60)
