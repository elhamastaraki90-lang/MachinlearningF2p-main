# MachinlearningF2p
# Machine Learning for Predicting the Proton Structure Function F₂ in QCD

**Shahin Atashbar Tehrani¹² and Elham Astaraki³

¹ School of Particles and Accelerators,  
Institute for Research in Fundamental Sciences (IPM),  
P.O. Box 19395-5531, Tehran, Iran  

² Department of Physics,  
Faculty of Nano and Bio Science and Technology,  
Persian Gulf University, 75169 Bushehr, Iran  

³ Department of Physics,  
Razi University, Kermanshah 67149, Iran  

---

## 📌 Project Overview

This repository contains the full machine learning pipeline used to analyze and predict the **proton structure function F₂** in Quantum Chromodynamics (QCD) using experimental DIS data.

The regression problem is reformulated as a **quantile-based classification task**, allowing the use of robust classification metrics such as **F1-score, Precision, Recall, Accuracy, and ROC–AUC**, followed by probability-weighted reconstruction of continuous predictions.

The project is designed to be:
- ✅ **Thesis-ready**
- ✅ **Journal-ready**
- ✅ **Reproducible**
- ✅ **Free of data leakage**

---

## 📂 Dataset

- **File:** `F2BCMS.csv`
- **Target variable:** `F2_exp`
- **Input features:**
  - `x`  → transformed to `logx`
  - `Q^2` → transformed to `logQ2`

Logarithmic transformations are applied to improve numerical stability and reflect the underlying physics.

---

## 🧠 Machine Learning Models

The following supervised learning algorithms are implemented and analyzed:

| Model | Type |
|-----|-----|
| Random Forest | Ensemble (Tree-based) |
| Decision Tree | Tree-based |
| k-Nearest Neighbors (KNN) | Distance-based |
| Logistic Regression | Linear classifier |

All models follow the **same preprocessing and evaluation protocol** to ensure fair comparison.

---

## 🔄 Methodology

### 1. Regression → Classification
- The continuous target variable `F2_exp` is discretized using **quantile-based binning**.
- Default configuration uses **3 classes**, but **9-class** variants are also supported for reduced information loss.

### 2. Training Protocol
- Train/test split: **80% / 20%**
- Standardization using `StandardScaler`
- No information leakage (all statistics computed on training data only)

### 3. Hyperparameter Optimization
Each model includes a dedicated hyperparameter sweep using **Stratified 5-Fold Cross-Validation**:
- Random Forest: `n_estimators`
- Decision Tree: `max_depth`
- KNN: `k`
- Logistic Regression: regularization strength `C`

Training and cross-validation F1-scores are analyzed to study generalization behavior.

---

## 📊 Evaluation Metrics

### Classification Metrics
Computed on the **test set** using weighted averaging:
- Precision
- Recall
- F1-score
- Accuracy
- ROC–AUC (One-vs-Rest, weighted)

### Regression Metrics (Reconstructed)
Continuous predictions are reconstructed using **probability-weighted median class centers**, and evaluated using:
- MAE
- MSE
- RMSE
- R²  
(Reported separately for Training and Test sets)

---

## 📈 Outputs

The codes automatically generate:

- ✅ Journal-quality plots (PDF & PNG)
- ✅ Summary tables saved as:
  - Excel (`.xlsx`)
  - PDF (`.pdf`)
- ✅ Hyperparameter optimization curves
- ✅ Confusion matrices (optional extension)

All figures are publication-ready (300 dpi).

---

## 📁 Repository Structure
```text
├── F2BCMS.csv
├── f1random.py              # Random Forest F1-score analysis
├── Decision Tree.py         # Decision Tree F1-score analysis
├── knn.py                 # KNN F1-score analysis
├── logesticf1.py            # Logistic Regression F1-score analysis
├── FINAL_9Class_*.py        # Unified 9-class regression & metrics
├── *.pdf                    # Publication-ready figures and tables
├── *.xlsx                   # Metric summary tables
└── README.md
▶️ How to Run
Example (Random Forest F1 analysis):

bash
python f1random.py
Unified 9-class regression metrics:

bash
python FINAL_9Class_4Models_FullMetrics.py
📌 Scientific Motivation
Quantile-based classification provides a physically meaningful discretization of the proton structure function, reducing sensitivity to outliers and improving robustness. Increasing the number of classes (e.g. from 3 to 9) significantly reduces information loss and leads to improved reconstructed regression performance.

📄 Citation
If you use this code in your research, please cite:

S. Atashbar Tehrani and E. Astaraki,

Machine Learning for Predicting the Proton Structure Function F₂ in QCD,

(Manuscript in preparation / under review).

📬 Contact
For questions or collaborations:

Shahin Atashbar Tehrani – IPM & Persian Gulf University (atashbar@ipm.ir)
Elham Astaraki – Razi University(E-Mail: astaraki.elham@razi.ac.ir)
