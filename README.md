# Machine Learning for Predicting the Proton Structure Function F₂ in QCD

Shahin Atashbar Tehrani¹² , Elham Astaraki³,Fatemeh Arbabifar*
¹ School of Particles and Accelerators, Institute for Research in Fundamental Sciences (IPM), Tehran, Iran.  
² Department of Physics, Faculty of Nano and Bio Science and Technology, Persian Gulf University, Bushehr, Iran.
³ Department of Physics, Razi University, Kermanshah, Iran.  
*Department of Physics Education, Farhangian University, Tehran, Iran.

📝 Abstract
We present a comparative study of four supervised machine learning (ML) regression algorithms—Support Vector Regression (SVR), Gradient Boosting (GBoost), Gaussian Process Regression (GPR), and Multilayer Perceptron (MLP) for predicting the proton structure function $F_2^p(x, Q^2)$ based on the high-precision BCDMS experimental data. Unlike traditional approaches that rely on solving the DGLAP evolution equations, our framework adopts a data-driven strategy to capture the complex non-linear dynamics of partonic structures. 

To ensure statistical robustness, we implemented a $k$-fold cross-validation pipeline and performed detailed hyperparameter optimization. Our results indicate that the Neural Network (MLP) and GPR models achieve superior predictive accuracy, with $R^2$ scores exceeding 0.72. Specifically, the MLP model demonstrates the highest sensitivity to non-linear gradients, while SVR exhibits the greatest stability against experimental uncertainties. The convergence of training and validation metrics confirms that the ML models effectively learn the underlying QCD physics without overfitting to statistical noise. This work demonstrates the potential of ML-based regression as a complementary tool for structure function analysis and kinematic extrapolation in high-energy physics.

---

🗂 Project Overview
This repository contains the full implementation of the models discussed in the paper. The project is structured to allow reproducibility of the results using the BCDMS dataset.


---

## Dataset

- File: `F2BCMS.csv`  
- Inputs: `x`, `Q^2` (log‑transformed to `logx`, `logQ2`)  
- Target: `F2_exp`

---

### Models Implemented:
- **SVR**: Focused on stability and noise resistance.
- **GBoost**: Optimized for capturing complex patterns via ensemble learning.
- **GPR**: Utilizing Gaussian processes for probabilistic regression.
- **MLP**: A deep learning approach for high-sensitivity non-linear mapping.

## 📊 Evaluation & Metrics
The performance of the models is evaluated using standard regression criteria:
- **$R^2$ Score** (exceeding 0.72 for top models)
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

> **Technical Note:**
"Although the primary objective and core of this project is based on Regression to predict the continuous values of the proton structure function $F_2^p(x, Q^2), the inclusion of classification metrics (such as F1-score, ROC-AUC, and Recall) in certain diagnostic scripts is intentional and serves specific analytical purposes.
Retaining these metrics in the code is instrumental for deeper personalized analysis; for instance, they are employed to evaluate the model’s success in classifying distinct data regions. This approach allows us to assess the precision with which the model distinguishes between high-$F_2^p(x, Q^2)and low-$F_2^p(x, Q^2) regions or its capability in identifying boundary points within specific kinematic spaces. In practice, these metrics serve strictly as auxiliary tools for bin-wise performance analysis and for evaluating model behavior across different data clusters. They should not be interpreted as the primary physical output or the final criteria for physical validation. The ultimate metric for the scientific validation of this project remains the regression-based quantities and the precise alignment of the model outputs with the BCDMS experimental data."

## Outputs

- Publication‑ready plots (`.png`, `.pdf`)  
- Summary tables (`.xlsx`)  
- Learning curves, residuals  
- Optional confusion matrices

---

## Repository Structure
├── F2BCMS.csv

├── svr.py

├── gboost.py

├── NN.py

├── GPR.py

├── R2.py

├── plots/

├── tables/

└── RE
---

## Running

---

## Citation

S. Atashbar Tehrani and E. Astaraki,  
*Machine Learning for Predicting the Proton Structure Function F₂ in QCD*,  
Manuscript in preparation / under review.	arXiv:2606.06414

---

## Contact

**Shahin Atashbar Tehrani** — atashbar@ipm.ir  
**Elham Astaraki** — astaraki.elham@razi.ac.ir

