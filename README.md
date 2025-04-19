#  SVM Parameter Optimization on Letter Recognition Dataset

This project implements **SVM parameter optimization** using `RandomizedSearchCV` on a multi-class dataset from the UCI Machine Learning Repository.

The notebook and code:
- Load the **Letter Recognition** dataset
- Create **10 different train-test splits** (70-30)
- Perform **SVM hyperparameter optimization** on each sample (with 100-step simulated convergence)
- Record the best performing configuration and test accuracy for each
- Visualize the **convergence graph** of the best-performing sample
- Save all results, tables, and a basic analytics summary

---

##  Dataset Used
- **Dataset**: [Letter Recognition](https://archive.ics.uci.edu/ml/datasets/letter+recognition)
- **Source**: UCI Machine Learning Repository
- **Samples**: 20,000
- **Features**: 16
- **Classes**: 26 (A–Z)

---

##  Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib

---

##  Optimization Settings

- **Model**: `SVC` (Support Vector Classification)
- **Parameter Tuning**: `RandomizedSearchCV` (10 combinations, 3-fold CV)
- **Parameters Tuned**:
  - Kernel: `['rbf', 'linear']`
  - C: `[0.1, 1, 10]`
  - Gamma: `['scale', 'auto']`
- **Convergence**: Simulated for 100 iterations by interpolating CV scores

---

##  Convergence Graph

The following plot shows the convergence behavior of the best-performing SVM configuration across 100 interpolated steps:

![image](https://github.com/user-attachments/assets/c127452f-163d-43a5-b2f2-5e8b306ae748)


---

##  Optimization Results Table

Summary of test accuracy and best SVM parameters for each of the 10 samples:

![image](https://github.com/user-attachments/assets/0669016d-8b21-4219-ab64-0041f2f2c9b1)


---

##  Data Analytics 

Basic insights from the dataset and optimization summary:

![image](https://github.com/user-attachments/assets/f9ce1a12-d0ff-478f-a695-0566084ce357)



---

