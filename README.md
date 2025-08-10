# 🛒 Store Sales Forecasting (1st Place Solution)


<img width="2048" height="640" alt="upscaled_market_2048x2048" src="https://github.com/user-attachments/assets/1b353996-5231-486c-8ff3-4e8e72259462" />


This repository contains the complete pipeline used for forecasting store-category-level sales using LightGBM with GPU acceleration. The solution applies cross-validation, log transformations, and robust ensemble strategies to produce accurate predictions.

## 📂 Project Structure
store-sales-forecasting-first-place-solution.ipynb – Jupyter notebook with full training and prediction pipeline.

<code>README.md</code>– Project documentation (you’re reading it!).

<code>data/</code> – Assumed folder for input data (X, y, X_test, etc.).

<code>submission.csv</code> – Final submission file with predicted sales.

## 🚀 Key Features
✅ LightGBM with GPU acceleration

✅ 6-Fold Cross-Validation

✅ Log-Transformation of Targets for improved regression accuracy

✅ Early Stopping & Logging to prevent overfitting

✅ Ensembling with Mean & Median predictions across folds

✅ GroupBy Aggregation for per-ID predictions

## 🔧 How to Use
1. Environment Setup
Make sure you have the required Python packages:

```bash
pip install lightgbm scikit-learn pandas numpy tqdm
```
For GPU support, ensure LightGBM is compiled with GPU enabled.

2. Prepare Data
Make sure you have the following data objects available in the notebook:

<code>X, y</code>: Training features and target

<code>X_test</code>: Test features

<code>cat_list</code>: List of categorical feature names

## 3. Run the Notebook
Launch the notebook and execute all cells. The training will perform 6-fold CV, and output validation RMSE. Predictions will be saved to a submission DataFrame.

## 📈 Output
submission.csv: File with predictions per ID

Format:

```cs
ID,target
year_week_2021_3_1,187.92
year_week_2021_3_2,211.56
...

```
🧠 Model Details
Parameters
```python
param = {
    "verbose": -100,
    "metric": "rmse",
    "device_type": "gpu",
    "gpu_use_dp": False,
    "random_state": 42
}
```
## Training Strategy
6-fold KFold cross-validation with shuffling and fixed seed

Log1p transformation on the target

Ensemble of models’ predictions via mean and median

## 📌 Notes
If not run on Kaggle or Colab, install packages first.

All predictions are inverse-transformed with np.expm1 to revert log scaling.

Clip predictions to ensure they are non-negative.

## 🏁 Final Step
To generate your final predictions:

```python
submission.to_csv("submission.csv", index=False)
```
📬 Contact
For questions or feedback, feel free to reach out or open an issue!

## Don't Just Have A Good Day. Have a Great Day!

