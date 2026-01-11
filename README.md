Student Performance Evaluation – MLflow Project

This project builds a complete machine learning pipeline to predict student performance using multiple regression models.
It uses MLflow to track experiments, compare models, and store trained models.

The pipeline automatically:
Loads and preprocesses student data
Trains multiple ML models
Evaluates each model
Logs all results to MLflow
Stores trained models for comparison

This allows easy selection of the best-performing model.

Models Used
The following models are trained and evaluated:
Linear Regression
Lasso
Ridge
K-Nearest Neighbors
Decision Tree
Random Forest
XGBoost
CatBoost
AdaBoost

Each model is logged as a separate MLflow run.

🔹 How to Run

Activate environment:
.venv\Scripts\activate


Start MLflow:
mlflow ui


Run training:
python -m src.pipeline.train_pipeline


Open MLflow in browser:
http://127.0.0.1:5000

What MLflow Tracks
For each model MLflow logs:
RMSE
MAE
R² score
Trained model

We can compare all models and choose the best one.

Tools Used:-
Python
Scikit-learn
XGBoost
CatBoost
MLflow

