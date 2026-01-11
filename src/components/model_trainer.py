import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class ModelTrainer:

    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    def initiate_model_trainer(self, train_array, test_array):

        X_train, y_train = train_array[:, :-1], train_array[:, -1]
        X_test, y_test = test_array[:, :-1], test_array[:, -1]

        models = {
            "LinearRegression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
            "CatBoost": CatBoostRegressor(verbose=False),
            "AdaBoost": AdaBoostRegressor()
        }

        best_model = None
        best_r2 = -999

        for model_name, model in models.items():

            with mlflow.start_run(run_name=model_name, nested=True):

                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_mae, train_rmse, train_r2 = self.evaluate(y_train, y_train_pred)
                test_mae, test_rmse, test_r2 = self.evaluate(y_test, y_test_pred)

                # Log hyperparameters
                mlflow.log_param("model", model_name)

                # Log metrics
                mlflow.log_metric("train_rmse", train_rmse)
                mlflow.log_metric("test_rmse", test_rmse)
                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("test_r2", test_r2)
                mlflow.log_metric("test_mae", test_mae)

                # Log model
                mlflow.sklearn.log_model(model, "model")

                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model = model

        return best_model, {"best_r2": best_r2}
