import mlflow
import mlflow.sklearn
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


class TrainPipeline:
    def run_pipeline(self):
        logging.info("Training pipeline started")

        # Force MLflow to use same database as mlflow ui
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("students_evaluation_experiment")

        with mlflow.start_run():   # 🔥 MLflow Run starts
            # ===== Data Ingestion =====
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train: {train_path}, Test: {test_path}")

            mlflow.log_param("train_path", train_path)
            mlflow.log_param("test_path", test_path)

            # ===== Data Transformation =====
            transformation = DataTransformation()
            train_array, test_array, _ = transformation.initiate_data_transformation(
                train_path, test_path
            )
            logging.info("Data transformation completed.")

            # ===== Model Training =====
            trainer = ModelTrainer()
            model, metrics = trainer.initiate_model_trainer(train_array, test_array)
            logging.info("Model training completed.")

            # ===== Log Metrics =====
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # ===== Log Model =====
            mlflow.sklearn.log_model(model, "model")

            logging.info("Training pipeline completed successfully")




if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
