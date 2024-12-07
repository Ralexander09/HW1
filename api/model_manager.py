# model_manager.py
import json
import os
import subprocess
import uuid
from io import BytesIO

import clearml
from clearml import Task, Logger

import joblib
import pandas as pd
from minio import Minio
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class ModelManager:
    def __init__(
            self, models_dir="saved_models", data_dir="data", bucket_name="my-bucket"
    ):
        self.models = {}
        self.models_dir = models_dir
        self.data_dir = data_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        # Инициализируем MinIO клиент
        self.minio_client = Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin123",
            secure=False,
        )

        # Проверяем наличие bucket, если нет - создаём
        found = self.minio_client.bucket_exists(bucket_name)
        if not found:
            self.minio_client.make_bucket(bucket_name)
        self.bucket_name = bucket_name

        self.model_classes = {
            "RandomForest": RandomForestRegressor,
            "LinearRegression": LinearRegression,
        }

        self.DEFAULT_DATA = [
            {"X": 230.1, "y": 22.1},
            {"X": 44.5, "y": 10.4},
            {"X": 17.2, "y": 9.3},
            {"X": 151.5, "y": 18.5},
            {"X": 180.8, "y": 12.9},
            {"X": 8.7, "y": 7.2},
            {"X": 57.5, "y": 11.8},
            {"X": 120.2, "y": 13.2},
            {"X": 8.6, "y": 4.8},
            {"X": 199.8, "y": 10.6},
            {"X": 66.1, "y": 8.6},
            {"X": 214.7, "y": 17.4},
            {"X": 23.8, "y": 9.2},
            {"X": 97.5, "y": 9.7},
            {"X": 204.1, "y": 19},
            {"X": 195.4, "y": 22.4},
            {"X": 67.8, "y": 12.5},
            {"X": 281.4, "y": 24.4},
            {"X": 69.2, "y": 11.3},
            {"X": 147.3, "y": 14.6},
            {"X": 218.4, "y": 18},
            {"X": 237.4, "y": 12.5},
            {"X": 13.2, "y": 5.6},
            {"X": 228.3, "y": 15.5},
            {"X": 62.3, "y": 9.7},
            {"X": 262.9, "y": 12},
            {"X": 142.9, "y": 15},
            {"X": 240.1, "y": 15.9},
            {"X": 248.8, "y": 18.9},
            {"X": 70.6, "y": 10.5},
            {"X": 292.9, "y": 21.4},
            {"X": 112.9, "y": 11.9},
            {"X": 97.2, "y": 9.6},
            {"X": 265.6, "y": 17.4},
            {"X": 95.7, "y": 9.5},
            {"X": 290.7, "y": 12.8},
            {"X": 266.9, "y": 25.4},
            {"X": 74.7, "y": 14.7},
            {"X": 43.1, "y": 10.1},
            {"X": 228, "y": 21.5},
            {"X": 202.5, "y": 16.6},
            {"X": 177, "y": 17.1},
            {"X": 293.6, "y": 20.7},
            {"X": 206.9, "y": 12.9},
            {"X": 25.1, "y": 8.5},
            {"X": 175.1, "y": 14.9},
            {"X": 89.7, "y": 10.6},
            {"X": 239.9, "y": 23.2},
            {"X": 227.2, "y": 14.8},
            {"X": 66.9, "y": 9.7},
            {"X": 199.8, "y": 11.4},
            {"X": 100.4, "y": 10.7},
            {"X": 216.4, "y": 22.6},
            {"X": 182.6, "y": 21.2},
            {"X": 262.7, "y": 20.2},
            {"X": 198.9, "y": 23.7},
            {"X": 7.3, "y": 5.5},
            {"X": 136.2, "y": 13.2},
            {"X": 210.8, "y": 23.8},
            {"X": 210.7, "y": 18.4},
            {"X": 53.5, "y": 8.1},
            {"X": 261.3, "y": 24.2},
            {"X": 239.3, "y": 15.7},
            {"X": 102.7, "y": 14},
            {"X": 131.1, "y": 18},
            {"X": 69, "y": 9.3},
            {"X": 31.5, "y": 9.5},
            {"X": 139.3, "y": 13.4},
            {"X": 237.4, "y": 18.9},
            {"X": 216.8, "y": 22.3},
            {"X": 199.1, "y": 18.3},
            {"X": 109.8, "y": 12.4},
            {"X": 26.8, "y": 8.8},
            {"X": 129.4, "y": 11},
            {"X": 213.4, "y": 17},
            {"X": 16.9, "y": 8.7},
            {"X": 27.5, "y": 6.9},
            {"X": 120.5, "y": 14.2},
            {"X": 5.4, "y": 5.3},
            {"X": 116, "y": 11},
            {"X": 76.4, "y": 11.8},
            {"X": 239.8, "y": 12.3},
            {"X": 75.3, "y": 11.3},
            {"X": 68.4, "y": 13.6},
            {"X": 213.5, "y": 21.7},
            {"X": 193.2, "y": 15.2},
            {"X": 76.3, "y": 12},
            {"X": 110.7, "y": 16},
            {"X": 88.3, "y": 12.9},
            {"X": 109.8, "y": 16.7},
            {"X": 134.3, "y": 11.2},
            {"X": 28.6, "y": 7.3},
            {"X": 217.7, "y": 19.4},
            {"X": 250.9, "y": 22.2},
            {"X": 107.4, "y": 11.5},
            {"X": 163.3, "y": 16.9},
            {"X": 197.6, "y": 11.7},
            {"X": 184.9, "y": 15.5},
            {"X": 289.7, "y": 25.4},
            {"X": 135.2, "y": 17.2},
            {"X": 222.4, "y": 11.7},
            {"X": 296.4, "y": 23.8},
            {"X": 280.2, "y": 14.8},
            {"X": 187.9, "y": 14.7},
            {"X": 238.2, "y": 20.7},
            {"X": 137.9, "y": 19.2},
            {"X": 25, "y": 7.2},
            {"X": 90.4, "y": 8.7},
            {"X": 13.1, "y": 5.3},
            {"X": 255.4, "y": 19.8},
            {"X": 225.8, "y": 13.4},
            {"X": 241.7, "y": 21.8},
            {"X": 175.7, "y": 14.1},
            {"X": 209.6, "y": 15.9},
            {"X": 78.2, "y": 14.6},
            {"X": 75.1, "y": 12.6},
            {"X": 139.2, "y": 12.2},
            {"X": 76.4, "y": 9.4},
            {"X": 125.7, "y": 15.9},
            {"X": 19.4, "y": 6.6},
            {"X": 141.3, "y": 15.5},
            {"X": 18.8, "y": 7},
            {"X": 224, "y": 11.6},
            {"X": 123.1, "y": 15.2},
            {"X": 229.5, "y": 19.7},
            {"X": 87.2, "y": 10.6},
            {"X": 7.8, "y": 6.6},
            {"X": 80.2, "y": 8.8},
            {"X": 220.3, "y": 24.7},
            {"X": 59.6, "y": 9.7},
            {"X": 0.7, "y": 1.6},
            {"X": 265.2, "y": 12.7},
            {"X": 8.4, "y": 5.7},
            {"X": 219.8, "y": 19.6},
            {"X": 36.9, "y": 10.8},
            {"X": 48.3, "y": 11.6},
            {"X": 25.6, "y": 9.5},
            {"X": 273.7, "y": 20.8},
            {"X": 43, "y": 9.6},
            {"X": 184.9, "y": 20.7},
            {"X": 73.4, "y": 10.9},
            {"X": 193.7, "y": 19.2},
            {"X": 220.5, "y": 20.1},
            {"X": 104.6, "y": 10.4},
            {"X": 96.2, "y": 11.4},
            {"X": 140.3, "y": 10.3},
            {"X": 240.1, "y": 13.2},
            {"X": 243.2, "y": 25.4},
            {"X": 38, "y": 10.9},
            {"X": 44.7, "y": 10.1},
            {"X": 280.7, "y": 16.1},
            {"X": 121, "y": 11.6},
            {"X": 197.6, "y": 16.6},
            {"X": 171.3, "y": 19},
            {"X": 187.8, "y": 15.6},
            {"X": 4.1, "y": 3.2},
            {"X": 93.9, "y": 15.3},
            {"X": 149.8, "y": 10.1},
            {"X": 11.7, "y": 7.3},
            {"X": 131.7, "y": 12.9},
            {"X": 172.5, "y": 14.4},
            {"X": 85.7, "y": 13.3},
            {"X": 188.4, "y": 14.9},
            {"X": 163.5, "y": 18},
            {"X": 117.2, "y": 11.9},
            {"X": 234.5, "y": 11.9},
            {"X": 17.9, "y": 8},
            {"X": 206.8, "y": 12.2},
            {"X": 215.4, "y": 17.1},
            {"X": 284.3, "y": 15},
            {"X": 50, "y": 8.4},
            {"X": 164.5, "y": 14.5},
            {"X": 19.6, "y": 7.6},
            {"X": 168.4, "y": 11.7},
            {"X": 222.4, "y": 11.5},
            {"X": 276.9, "y": 27},
            {"X": 248.4, "y": 20.2},
            {"X": 170.2, "y": 11.7},
            {"X": 276.7, "y": 11.8},
            {"X": 165.6, "y": 12.6},
            {"X": 156.6, "y": 10.5},
            {"X": 218.5, "y": 12.2},
            {"X": 56.2, "y": 8.7},
            {"X": 287.6, "y": 26.2},
            {"X": 253.8, "y": 17.6},
            {"X": 205, "y": 22.6},
            {"X": 139.5, "y": 10.3},
            {"X": 191.1, "y": 17.3},
            {"X": 286, "y": 15.9},
            {"X": 18.7, "y": 6.7},
            {"X": 39.5, "y": 10.8},
            {"X": 75.5, "y": 9.9},
            {"X": 17.2, "y": 5.9},
            {"X": 166.8, "y": 19.6},
            {"X": 149.7, "y": 17.3},
            {"X": 38.2, "y": 7.6},
            {"X": 94.2, "y": 9.7},
            {"X": 177, "y": 12.8},
            {"X": 283.6, "y": 25.5},
            {"X": 232.1, "y": 13.4},
        ]

    def get_available_models(self):
        return list(self.model_classes.keys())

    def train_model(self, model_type, hyperparams, data=None):
        if model_type not in self.model_classes:
            raise ValueError(f"Model type '{model_type}' is not supported.")

        # Создаём уникальный ID модели
        model_id = str(uuid.uuid4())

        # Инициализируем задачу ClearML
        task = Task.init(
            project_name='MLOps_HW2',
            task_name='Model' + ' ' + model_id,
            tags=[model_type])

        # Определяем пути для данных
        data_path = os.path.join(self.data_dir, f"train_{model_id}.json")

        # Загружаем путь к данным в задачу ClearML
        task.upload_artifact(name='data_train.path', artifact_object=data_path)

        # Загружаем гиперпараметры в задачу ClearML
        task.connect(hyperparams)

        # Сохраняем датасет
        with open(data_path, "w") as f:
            json.dump(data if data is not None else self.DEFAULT_DATA, f)

        try:
            # Добавляем датасет под контроль DVC
            subprocess.run(
                ["dvc", "add", data_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Добавляем только .dvc файл и .gitignore в Git
            subprocess.run(
                ["git", "add", f"{data_path}.dvc", ".gitignore"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Коммитим изменения
            subprocess.run(
                ["git", "commit", "-m", f"Add training dataset for model {model_id}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Пушим датасет в удалённое хранилище
            subprocess.run(
                ["dvc", "push"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode().strip()
            raise RuntimeError(f"Ошибка при выполнении команд DVC/Git: {error_message}")

        # Загружаем данные
        df = pd.read_json(data_path)
        if df.shape[1] < 2:
            raise ValueError(
                "Data must contain at least two columns (features and target)."
            )

        # Загружаем сами данные в задачу ClearML
        task.upload_artifact(name='data_train.data', artifact_object=df)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        ModelClass = self.model_classes[model_type]
        model = ModelClass(**hyperparams)
        model.fit(X, y)

        prediction = model.predict(X)
        mse = round(mean_squared_error(y, prediction), 3) if y is not None else None
        max_err = round(max_error(y, prediction), 3) if y is not None else None
        r2 = round(r2_score(y, prediction), 3) if y is not None else None
        mape = round(mean_absolute_percentage_error(y, prediction), 3) if y is not None else None

        # Логируем метрики на обучении в ClearML
        log = Logger.current_logger()
        log.report_single_value(name='Train MSE', value=mse)
        log.report_single_value(name='Train Maximum Error', value=max_err)
        log.report_single_value(name='Train Determination coef.', value=r2)
        log.report_single_value(name='Train MAPE', value=mape)
        log.report_single_value(name='Train Rows', value=X.shape[0])
        log.report_single_value(name='Train Ratio', value=round(y.mean(), 3))

        self.models[model_id] = model

        # Сохраняем модель локально
        model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
        joblib.dump(model, model_path)

        # Сохраняем модель в ClearML
        model.save_model(f"{model_id}.cbm")

        # Загружаем модель в MinIO
        try:
            self.minio_client.fput_object(
                self.bucket_name, f"{model_id}.joblib", model_path
            )
        except Exception as e:
            os.remove(model_path)
            raise RuntimeError(f"Ошибка при загрузке модели в MinIO: {str(e)}")

        # Удаляем локальную копию модели
        os.remove(model_path)

        return model_id

    def load_model(self, model_id):
        if model_id in self.models:
            return self.models[model_id]
        # Попытаемся загрузить из MinIO
        try:
            response = self.minio_client.get_object(
                self.bucket_name, f"{model_id}.joblib"
            )
            model_bytes = response.read()
            model = joblib.load(BytesIO(model_bytes))
            self.models[model_id] = model
            return model
        except Exception:
            return None

    def predict(self, model_id, data=None):
        model = self.load_model(model_id)
        if not model:
            raise KeyError("Model not found")

        # Загружаем задачу в ClearML для выбранного ID модели
        task = Task.get_task(project_name='MLOps_HW2', task_name='Model' + ' ' + model_id)

        # Загружаем данные
        if data is not None:
            df = pd.DataFrame(data)
        else:
            # Можно взять дефолтный датасет
            df = pd.DataFrame(self.DEFAULT_DATA)

        if df.empty:
            raise ValueError("No data provided for prediction.")

        # Загружаем сами данные в задачу ClearML
        task.upload_artifact(name='data_test.data', artifact_object=df)

        has_y = "y" in df.columns
        if has_y:
            X = df.drop(columns=["y"])
            y_true = df["y"]
        else:
            X = df
            y_true = None

        prediction = model.predict(X)
        mse = round(mean_squared_error(y_true, prediction), 3) if y_true is not None else None
        max_err = round(max_error(y_true, prediction), 3) if y_true is not None else None
        r2 = round(r2_score(y_true, prediction), 3) if y_true is not None else None
        mape = round(mean_absolute_percentage_error(y_true, prediction), 3) if y_true is not None else None

        # Логируем метрики на обучении в ClearML
        log = Logger.current_logger()
        log.report_single_value(name='Predict MSE', value=mse)
        log.report_single_value(name='Predict Maximum Error', value=max_err)
        log.report_single_value(name='Predict Determination coef.', value=r2)
        log.report_single_value(name='Predict MAPE', value=mape)
        log.report_single_value(name='Predict Rows', value=X.shape[0])
        log.report_single_value(name='Predict Ratio', value=round(y_true.mean(), 3))

        return prediction.tolist(), mse

    def delete_model(self, model_id):
        # Удаляем модель из памяти, если она загружена
        if model_id in self.models:
            del self.models[model_id]
        # Удаляем модель из MinIO
        try:
            self.minio_client.remove_object(self.bucket_name, f"{model_id}.joblib")
            return True
        except Exception:
            return False

    def list_models(self):
        # Получаем список объектов в bucket и фильтруем модели
        objects = self.minio_client.list_objects(self.bucket_name, recursive=True)
        model_ids = [
            obj.object_name.replace(".joblib", "")
            for obj in objects
            if obj.object_name.endswith(".joblib")
        ]
        return model_ids
