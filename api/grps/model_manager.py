import uuid
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class ModelManager:
    def __init__(self):
        # Словарь для хранения обученных моделей
        self.models = {}
        # Доступные классы моделей и их гиперпараметры
        self.model_classes = {
            "RandomForest": RandomForestRegressor,
            "LinearRegression": LinearRegression
        }
        # self.data = pd.read_csv(data_path)

    def transform_data(self):
        print('Entered fransform_data function')
        X, y = self.data.iloc[:, :-1], self.data.iloc[:, -1]
        return X, y

    def get_available_models(self):
        """Возвращает список доступных классов моделей."""
        return list(self.model_classes.keys())

    def train_model(self, model_type, hyperparams, data_path):
        """
        Обучает модель заданного типа с переданными гиперпараметрами.
        Возвращает уникальный ID модели.
        """
        if model_type not in self.model_classes:
            raise ValueError(f"Модель {model_type} не поддерживается.")

        # Создаем и настраиваем модель с переданными гиперпараметрами
        ModelClass = self.model_classes[model_type]
        model = ModelClass(**hyperparams)

        # Генерируем обучающие данные (замените на собственные данные)
        # X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        # X, y = self.transform_data()
        data = pd.read_csv(data_path)
        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        print('Successfully transformed data')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучаем модель
        model.fit(X_train, y_train)
        print('Trained model')

        # Генерируем уникальный ID для модели и сохраняем её
        model_id = str(uuid.uuid4())
        self.models[model_id] = model
        return model_id

    def predict(self, model_id, data_path):
        """
        Выполняет предсказание на основе входных данных для конкретной модели.
        """
        model = self.models.get(model_id)
        data = pd.read_csv(data_path)
        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        # X, y = self.transform_data()
        print('Successfully transformed data')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if not model:
            return None

        # Преобразуем входные данные в numpy-массив для предсказания
        # input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(X_test)
        return prediction.tolist(), mean_squared_error(y_test, prediction)
        # return prediction.tolist()  # Преобразуем в список для JSON-сериализации

    def delete_model(self, model_id):
        """
        Удаляет модель по её ID.
        """
        if model_id in self.models:
            del self.models[model_id]
            return True
        return False

    def list_models(self):
        """Возвращает список всех обученных моделей с их ID."""
        return list(self.models.keys())