# app.py
import json

import requests
import streamlit as st

# Default JSON data для отображения (опционально)
DEFAULT_DATA = [
    {"X": 230.1, "y": 22.1},
    {"X": 44.5, "y": 10.4},
    {"X": 232.1, "y": 13.4},
]

# Заголовок приложения
st.title("Model Training Dashboard")

# Выбор модели
model_type = st.selectbox("Select Model Type", ["RandomForest", "LinearRegression"])
hyperparams = {}

# Настройка гиперпараметров в зависимости от выбранной модели
if model_type == "RandomForest":
    hyperparams["n_estimators"] = st.slider(
        "Number of Trees (n_estimators)", 10, 100, 10
    )
elif model_type == "LinearRegression":
    hyperparams["fit_intercept"] = st.selectbox("Fit Intercept", [True, False])

st.markdown("## Train a Model")

# Опция использования дефолтных данных или загрузки JSON
use_default_train_data = st.checkbox("Use Default Training Data", value=True)

if not use_default_train_data:
    uploaded_train_file = st.file_uploader(
        "Upload JSON File for Training", type=["json"]
    )
    if uploaded_train_file is not None:
        try:
            train_data = json.load(uploaded_train_file)
            st.write("Training Data:")
            st.json(train_data)
        except Exception as e:
            st.error(f"Error loading JSON: {e}")
            train_data = None
else:
    train_data = None  # API использует дефолтные данные

# Кнопка для начала обучения
if st.button("Train Model"):
    try:
        if not use_default_train_data and uploaded_train_file is not None:
            payload = {
                "model_type": model_type,
                "hyperparams": hyperparams,
                "data": train_data,
            }
        else:
            payload = {
                "model_type": model_type,
                "hyperparams": hyperparams,
                # Поле 'data' отсутствует; API использует дефолтные данные
            }

        response = requests.post(
            "http://127.0.0.1:8000/train",
            json=payload,
        )
        response.raise_for_status()  # Проверка на HTTP ошибки
        result = response.json()
        st.success("Model trained successfully!")
        st.write(f"Model ID: {result['model_id']}")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error: {http_err}")
        if response.content:
            try:
                st.write(response.json())
            except ValueError:
                st.write(response.text)
    except Exception as err:
        st.error(f"An error occurred: {err}")

st.markdown("---")  # Разделитель

st.markdown("## Make Predictions")

# Опция использования дефолтных данных или загрузки JSON для предсказаний
use_default_predict_data = st.checkbox("Use Default Prediction Data", value=True)

if not use_default_predict_data:
    uploaded_predict_file = st.file_uploader(
        "Upload JSON File for Prediction", type=["json"]
    )
    if uploaded_predict_file is not None:
        try:
            predict_data = json.load(uploaded_predict_file)
            st.write("Prediction Data:")
            st.json(predict_data)
        except Exception as e:
            st.error(f"Error loading JSON: {e}")
            predict_data = None
else:
    predict_data = None  # API использует дефолтные данные

# Поле ввода для Model ID
model_id = st.text_input("Enter Model ID for Prediction")

# Кнопка для начала предсказания
if st.button("Get Predictions"):
    if not model_id:
        st.error("Please enter a Model ID.")
    else:
        try:
            if not use_default_predict_data and uploaded_predict_file is not None:
                payload = {
                    "data": predict_data,
                }
            else:
                payload = {
                    # Поле 'data' отсутствует; API использует дефолтные данные
                }

            response = requests.post(
                f"http://127.0.0.1:8000/predict/{model_id}",
                json=payload,
            )
            response.raise_for_status()  # Проверка на HTTP ошибки
            result = response.json()
            if result["prediction"] is None:
                st.error("Model not found or prediction failed.")
            else:
                st.success("Predictions obtained successfully!")
                st.write(f"Predictions: {result['prediction']}")
                if "MSE" in result:
                    st.write(f"MSE: {result['MSE']}")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error: {http_err}")
            if response.content:
                try:
                    st.write(response.json())
                except ValueError:
                    st.write(response.text)
        except Exception as err:
            st.error(f"An error occurred: {err}")

st.markdown("---")  # Разделитель

st.markdown("## Delete a Model")

# Поле ввода для Model ID для удаления
delete_model_id = st.text_input("Enter Model ID to Delete")

# Кнопка для удаления модели
if st.button("Delete Model"):
    if not delete_model_id:
        st.error("Please enter a Model ID to delete.")
    else:
        try:
            response = requests.delete(f"http://127.0.0.1:8000/model/{delete_model_id}")
            response.raise_for_status()  # Проверка на HTTP ошибки
            result = response.json()
            st.success(f"Model with ID {delete_model_id} deleted successfully!")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error: {http_err}")
            if response.content:
                try:
                    st.write(response.json())
                except ValueError:
                    st.write(response.text)
        except Exception as err:
            st.error(f"An error occurred: {err}")

st.markdown("---")  # Разделитель

st.markdown("## Service Status")

# Кнопка для проверки статуса сервиса
if st.button("Check Service Status"):
    try:
        response = requests.get("http://127.0.0.1:8000/status")
        response.raise_for_status()
        status = response.json()
        st.write(f"Service Status: {status['status']}")
        st.write("List of Models:")
        for model in status.get("models", []):
            st.write(f"- {model}")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error: {http_err}")
        if response.content:
            try:
                st.write(response.json())
            except ValueError:
                st.write(response.text)
    except Exception as err:
        st.error(f"An error occurred: {err}")
