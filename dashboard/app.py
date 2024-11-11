import streamlit as st
import requests

st.title("Model Training Dashboard")

# Выбор модели
model_type = st.selectbox("Выберите модель", ["RandomForest", "LinearRegression"])
hyperparams = {}

if model_type == "RandomForest":
    hyperparams["n_estimators"] = st.slider("n_estimators", 10, 100, 10)
# elif model_type == "LinearRegression":
#     hyperparams["C"] = st.slider("C", 0.01, 1.0, 0.1)

data_path_train = st.text_input('Ввести путь к данным для обучения')

if st.button("Запустить обучение"):
    # response = requests.post("http://localhost:8000/train", json={"model_type": model_type, "hyperparams": hyperparams})

    response = requests.post("http://127.0.0.1:8000/train", json={"model_type": model_type, "hyperparams": hyperparams,
                                                                  "data_path": data_path_train})
    print(response)
    st.write(response)
    st.write(response.json())

model_id = st.text_input('Ввести id модели для получения предсказаний')
data_path_predict = st.text_input('Ввести путь к данным для получения предсказаний')

if st.button("Получить предсказания"):
    # response = requests.post("http://localhost:8000/train", json={"model_type": model_type, "hyperparams": hyperparams})

    response = requests.post(f"http://127.0.0.1:8000/predict/{model_id}", json={"model_id": model_id,
                                                                                "data_path": data_path_predict})
    print(response)
    st.write(response)
    st.write(response.json())
