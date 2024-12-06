# Домашняя работа №2 по MLOps

Выполнили: 
- Романенко Александр
- Староверова Анна
- Новицкий Олег

## Как запустить:
### Env
```python
pyenv virtualenv 3.9.5 mlopshw2   
pyenv activate mlopshw2 
poetry env use $(pyenv which python)
poetry shell
poetry install
```

### Rest:
    poetry run python api/main.py

### Streamlit:
    streamlit run dashboard/app.py

Сваггер открывается по ссылке http://127.0.0.1:8000/docs

<img width="669" alt="Снимок экрана 2024-11-11 в 23 50 36" src="https://github.com/user-attachments/assets/6f10d708-803b-4f48-b0f1-b1c7562c9312">


Как устроен gPRC:

Сама реализация находится в api/grpc. Исходный proto файл, на основании которого происходила инициализация, находится в папке protos. В файле main находится реализация Сервера, а Клиент и проверка работы методов находятся в файле example.ipynb

Для того чтобы запустить gRPC достаточно поднять сервер через Терминал (прогрузить файл main.py), далее запустить тетрадку example.ipynb. При этом используется пример данных, находящийся в папке data_example - его можно поменять на любые данные, в которых X размечен как фича, а y как таргет
