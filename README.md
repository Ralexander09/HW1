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
pip install "dvc[s3]"
```
### Minio
    docker run -p 9000:9000 -p 9090:9090 \
    -e "MINIO_ROOT_USER=minioadmin" \
    -e "MINIO_ROOT_PASSWORD=minioadmin123" \
    quay.io/minio/minio server /data --console-address ":9090"

Далее переходим на http://127.0.0.1:9090 login minioadmin, пароль minioadmin123 и создаем бакет my-bucket (если его нет)

Далее я инициализировал dvc
### DVC
    dvc init
    git add .dvc .gitignore
    git commit -m "Initialize DVC"

    dvc remote add -d minio s3://my-bucket
    dvc remote modify minio endpointurl http://127.0.0.1:9000
    dvc remote modify minio access_key_id minioadmin
    dvc remote modify minio secret_access_key minioadmin123
    dvc remote modify minio region us-east-1
    dvc remote modify minio use_ssl false

тут добавляем и отправляем данные в minio
    dvc add /Users/oanovitskij/Desktop/hse/mlops_hw2/HW1/data/
    git add /Users/oanovitskij/Desktop/hse/mlops_hw2/HW1/data.dvc
    git commit -m "Добавлены данные с использованием DVC"
    dvc push
    
### ClearML
#### Windows
Создаем директории для данных и логов при их отсутствии:
```python
mkdir c:\opt\clearml\data
mkdir c:\opt\clearml\logs
```
    
Запускаем docker-compose:
```python
docker-compose -f clearml\docker-compose-win10.yml up
```

#### MacOS
Создаем директории для данных и логов при их отсутствии:
```python
sudo mkdir -p /opt/clearml/data/elastic_7
sudo mkdir -p /opt/clearml/data/mongo_4/db
sudo mkdir -p /opt/clearml/data/mongo_4/configdb        
sudo mkdir -p /opt/clearml/data/redis
sudo mkdir -p /opt/clearml/logs
sudo mkdir -p /opt/clearml/config
sudo mkdir -p /opt/clearml/data/fileserver
```
  
Запускаем docker-compose:
```python
docker-compose -f clearml\docker-compose.yml up
```

Сервер открывается по ссылке:
http://localhost:8080/

### Rest:
    poetry run python api/main.py

### Streamlit:
    streamlit run dashboard/app.py

Сваггер открывается по ссылке http://127.0.0.1:8000/docs

<img width="669" alt="Снимок экрана 2024-11-11 в 23 50 36" src="https://github.com/user-attachments/assets/6f10d708-803b-4f48-b0f1-b1c7562c9312">


### gPRC:

Сама реализация находится в api/grpc. Исходный proto файл, на основании которого происходила инициализация, находится в папке protos. В файле main находится реализация Сервера, а Клиент и проверка работы методов находятся в файле example.ipynb

Для того чтобы запустить gRPC достаточно поднять сервер через Терминал (прогрузить файл main.py), далее запустить тетрадку example.ipynb. При этом используется пример данных, находящийся в папке data_example - его можно поменять на любые данные, в которых X размечен как фича, а y как таргет


## Examples

```
curl -X POST http://127.0.0.1:8000/train \
     -H "Content-Type: application/json" \
     -d '{
           "model_type": "LinearRegression",
           "hyperparams": {
               "fit_intercept": true
           },
           "data": [
               {"X": 230.1, "y": 22.1},
               {"X": 44.5, "y": 10.4}
           ]
         }'
```

```
curl -X POST http://127.0.0.1:8000/predict/MODEL_ID \
     -H "Content-Type: application/json" \
     -d '{
           "data": [
               {"X": 150.0},
               {"X": 200.0}
           ]
         }'

```

```
curl -X 'GET' \
  'http://127.0.0.1:8000/models' \
  -H 'accept: application/json'
```

```
curl -X 'GET' \
  'http://127.0.0.1:8000/status' \
  -H 'accept: application/json'
```

```
curl -X 'DELETE' \
  'http://127.0.0.1:8000/model/MODEL_ID' \
  -H 'accept: application/json'
```
