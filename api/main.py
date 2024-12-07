# main.py
from fastapi import FastAPI, HTTPException
from model_manager import ModelManager
from pydantic import BaseModel

app = FastAPI()
model_manager = ModelManager()


@app.get("/")
async def root():
    return {"message": "Welcome to the Model Training API"}


class TrainRequest(BaseModel):
    model_type: str
    hyperparams: dict
    data: list = None  # Поле для JSON данных

    class Config:
        extra = "forbid"


@app.post("/train")
def train_model(request: TrainRequest):
    """
    Обучить новую модель с указанным типом, гиперпараметрами и данными.
    Если данные не предоставлены, используются дефолтные JSON данные.
    """
    model_type = request.model_type
    hyperparams = request.hyperparams
    data = request.data  # Может быть None

    try:
        model_id = model_manager.train_model(
            model_type=model_type,
            hyperparams=hyperparams,
            data=data,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"model_id": model_id}


@app.get("/models")
def get_model_types():
    """
    Получить список доступных типов моделей.
    """
    try:
        available_models = model_manager.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"available_models": available_models}


class PredictRequest(BaseModel):
    data: list = None  # Поле для JSON данных

    class Config:
        extra = "forbid"


@app.post("/predict/{model_id}")
def predict(model_id: str, request: PredictRequest):
    """
    Сделать предсказание с использованием указанной модели и данных.
    Если данные не предоставлены, используются дефолтные JSON данные.
    """
    data = request.data  # Может быть None

    try:
        prediction, mse = model_manager.predict(model_id=model_id, data=data)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction failed")

    response = {"prediction": prediction}
    if mse is not None:
        response["MSE"] = mse

    return response


@app.delete("/model/{model_id}")
def delete_model(model_id: str):
    """
    Удалить указанную модель.
    """
    try:
        success = model_manager.delete_model(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted"}


@app.get("/status")
def service_status():
    """
    Получить текущий статус сервиса и список моделей.
    """
    try:
        models = model_manager.list_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "running", "models": models}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
