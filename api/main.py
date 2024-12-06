import json

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
    data: list = None  # Field for JSON data

    class Config:
        extra = "forbid"


@app.post("/train")
def train_model(request: TrainRequest):
    """
    Train a new model with the specified type, hyperparameters, and data.
    If data is not provided, default JSON data is used.
    """
    model_type = request.model_type
    hyperparams = request.hyperparams
    data = request.data  # Can be None

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
    Retrieve a list of available model types.
    """
    try:
        available_models = model_manager.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return available_models


class PredictRequest(BaseModel):
    data: list = None  # Field for JSON data

    class Config:
        extra = "forbid"


@app.post("/predict/{model_id}")
def predict(model_id: str, request: PredictRequest):
    """
    Make a prediction using the specified model and data.
    If data is not provided, default JSON data is used.
    """
    data = request.data  # Can be None

    try:
        prediction, mse = model_manager.predict(model_id=model_id, data=data)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction failed")

    return {"prediction": prediction, "MSE": mse}


@app.delete("/model/{model_id}")
def delete_model(model_id: str):
    """
    Delete the specified model.
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
    Get the current status of the service and list of models.
    """
    try:
        models = model_manager.list_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "running", "models": models}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
