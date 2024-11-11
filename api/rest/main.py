from fastapi import FastAPI, HTTPException, Request
from models import model_manager
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import json

app = FastAPI()
model_manager = model_manager.ModelManager('Advertising.csv')

@app.get("/")
async def root():
    return {"message": "Welcome to the Model Training API"}

class TrainRequest(BaseModel):
    model_type: str
    hyperparams: dict

@app.post("/train")
def train_model(request: TrainRequest):
    # model_type: str, hyperparams: dict
    # print(request)
    # data = json.loads(request)
    # model_type = data['model_type']
    # hyperparams = data['hyperparams']

    model_type = request.model_type
    hyperparams = request.hyperparams

    model_id = model_manager.train_model(model_type, hyperparams)
    return {"model_id": model_id}
    # return JSONResponse(content={"model_id": model_id}, status_code=200)

@app.get("/models")
def get_model_types():
    return model_manager.get_available_models()

@app.post("/predict/{model_id}")
def predict(model_id: str):
    prediction, mse = model_manager.predict(model_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"prediction": prediction, "MSE": mse}

@app.delete("/model/{model_id}")
def delete_model(model_id: str):
    success = model_manager.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted"}

@app.get("/status")
def service_status():
    return {"status": "running", "models": model_manager.list_models()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="http://127.0.0.1:8000", port=8000)
