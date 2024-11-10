from concurrent import futures
import grpc
import uuid

from models.model_manager import ModelManager
#from model_manager import ModelManager
from grpc.framework.interfaces.base import base

model_manager = ModelManager()

class ModelService(base.Servicer):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    # Функция для обучения модели
    def train_model(self, request, context):
        model_type = request.model_type
        hyperparams = request.hyperparams
        model_id = self.model_manager.train_model(model_type, hyperparams)
        return {"model_id": model_id}

    # Функция для получения доступных моделей
    def get_model_types(self, _):
        return self.model_manager.get_available_models()

    # Функция для прогнозирования
    def predict(self, request, context):
        model_id = request.model_id
        input_data = request.input_data
        prediction = self.model_manager.predict(model_id, input_data)
        if prediction is None:
            raise context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
        return {"prediction": prediction}

    # Функция для удаления модели
    def delete_model(self, request, context):
        model_id = request.model_id
        success = self.model_manager.delete_model(model_id)
        if not success:
            raise context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            
    # Функция для определения текущего статуса
    def service_status(self, _):
        return {"status": "running", "models": self.model_manager.get_available_models()}

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server is running on port 50051.")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()