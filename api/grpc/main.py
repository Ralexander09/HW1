import uuid
from concurrent import futures

import grpc
import main_pb2
import main_pb2_grpc

# from models.model_manager import ModelManager
from model_manager import ModelManager

model_manager = ModelManager()


class ModelServiceStub(main_pb2_grpc.ModelServiceStub):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    # Функция для обучения модели
    def train_model(self, request, context):
        model_type = request.model_type
        hyperparams = request.hyperparams
        data_path = request.data_path
        model_id = self.model_manager.train_model(model_type, hyperparams, data_path)
        output = {"model_id": model_id}
        return main_pb2.train_response(**output)

    # Функция для получения доступных моделей
    def get_model_types(self, request, context):
        output = {"model_types": self.model_manager.get_available_models()}
        return main_pb2.model_list(**output)

    # Функция для прогнозирования
    def predict(self, request, context):
        model_id = request.model_id
        data_path = request.data_path
        prediction, mse = self.model_manager.predict(model_id, data_path)
        if prediction is None:
            raise context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")

        output = {"prediction": prediction, "MSE": mse}
        return main_pb2.predict_response(**output)

    # Функция для удаления модели
    def delete_model(self, request, context):
        model_id = request.model_id
        success = self.model_manager.delete_model(model_id)
        if not success:
            raise context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")

        output = {"status": "deleted"}
        return main_pb2.delete_response(**output)

    # Функция для определения текущего статуса
    def service_status(self, request, context):
        output = {"status": "running", "models": self.model_manager.list_models()}
        return main_pb2.status_response(**output)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    main_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServiceStub(model_manager), server
    )
    server.add_insecure_port("[::]:50055")
    server.start()
    print("gRPC server is running on port 50055.")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
