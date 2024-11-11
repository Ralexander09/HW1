from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class train_request(_message.Message):
    __slots__ = ("model_type", "hyperparams", "data_path")
    class HyperparamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    HYPERPARAMS_FIELD_NUMBER: _ClassVar[int]
    DATA_PATH_FIELD_NUMBER: _ClassVar[int]
    model_type: str
    hyperparams: _containers.ScalarMap[str, str]
    data_path: str
    def __init__(self, model_type: _Optional[str] = ..., hyperparams: _Optional[_Mapping[str, str]] = ..., data_path: _Optional[str] = ...) -> None: ...

class train_response(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class predict_request(_message.Message):
    __slots__ = ("model_id", "data_path")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_PATH_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    data_path: str
    def __init__(self, model_id: _Optional[str] = ..., data_path: _Optional[str] = ...) -> None: ...

class predict_response(_message.Message):
    __slots__ = ("prediction", "MSE")
    PREDICTION_FIELD_NUMBER: _ClassVar[int]
    MSE_FIELD_NUMBER: _ClassVar[int]
    prediction: _containers.RepeatedScalarFieldContainer[float]
    MSE: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, prediction: _Optional[_Iterable[float]] = ..., MSE: _Optional[_Iterable[float]] = ...) -> None: ...

class delete_request(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class delete_response(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...

class status_response(_message.Message):
    __slots__ = ("status", "models")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    status: str
    models: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, status: _Optional[str] = ..., models: _Optional[_Iterable[str]] = ...) -> None: ...

class model_list(_message.Message):
    __slots__ = ("model_types",)
    MODEL_TYPES_FIELD_NUMBER: _ClassVar[int]
    model_types: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, model_types: _Optional[_Iterable[str]] = ...) -> None: ...

class empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
