from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def reset(self):
        self.model = None

    @abstractmethod
    def fit(y, param:dict, X=None):
        pass

    @abstractmethod
    def forecast(steps):
        pass