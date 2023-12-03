from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def reset(self):
        self.model = None

    @abstractmethod
    def fit(y: pd.Series, param:dict, X=pd.DataFrame):
        """Fit the model
        [Warning] you should use `self.model` attribute to define your model

        Args:
            y (np.array): Time series data
            param (dict): a dictionary with keys (parameter name), values (parameter)
            X (np.array, optional): Exogenous variables. Defaults to None.
        """
        pass

    @abstractmethod
    def forecast(steps: int):
        """Forecast `steps` ahead

        Args:
            steps (int): the number of forecasts
        """
        pass