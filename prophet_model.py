import pandas as pd

from base_model import BaseModel
from prophet import Prophet

class ProphetModel(BaseModel):
    def __init__(self):
        super().__init__("Prophet")

    def fit(self, y: pd.Series, param:dict, X=pd.DataFrame):
        y.index.name = 'ds'
        y.name = 'y'
        new_y = y.reset_index()
        
        self.model = Prophet()

        self.model.fit(new_y)

    def forecast(self, steps):
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return forecast.yhat.iloc[-steps:]