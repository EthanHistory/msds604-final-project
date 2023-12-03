from base_model import BaseModel
from statsmodels.tsa.arima.model import ARIMA

class SARIMAX(BaseModel):
    def __init__(self, m):
        super().__init__("SARIMAX")
        self.m = m

    def fit(self, y, param:dict, X=None):
        p = param['p']
        d = param['d']
        q = param['q']
        
        P = param['P']
        D = param['D']
        Q = param['Q']

        t = param['t']
        self.model = ARIMA(y, exog=X,
            order=(p,d,q),
            seasonal_order=(P,D,Q,self.m),
            trend=t, 
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()

    def forecast(self, steps):
        return self.model.forecast(steps)