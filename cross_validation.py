from typing import List, Dict
from itertools import product
from base_model import BaseModel
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

class BaseTimeSeriesModelCrossValidation:
    def __init__(self, num_fold: int = 5, rolling_size: int = 10, validation_size: int = 14):
        self.num_fold = num_fold
        self.rolling_size = rolling_size
        self.validation_size = validation_size
    
    def cross_validation(self, y, model:BaseModel, params: List[Dict], X=None):
        result = []
        best_rmse, best_p = float('inf'), None
        print('[%s] Start cross-validation' % (model.model_name))
        for p in tqdm(params):
            rmse = 0
            initial_train_size = len(y) - self.validation_size - (self.num_fold-1) * self.rolling_size
            for k in range(self.num_fold):
                y_train = y[:initial_train_size+k*self.rolling_size]
                y_val = y[initial_train_size+k*self.rolling_size:initial_train_size+k*self.rolling_size+self.validation_size]

                if X is not None:
                    X_train = X[:initial_train_size+k*self.rolling_size]
                else:
                    X_train = None

                model.reset()
                model.fit(y_train, p, X_train)
                y_preds = model.forecast(self.validation_size)

                rmse = rmse+np.sqrt(mean_squared_error(y_val, y_preds))
            rmse_avg = rmse / float(self.num_fold)
            result.append((p, rmse_avg))
            if rmse_avg < best_rmse:
                best_rmse, best_p = rmse_avg, p
                print('[%s] P=%s RMSE=%.3f' % (model.model_name, p, rmse_avg))
        print('[%s] P=%s Best RMSE=%.3f' % (model.model_name, best_p, best_rmse))
        return result, best_p
    

def parameter_mixer(parameter_names: List[str], parameter_lists: List[List]):
    assert(len(parameter_names) == len(parameter_lists))
    params = [
        {
            parameter_names[i] : param[i] for i in range(len(parameter_names))
        }
        for param in product(
            *parameter_lists
        )
    ]
    return params