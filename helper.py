from typing import List
from itertools import product
from base_model import BaseModel

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