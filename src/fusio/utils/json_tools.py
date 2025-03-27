import numpy as np

def serialize(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x

def deserialize(x):
    if isinstance(x, list) and len(x) >= 1:
        if isinstance(x[0], (float, int)):
            return np.array(x)
    return x

