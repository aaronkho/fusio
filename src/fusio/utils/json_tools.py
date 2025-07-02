import numpy as np
from ..classes.io import Any

def serialize(x: Any) -> Any:
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x

def deserialize(x: Any) -> Any:
    if isinstance(x, list) and len(x) >= 1:
        if isinstance(x[0], (float, int)):
            return np.array(x)
    return x

