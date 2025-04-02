import copy
import logging
import importlib
from pathlib import Path
import numpy as np

logger = logging.getLogger('fusio')


class io():

    _supported_formats = [
        'gacode',
        'imas',
    ]

    def __init__(self, *args, **kwargs):
        self._data = {}

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @property
    def format(self):
        return self.__class__.__name__

    @property
    def empty(self):
        return (not bool(self._data))

    def to(self, fmt):
        try:
            mod = importlib.import_module(f'fusio.classes.{fmt}')
            cls = getattr(mod, f'{fmt}')
            return cls._from(self)
        except:
            raise NotImplementedError(f'Direct conversion to {fmt} not implemented.')

    @classmethod
    def _from(cls, obj):
        newobj = None
        if isinstance(obj, io):
            if hasattr(cls, f'from_{obj.format}'):
                generator = getattr(cls, f'from_{obj.format}')
                newobj = generator(obj)
            else:
                raise NotImplementedError(f'Direct conversion from {obj.format} to {cls.__name__} not implemented.')
        return newobj

