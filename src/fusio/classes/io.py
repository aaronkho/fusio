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
        'torax',
    ]

    def __init__(self, *args, **kwargs):
        self._input = {}
        self._output = {}

    @property
    def has_input(self):
        return bool(self._input)

    @property
    def has_output(self):
        return bool(self._output)

    @property
    def input(self):
        return copy.deepcopy(self._input)

    @property
    def output(self):
        return copy.deepcopy(self._output)

    @property
    def format(self):
        return self.__class__.__name__[:-3] if self.__class__.__name__.endswith('_io') else self.__class__.__name__

    @property
    def empty(self):
        return (not bool(self._input) and not bool(self._output))

    # These functions always assume data is placed on input side of target format

    def to(self, fmt):
        try:
            mod = importlib.import_module(f'fusio.classes.{fmt}')
            cls = getattr(mod, f'{fmt}_io')
            return cls._from(self)
        except:
            raise NotImplementedError(f'Direct conversion to {fmt} not implemented.')

    @classmethod
    def _from(cls, obj, side='output'):
        newobj = None
        if isinstance(obj, io):
            if hasattr(cls, f'from_{obj.format}'):
                generator = getattr(cls, f'from_{obj.format}')
                checker = getattr(cls, f'has_{side}')
                if checker:
                    newobj = generator(obj, side=side)
            else:
                raise NotImplementedError(f'Direct conversion from {obj.format} to {cls.__name__} not implemented.')
        return newobj

