import logging
import importlib
from pathlib import Path
import xarray as xr

logger = logging.getLogger('fusio')


class io():

    _supported_formats = [
        'gacode',
        'imas',
        'torax',
    ]

    def __init__(self, *args, **kwargs):
        self._tree = xr.DataTree(name='root', data=xr.Dataset(attrs={'class': f'{self.__class__.__name__}'}))
        self._input = xr.DataTree(name='input', parent=self._tree)
        self._output = xr.DataTree(name='output', parent=self._tree)

    @property
    def has_input(self):
        return (not self._input.is_empty)

    @property
    def has_output(self):
        return (not self._output.is_empty)

    @property
    def input(self):
        return self._input.copy(deep=True)

    @property.setter
    def input(self, data):
        if isinstance(data, xr.Dataset):
            self._input = xr.DataTree(name='input', parent=self._tree, data=data.copy(deep=True))

    @property
    def output(self):
        return self._output.copy(deep=True)

    @property.setter
    def output(self):
        if isinstance(data, xr.Dataset):
            self._output = xr.DataTree(name='output', parent=self._tree, data=data.copy(deep=True))

    @property
    def format(self):
        return self.__class__.__name__[:-3] if self.__class__.__name__.endswith('_io') else self.__class__.__name__

    def autoformat(self):
        self._tree.attrs['class'] = self.__class__.__name__[:-3] if self.__class__.__name__.endswith('_io') else self.__class__.__name__

    @property
    def is_empty(self):
        return (self._input.is_empty and self._output.is_empty)

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

    # These functions assume that the path has been checked

    def dump(self, path, overwrite=False):
        if isinstance(path, (str, Path)):
            dump_path = Path(path)
            if overwrite or not dump_path.exists():
                self._tree.to_netcdf(dump_path)
            else:
                logger.warning('Requested dump path, {dump_path.resolve()}, already exists! Aborting dump...')
        else:
            logger.warning('Invalid path argument given to dump function! Aborting dump...')

    @classmethod
    def load(cls, path):
        if isinstance(path, (str, Path)):
            load_path = Path(path)
            if load_path.exists():
                tree = xr.open_datatree(path)
                try:
                    fmt = tree.get('root').get('class')
                    mod = importlib.import_module(f'fusio.classes.{fmt}')
                    newcls = getattr(mod, f'{fmt}_io')
                    return newcls._from(self)
                except:
                    raise NotImplementedError(f'File contains data for {fmt} but this format is not yet implemented.')
            else:
                logger.warning('Requested load path, {load_path}, does not exist! Returning empty base class...')
        else:
            logger.warning('Invalid path argument given to load function! Returning empty base class...')
        return cls()
