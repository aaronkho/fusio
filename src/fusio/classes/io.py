from .. import Version, __python_version_object__
import importlib
import copy
import logging
from pathlib import Path
from typing import Any      # Available in Python 3.5+
from typing import Final    # Available in Python 3.8+
Self: Any
if __python_version_object__ > Version("3.11"):
    from typing import Self  # type: ignore
else:
    from typing_extensions import Self
from collections.abc import Mapping, Sequence, Iterable
import xarray as xr

logger = logging.getLogger('fusio')


class io():

    _supported_formats: Final[Sequence[str]] = [
        'gacode',
        'imas',
        'omas',
        'torax',
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._tree = xr.DataTree(
            name='root',
            children={'input': xr.DataTree(name='input'), 'output': xr.DataTree(name='output')},
            dataset=xr.Dataset(attrs={'class': f'{self.__class__.__name__}'})
        )

    @property
    def has_input(self) -> bool:
        return (not self._tree['input'].is_empty)

    @property
    def has_output(self) -> bool:
        return (not self._tree['output'].is_empty)

    @property
    def input(self) -> xr.Dataset:
        return self._tree['input'].copy(deep=True).to_dataset()

    @input.setter
    def input(self, data: xr.Dataset) -> None:
        if isinstance(data, xr.Dataset):
            self._tree['input'] = xr.DataTree(name='input', dataset=data.copy(deep=True))

    @property
    def output(self) -> xr.Dataset:
        return self._tree['output'].copy(deep=True).to_dataset()

    @output.setter
    def output(self, data: xr.Dataset) -> None:
        if isinstance(data, xr.Dataset):
            self._tree['output'] = xr.DataTree(name='output', dataset=data.copy(deep=True))

    @property
    def format(self) -> str:
        return self.__class__.__name__[:-3] if self.__class__.__name__.endswith('_io') else self.__class__.__name__

    def autoformat(self) -> None:
        self._tree.attrs['class'] = self.__class__.__name__[:-3] if self.__class__.__name__.endswith('_io') else self.__class__.__name__

    @property
    def is_empty(self) -> bool:
        return (self._tree['input'].is_empty and self._tree['output'].is_empty)

    def update_input_coords(self, data: Mapping) -> None:
        if isinstance(data, dict):
            self.input = self._tree['input'].to_dataset().assign_coords(data)

    def update_output_coords(self, data: Mapping) -> None:
        if isinstance(data, dict):
            self.output = self._tree['output'].to_dataset().assign_coords(data)

    def update_input_data_vars(self, data: Mapping) -> None:
        if isinstance(data, dict):
            self.input = self._tree['input'].to_dataset().assign(data)

    def update_output_data_vars(self, data: Mapping) -> None:
        if isinstance(data, dict):
            self.output = self._tree['output'].to_dataset().assign(data)

    def delete_input_data_vars(self, data: Iterable) -> None:
        self.input = self._tree['input'].to_dataset().drop_vars([key for key in data], errors='ignore')

    def delete_output_data_vars(self, data: Iterable) -> None:
        self.output = self._tree['output'].to_dataset().drop_vars([key for key in data], errors='ignore')

    def update_input_attrs(self, data: Mapping) -> None:
        if isinstance(data, dict):
            self._tree['input'].attrs.update(data)

    def update_output_attrs(self, data: Mapping) -> None:
        if isinstance(data, dict):
            self._tree['output'].attrs.update(data)

    def delete_input_attrs(self, data: Iterable) -> None:
        for key in data:
            self._tree['input'].attrs.pop(key, None)

    def delete_output_attrs(self, data: Iterable) -> None:
        for key in data:
            self._tree['output'].attrs.pop(key, None)

    # These functions always assume data is placed on input side of target format

    def to(self, fmt: str, **kwargs: Any) -> Self:
        try:
            mod = importlib.import_module(f'fusio.classes.{fmt}')
            cls = getattr(mod, f'{fmt}_io')
            return cls._from(self, **kwargs)
        except:
            raise NotImplementedError(f'Direct conversion to {fmt} not implemented.')

    @classmethod
    def _from(cls, obj: Self, side: str = 'output', **kwargs: Any) -> Self | None:
        newobj = None
        if isinstance(obj, io):
            if hasattr(cls, f'from_{obj.format}'):
                generator = getattr(cls, f'from_{obj.format}')
                checker = getattr(cls, f'has_{side}')
                if checker:
                    newobj = generator(obj, side=side, **kwargs)
            else:
                raise NotImplementedError(f'Direct conversion from {obj.format} to {cls.__name__} not implemented.')
        return newobj

    # These functions assume that the path has been checked

    def dump(self, path: str | Path, overwrite: bool = False) -> None:
        if isinstance(path, (str, Path)):
            dump_path = Path(path)
            if overwrite or not dump_path.exists():
                self._tree.to_netcdf(dump_path)
            else:
                logger.warning(f'Requested dump path, {dump_path.resolve()}, already exists! Aborting dump...')
        else:
            logger.warning(f'Invalid path argument given to dump function! Aborting dump...')

    @classmethod
    def load(cls, path: str | Path) -> Self:
        if isinstance(path, (str, Path)):
            load_path = Path(path)
            if load_path.exists():
                tree = xr.open_datatree(path)
                root = tree.get('root')
                if isinstance(root, xr.DataTree):
                    fmt = root.to_dataset().attrs.get('class', 'unknown')
                    try:
                        mod = importlib.import_module(f'fusio.classes.{fmt}')
                    except:
                        raise NotImplementedError(f'File contains data for {fmt} but this format is not yet implemented.')
                    newcls = getattr(mod, f'{fmt}_io')
                    newobj = newcls()
                    newobj.input = tree.get('input')
                    newobj.output = tree.get('output')
                    return newobj
                else:
                    logger.warning('Requested load path, {load_path}, contains data which is incompatible with fusio! Returning empty base class...')
            else:
                logger.warning('Requested load path, {load_path}, does not exist! Returning empty base class...')
        else:
            logger.warning('Invalid path argument given to load function! Returning empty base class...')
        return cls()
