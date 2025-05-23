from pathlib import Path
import logging
import numpy as np
import xarray as xr
from .io import io

logger = logging.getLogger('fusio')


class imas_io(io):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ipath = None
        opath = None
        for arg in args:
            if ipath is None and isinstance(arg, (str, Path)):
                ipath = Path(arg)
            elif opath is None and isinstance(arg, (str, Path)):
                opath = Path(arg)
        for key, kwarg in kwargs.items():
            if ipath is None and key in ['input'] and isinstance(kwarg, (str, Path)):
                ipath = Path(kwarg)
            if opath is None and key in ['path', 'file', 'output'] and isinstance(kwarg, (str, Path)):
                opath = Path(kwarg)
        if ipath is not None:
            self.read(ipath, side='input')
        if opath is not None:
            self.read(opath, side='output')
        self.autoformat()


    def read(self, path, side='output'):
        if side == 'input':
            self.input = self._read_imas_file(path)
        else:
            self.output = self._read_imas_file(path)


    def write(self, path, side='input', overwrite=False):
        if side == 'input':
            self._write_imas_file(path, self.input, overwrite=overwrite)
        else:
            self._write_imas_file(path, self.output, overwrite=overwrite)


    def _read_imas_file(self, path):

        coords = {}
        data_vars = {}
        attrs = {}

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


    def _write_imas_file(self, data, path, overwrite=False):

        if isinstance(data, xr.DataTree):
            data = data.to_dataset().sel(n=0, drop=True) if not data.is_empty else None

        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            opath = Path(path)
            logger.info(f'Saved {self.format} data into {opath.resolve()}')
            #else:
            #    logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    @classmethod
    def from_file(cls, path=None, input=None, output=None):
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


    # Assumed that the self creation method transfers output to input
    @classmethod
    def from_gacode(cls, obj, side='output'):
        newobj = cls()
        if isinstance(obj, io):
            newobj.input = obj.input if side == 'input' else obj.output
        return newobj

