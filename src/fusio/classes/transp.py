import copy
import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

import datetime
from .io import io

logger = logging.getLogger('fusio')


class transp_io(io):


    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
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


    def read(
        self,
        path: str | Path,
        side: str = 'output',
    ) -> None:
        if side == 'input':
            self.input = self._read_transp_ufile_file(path)
        else:
            self.output = self._read_transp_netcdf_file(path)


    def write(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False
    ) -> None:
        if side == 'input':
            self._write_transp_ufile_file(path, self.input, overwrite=overwrite)
        else:
            self._write_transp_netcdf_file(path, self.output, overwrite=overwrite)


    def _read_transp_netcdf_file(
        self,
        path: str | Path
    ) -> xr.Dataset:
        data = xr.Dataset()
        if isinstance(path, (str, Path)):
            load_path = Path(path)
            if load_path.exists():
                data = xr.open_dataset(path, engine='netcdf4')
        return data


    def _write_transp_netcdf_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False
    ) -> None:
        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            opath = Path(path)
            if not opath.exists() or overwrite:
                data.to_netcdf(opath, mode='w', format='NETCDF4')
                logger.info(f'Saved {self.format} data into {opath.resolve()}')
            else:
                logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    def _read_transp_ufile_file(
        self,
        path: str | Path
    ) -> xr.Dataset:
        raise NotImplementedError('TRANSP U-FILE read not yet implemented!')
    

    def _write_transp_ufile_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False
    ) -> None:
        raise NotImplementedError('TRANSP U-FILE write not yet implemented!')


    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        input: str | Path | None = None,
        output: str | Path | None = None,
    ) -> Self:
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


    @classmethod
    def from_plasma(
        cls,
        obj: io,
        side: str = 'output',
        window: Sequence[int | float] | None = None,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError('Conversion from plasma class not implemented yet!')