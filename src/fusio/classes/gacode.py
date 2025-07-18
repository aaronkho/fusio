import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

import datetime
from .io import io
from ..utils.plasma_tools import define_ion_species
from ..utils.eqdsk_tools import (
    define_cocos_converter,
    read_eqdsk,
    trace_flux_surfaces,
    calculate_mxh_coefficients,
)

logger = logging.getLogger('fusio')


class gacode_io(io):

    basevars: Final[Sequence[str]] = [
        'nexp',
        'nion',
        'shot',
        'name',
        'type',
        'masse',
        'mass',
        'ze',
        'z',
        'torfluxa',
        'rcentr',
        'bcentr',
        'current',
        'time',
        'polflux',
        'q',
        'rmin',
        'rmaj',
        'zmag',
        'kappa',
        'delta',
        'zeta',
        'shape_cos',
        'shape_sin',
        'ni',
        'ti',
        'ne',
        'te',
        'z_eff',
        'qohme',
        'qbeame',
        'qbeami',
        'qrfe',
        'qrfi',
        'qsync',
        'qbrem',
        'qline',
        'qfuse',
        'qfusi',
        'qei',
        'qione',
        'qioni',
        'qcxi',
        'johm',
        'jbs',
        'jbstor',
        'jrf',
        'jnb',
        'vtor',
        'vpol',
        'omega0',
        'ptot',
        'qpar_beam',
        'qpar_wall',
        'qmom',
    ]
    titles_singleInt: Final[Sequence[str]] = [
        'nexp',
        'nion',
        'shot',
    ]
    titles_singleStr: Final[Sequence[str]] = [
        'name',
        'type',
    ]
    titles_singleFloat: Final[Sequence[str]] = [
        'masse',
        'mass',
        'ze',
        'z',
        'torfluxa',
        'rcentr',
        'bcentr',
        'current',
        'time',
    ]
    units: Final[Mapping[str, str]] = {
        'torfluxa': 'Wb/radian',
        'rcentr': 'm',
        'bcentr': 'T',
        'current': 'MA',
        'polflux': 'Wb/radian',
        'rmin': 'm',
        'rmaj': 'm',
        'zmag': 'm',
        'ni': '10^19/m^3',
        'ti': 'keV',
        'ne': '10^19/m^3',
        'te': 'keV',
        'qohme': 'MW/m^3',
        'qbeame': 'MW/m^3',
        'qbeami': 'MW/m^3',
        'qrfe': 'MW/m^3',
        'qrfi': 'MW/m^3',
        'qsync': 'MW/m^3',
        'qbrem': 'MW/m^3',
        'qline': 'MW/m^3',
        'qfuse': 'MW/m^3',
        'qfusi': 'MW/m^3',
        'qei': 'MW/m^3',
        'qione': 'MW/m^3',
        'qioni': 'MW/m^3',
        'qcxi': 'MW/m^3',
        'johm': 'MA/m^2',
        'jbs': 'MA/m^2',
        'jbstor': 'MA/m^2',
        'jrf': 'MA/m^2',
        'jnb': 'MA/m^2',
        'vtor': 'm/s',
        'vpol': 'm/s',
        'omega0': 'rad/s',
        'ptot': 'Pa',
        'qpar_beam': '1/m^3/s',
        'qpar_wall': '1/m^3/s',
        'qmom': 'N/m^2',
    }


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


    def make_file_header(
        self,
    ) -> str:
        now = datetime.datetime.now()
        gacode_header = [
            f'#  *original : {now.strftime("%a %b %-d %H:%M:%S %Z %Y")}',
            f'# *statefile : null',
            f'#     *gfile : null',
            f'#   *cerfile : null',
            f'#      *vgen : null',
            f'#     *tgyro : null',
            f'#',
        ]
        return '\n'.join(gacode_header)


    def correct_magnetic_fluxes(
        self,
        exponent: int = -1,
        side: str = 'input',
    ) -> None:
        if side == 'input':
            if 'polflux' in self.input:
                self._tree['input']['polflux'] *= np.power(2.0 * np.pi, exponent)
            if 'torfluxa' in self.input:
                self._tree['input']['torfluxa'] *= np.power(2.0 * np.pi, exponent)
        else:
            if 'polflux' in self.output:
                self._tree['output']['polflux'] *= np.power(2.0 * np.pi, exponent)
            if 'torfluxa' in self.output:
                self._tree['output']['torfluxa'] *= np.power(2.0 * np.pi, exponent)


    def add_geometry_from_eqdsk(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False,
    ) -> None:
        data = self.input.to_dataset() if side == 'input' else self.output.to_dataset()
        if isinstance(path, (str, Path)) and 'polflux' in data:
            eqdsk_data = read_eqdsk(path)
            mxh_data = self._calculate_geometry_from_eqdsk(eqdsk_data, data.isel(n=0)['polflux'].to_numpy().flatten())
            newvars = {}
            if overwrite or np.abs(data.get('rmaj', np.array([0.0]))).sum() == 0.0:
                newvars['rmaj'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['rmaj']), axis=0))
            if overwrite or np.abs(data.get('rmin', np.array([0.0]))).sum() == 0.0:
                newvars['rmin'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['rmin']), axis=0))
            if overwrite or np.abs(data.get('zmag', np.array([0.0]))).sum() == 0.0:
                newvars['zmag'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['zmag']), axis=0))
            if overwrite or np.abs(data.get('kappa', np.array([0.0]))).sum() == 0.0:
                newvars['kappa'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['kappa']), axis=0))
            if overwrite or np.abs(data.get('delta', np.array([0.0]))).sum() == 0.0:
                newvars['delta'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['delta']), axis=0))
            if overwrite or np.abs(data.get('zeta', np.array([0.0]))).sum() == 0.0:
                newvars['zeta'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['zeta']), axis=0))
            if overwrite or np.abs(data.get('shape_sin3', np.array([0.0]))).sum() == 0.0:
                newvars['shape_sin3'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin3']), axis=0))
            if overwrite or np.abs(data.get('shape_sin4', np.array([0.0]))).sum() == 0.0:
                newvars['shape_sin4'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin4']), axis=0))
            if overwrite or np.abs(data.get('shape_sin5', np.array([0.0]))).sum() == 0.0:
                newvars['shape_sin5'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin5']), axis=0))
            if overwrite or np.abs(data.get('shape_sin6', np.array([0.0]))).sum() == 0.0:
                newvars['shape_sin6'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin6']), axis=0))
            if overwrite or np.abs(data.get('shape_cos0', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos0'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos0']), axis=0))
            if overwrite or np.abs(data.get('shape_cos1', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos1'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos1']), axis=0))
            if overwrite or np.abs(data.get('shape_cos2', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos2'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos2']), axis=0))
            if overwrite or np.abs(data.get('shape_cos3', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos3'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos3']), axis=0))
            if overwrite or np.abs(data.get('shape_cos4', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos4'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos4']), axis=0))
            if overwrite or np.abs(data.get('shape_cos5', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos5'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos5']), axis=0))
            if overwrite or np.abs(data.get('shape_cos6', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos6'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos6']), axis=0))
            if newvars:
                if side == 'input':
                    self.update_input_data_vars(newvars)
                else:
                    self.update_output_data_vars(newvars)


    # This could probably be generalized and moved to eqdsk_tools
    def _calculate_geometry_from_eqdsk(
        self,
        eqdsk_data: MutableMapping[str, Any],
        psivec: ArrayLike,
    ) -> MutableMapping[str, list[int | float]]:
        mxh_data: dict[str, list[int| float]] = {
            'rmaj': [],
            'rmin': [],
            'zmag': [],
            'kappa': [],
            'delta': [],
            'zeta': [],
            'sin3': [],
            'sin4': [],
            'sin5': [],
            'sin6': [],
            'cos0': [],
            'cos1': [],
            'cos2': [],
            'cos3': [],
            'cos4': [],
            'cos5': [],
            'cos6': [],
        }
        if isinstance(eqdsk_data, dict) and isinstance(psivec, np.ndarray):
            faxis = False
            rvec = np.linspace(eqdsk_data['rleft'], eqdsk_data['rleft'] + eqdsk_data['rdim'], eqdsk_data['nr'])
            zvec = np.linspace(eqdsk_data['zmid'] - 0.5 * eqdsk_data['zdim'], eqdsk_data['zmid'] + 0.5 * eqdsk_data['zdim'], eqdsk_data['nz'])
            if np.isclose(eqdsk_data['psi'][0, 0], eqdsk_data['psi'][-1, -1]) and np.isclose(eqdsk_data['psi'][0, -1], eqdsk_data['psi'][-1, 0]):
                if eqdsk_data['simagx'] > eqdsk_data['sibdry'] and psivec[-1] < eqdsk_data['psi'][0, 0]:
                    psivec[-1] = eqdsk_data['psi'][0, 0] + 1.0e-6
                elif eqdsk_data['simagx'] < eqdsk_data['sibdry'] and psivec[-1] > eqdsk_data['psi'][0, 0]:
                    psivec[-1] = eqdsk_data['psi'][0, 0] - 1.0e-6
            if eqdsk_data['simagx'] > eqdsk_data['sibdry'] and psivec[0] >= eqdsk_data['simagx']:
                faxis = True
                psivec[0] = eqdsk_data['simagx'] - 1.0e-6
            elif eqdsk_data['simagx'] < eqdsk_data['sibdry'] and psivec[0] <= eqdsk_data['simagx']:
                faxis = True
                psivec[0] = eqdsk_data['simagx'] + 1.0e-6
            rmesh, zmesh = np.meshgrid(rvec, zvec)
            axis = [eqdsk_data['rmagx'], eqdsk_data['zmagx']]
            fs = trace_flux_surfaces(rmesh, zmesh, eqdsk_data['psi'], psivec, axis=axis)
            mxh = {psi: calculate_mxh_coefficients(c[:, 0], c[:, 1], n=6) for psi, c in fs.items()}
            for psi in psivec:
                mxh_data['rmaj'].append(mxh[psi][2][0] if psi in mxh else np.nan)
                mxh_data['rmin'].append(mxh[psi][2][1] if psi in mxh else np.nan)
                mxh_data['zmag'].append(mxh[psi][2][2] if psi in mxh else np.nan)
                mxh_data['kappa'].append(mxh[psi][2][3] if psi in mxh else np.nan)
                mxh_data['delta'].append(np.sin(mxh[psi][1][1]) if psi in mxh else np.nan)
                mxh_data['zeta'].append(-mxh[psi][1][2] if psi in mxh else np.nan)
                mxh_data['sin3'].append(mxh[psi][1][3] if psi in mxh else np.nan)
                mxh_data['sin4'].append(mxh[psi][1][4] if psi in mxh else np.nan)
                mxh_data['sin5'].append(mxh[psi][1][5] if psi in mxh else np.nan)
                mxh_data['sin6'].append(mxh[psi][1][6] if psi in mxh else np.nan)
                mxh_data['cos0'].append(mxh[psi][0][0] if psi in mxh else np.nan)
                mxh_data['cos1'].append(mxh[psi][0][1] if psi in mxh else np.nan)
                mxh_data['cos2'].append(mxh[psi][0][2] if psi in mxh else np.nan)
                mxh_data['cos3'].append(mxh[psi][0][3] if psi in mxh else np.nan)
                mxh_data['cos4'].append(mxh[psi][0][4] if psi in mxh else np.nan)
                mxh_data['cos5'].append(mxh[psi][0][5] if psi in mxh else np.nan)
                mxh_data['cos6'].append(mxh[psi][0][6] if psi in mxh else np.nan)
            for key in mxh_data:
                arr = np.array(mxh_data[key])
                mask = np.isfinite(arr)
                if not np.all(mask) and np.any(mask):
                    inner_idx = np.argmax(mask)
                    mask |= (np.arange(len(mask)) >= inner_idx)
                    if inner_idx > 0:
                        if key == 'rmaj':
                            arr[~mask] = eqdsk_data['rmagx'] + (psivec[~mask] - psivec[0]) * (arr[inner_idx] - eqdsk_data['rmagx']) / (psivec[inner_idx] - psivec[0])
                        elif key == 'zmag':
                            arr[~mask] = eqdsk_data['zmagx'] + (psivec[~mask] - psivec[0]) * (arr[inner_idx] - eqdsk_data['zmagx']) / (psivec[inner_idx] - psivec[0])
                        else:
                            arr[~mask] = arr[inner_idx] + (psivec[~mask] - psivec[inner_idx]) * (arr[inner_idx + 1] - arr[inner_idx]) / (psivec[inner_idx + 1] - psivec[inner_idx])
                        for i in range(inner_idx):
                            mxh_data[key][i] = arr.item(i)
        return mxh_data


    def read(
        self,
        path: str | Path,
        side: str = 'output',
    ) -> None:
        if side == 'input':
            self.input = self._read_gacode_file(path)
        else:
            self.output = self._read_gacode_file(path)


    def write(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False
    ) -> None:
        if side == 'input':
            self._write_gacode_file(path, self.input.to_dataset(), overwrite=overwrite)
        else:
            self._write_gacode_file(path, self.output.to_dataset(), overwrite=overwrite)


    def _read_gacode_file(
        self,
        path: str | Path
    ) -> xr.Dataset:

        coords = {}
        data_vars = {}
        attrs: MutableMapping[str, Any] = {}

        if isinstance(path, (str, Path)):
            ipath = Path(path)
            lines = []
            titles_single: list[str] = []
            if ipath.is_file():
                titles_single.extend(self.titles_singleInt)
                titles_single.extend(self.titles_singleStr)
                titles_single.extend(self.titles_singleFloat)
                with open(ipath, 'r') as f:
                    lines = f.readlines()

            istartProfs = None
            for i in range(len(lines)):
                if '# nexp' in lines[i]:
                    istartProfs = i
                    break
            header = lines[:istartProfs]
            while len(header) > 0 and not header[-1].strip():
                header = header[:-1]
            attrs['header'] = ''.join(header).strip()

            singleLine = False
            title = ''
            var: list[list[int | float]] = []
            found = False
            singles: dict[str, NDArray] = {}
            profiles: dict[str, NDArray] = {}
            for i in range(len(lines)):

                if lines[i].startswith('#') and not lines[i + 1].startswith('#'):
                    # previous
                    if found and not singleLine:
                        profiles[title] = np.array(var)
                        if profiles[title].shape[1] == 1:
                            profiles[title] = profiles[title][:, 0]
                    linebr = lines[i].split('#')[1].split('\n')[0].split()
                    title = linebr[0]
                    #title_orig = linebr[0]
                    #aif len(linebr) > 1:
                    #    unit = lines[i].split('#')[1].split('\n')[0].split()[2]
                    #    title = title_orig
                    #else:
                    #    title = title_orig
                    found, var = True, []
                    if title in titles_single:
                        singleLine = True
                    else:
                        singleLine = False

                elif found:
                    var0 = lines[i].split()
                    if singleLine:
                        if title in self.titles_singleFloat:
                            singles[title] = np.array(var0, dtype=float)
                        elif title in self.titles_singleInt:
                            singles[title] = np.array(var0, dtype=int)
                        else:
                            singles[title] = np.array(var0, dtype=str)
                    else:
                        varT = [float(j) if (j[-4].upper() == "E" or "." in j) else 0.0 for j in var0[1:]]
                        var.append(varT)

            # last
            if not singleLine:
                while len(var[-1]) < 1:
                    var = var[:-1]  # Sometimes there's an extra space, remove
                profiles[title] = np.array(var)
                if profiles[title].shape[1] == 1:
                    profiles[title] = profiles[title][:, 0]

            ncoord = 'n'
            rcoord = 'rho' if 'rho' in profiles else 'polflux'
            scoord = 'name' if 'name' in singles else 'z'
            coords[ncoord] = np.atleast_1d([0])
            if rcoord in profiles:
                coords[rcoord] = profiles.pop(rcoord)
            if scoord in singles:
                coords[scoord] = singles.pop(scoord)
            for key, val in profiles.items():
                if key in ['rho', 'polflux', 'rmin']:
                    data_vars[key] = ([ncoord, rcoord], np.expand_dims(val, axis=0))
                elif key in ['ni', 'ti', 'vtor', 'vpol']:
                    data_vars[key] = ([ncoord, rcoord, scoord], np.expand_dims(val, axis=0))
                elif key in ['w0']:
                    data_vars['omega0'] = ([ncoord, rcoord], np.expand_dims(val, axis=0))
                else:
                    data_vars[key] = ([ncoord, rcoord], np.expand_dims(val, axis=0))
            for key, val in singles.items():
                if key in ['name', 'z', 'mass', 'type']:
                    data_vars[key] = ([ncoord, scoord], np.expand_dims(val, axis=0))
                #elif key in ['header']:
                #    attrs[key] = val
                else:
                    data_vars[key] = ([ncoord], val)

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


    def _write_gacode_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        item: int = 0,
        overwrite: bool = False
    ) -> None:

        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            wdata = data.sel(n=item, drop=True)
            opath = Path(path)
            processed_titles = []
            header = wdata.attrs.get('header', '').split('\n')
            lines = [f'{line:<70}\n' for line in header]
            lines += ['#\n']
            processed_titles.append('header')
            for title in self.titles_singleInt:
                newlines = []
                if title in wdata:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(f'{wdata[title]:d}\n')
                    processed_titles.append(title)
                lines += newlines
            for title in self.titles_singleStr:
                newlines = []
                if title in wdata:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(' '.join([f'{val}' for val in np.atleast_1d(wdata[title].values)]) + '\n')
                    processed_titles.append(title)
                lines += newlines
            for title in self.titles_singleFloat:
                newlines = []
                if title in wdata:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(' '.join([f'{val:14.7E}' for val in np.atleast_1d(wdata[title].values)]) + '\n')
                    processed_titles.append(title)
                lines += newlines
            for title in list(wdata.coords) + list(wdata.data_vars):
                newlines = []
                if title not in processed_titles:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    else:
                        newtitle += f' | -'
                    newlines.append(f'# {newtitle}\n')
                    rcoord = [f'{dim}' for dim in wdata[title].dims if dim in ['rho', 'polflux', 'rmin']]
                    if len(rcoord) > 0:
                        for ii in range(len(wdata[rcoord[0]])):
                            nstr = [f'{ii + 1:3d}']
                            nstr.extend([f'{val:14.7E}' for val in np.atleast_1d(wdata[title].isel({f'{rcoord[0]}': ii}).values)])
                            newlines.append(' '.join(nstr) + '\n')
                    processed_titles.append(title)
                lines += newlines

            with open(opath, 'w') as f:
                f.writelines(lines)
            logger.info(f'Saved {self.format} data into {opath.resolve()}')
            #else:
            #    logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        input: str | Path | None = None,
        output: str | Path | None = None,
    ) -> Self:
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


    # Assumed that the self creation method transfers output to input
    @classmethod
    def from_gacode(
        cls,
        obj: io,
        side: str = 'output',
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            newobj.input = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
        return newobj


    @classmethod
    def from_torax(
        cls,
        obj: io,
        side: str = 'output',
        window: Sequence[int | float] | None = None,
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            if 'rho_norm' in data.coords:
                data = data.isel(time=-1)
                zeros = np.zeros_like(data.coords['rho_norm'].to_numpy().flatten())
                coords = {}
                data_vars = {}
                attrs: MutableMapping[str, Any] = {}
                name: list[str] = []
                coords['n'] = np.array([0], dtype=int)
                if 'rho_norm' in data.coords:
                    coords['rho'] = data.coords['rho_norm'].to_numpy().flatten()
                    data_vars['nexp'] = (['n'], np.array([len(coords['rho'])], dtype=int))
                if 'psi' in data:
                    data_vars['polflux'] = (['n', 'rho'], np.expand_dims(data['psi'].to_numpy().flatten(), axis=0))
                if 'r_mid' in data:
                    data_vars['rmin'] = (['n', 'rho'], np.expand_dims(data['r_mid'].to_numpy().flatten(), axis=0))
                data_vars['shot'] = (['n'], np.atleast_1d([0]))
                data_vars['masse'] = (['n'], np.atleast_1d([5.4488748e-04]))
                data_vars['ze'] = (['n'], np.atleast_1d([-1.0]))
                if 'Phi_b' in data:
                    data_vars['torfluxa'] = (['n'], data['Phi_b'].to_numpy().flatten())
                #if 'R_major' in data:
                #    data_vars['rcentr'] = (['n'], data['R_major'].to_numpy().flatten())
                if 'R_out' in data:
                    data_vars['rcentr'] = (['n'], data['R_out'].isel(rho_norm=0).to_numpy().flatten())
                if 'F' in data and 'R_out' in data:
                    data_vars['bcentr'] = (['n'], (data['F'] / data['R_out']).isel(rho_norm=0).to_numpy().flatten())
                if 'Ip' in data:
                    data_vars['current'] = (['n'], 1.0e-6 * data['Ip'].to_numpy().flatten())
                if 'q' in data and 'rho_norm' in data and 'rho_face_norm' in data:
                    q = np.interp(data['rho_norm'].to_numpy().flatten(), data['rho_face_norm'].to_numpy().flatten(), data['q'].to_numpy().flatten())
                    data_vars['q'] = (['n', 'rho'], np.expand_dims(q, axis=0))
                if 'R_in' in data and 'R_out' in data:
                    rmaj = (data['R_in'] + data['R_out']).to_numpy().flatten() / 2.0
                    data_vars['rmaj'] = (['n', 'rho'], np.expand_dims(rmaj, axis=0))
                    data_vars['zmag'] = (['n', 'rho'], np.expand_dims(np.zeros_like(zeros), axis=0))
                if 'elongation' in data:
                    data_vars['kappa'] = (['n', 'rho'], np.expand_dims(data['elongation'].to_numpy().flatten(), axis=0))
                if 'delta' in data:
                    delta = data['delta'].to_numpy().flatten()
                    delta = np.concatenate([np.array([delta[0]]), delta[:-1] + 0.5 * np.diff(delta), np.array([delta[-1]])], axis=0)
                    data_vars['delta'] = (['n', 'rho'], np.expand_dims(delta, axis=0))
                data['zeta'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos0'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos1'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos2'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos3'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos4'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos5'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos6'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_sin3'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_sin4'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_sin5'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_sin6'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                if 'n_i' in data and 'n_e' in data:
                    split_dt = True
                    ne = np.expand_dims(1.0e-19 * data['n_e'].to_numpy().flatten(), axis=-1)
                    ni = np.expand_dims(1.0e-19 * data['n_i'].to_numpy().flatten(), axis=-1)
                    zeff = ni / ne
                    zimps = []
                    if 'n_impurity' in data:
                        nimp = np.expand_dims(1.0e-19 * data['n_impurity'].to_numpy().flatten(), axis=-1)
                        if 'Z_impurity' in data and 'n_e' in data:
                            zimp = np.expand_dims(data['Z_impurity'].to_numpy().flatten(), axis=-1)
                            zimps = [zimp[0, 0]]
                        if split_dt:
                            ni = np.concatenate([0.5 * ni, 0.5 * ni], axis=-1)
                        if 'config' in data.attrs:
                            impdict = data.attrs['config'].get('plasma_composition', {}).get('impurity', {})
                            multi_nimp = []
                            multi_zimp = []
                            for key in impdict:
                                fraction = impdict[key].get('value', ['float', [0.0]])[1][-1]
                                impname, impa, impz = define_ion_species(short_name=key)
                                multi_zimp.append(impz)
                                multi_nimp.append(fraction * nimp)
                            if len(multi_nimp) > 0:
                                nimp = np.concatenate(multi_nimp, axis=-1)
                                zimps = multi_zimp
                        ni = np.concatenate([ni, nimp], axis=-1)
                    names = ['D']
                    types = ['[therm]']
                    masses = [2.0]
                    zs = [1.0]
                    if split_dt:
                        names.append('T')
                        types.append('[therm]')
                        masses.append(3.0)
                        zs.append(1.0)
                    ii = len(names)
                    for zz in range(len(zimps)):
                        impname, impa, impz = define_ion_species(z=zimps[zz])
                        names.append(impname)
                        types.append('[therm]')
                        masses.append(impa)
                        zs.append(impz)
                        zeff += np.expand_dims(ni[:, zz+ii], axis=-1) * (impz ** 2.0) / ne
                    coords['name'] = np.array(names)
                    data_vars['ni'] = (['n', 'rho', 'name'], np.expand_dims(ni, axis=0))
                    data_vars['nion'] = (['n'], np.array([len(names)], dtype=int))
                    data_vars['type'] = (['n', 'name'], np.expand_dims(types, axis=0))
                    data_vars['mass'] = (['n', 'name'], np.expand_dims(masses, axis=0))
                    data_vars['z'] = (['n', 'name'], np.expand_dims(zs, axis=0))
                    data_vars['z_eff'] = (['n', 'rho'], np.expand_dims(zeff.flatten(), axis=0))
                if 'T_i' in data:
                    ti = np.expand_dims(data['T_i'].to_numpy().flatten(), axis=-1)
                    if 'name' in coords and len(coords['name']) > 1:
                        ti = np.repeat(ti, len(coords['name']), axis=-1)
                    data_vars['ti'] = (['n', 'rho', 'name'], np.expand_dims(ti, axis=0))
                if 'n_e' in data:
                    data_vars['ne'] = (['n', 'rho'], np.expand_dims(1.0e-19 * data['n_e'].to_numpy().flatten(), axis=0))
                if 'T_e' in data:
                    data_vars['te'] = (['n', 'rho'], np.expand_dims(data['T_e'].to_numpy().flatten(), axis=0))
                if 'p_ohmic_e' in data:
                    dvec = data['p_ohmic_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qohme'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_generic_heat_e' in data:
                    dvec = data['p_generic_heat_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfe'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                    #data_vars['qbeame'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_generic_heat_i' in data:
                    dvec = data['p_generic_heat_i'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfi'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                    #data_vars['qbeami'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_icrh_e' in data:
                    dvec = data['p_icrh_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfe'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_icrh_i' in data:
                    dvec = data['p_icrh_i'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfi'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_ecrh_e' in data:
                    dvec = data['p_ecrh_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfe'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_ecrh_i' in data:
                    dvec = data['p_ecrh_i'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfi'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_cyclotron_radiation_e' in data:
                    dvec = data['p_cyclotron_radiation_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qsync'] = (['n', 'rho'], np.expand_dims(-1.0e-6 * dvec, axis=0))
                if 'p_bremsstrahlung_e' in data:
                    dvec = data['p_bremsstrahlung_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qbrem'] = (['n', 'rho'], np.expand_dims(-1.0e-6 * dvec, axis=0))
                if 'p_impurity_radiation_e' in data:
                    dvec = data['p_impurity_radiation_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qline'] = (['n', 'rho'], np.expand_dims(-1.0e-6 * dvec, axis=0))
                if 'p_alpha_e' in data:
                    dvec = data['p_alpha_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qfuse'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_alpha_i' in data:
                    dvec = data['p_alpha_i'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qfusi'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'ei_exchange' in data:
                    dvec = data['ei_exchange'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qei'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'j_ohmic' in data:
                    dvec = data['j_ohmic'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['johm'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'j_bootstrap' in data:
                    #dvec = np.concatenate([np.array([np.nan]), data['j_bootstrap'].to_numpy().flatten(), np.array([np.nan])], axis=0)
                    data_vars['jbs'] = (['n', 'rho'], np.expand_dims(1.0e-6 * data['j_bootstrap'].to_numpy().flatten(), axis=0))
                    #data_vars['jbstor'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'j_ecrh' in data:
                    dvec = data['j_ecrh'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['jrf'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'j_external' in data:
                    dvec = data['j_external'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['jrf'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                    #data_vars['jnb'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'j_generic_current' in data:
                    dvec = data['j_generic_current'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['jrf'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                #    data_vars['jnb'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'pressure_thermal_total' in data and 'rho_norm' in data and 'rho_face_norm' in data:
                    data_vars['ptot'] = (['n', 'rho'], np.expand_dims(data['pressure_thermal_total'].to_numpy().flatten(), axis=0))
                if 's_gas_puff' in data:
                    dvec = data['s_gas_puff'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qpar_wall'] = (['n', 'rho'], np.expand_dims(dvec, axis=0))
                if 's_pellet' in data:
                    dvec = data['s_pellet'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qpar_wall'] = (['n', 'rho'], np.expand_dims(dvec, axis=0))
                if 's_generic_particle' in data:
                    dvec = data['s_generic_particle'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qpar_beam'] = (['n', 'rho'], np.expand_dims(dvec, axis=0))
                #'qione'
                #'qioni'
                #'qcxi'
                #'vtor'
                #'vpol'
                #'omega0'
                #'qmom'
                attrs['header'] = newobj.make_file_header()
                newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj


    @classmethod
    def from_imas(
        cls,
        obj: io,
        side: str = 'output',
        window: Sequence[int | float] | None = None,
        transpose_equilibrium: bool = False,
        **kwargs: Any,
    ) -> Self:

        newobj = cls()
        if isinstance(obj, io):

            data: xr.Dataset = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            time_cp = 'core_profiles.time'
            rho_cp_i = 'core_profiles.profiles_1d.grid.rho_tor_norm:i'
            rho_cp = 'core_profiles.profiles_1d.grid.rho_tor_norm'
            ion_cp_i = 'core_profiles.profiles_1d.ion:i'
            ion_cp = 'core_profiles.profiles_1d.ion.label'
            time_eq = 'equilibrium.time'
            psi_eq_i = 'equilibrium.time_slice.profiles_1d.psi:i'
            psi_eq = 'equilibrium.time_slice.profiles_1d.psi'
            rho_eq = 'equilibrium.time_slice.profiles_1d.rho_tor_norm'
            time_cs = 'core_sources.time'
            src_cs_i = 'core_sources.source:i'
            src_cs = 'core_sources.source.identifier.name'
            rho_cs_i = 'core_sources.source.profiles_1d.grid.rho_tor_norm:i'
            rho_cs = 'core_sources.source.profiles_1d.grid.rho_tor_norm'
            ion_cs_i = 'core_sources.source.profiles_1d.ion:i'
            ion_cs = 'core_sources.source.profiles_1d.ion.label'
            cocos = define_cocos_converter(17, 2)  # Assumed IMAS=17 -> GACODE=2
            ikwargs = {'fill_value': 'extrapolate'}

            dsvec = []

            if time_cp in data.coords:

                time_indices = [-1]
                time = np.array([data.get(time_cp, xr.DataArray()).to_numpy().flatten()[time_indices]]).flatten()  #TODO: Use window argument
                for i, time_index in enumerate(time_indices):

                    coords = {}
                    data_vars = {}
                    attrs: MutableMapping[str, Any] = {}

                    if rho_cp_i in data.dims and rho_cp in data:
                        data = data.isel({time_cp: time_index}).swap_dims({rho_cp_i: rho_cp})
                        if ion_cp_i in data.dims and ion_cp in data:
                            data = data.swap_dims({ion_cp_i: ion_cp})
                        coords['n'] = np.array([i], dtype=int)
                        coords['rho'] = data[rho_cp].to_numpy().flatten()
                        data_vars['nexp'] = (['n'], np.array([len(coords['rho'])], dtype=int))
                        data_vars['shot'] = (['n'], np.atleast_1d([0]))
                        data_vars['masse'] = (['n'], np.atleast_1d([5.4488748e-04]))
                        data_vars['ze'] = (['n'], np.atleast_1d([-1.0]))
                        if ion_cp in data.coords:
                            coords['name'] = data[ion_cp].to_numpy().flatten()
                            data_vars['nion'] = (['n'], np.array([len(coords['name'])], dtype=int))
                            ni = None
                            zi = None
                            tag = 'core_profiles.profiles_1d.ion.density_thermal'
                            if tag in data:
                                types = []
                                for name in coords['name']:
                                    types.extend(['[therm]' if data[tag].sel({ion_cp: name}).sum() > 0.0 else '[fast]'])
                                ni = data[tag]
                                data_vars['ni'] = (['n', 'rho', 'name'], 1.0e-19 * np.expand_dims(ni.to_numpy().T, axis=0))
                                data_vars['type'] = (['n', 'name'], np.expand_dims(types, axis=0))
                            tag = 'core_profiles.profiles_1d.ion.temperature'
                            if tag in data:
                                ti = data[tag]
                                data_vars['ti'] = (['n', 'rho', 'name'], 1.0e-3 * np.expand_dims(ti.to_numpy().T, axis=0))
                            eltag = 'core_profiles.profiles_1d.ion.element:i'
                            tag = 'core_profiles.profiles_1d.ion.element.a'
                            if tag in data:
                                data_vars['mass'] = (['n', 'name'], np.expand_dims(data[tag].isel({eltag: 0}).to_numpy(), axis=0))
                            tag = 'core_profiles.profiles_1d.ion.element.z_n'
                            if tag in data:
                                zi = data[tag].isel({eltag: 0})
                                data_vars['z'] = (['n', 'name'], np.expand_dims(zi.to_numpy(), axis=0))
                            tag = 'core_profiles.profiles_1d.ion.z_ion_1d'  # Potential source of mismatch
                            if tag in data:
                                zi = data[tag]
                            tag = 'core_profiles.profiles_1d.electrons.density_thermal'
                            if tag in data and ni is not None and zi is not None:
                                zeff = (ni * zi * zi).sum(ion_cp) / data[tag]
                                data_vars['z_eff'] = (['n', 'rho'], np.expand_dims(zeff.to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.electrons.density_thermal'
                        if tag in data:
                            ne = data[tag]
                            data_vars['ne'] = (['n', 'rho'], 1.0e-19 * np.expand_dims(ne.to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.electrons.temperature'
                        if tag in data:
                            te = data[tag]
                            data_vars['te'] = (['n', 'rho'], 1.0e-3 * np.expand_dims(te.to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.pressure_thermal'
                        if tag in data:
                            data_vars['ptot'] = (['n', 'rho'], np.expand_dims(data[tag].to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.q'
                        if tag in data:
                            data_vars['q'] = (['n', 'rho'], cocos['spol'] * np.expand_dims(data[tag].to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.j_ohmic'
                        if tag in data:
                            data_vars['johm'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * np.expand_dims(data[tag].to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.j_bootstrap'
                        if tag in data:
                            data_vars['jbs'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * np.expand_dims(data[tag].to_numpy(), axis=0))
                        #tag = 'core_profiles.profiles_1d.momentum_tor'
                        tag = 'core_profiles.profiles_1d.ion.velocity.toroidal'
                        if tag in data:
                            data_vars['vtor'] = (['n', 'rho', 'name'], cocos['scyl'] * np.expand_dims(data[tag].to_numpy().T, axis=0))
                        tag = 'core_profiles.profiles_1d.ion.velocity.poloidal'
                        if tag in data:
                            data_vars['vpol'] = (['n', 'rho', 'name'], cocos['spol'] * np.expand_dims(data[tag].to_numpy().T, axis=0))
                        tag = 'core_profiles.profiles_1d.rotation_frequency_tor_sonic'
                        if tag in data:
                            data_vars['omega0'] = (['n', 'rho'], cocos['scyl'] * np.expand_dims(data[tag].to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.grid.rho_tor'
                        if tag in data and 'core_profiles.vacuum_toroidal_field.b0' in data:
                            torflux = data[tag].interp({rho_cp: np.array([1.0])}, kwargs=ikwargs) ** 2.0 / (np.pi * data['core_profiles.vacuum_toroidal_field.b0'])
                            data_vars['torfluxa'] = (['n'], torflux.to_numpy().flatten())

                    if time_eq in data.coords and psi_eq_i in data.dims and rho_eq in data and 'rho' in coords:
                        data = data.interp({time_eq: time.item(i)}, kwargs=ikwargs).swap_dims({psi_eq_i: rho_eq})
                        eqdsk_data = obj.to_eqdsk(time_index=time_index, side=side, transpose=transpose_equilibrium) if hasattr(obj, 'to_eqdsk') else {}
                        rhovec = data.get(rho_eq, xr.DataArray()).to_numpy().flatten()
                        tag = 'equilibrium.time_slice.profiles_1d.psi'
                        if tag in data:
                            #ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'psi': (['rho_int'], data[tag].to_numpy().flatten())})
                            #data_vars['polflux'] = (['n', 'rho'], np.expand_dims(ndata['psi'].interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['polflux'] = (['n', 'rho'], np.expand_dims(data[tag].interp({rho_eq: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        tag = 'equilibrium.vacuum_toroidal_field.r0'
                        if tag in data:
                            data_vars['rcentr'] = (['n'], np.atleast_1d(data[tag].to_numpy()))
                        tag = 'equilibrium.vacuum_toroidal_field.b0'
                        if tag in data:
                            data_vars['bcentr'] = (['n'], cocos['scyl'] * np.atleast_1d(data[tag].to_numpy()))
                        tag = 'equilibrium.time_slice.profiles_1d.pressure'
                        if tag in data and 'ptot' not in data_vars:
                            #ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'pressure': (['rho_int'], data[tag].to_numpy().flatten())})
                            #data_vars['ptot'] = (['n', 'rho'], np.expand_dims(ndata['pressure'].interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['ptot'] = (['n', 'rho'], np.expand_dims(data[tag].interp({rho_cp: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        tag = 'equilibrium.time_slice.profiles_1d.q'
                        if tag in data and 'q' not in data_vars:
                            #ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'q': (['rho_int'], data[tag].to_numpy().flatten())})
                            #data_vars['q'] = (['n', 'rho'], np.expand_dims(ndata['q'].interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['q'] = (['n', 'rho'], np.expand_dims(data[tag].interp({rho_cp: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        if eqdsk_data:
                            psivec = data_vars['polflux'][1].flatten() if 'polflux' in data_vars else np.linspace(eqdsk_data['simagx'], eqdsk_data['sibdry'], len(coords['rho']))
                            mxh_data = newobj._calculate_geometry_from_eqdsk(eqdsk_data, psivec)
                            data_vars['rmaj'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['rmaj']), axis=0))
                            data_vars['rmin'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['rmin']), axis=0))
                            data_vars['zmag'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['zmag']), axis=0))
                            data_vars['kappa'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['kappa']), axis=0))
                            data_vars['delta'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['delta']), axis=0))
                            data_vars['zeta'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['zeta']), axis=0))
                            data_vars['shape_sin3'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin3']), axis=0))
                            data_vars['shape_sin4'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin4']), axis=0))
                            data_vars['shape_sin5'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin5']), axis=0))
                            data_vars['shape_sin6'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin6']), axis=0))
                            data_vars['shape_cos0'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos0']), axis=0))
                            data_vars['shape_cos1'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos1']), axis=0))
                            data_vars['shape_cos2'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos2']), axis=0))
                            data_vars['shape_cos3'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos3']), axis=0))
                            data_vars['shape_cos4'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos4']), axis=0))
                            data_vars['shape_cos5'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos5']), axis=0))
                            data_vars['shape_cos6'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos6']), axis=0))
                        tag = 'equilibrium.time_slice.global_quantities.ip'
                        if tag in data:
                            data_vars['current'] = (['n'], np.atleast_1d(data[tag].to_numpy()))
                        itag = 'equilibrium.time_slice.profiles_1d.r_inboard'
                        otag = 'equilibrium.time_slice.profiles_1d.r_outboard'
                        if itag in data and otag in data and ('rmaj' not in data_vars or 'rmin' not in data_vars):
                            #ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={
                            #    'r_inboard': (['rho_int'], data[itag].to_numpy().flatten()),
                            #    'r_outboard': (['rho_int'], data[otag].to_numpy().flatten())
                            #})
                            #data_vars['rmin'] = (['n', 'rho'], np.expand_dims((0.5 * (ndata['r_outboard'] - ndata['r_inboard'])).interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            #data_vars['rmaj'] = (['n', 'rho'], np.expand_dims((0.5 * (ndata['r_outboard'] + ndata['r_inboard'])).interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['rmin'] = (['n', 'rho'], np.expand_dims((0.5 * (data[otag] - data[itag])).interp({rho_cp: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['rmaj'] = (['n', 'rho'], np.expand_dims((0.5 * (data[otag] + data[itag])).interp({rho_cp: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        #tag = 'equilibrium.time_slice.global_quantities.magnetic_axis.z'
                        #if tag in data and 'zmag' not in data_vars:
                        #    data_vars['zmag'] = (['n', 'rho'], np.expand_dims(np.repeat(data[tag].to_numpy().flatten(), len(coords['rho']), axis=0), axis=0))
                        tag = 'equilibrium.time_slice.profiles_1d.elongation'
                        if tag in data and 'kappa' not in data_vars:
                            #ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'elongation': (['rho_int'], data[tag].to_numpy().flatten())})
                            #data_vars['kappa'] = (['n', 'rho'], np.expand_dims(ndata['elongation'].interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['kappa'] = (['n', 'rho'], np.expand_dims(data[tag].interp({rho_cp: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        #if 'equilibrium.time_slice.profiles_1d.triangularity_upper' in data or 'equilibrium.time_slice.profiles_1d.triangularity_lower' in data and 'delta' not in data_vars:
                            #tri = np.zeros(data['rho(-)'].shape)
                            #itri = 0
                            #if hasattr(time_struct.profiles_1d, 'triangularity_upper'):
                            #    tri += time_struct.profiles_1d.triangularity_upper.flatten()
                            #    itri += 1
                            #if hasattr(time_struct.profiles_1d, 'triangularity_lower') and len(time_struct.profiles_1d.triangularity_lower) == data['nexp']:
                            #    tri += time_struct.profiles_1d.triangularity_lower.flatten()
                            #    itri += 1
                            #data['delta(-)'] = tri / float(itri) if itri > 0 else tri

                    if time_cs in data.coords and src_cs_i in data.dims and src_cs in data and rho_cs_i in data.dims and rho_cs in data and 'rho' in coords:
                        data = data.interp({time_cs: time.item(i)}, kwargs=ikwargs).swap_dims({src_cs_i: src_cs})
                        if ion_cs_i in data.dims and ion_cs in data:
                            data = data.swap_dims({ion_cs_i: ion_cs})
                        srclist = data[src_cs].to_numpy().tolist()
                        qrfe = np.zeros((len(coords['rho']), ))
                        qrfi = np.zeros((len(coords['rho']), ))
                        jrf = np.zeros((len(coords['rho']), ))
                        tag = 'core_sources.source.profiles_1d.electrons.energy'
                        if tag in data:
                            srctag = 'ohmic'
                            if srctag in srclist:
                                data_vars['qohme'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'ec'
                            if srctag in srclist:
                                qrfe += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'ic'
                            if srctag in srclist:
                                qrfe += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'lh'
                            if srctag in srclist:
                                qrfe += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'nbi'
                            if srctag in srclist:
                                data_vars['qbeame'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'synchrotron_radiation'
                            if srctag in srclist:
                                data_vars['qsync'] = (['n', 'rho'], -1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'radiation'
                            if srctag in srclist:
                                data_vars['qline'] = (['n', 'rho'], -1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'bremsstrahlung'
                            if srctag in srclist:
                                data_vars['qbrem'] = (['n', 'rho'], -1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'fusion'
                            if srctag in srclist:
                                data_vars['qfuse'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'collisional_equipartition'
                            if srctag in srclist:
                                data_vars['qei'] = (['n', 'rho'], -1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        if 'qbrem' not in data_vars:  # Why single this one out randomly?
                            data_vars['qbrem'] = (['n', 'rho'], np.expand_dims(np.zeros_like(coords['rho']), axis=0))
                        tag = 'core_sources.source.profiles_1d.total_ion_energy'
                        if tag in data:
                            srctag = 'ic'
                            if srctag in srclist:
                                qrfi += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'lh'
                            if srctag in srclist:
                                qrfi += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'nbi'
                            if srctag in srclist:
                                data_vars['qbeami'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'charge_exchange'
                            if srctag in srclist:
                                data_vars['qcxi'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'fusion'
                            if srctag in srclist:
                                data_vars['qfusi'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        tag = 'core_sources.source.profiles_1d.j_parallel'
                        if tag in data:
                            srctag = 'ohmic'
                            if srctag in srclist and 'johm' not in data_vars:
                                data_vars['johm'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'j_bootstrap'
                            if srctag in srclist and 'jbs' not in data_vars:
                                data_vars['jbs'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                                #data_vars['jbstor'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                            srctag = 'ec'
                            if srctag in srclist:
                                jrf += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'ic'
                            if srctag in srclist:
                                jrf += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'lh'
                            if srctag in srclist:
                                jrf += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'nbi'
                            if srctag in srclist:
                                data_vars['jnb'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        tag = 'core_sources.source.profiles_1d.ion.particles'
                        if tag in data and ion_cs in data.coords:
                            srctag = 'cold_neutrals'
                            if srctag in srclist:
                                data_vars['qpar_wall'] = (['n', 'rho'], np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).sum(ion_cs).to_numpy(), axis=0))
                            srctag = 'nbi'
                            if srctag in srclist:
                                data_vars['qpar_beam'] = (['n', 'rho'], np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).sum(ion_cs).to_numpy(), axis=0))
                        tag = 'core_sources.source.profiles_1d.momentum_tor'
                        if tag in data:
                            srctag = 'nbi'
                            if srctag in srclist:
                                data_vars['qmom'] = (['n', 'rho'], cocos['scyl'] * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        if np.abs(qrfe).sum() > 0.0:
                            data_vars['qrfe'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(qrfe, axis=0))
                        if np.abs(qrfi).sum() > 0.0:
                            data_vars['qrfi'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(qrfi, axis=0))
                        if np.abs(jrf).sum() > 0.0:
                            data_vars['jrf'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * np.expand_dims(jrf, axis=0))

                    dsvec.append(xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs))

            if len(dsvec) > 0:
                newobj.input = xr.concat(dsvec, dim='n').assign_attrs({'header': newobj.make_file_header()})

        return newobj


    @classmethod
    def from_astra(
        cls,
        obj: io,
        side: str = 'output',
        window: Sequence[int | float] | None = None,
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            coords = {}
            data_vars = {}
            attrs: MutableMapping[str, Any] = {}
            if 'xrho' in data.coords:
                data = data.isel(time=-1)
                zeros = np.zeros_like(data.coords['xrho'].to_numpy().flatten())
                #name = []
                coords['n'] = np.array([0], dtype=int)
                coords['rho'] = data.coords['xrho'].to_numpy().flatten()
                data_vars['nexp'] = (['n'], np.array([len(coords['rho'])], dtype=int))
                if 'te' in data:
                    data_vars['te'] = (['n', 'rho'], np.expand_dims(data['te'].to_numpy().flatten(), axis=0))
                if 'ti' in data:
                    data_vars['ti'] = (['n', 'rho'], np.expand_dims(data['ti'].to_numpy().flatten(), axis=0))
            attrs['header'] = newobj.make_file_header()
            newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj
