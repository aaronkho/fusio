from pathlib import Path
import logging
import numpy as np
import xarray as xr
from fusio.classes.io import io

logger = logging.getLogger('fusio')


class gacode_io(io):

    basevars = [
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
    titles_singleInt = [
        'nexp',
        'nion',
        'shot',
    ]
    titles_singleStr = [
        'name',
        'type',
    ]
    titles_singleFloat = [
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
    units = {
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


    def correct_magnetic_fluxes(self, exponent=-1, side='input'):
        if side == 'input':
            if 'polflux' in self._input:
                self._tree['input']['polflux'] *= np.power(2.0 * np.pi, exponent)
            if 'torfluxa' in self._input:
                self._tree['input']['torfluxa'] *= np.power(2.0 * np.pi, exponent)
        else:
            if 'polflux' in self._output:
                self._tree['output']['polflux'] *= np.power(2.0 * np.pi, exponent)
            if 'torfluxa' in self._output:
                self._tree['output']['torfluxa'] *= np.power(2.0 * np.pi, exponent)


    def read(self, path, side='input'):
        if side == 'input':
            self.input = self._read_gacode_file(path)
        else:
            self.output = self._read_gacode_file(path)


    def write(self, path, side='input', overwrite=False):
        if side == 'input':
            self._write_gacode_file(path, self.input, overwrite=overwrite)
        else:
            self._write_gacode_file(path, self.output, overwrite=overwrite)


    def _read_gacode_file(self, path):

        ipath = Path(path) if isinstance(path, (str, Path)) else None
        coords = {}
        data_vars = {}
        attrs = {}
        titles_single = self.titles_singleInt + self.titles_singleStr + self.titles_singleFloat

        if ipath is not None and ipath.is_file():
            with open(ipath, 'r') as f:
                lines = f.readlines()

            for i in range(len(lines)):
                if "# nexp" in lines[i]:
                    istartProfs = i
                    break
            header = lines[:istartProfs]
            if header[-1].strip() == '#':
                header = header[:-1]
            attrs['header'] = '\n'.join(header)

            singleLine, title, var = None, None, None
            found = False
            profiles = {}
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
                            attrs[title] = np.array(var0, dtype=float)
                        elif title in self.titles_singleInt:
                            attrs[title] = np.array(var0, dtype=int)
                        else:
                            attrs[title] = np.array(var0, dtype=str)
                    else:
                        varT = [
                            float(j) if (j[-4].upper() == "E" or "." in j) else 0.0
                            for j in var0[1:]
                        ]
                        var.append(varT)

            # last
            if not singleLine:
                while len(var[-1]) < 1:
                    var = var[:-1]  # Sometimes there's an extra space, remove
                profiles[title] = np.array(var)
                if profiles[title].shape[1] == 1:
                    profiles[title] = profiles[title][:, 0]

            base_coord = 'rho' if 'rho' in profiles else 'polflux'
            if base_coord in profiles:
                coords[base_coord] = profiles.pop(base_coord)
            for key, val in profiles.items():
                if key in ['rho', 'polflux', 'rmin']:
                    coords[key] = ([base_coord], profiles[key])
                else:
                    data_vars[key] = ([base_coord], profiles[key])

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


    def _write_gacode_file(self, path, data, overwrite=False):

        opath = Path(path) if isinstance(path, (str, Path)) else None
        processed_titles = []

        if isinstance(data, (xr.Dataset, xr.DataTree)) and not data.is_empty:
            header = data['header'].split('\n')
            lines = [f'{line:<70}\n' for line in header]
            lines += ['#\n']
            processed_titles.append('header')
            for title in titles_singleInt:
                newlines = []
                if title in data:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(f'{data[title]:d}\n')
                    processed_titles.append(title)
                lines += newlines
            for title in titles_singleStr:
                newlines = []
                if title in data:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(' '.join([f'{val}' for val in data[title].flatten().tolist()]) + '\n')
                    processed_titles.append(title)
                lines += newlines
            for title in titles_singleFloat:
                newlines = []
                if title in data:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(' '.join([f'{val:14.7E}' for val in data[title].flatten().tolist()]) + '\n')
                    processed_titles.append(title)
                lines += newlines
            for title in data:
                newlines = []
                if title not in processed_titles:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    else:
                        newtitle += f' | -'
                    newlines.append(f'# {newtitle}\n')
                    if profiles[title].ndim > 1:
                        for ii in range(profiles[title].shape[0]):
                            newlines.append(' '.join([f'{ii:3d}'] + [f'{val:14.7E}' for val in data[title][ii].flatten().tolist()]) + '\n')
                    else:
                        newlines.extend([f'{ii:3d} {val:14.7E}\n' for ii, val in enumerate(data[title].flatten().tolist())])
                    processed_titles.append(title)
                lines += newlines

            with open(opath, 'w') as f:
                f.writelines(lines)
            logger.info(f'Saved {self.format} data into {path.resolve()}')

        else:
            logger.error(f'Attempting to write empty {self.format} class instance... Failed!')


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
