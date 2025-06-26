import datetime
from pathlib import Path
import logging
import numpy as np
import xarray as xr
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


    def make_file_header(self):
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


    def add_geometry_from_eqdsk(self, path, side='input', overwrite=False):
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
    def _calculate_geometry_from_eqdsk(self, eqdsk_data, psivec):
        mxh_data = {
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
            rvec = np.linspace(eqdsk_data['rleft'], eqdsk_data['rleft'] + eqdsk_data['rdim'], eqdsk_data['nr'])
            zvec = np.linspace(eqdsk_data['zmid'] - 0.5 * eqdsk_data['zdim'], eqdsk_data['zmid'] + 0.5 * eqdsk_data['zdim'], eqdsk_data['nz'])
            if np.isclose(eqdsk_data['psi'][0, 0], eqdsk_data['psi'][-1, -1]) and np.isclose(eqdsk_data['psi'][0, -1], eqdsk_data['psi'][-1, 0]):
                if eqdsk_data['simagx'] > eqdsk_data['sibdry'] and psivec[-1] < eqdsk_data['psi'][0, 0]:
                    psivec[-1] = eqdsk_data['psi'][0, 0] + 1.0e-6
                elif eqdsk_data['simagx'] < eqdsk_data['sibdry'] and psivec[-1] > eqdsk_data['psi'][0, 0]:
                    psivec[-1] = eqdsk_data['psi'][0, 0] - 1.0e-6
            if psivec[0] == eqdsk_data['simagx']:
                if eqdsk_data['simagx'] > eqdsk_data['sibdry']:
                    psivec[0] = eqdsk_data['simagx'] - 1.0e-6
                elif eqdsk_data['simagx'] < eqdsk_data['sibdry']:
                    psivec[0] = eqdsk_data['simagx'] + 1.0e-6
            rmesh, zmesh = np.meshgrid(rvec, zvec, indexing='ij')
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
        return mxh_data


    def read(self, path, side='output'):
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

        coords = {}
        data_vars = {}
        attrs = {}

        if isinstance(path, (str, Path)):
            ipath = Path(path)
            lines = []
            if ipath.is_file():
                titles_single = self.titles_singleInt + self.titles_singleStr + self.titles_singleFloat
                with open(ipath, 'r') as f:
                    lines = f.readlines()

            istartProfs = None
            for i in range(len(lines)):
                if "# nexp" in lines[i]:
                    istartProfs = i
                    break
            header = lines[:istartProfs]
            if header[-1].strip() == '#':
                header = header[:-1]
            attrs['header'] = ''.join(header).strip()

            singleLine, title, var = None, None, None
            found = False
            singles = {}
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
                            singles[title] = np.array(var0, dtype=float)
                        elif title in self.titles_singleInt:
                            singles[title] = np.array(var0, dtype=int)
                        else:
                            singles[title] = np.array(var0, dtype=str)
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

            ncoord = 'n'
            rcoord = 'rho' if 'rho' in profiles else 'polflux'
            scoord = 'name' if 'name' in singles else 'z'
            coords[ncoord] = [0]
            if rcoord in profiles:
                coords[rcoord] = profiles.pop(rcoord)
            if scoord in singles:
                coords[scoord] = singles.pop(scoord)
            for key, val in profiles.items():
                if key in ['rho', 'polflux', 'rmin']:
                    coords[key] = ([ncoord, rcoord], np.expand_dims(val, axis=0))
                elif key in ['ni', 'ti', 'vtor', 'vpol']:
                    data_vars[key] = ([ncoord, rcoord, scoord], np.expand_dims(val, axis=0))
                elif key in ['w0']:
                    data_vars['omega0'] = ([ncoord, rcoord], np.expand_dims(val, axis=0))
                else:
                    data_vars[key] = ([ncoord, rcoord], np.expand_dims(val, axis=0))
            for key, val in singles.items():
                if key in ['name', 'z', 'mass', 'type']:
                    coords[key] = ([ncoord, scoord], np.expand_dims(val, axis=0))
                elif key in ['header']:
                    attrs[key] = val
                else:
                    data_vars[key] = ([ncoord], val)

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


    def _write_gacode_file(self, path, data, overwrite=False):

        if isinstance(data, xr.DataTree):
            data = data.to_dataset().sel(n=0, drop=True) if not data.is_empty else None

        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            opath = Path(path)
            processed_titles = []
            header = data.attrs.get('header', '').split('\n')
            lines = [f'{line:<70}\n' for line in header]
            lines += ['#\n']
            processed_titles.append('header')
            for title in self.titles_singleInt:
                newlines = []
                if title in data:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(f'{data[title]:d}\n')
                    processed_titles.append(title)
                lines += newlines
            for title in self.titles_singleStr:
                newlines = []
                if title in data:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(' '.join([f'{val}' for val in data[title].to_numpy().flatten().tolist()]) + '\n')
                    processed_titles.append(title)
                lines += newlines
            for title in self.titles_singleFloat:
                newlines = []
                if title in data:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(' '.join([f'{val:14.7E}' for val in data[title].to_numpy().flatten().tolist()]) + '\n')
                    processed_titles.append(title)
                lines += newlines
            for title in list(data.coords) + list(data.data_vars):
                newlines = []
                if title not in processed_titles:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    else:
                        newtitle += f' | -'
                    newlines.append(f'# {newtitle}\n')
                    rcoord = [f'{dim}' for dim in data[title].dims if dim in ['rho', 'polflux', 'rmin']]
                    for ii in range(len(data[rcoord[0]])):
                        newlines.append(' '.join([f'{ii+1:3d}'] + [f'{val:14.7E}' for val in data[title].isel(**{f'{rcoord[0]}': ii}).to_numpy().flatten().tolist()]) + '\n')
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
    def from_file(cls, path=None, input=None, output=None):
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


    # Assumed that the self creation method transfers output to input
    @classmethod
    def from_gacode(cls, obj, side='output'):
        newobj = cls()
        if isinstance(obj, io):
            newobj.input = obj.input if side == 'input' else obj.output
        return newobj


    @classmethod
    def from_torax(cls, obj, side='output', window=None):
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            if 'rho_norm' in data.coords:
                data = data.isel(time=-1)
                zeros = np.zeros_like(data.coords['rho_norm'].to_numpy().flatten())
                coords = {}
                data_vars = {}
                attrs = {}
                name = []
                coords['n'] = [0]
                if 'rho_norm' in data.coords:
                    coords['rho'] = data.coords['rho_norm'].to_numpy().flatten()
                    data_vars['nexp'] = (['n'], [len(coords['rho'])])
                if 'psi' in data:
                    coords['polflux'] = (['n', 'rho'], np.expand_dims(data['psi'].to_numpy().flatten(), axis=0))
                if 'r_mid' in data:
                    coords['rmin'] = (['n', 'rho'], np.expand_dims(data['r_mid'].to_numpy().flatten(), axis=0))
                data_vars['shot'] = (['n'], [0])
                data_vars['masse'] = (['n'], [5.4488748e-04])
                data_vars['ze'] = (['n'], [-1.0])
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
                    coords['name'] = names
                    data_vars['ni'] = (['n', 'rho', 'name'], np.expand_dims(ni, axis=0))
                    data_vars['nion'] = (['n'], [len(names)])
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
    def from_imas(cls, obj, side='output', window=None):
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            cocos = define_cocos_converter(17, 2)  # Assumed IMAS=17 -> GACODE=2
            time_index = -1
            time = np.array([data.get('time_cp').to_numpy().flatten()[time_index]])  #TODO: Use window argument
            ikwargs = {'fill_value': 'extrapolate'}
            coords = {}
            data_vars = {}
            attrs = {}
            if 'rho_cp' in data.coords:
                coords['n'] = [0]
                coords['rho'] = data.coords['rho_cp'].to_numpy().flatten()
                data_vars['nexp'] = (['n'], [len(coords['rho'])])
                data_vars['shot'] = (['n'], [0])
                data_vars['masse'] = (['n'], [5.4488748e-04])
                data_vars['ze'] = (['n'], [-1.0])
                if 'ion_cp' in data.coords:
                    coords['name'] = data.coords['ion_cp'].to_numpy().flatten()
                    data_vars['nion'] = (['n'], [len(coords['name'])])
                    ni = None
                    zi = None
                    tag = 'core_profiles&profiles_1d[]&ion[]&density_thermal'
                    if tag in data:
                        types = []
                        for j in range(len(coords['name'])):
                            types.extend(['[therm]' if data[tag].isel(ion_cp=j).interp(time_cp=time).sum() > 0.0 else '[fast]'])
                        ni = data[tag].interp(time_cp=time)
                        data_vars['ni'] = (['n', 'rho', 'name'], 1.0e-19 * np.transpose(ni.to_numpy(), axes=(0, 2, 1)))
                        data_vars['type'] = (['n', 'name'], np.expand_dims(types, axis=0))
                    tag = 'core_profiles&profiles_1d[]&ion[]&temperature'
                    if tag in data:
                        ti = data[tag].interp(time_cp=time)
                        data_vars['ti'] = (['n', 'rho', 'name'], 1.0e-3 * np.transpose(ti.to_numpy(), axes=(0, 2, 1)))
                    tag = 'core_profiles&profiles_1d[]&ion[]&element[]&a'
                    if tag in data:
                        data_vars['mass'] = (['n', 'name'], data[tag].interp(time_cp=time).to_numpy())
                    tag = 'core_profiles&profiles_1d[]&ion[]&element[]&z_n'
                    if tag in data:
                        zi = data[tag].interp(time_cp=time)
                        data_vars['z'] = (['n', 'name'], zi.to_numpy())
                    tag = 'core_profiles&profiles_1d[]&ion[]&z_ion_1d'  # Potential source of mismatch
                    if tag in data:
                        zi = data[tag].interp(time_cp=time)
                    tag = 'core_profiles&profiles_1d[]&electrons&density_thermal'
                    if ni is not None and zi is not None and tag in data:
                        zeff = (ni * zi * zi).sum('ion_cp') / data[tag].interp(time_cp=time)
                        data_vars['z_eff'] = (['n', 'rho'], zeff.to_numpy())
                tag = 'core_profiles&profiles_1d[]&electrons&density_thermal'
                if tag in data:
                    ne = data[tag].interp(time_cp=time)
                    data_vars['ne'] = (['n', 'rho'], 1.0e-19 * ne.to_numpy())
                tag = 'core_profiles&profiles_1d[]&electrons&temperature'
                if tag in data:
                    te = data[tag].interp(time_cp=time)
                    data_vars['te'] = (['n', 'rho'], 1.0e-3 * te.to_numpy())
                tag = 'core_profiles&profiles_1d[]&pressure_thermal'
                if tag in data:
                    data_vars['ptot'] = (['n', 'rho'], data[tag].interp(time_cp=time).to_numpy())
                tag = 'core_profiles&profiles_1d[]&q'
                if tag in data:
                    data_vars['q'] = (['n', 'rho'], cocos['spol'] * data[tag].interp(time_cp=time).to_numpy())
                tag = 'core_profiles&profiles_1d[]&j_ohmic'
                if tag in data:
                    data_vars['johm'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * data[tag].interp(time_cp=time).to_numpy())
                tag = 'core_profiles&profiles_1d[]&j_bootstrap'
                if tag in data:
                    data_vars['jbs'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * data[tag].interp(time_cp=time).to_numpy())
                #tag = 'core_profiles&profiles_1d[]&momentum_tor'
                tag = 'core_profiles&profiles_1d[]&ion[]&velocity&toroidal'
                if tag in data:
                    data_vars['vtor'] = (['n', 'rho', 'name'], cocos['scyl'] * np.transpose(data[tag].interp(time_cp=time).to_numpy(), axes=(0, 2, 1)))
                tag = 'core_profiles&profiles_1d[]&ion[]&velocity&poloidal'
                if tag in data:
                    data_vars['vpol'] = (['n', 'rho', 'name'], cocos['spol'] * np.transpose(data[tag].interp(time_cp=time).to_numpy(), axes=(0, 2, 1)))
                tag = 'core_profiles&profiles_1d[]&rotation_frequency_tor_sonic'
                if tag in data:
                    data_vars['omega0'] = (['n', 'rho'], cocos['scyl'] * data[tag].interp(time_cp=time).to_numpy())
                tag = 'core_profiles&profiles_1d[]&grid&rho_tor'
                if tag in data and 'core_profiles&vacuum_toroidal_field&b0' in data:
                    torflux = data[tag].interp(time_cp=time, rho_cp=np.array([1.0]), kwargs=ikwargs) ** 2.0 / (np.pi * data['core_profiles&vacuum_toroidal_field&b0'].interp(time_cp=time))
                    data_vars['torfluxa'] = (['n'], torflux.to_numpy().flatten())
            if 'rho_eq' in data.coords or 'equilibrium&time_slice[]&profiles_1d[]&rho_tor_norm' in data:
                eqdsk_data = obj.to_eqdsk(time_index=time_index, side=side)
                rhovec = data.get('rho_eq')
                if rhovec is None:
                    rhovec = data['equilibrium&time_slice[]&profiles_1d[]&rho_tor_norm'].interp(time_eq=time, kwargs=ikwargs)
                rhovec = rhovec.to_numpy().flatten()
                tag = 'equilibrium&time_slice[]&profiles_1d&psi'
                if tag in data:
                    ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'psi': (['rho_int'], data[tag].interp(time_eq=time, kwargs=ikwargs).to_numpy().flatten())})
                    data_vars['polflux'] = (['n', 'rho'], np.expand_dims(ndata['psi'].interp(rho_int=coords['rho'], kwargs=ikwargs).to_numpy(), axis=0))
                tag = 'equilibrium&vacuum_toroidal_field&r0'
                if tag in data:
                    data_vars['rcentr'] = (['n'], data[tag].interp(time_eq=time, kwargs=ikwargs).to_numpy())
                tag = 'equilibrium&vacuum_toroidal_field&b0'
                if tag in data:
                    data_vars['bcentr'] = (['n'], cocos['scyl'] * data[tag].interp(time_eq=time, kwargs=ikwargs).to_numpy())
                tag = 'equilibrium&time_slice[]&profiles_1d&pressure'
                if tag in data and 'ptot' not in data_vars:
                    ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'pressure': (['rho_int'], data[tag].interp(time_eq=time, kwargs=ikwargs).to_numpy().flatten())})
                    data_vars['ptot'] = (['n', 'rho'], np.expand_dims(ndata['pressure'].interp(rho_int=coords['rho'], kwargs=ikwargs).to_numpy(), axis=0))
                tag = 'equilibrium&time_slice[]&profiles_1d&q'
                if tag in data and 'q' not in data_vars:
                    ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'q': (['rho_int'], data[tag].interp(time_eq=time, kwargs=ikwargs).to_numpy().flatten())})
                    data_vars['q'] = (['n', 'rho'], np.expand_dims(ndata['q'].interp(rho_int=coords['rho'], kwargs=ikwargs).to_numpy(), axis=0))
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
                tag = 'equilibrium&time_slice[]&global_quantities&ip'
                if tag in data:
                    data_vars['current'] = (['n'], data[tag].interp(time_eq=time, kwargs=ikwargs).to_numpy().flatten())
                if 'equilibrium&time_slice[]&profiles_1d[]&r_inboard' in data and 'equilibrium&time_slice[]&profiles_1d[]&r_outboard' in data and ('rmaj' not in data_vars or 'rmin' not in data_vars):
                    ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={
                        'r_inboard': (['rho_int'], data['equilibrium&time_slice[]&profiles_1d[]&r_inboard'].interp(time_eq=time, kwargs=ikwargs).to_numpy().flatten()),
                        'r_outboard': (['rho_int'], data['equilibrium&time_slice[]&profiles_1d[]&r_outboard'].interp(time_eq=time, kwargs=ikwargs).to_numpy().flatten())
                    })
                    data_vars['rmin'] = (['n', 'rho'], np.expand_dims((0.5 * (ndata['r_outboard'] - ndata['r_inboard'])).interp(rho_int=coords['rho'], kwargs=ikwargs).to_numpy(), axis=0))
                    data_vars['rmaj'] = (['n', 'rho'], np.expand_dims((0.5 * (ndata['r_outboard'] + ndata['r_inboard'])).interp(rho_int=coords['rho'], kwargs=ikwargs).to_numpy(), axis=0))
                tag = 'equilibrium&time_slice[]&global_quantities&magnetic_axis&z'
                if tag in data and 'zmag' not in data_vars:
                    data_vars['zmag'] = (['n', 'rho'], np.expand_dims(np.repeat(data[tag].interp(time_eq=time, kwargs=ikwargs).to_numpy().flatten(), len(coords['rho']), axis=0), axis=0))
                tag = 'equilibrium&time_slice[]&profiles_1d&elongation'
                if tag in data and 'kappa' not in data_vars:
                    ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'elongation': (['rho_int'], data[tag].interp(time_eq=time, kwargs=ikwargs).to_numpy().flatten())})
                    data_vars['kappa'] = (['n', 'rho'], np.expand_dims(ndata['elongation'].interp(rho_int=coords['rho'], kwargs=ikwargs).to_numpy(), axis=0))
                if 'equilibrium&time_slice[]&profiles_1d[]&triangularity_upper' in data or 'equilibrium&time_slice[]&profiles_1d[]&triangularity_lower' in data and 'delta' not in data_vars:
                    #tri = np.zeros(data['rho(-)'].shape)
                    #itri = 0
                    #if hasattr(time_struct.profiles_1d, 'triangularity_upper'):
                    #    tri += time_struct.profiles_1d.triangularity_upper.flatten()
                    #    itri += 1
                    #if hasattr(time_struct.profiles_1d, 'triangularity_lower') and len(time_struct.profiles_1d.triangularity_lower) == data['nexp']:
                    #    tri += time_struct.profiles_1d.triangularity_lower.flatten()
                    #    itri += 1
                    #data['delta(-)'] = tri / float(itri) if itri > 0 else tri
                    pass
            #    data['polflux(Wb/radian)'] = cocos['sBp'] * time_struct.profiles_1d.psi.flatten() * np.power(2.0 * np.pi, cocos['eBp'])
            #if hasattr(time_struct.profiles_1d, 'phi') and 'torfluxa(Wb/radian)' not in data:
            #    data['torfluxa(Wb/radian)'] = cocos['scyl'] * np.array([time_struct.profiles_1d.phi[-1]], dtype=float) * np.power(2.0 * np.pi, cocos['eBp'])
            if 'rho_cs' in data.coords:
                tag = 'core_sources&ohmic&profiles_1d[]&electrons&energy'
                if tag in data:
                    data_vars['qohme'] = (['n', 'rho'], 1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                qrfe = np.zeros((1, len(coords['rho'])))
                tag = 'core_sources&ec&profiles_1d[]&electrons&energy'
                if tag in data:
                    qrfe += data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy()
                tag = 'core_sources&ic&profiles_1d[]&electrons&energy'
                if tag in data:
                    qrfe += data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy()
                tag = 'core_sources&lh&profiles_1d[]&electrons&energy'
                if tag in data:
                    qrfe += data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy()
                if np.abs(qrfe).sum() > 0.0:
                    data_vars['qrfe'] = (['n', 'rho'], 1.0e-6 * qrfe)
                qrfi = np.zeros((1, len(coords['rho'])))
                tag = 'core_sources&ec&profiles_1d[]&total_ion_energy'
                if tag in data:
                    qrfi += data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy()
                tag = 'core_sources&ic&profiles_1d[]&total_ion_energy'
                if tag in data:
                    qrfi += data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy()
                tag = 'core_sources&lh&profiles_1d[]&total_ion_energy'
                if tag in data:
                    qrfi += data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy()
                if np.abs(qrfi).sum() > 0.0:
                    data_vars['qrfi'] = (['n', 'rho'], 1.0e-6 * qrfi)
                tag = 'core_sources&nbi&profiles_1d[]&electrons&energy'
                if tag in data:
                    data_vars['qbeame'] = (['n', 'rho'], 1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                tag = 'core_sources&nbi&profiles_1d[]&total_ion_energy'
                if tag in data:
                    data_vars['qbeami'] = (['n', 'rho'], 1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                tag = 'core_sources&synchrotron_radiation&profiles_1d[]&electrons&energy'
                if tag in data:
                    data_vars['qsync'] = (['n', 'rho'], -1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                tag = 'core_sources&radiation&profiles_1d[]&electrons&energy'
                if tag in data:
                    data_vars['qline'] = (['n', 'rho'], -1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                    data_vars['qbrem'] = (['n', 'rho'], np.expand_dims(np.zeros_like(coords['rho']), axis=0))
                tag = 'core_sources&charge_exchange&profiles_1d[]&total_ion_energy'
                if tag in data:
                    data_vars['qcxi'] = (['n', 'rho'], 1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                tag = 'core_sources&fusion&profiles_1d[]&electrons&energy'
                if tag in data:
                    data_vars['qfuse'] = (['n', 'rho'], 1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                tag = 'core_sources&fusion&profiles_1d[]&total_ion_energy'
                if tag in data:
                    data_vars['qfusi'] = (['n', 'rho'], 1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                tag = 'core_sources&collisional_equipartition&profiles_1d[]&electrons&energy'
                if tag in data:
                    data_vars['qei'] = (['n', 'rho'], -1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                tag = 'core_sources&ohmic&profiles_1d[]&j_parallel'
                if tag in data and 'johm' not in data_vars:
                    data_vars['johm'] = (['n', 'rho'], 1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                tag = 'core_sources&bootstrap_current$profiles_1d[]&j_parallel'
                if tag in data and 'jbs' not in data_vars:
                    data_vars['jbs'] = (['n', 'rho'], 1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                    #data_vars['jbstor'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                jrf = np.zeros((1, len(coords['rho'])))
                tag = 'core_sources&ec&profiles_1d[]&j_parallel'
                if tag in data:
                    jrf += data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy()
                tag = 'core_sources&ic&profiles_1d[]&j_parallel'
                if tag in data:
                    jrf += data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy()
                tag = 'core_sources&lh&profiles_1d[]&j_parallel'
                if tag in data:
                    jrf += data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy()
                if np.abs(jrf).sum() > 0.0:
                    data_vars['jrf'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * jrf)
                tag = 'core_sources&nbi&profiles_1d[]&j_parallel'
                if tag in data:
                    data_vars['jnb'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
                tag = 'core_sources&cold_neutrals&profiles_1d[]&ion[]&particles'
                if tag in data:
                    data_vars['qpar_wall'] = (['n', 'rho'], data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).sum('ion_cs').to_numpy())
                tag = 'core_sources&nbi&profiles_1d[]&ion[]&particles'
                if tag in data:
                    data_vars['qpar_beam'] = (['n', 'rho'], data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).sum('ion_cs').to_numpy())
                tag = 'core_sources&nbi&profiles_1d[]&momentum_tor'
                if tag in data:
                    data_vars['qmom'] = (['n', 'rho'], cocos['scyl'] * data[tag].interp(time_cs=time, rho_cs=coords['rho'], kwargs=ikwargs).to_numpy())
            attrs['header'] = newobj.make_file_header()
            newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj


    @classmethod
    def from_astra(cls, obj, side='output', window=None):
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            if 'xrho' in data.coords:
                data = data.isel(time=-1)
                zeros = np.zeros_like(data.coords['xrho'].to_numpy().flatten())
                coords = {}
                data_vars = {}
                attrs = {}
                name = []
                coords['n'] = [0]
                coords['rho'] = data.coords['xrho'].to_numpy().flatten()
                data_vars['nexp'] = (['n'], [len(coords['rho'])])
                if 'te' in data:
                    data_vars['te'] = (['n', 'rho'], np.expand_dims(data['te'].to_numpy().flatten(), axis=0))
                if 'ti' in data:
                    data_vars['ti'] = (['n', 'rho'], np.expand_dims(data['ti'].to_numpy().flatten(), axis=0))
        return newobj
