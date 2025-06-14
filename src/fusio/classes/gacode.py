from pathlib import Path
import logging
import numpy as np
import xarray as xr
from .io import io
from ..utils.plasma_tools import define_ion_species

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
            attrs['header'] = ''.join(header)

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
                    ptot = np.interp(data['rho_norm'].to_numpy().flatten(), data['rho_face_norm'].to_numpy().flatten(), data['pressure_thermal_total'].to_numpy().flatten())
                    data_vars['ptot'] = (['n', 'rho'], np.expand_dims(ptot, axis=0))
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
                newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj
