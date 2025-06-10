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
            self.input = self._read_omas_json_file(path)
        else:
            self.output = self._read_omas_json_file(path)


    def write(self, path, side='input', overwrite=False):
        if side == 'input':
            self._write_omas_json_file(path, self.input, overwrite=overwrite)
        else:
            self._write_omas_json_file(path, self.output, overwrite=overwrite)


    def _read_omas_json_file(self, path):

        dsvec = []

        if isinstance(path, (str, Path)):
            ipath = Path(path)
            if ipath.exists():

                with open(ipath, 'r') as jsonfile:
                    data = json.load(jsonfile)

                if 'core_profiles' in data:
                    cp_coords = {}
                    cp_attrs = {}
                    if 'time' in data['core_profiles']:
                        cp_coords['time_cp'] = np.atleast_1d(data['core_profiles'].pop('time'))
                    profs = data['core_profiles'].pop('profiles_1d', [])
                    if len(profs) > 0 and 'rho_tor_norm' in profs[0].get('grid', {}):
                        cp_coords['rho_cp'] = np.atleast_1d(profs[0]['grid'].pop('rho_tor_norm'))
                    if len(profs) > 0 and 'ion' in profs[0] and len(profs[0]['ion']) > 0:
                        ionlist = []
                        for ii, ion in enumerate(profs[0]['ion']):
                            ionlist.append(ion.pop('label', 'UNKNOWN'))
                        cp_coords['ion_cp'] = ionlist
                    if 'code' in data['core_profiles']:
                        cp_attrs['core_profiles.code'] = data['core_profiles'].pop('code')
                    cp_ds = xr.Dataset(coords=cp_coords, attrs=cp_attrs) if cp_coords else None
                    if 'time_cp' in cp_coords and 'rho_cp' in cp_coords:
                        cp_dsvec = []
                        for ii, cpp in enumerate(profs):
                            cp_data_vars = {}
                            time = cp_coords['time_cp'][ii] if 'time_cp' in cp_coords and len(cp_coords['time_cp']) > ii else float(cpp['time'])
                            grid = cpp.pop('grid', {})
                            for key, val in grid.items():
                                tag = f'core_profiles.profiles_1d.grid.{key}'
                                if isinstance(val, list):
                                    cp_data_vars[tag] = (['time_cp', 'rho_cp'], np.expand_dims(np.atleast_1d(val), axis=0))
                            elec = cpp.pop('electrons', {})
                            for key, val in elec.items():
                                tag = f'core_profiles.profiles_1d.electrons.{key}'
                                if isinstance(val, list):
                                    cp_data_vars[tag] = (['time_cp', 'rho_cp'], np.expand_dims(np.atleast_1d(val), axis=0))
                            if 'ion_cp' in cp_coords:
                                ions = cpp.pop('ions', [])
                                iondict = {}
                                for jj, ion in enumerate(ions):
                                    for key, val in ion.items():
                                        tag = f'core_profiles.profiles_1d.ions.{key}'
                                        if isinstance(val, list):
                                            iondict[tag] = np.concatenate([iondict[tag], np.atleast_2d(val)], axis=-1) if tag in iondict else np.atleast_2d(val)
                                for tag, val in iondict.items():
                                    cp_data_vars[tag] = (['time_cp', 'rho_cp', 'ion_cp'], np.expand_dims(val, axis=0))
                            for key, val in cpp.items():
                                tag = f'core_profiles.profiles_1d.{key}'
                                if isinstance(val, list):
                                    cp_data_vars[tag] = (['time_cp', 'rho_cp'], np.expand_dims(np.atleast_1d(val), axis=0))
                            if cp_data_vars:
                                cp_dsvec.append(xr.Dataset(coords=cp_coords, data_vars=cp_data_vars, attrs=cp_attrs))
                        if len(cp_dsvec) > 0:
                            cp_ds = xr.merge(cp_dsvec)
                    if 'time_cp' in cp_coords:
                        globs = data['core_profiles'].pop('global_quantities', {})
                        if 'ion_cp' in cp_coords:
                            ions = globs.pop('ions', [])
                            iondict = {}
                            for jj, ion in enumerate(ions):
                                for key, val in ion.items():
                                    tag = f'core_profiles.global_quantities.ions.{key}'
                                    if isinstance(val, list):
                                        iondict[tag] = np.concatenate([iondict[tag], np.atleast_1d(val)], axis=-1) if tag in iondict else np.atleast_1d(val)
                            for tag, val in iondict.items():
                                cp_ds[tag] = (['time_cp', 'ion_cp'], val)
                        for key, val in globs.items():
                            tag = f'core_profiles.global_quantities.{key}'
                            if isinstance(val, list):
                                cp_ds[tag] = (['time_cp'], np.atleast_1d(val))
                    if cp_ds is not None:
                        dsvec.append(cp_ds)

                if 'core_sources' in data:
                    srcs = data['core_sources'].pop('sources', [])
                    cs_dsvec = []
                    for ii, src in enumerate(srcs):
                        cs_coords = {}
                        cs_attrs = {}
                        sid = src.pop('identifier', {})
                        srctag = sid['name'] if 'name' in sid else f'source_index_{sid.get("index", ii):d}'
                        if 'time' in src:
                            cs_coords['time_cs'] = np.atleast_1d(src.pop('time'))
                        profs = src.pop('profiles_1d', [])
                        if len(profs) > 0 and 'rho_tor_norm' in profs[0].get('grid', {}):
                            cs_coords['rho_cs'] = np.atleast_1d(profs[0]['grid'].pop('rho_tor_norm'))
                        if len(profs) > 0 and 'ion' in profs[0] and len(profs[0]['ion']) > 0:
                            ionlist = []
                            for ii, ion in enumerate(profs[0]['ion']):
                                ionlist.append(ion.pop('label', 'UNKNOWN'))
                            cs_coords['ion_cs'] = ionlist
                        if 'code' in src:
                            cs_attrs[f'core_sources.{srctag}.code'] = src.pop('code')
                        cs_ds = xr.Dataset(coords=cs_coords, attrs=cs_attrs) if cs_coords else None
                        if 'time_cs' in cs_coords and 'rho_cs' in cs_coords:
                            csp_dsvec = []
                            for ii, csp in enumerate(profs):
                                cs_data_vars = {}
                                time = cs_coords['time_cs'][ii] if 'time_cs' in cs_coords and len(cs_coords['time_cs']) > ii else float(csp['time'])
                                grid = csp.pop('grid', {})
                                for key, val in grid.items():
                                    tag = f'core_sources.{srctag}.profiles_1d.grid.{key}'
                                    if isinstance(val, list):
                                        cs_data_vars[tag] = (['time_cs', 'rho_cs'], np.expand_dims(np.atleast_1d(val), axis=0))
                                elec = csp.pop('electrons', {})
                                for key, val in elec.items():
                                    tag = f'core_sources.{srctag}.profiles_1d.electrons.{key}'
                                    if isinstance(val, list):
                                        cs_data_vars[tag] = (['time_cs', 'rho_cs'], np.expand_dims(np.atleast_1d(val), axis=0))
                                if 'ion_cs' in cs_coords:
                                    ions = csp.pop('ions', [])
                                    iondict = {}
                                    for jj, ion in enumerate(ions):
                                        for key, val in ion.items():
                                            tag = f'core_sources.{srctag}.profiles_1d.ions.{key}'
                                            if isinstance(val, list):
                                                iondict[tag] = np.concatenate([iondict[tag], np.atleast_2d(val)], axis=-1) if tag in iondict else np.atleast_2d(val)
                                    for tag, val in iondict.items():
                                        cs_data_vars[tag] = (['time_cs', 'rho_cs', 'ion_cs'], np.expand_dims(val, axis=0))
                                for key, val in csp.items():
                                    tag = f'core_sources.{srctag}.profiles_1d.{key}'
                                    if isinstance(val, list):
                                        cs_data_vars[tag] = (['time_cs', 'rho_cs'], np.expand_dims(np.atleast_1d(val), axis=0))
                                if cs_data_vars:
                                    csp_dsvec.append(xr.Dataset(coords=cs_coords, data_vars=cs_data_vars, attrs=cs_attrs))
                            if len(csp_dsvec) > 0:
                                cs_ds = xr.merge(csp_dsvec)
                        if 'time_cs' in cs_coords:
                            globs = src.pop('global_quantities', {})
                            if 'ion_cs' in cs_coords:
                                ions = globs.pop('ions', [])
                                iondict = {}
                                for jj, ion in enumerate(ions):
                                    for key, val in ion.items():
                                        tag = f'core_sources.{srctag}.global_quantities.ions.{key}'
                                        if isinstance(val, list):
                                            iondict[tag] = np.concatenate([iondict[tag], np.atleast_1d(val)], axis=-1) if tag in iondict else np.atleast_1d(val)
                                for tag, val in iondict.items():
                                    cs_ds[tag] = (['time_cs', 'ion_cs'], val)
                            for key, val in globs.items():
                                tag = f'core_sources.{srctag}.global_quantities.{key}'
                                if isinstance(val, list):
                                    cs_ds[tag] = (['time_cs'], np.atleast_1d(val))
                        if cs_ds is not None:
                            cs_dsvec.append(cs_ds)
                    if len(cs_dsvec) > 0:
                        dsvec.append(xr.merge(cs_dsvec))

                if 'core_transport' in data:
                    pass

                if 'equilibrium' in data:
                    #data['equilibrium']['time_slice'][0].keys()
                    #['boundary', 'constraints', 'global_quantities', 'profiles_1d', 'profiles_2d', 'time']
                    cp_coords = {}
                    cp_attrs = {}
                    if 'time' in data['core_profiles']:
                        cp_coords['time_cp'] = np.atleast_1d(data['core_profiles'].pop('time'))
                    profs = data['core_profiles'].pop('profiles_1d', [])
                    if len(profs) > 0 and 'rho_tor_norm' in profs[0].get('grid', {}):
                        cp_coords['rho_cp'] = np.atleast_1d(profs[0]['grid'].pop('rho_tor_norm'))
                    pass

                if 'wall' in data:
                    #data['wall']['description_2d'][-1]['limiter']['unit'][0]['outline']['r', 'z']
                    pass

                if 'summary' in data:
                    sm_coords = {}
                    sm_data_vars = {}
                    sm_attrs = {}
                    if 'time' in data['summary']:
                        sm_coords['time_sum'] = np.atleast_1d(data['summary'].pop('time'))
                    if 'code' in data['summary']:
                        cp_attrs['summary.code'] = data['summary'].pop('code')
                    if 'time_sum' in sm_coords:
                        for key, val in data:
                            tag = f'summary.{key}'
                            if isinstance(val, list):
                                sm_data_vars[tag] = (['time_sum'], np.atleast_1d(val))
                    dsvec.append(xr.Dataset(coords=sm_coords, data_vars=sm_data_vars, attrs=sm_attrs))

                if 'pulse_schedule' in data:
                    #data['pulse_schedule'].keys()
                    #['density_control', 'flux_control', 'ic']
                    pass

        return xr.merge(dsvec)


    def _write_omas_json_file(self, data, path, overwrite=False):
        pass


    @classmethod
    def from_file(cls, path=None, input=None, output=None):
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


