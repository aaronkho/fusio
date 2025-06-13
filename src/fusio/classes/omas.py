import json
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from .io import io

logger = logging.getLogger('fusio')


class omas_io(io):


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
                        cp_attrs['core_profiles.code.name'] = data['core_profiles'].pop('code').get('name', '')
                    cp_ds = xr.Dataset(coords=cp_coords, attrs=cp_attrs) if cp_coords else None
                    if 'time_cp' in cp_coords and 'rho_cp' in cp_coords:
                        timevec = cp_coords.pop('time_cp', [])
                        cp_dsvec = []
                        for ii, cpp in enumerate(profs):
                            cp_data_vars = {}
                            cp_coords['time_cp'] = np.atleast_1d([timevec[ii]]) if len(timevec) > ii else np.atleast_1d([float(cpp['time'])])
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
                                ions = cpp.pop('ion', [])
                                iondict = {}
                                for jj, ion in enumerate(ions):
                                    for key, val in ion.items():
                                        tag = f'core_profiles.profiles_1d.ion.{key}'
                                        if isinstance(val, list):
                                            iondict[tag] = np.concatenate([iondict[tag], np.atleast_2d(val)], axis=0) if tag in iondict else np.atleast_2d(val)
                                for tag in iondict:
                                    while iondict[tag].shape[0] < len(cp_coords['ion_cp']):
                                        iondict[tag] = np.concatenate([iondict[tag], np.atleast_2d(iondict[tag][-1, :])], axis=0)
                                for tag, val in iondict.items():
                                    cp_data_vars[tag] = (['time_cp', 'ion_cp', 'rho_cp'], np.expand_dims(val, axis=0))
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
                            ions = globs.pop('ion', [])
                            iondict = {}
                            for jj, ion in enumerate(ions):
                                for key, val in ion.items():
                                    tag = f'core_profiles.global_quantities.ion.{key}'
                                    if isinstance(val, list):
                                        iondict[tag] = np.concatenate([iondict[tag], np.atleast_2d(val).T], axis=-1) if tag in iondict else np.atleast_2d(val).T
                            for tag, val in iondict.items():
                                cp_ds[tag] = (['time_cp', 'ion_cp'], val)
                        for key, val in globs.items():
                            tag = f'core_profiles.global_quantities.{key}'
                            if isinstance(val, list):
                                cp_ds[tag] = (['time_cp'], np.atleast_1d(val))
                    if cp_ds is not None:
                        dsvec.append(cp_ds)

                if 'core_sources' in data:
                    srcs = data['core_sources'].pop('source', [])
                    cs_dsvec = []
                    for ii, src in enumerate(srcs):
                        cs_coords = {}
                        cs_attrs = {}
                        sid = src.pop('identifier', {})
                        srctag = sid['name'] if 'name' in sid else f'source_index_{sid.get("index", ii):d}'
                        if 'global_quantities' in src:
                            timevec = []
                            for jj in range(len(src['global_quantities'])):
                                if 'time' in src['global_quantities'][jj]:
                                    timevec.append(src['global_quantities'][jj]['time'])
                            cs_coords['time_cs'] = np.atleast_1d(timevec)
                        profs = src.pop('profiles_1d', [])
                        if len(profs) > 0 and 'rho_tor_norm' in profs[0].get('grid', {}):
                            cs_coords['rho_cs'] = np.atleast_1d(profs[0]['grid'].pop('rho_tor_norm'))
                        if len(profs) > 0 and 'ion' in profs[0] and len(profs[0]['ion']) > 0:
                            ionlist = []
                            for ii, ion in enumerate(profs[0]['ion']):
                                ionlist.append(ion.pop('label', 'UNKNOWN'))
                            cs_coords['ion_cs'] = ionlist
                        if 'code' in src:
                            cs_attrs[f'core_sources.{srctag}.code.name'] = src.pop('code').get('name', '')
                        cs_ds = xr.Dataset(coords=cs_coords, attrs=cs_attrs) if cs_coords else None
                        if 'time_cs' in cs_coords and 'rho_cs' in cs_coords:
                            timevec = cs_coords.pop('time_cs', [])
                            csp_dsvec = []
                            for jj, csp in enumerate(profs):
                                cs_data_vars = {}
                                cs_coords['time_cs'] = np.atleast_1d([timevec[jj]]) if len(timevec) > jj else np.atleast_1d([float(csp['time'])])
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
                                    for kk, ion in enumerate(ions):
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
                                csp_ds = xr.merge(csp_dsvec)
                                cs_ds = cs_ds.assign_coords(csp_ds.coords).assign(csp_ds.data_vars).assign_attrs(**csp_ds.attrs)
                            if len(timevec) > 0:
                                cs_coords['time_cs'] = timevec
                        if 'time_cs' in cs_coords:
                            globs = src.pop('global_quantities', [])
                            timevec = cs_coords.pop('time_cs', [])
                            csg_dsvec = []
                            for jj, csg in enumerate(globs):
                                cs_data_vars = {}
                                cs_coords['time_cs'] = np.atleast_1d([timevec[jj]]) if len(timevec) > jj else np.atleast_1d([float(csg['time'])])
                                if 'ion_cs' in cs_coords:
                                    ions = csg.pop('ions', [])
                                    iondict = {}
                                    for kk, ion in enumerate(ions):
                                        for key, val in ion.items():
                                            tag = f'core_sources.{srctag}.global_quantities.ions.{key}'
                                            if isinstance(val, list):
                                                iondict[tag] = np.concatenate([iondict[tag], np.atleast_1d(val)], axis=-1) if tag in iondict else np.atleast_1d(val)
                                    for tag, val in iondict.items():
                                        cs_data_vars[tag] = (['time_cs', 'ion_cs'], val)
                                for key, val in csg.items():
                                    tag = f'core_sources.{srctag}.global_quantities.{key}'
                                    if isinstance(val, list):
                                        cs_data_vars[tag] = (['time_cs'], np.atleast_1d(val))
                                if cs_data_vars:
                                    csg_dsvec.append(xr.Dataset(coords=cs_coords, data_vars=cs_data_vars))
                            if len(csg_dsvec) > 0:
                                csg_ds = xr.merge(csg_dsvec)
                                cs_ds = cs_ds.assign_coords(csg_ds.coords).assign(csg_ds.data_vars).assign_attrs(**csg_ds.attrs)
                        if cs_ds is not None:
                            cs_dsvec.append(cs_ds)
                    if len(cs_dsvec) > 0:
                        dsvec.append(xr.merge(cs_dsvec))

                #if 'core_transport' in data:
                #    #data['core_transport']['model'][0]['profiles_1d'][0]['ion'][0]['energy']
                #    ct_coords = {}
                #    ct_attrs = {}
                #    if 'time' in data['core_transport']:
                #        ct_coords['time_ct'] = np.atleast_1d(data['core_transport'].pop('time'))
                #    models = data['core_transport'].pop('model', [])
                #    ct_dsvec = []
                #    for ii, model in enumerate(models):
                #        modtag = model['code'].lower() if 'code' in model else f'model_index_{ii}'
                #        if 'code' in model:
                #            ct_attrs[f'core_transport.{modtag.lower()}.code'] = model.pop('code')
                #        profs = model.pop('profiles_1d', [])
                #        if len(profs) > 0:
                #            if 'rho_tor_norm' in profs[0].get('grid_d', {}):
                #                ct_coords['rho_d_ct'] = np.atleast_1d(profs[0]['grid_d']['rho_tor_norm'])
                #            if 'rho_tor_norm' in profs[0].get('grid_v', {}):
                #                ct_coords['rho_v_ct'] = np.atleast_1d(profs[0]['grid_v']['rho_tor_norm'])
                #            if 'rho_tor_norm' in profs[0].get('grid_flux', {}):
                #                ct_coords['rho_flux_ct'] = np.atleast_1d(profs[0]['grid_flux']['rho_tor_norm'])
                #        if 'time_ct' in ct_coords and ('rho_d_ct' in ct_coords or 'rho_v_ct' in ct_coords or 'rho_flux_ct' in ct_coords):
                #            timevec = ct_coords['time_ct'].copy()
                #            ctp_dsvec = []
                #            for jj, ctp in enumerate(profs):
                #                ct_data_vars = {}
                #                ct_coords['time_ct'] = np.atleast_1d([timevec[ii]]) if len(timevec) else np.atleast_1d([ctp['time']])

                if 'equilibrium' in data:
                    eq_coords = {}
                    eq_attrs = {}
                    if 'time' in data['equilibrium']:
                        eq_coords['time_eq'] = np.atleast_1d(data['equilibrium'].pop('time'))
                    slices = data['equilibrium'].pop('time_slice', [])
                    if len(slices) > 0 and 'rho_tor_norm' in slices[0].get('profiles_1d', {}):
                        eq_coords['rho_eq'] = np.atleast_1d(slices[0]['profiles_1d'].pop('rho_tor_norm'))
                    if len(slices) > 0 and len(slices[0].get('profiles_2d', [])) > 0:
                        for cc in range(len(slices[0]['profiles_2d'])):
                            gtype = slices[0]['profiles_2d'][cc].get('grid_type', {})
                            if 'name' in gtype and gtype['name'] == 'rectangular':
                                eq_coords['r_eq'] = np.atleast_1d(slices[0]['profiles_2d'][cc]['grid']['dim1'])
                                eq_coords['z_eq'] = np.atleast_1d(slices[0]['profiles_2d'][cc]['grid']['dim2'])
                                eq_attrs['equilibrium.profiles_2d.grid_type.name'] = gtype['name']
                                if 'description' in gtype:
                                    eq_attrs['equilibrium.profiles_2d.grid_type.description'] = gtype['description']
                    eq_ds = xr.Dataset(coords=eq_coords, attrs=eq_attrs) if eq_coords else None
                    if 'time_eq' in eq_coords and 'r_eq' in eq_coords and 'z_eq' in eq_coords:
                        timevec = eq_coords.pop('time_eq', [])
                        eq_dsvec = []
                        for ii, eqs in enumerate(slices):
                            # Does not transfer boundary or constraints
                            eq_data_vars = {}
                            eq_coords['time_eq'] = np.atleast_1d([timevec[ii]]) if len(timevec) > ii else np.atleast_1d([float(eqs['time'])])
                            equil = eqs.pop('profiles_2d', [])
                            for cc in range(len(equil)):
                                if 'grid_type' in equil[cc] and equil[cc]['grid_type'] == 'rectangular':
                                    for key, val in equil[cc]:
                                        tag = f'equilibrium.time_slice.profiles_2d.{key}'
                                        if isinstance(val, list):
                                            eq_data_vars[tag] = (['time_eq', 'r_eq', 'z_eq'], np.expand_dims(np.atleast_2d(val), axis=0))
                            profs = eqs.pop('profiles_1d', {})
                            for key, val in profs.items():
                                tag = f'equilibrium.time_slice.profiles_1d.{key}'
                                if isinstance(val, list):
                                    eq_data_vars[tag] = (['time_eq', 'rho_eq'], np.expand_dims(np.atleast_1d(val), axis=0))
                            globs = eqs.pop('global_quantities', {})
                            for key, val in globs.items():
                                tag = 'equilibrium.time_slice.global_quantities.{key}'
                                if isinstance(val, (float, int)):
                                    eq_data_vars[tag] = (['time_eq'], np.atleast_1d([val]))
                            eq_dsvec.append(xr.Dataset(coords=eq_coords, data_vars=eq_data_vars, attrs=eq_attrs))
                        if len(eq_dsvec) > 0:
                            eq_ds = xr.merge(eq_dsvec)
                    if eq_ds is not None:
                        dsvec.append(eq_ds)

                #if 'wall' in data:
                #    #data['wall']['description_2d'][-1]['limiter']['unit'][0]['outline']['r', 'z']
                #    pass

                if 'summary' in data:
                    sm_coords = {}
                    sm_data_vars = {}
                    sm_attrs = {}
                    if 'time' in data['summary']:
                        sm_coords['time_sum'] = np.atleast_1d(data['summary'].pop('time'))
                    if 'code' in data['summary']:
                        cp_attrs['summary.code'] = data['summary'].pop('code')
                    if 'time_sum' in sm_coords:
                        globs = data['summary'].pop('global_quantities', {})
                        for key, val in globs.items():
                            tag = f'summary.global_quantities.{key}.value'
                            if 'value' in val and isinstance(val['value'], list):
                                sm_data_vars[tag] = (['time_sum'], np.atleast_1d(val['value']))
                            if 'source' in val:
                                sm_attrs[f'summary.global_quantities.{key}.source'] = val['source']
                    if sm_coords:
                        dsvec.append(xr.Dataset(coords=sm_coords, data_vars=sm_data_vars, attrs=sm_attrs))

                if 'pulse_schedule' in data:
                    ps_coords = {}
                    ps_data_vars = {}
                    ps_attrs = {}
                    ii = 0
                    for var, obj in data['pulse_schedule'].items():
                        itag = f'{ii:d}'
                        if 'time' in obj:
                            ps_coords[f'time_{itag}_ps'] = np.atleast_1d(obj.pop('time'))
                            ps_attrs[f'pulse_schedule.index_{itag}'] = var
                            ii += 1
                        for key, val in obj.items():
                            tag = f'pulse_schedule.{var}.{key}.reference'
                            if 'reference' in val and isinstance(val['reference'], list):
                                ps_data_vars[tag] = ([f'time_{itag}_ps'], np.atleast_1d(val['reference']))
                    if ps_coords:
                        dsvec.append(xr.Dataset(coords=ps_coords, data_vars=ps_data_vars, attrs=ps_attrs))

        ds = xr.Dataset()
        for dss in dsvec:
            ds = ds.assign_coords(dss.coords).assign(dss.data_vars).assign_attrs(**dss.attrs)

        return ds


    def _write_omas_json_file(self, data, path, overwrite=False):
        pass


    @classmethod
    def from_file(cls, path=None, input=None, output=None):
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


