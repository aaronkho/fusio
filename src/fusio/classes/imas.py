import h5py
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from .io import io
from ..utils.eqdsk_tools import write_eqdsk

logger = logging.getLogger('fusio')


class imas_io(io):


    empty_int = -999999999
    empty_float = -9.0e40
    empty_complex = -9.0e40-9.0e40j
    int_types = (int, np.int8, np.int16, np.int32, np.int64)
    float_types = (float, np.float16, np.float32, np.float64, np.float128)
    complex_types = (complex, np.complex64, np.complex128, np.complex256)


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
            self.input = self._read_imas_directory(path)
        else:
            self.output = self._read_imas_directory(path)


    def write(self, path, side='input', overwrite=False):
        if side == 'input':
            self._write_imas_file(path, self.input, overwrite=overwrite)
        else:
            self._write_imas_file(path, self.output, overwrite=overwrite)


    def _read_imas_directory(self, path):

        dsvec = []

        if isinstance(path, (str, Path)):
            ipath = Path(path)
            if ipath.is_dir():

                core_profiles_path = ipath / 'core_profiles.h5'
                core_profiles_data = {}
                if core_profiles_path.exists():
                    core_profiles_data = h5py.File(ipath / 'core_profiles.h5')
                if 'core_profiles' in core_profiles_data:
                    data = {k: v[()] for k, v in core_profiles_data['core_profiles'].items()}
                    cp_coords = {}
                    cp_data_vars = {}
                    cp_attrs = {}
                    if 'time' in data:
                        cp_coords['time_cp'] = np.atleast_1d(data.pop('time')).flatten()
                    for key in list(data.keys()):
                        if key.endswith('&AOS_SHAPE'):
                            val = data.pop(key)
                            if key == 'profiles_1d[]&ion[]&AOS_SHAPE' and 'profiles_1d[]&ion[]&label' in data:
                                cp_coords['ion_cp'] = [label.decode('utf-8') for label in data['profiles_1d[]&ion[]&label'][0]]
                                data.pop('profiles_1d[]&ion[]&label')
                            if key == 'profiles_1d[]&neutral[]&AOS_SHAPE' and 'profiles_1d[]&neutral[]&label' in data:
                                cp_coords['neut_cp'] = [label.decode('utf-8') for label in data['profiles_1d[]&neutral[]&label'][0]]
                                data.pop('profiles_1d[]&neutral[]&label')
                        elif 'state[]' in key:
                            data.pop(key)
                        elif key == 'profiles_1d[]&grid&rho_tor_norm':  # Assumes rho_tor_norm is the constant radial coordinate
                            cp_coords['rho_cp'] = data.pop(key)[0].flatten()
                            data.pop(f'{key}_SHAPE')
                    if 'time_cp' in cp_coords and 'rho_cp' in cp_coords:
                        timevec = cp_coords['time_cp']
                        for key, val in data.items():
                            if not key.endswith('_SHAPE'):
                                if isinstance(val, np.ndarray):
                                    if val.dtype in self.int_types:
                                        val = np.where(val == self.empty_int, np.nan, val)
                                    if val.dtype in self.float_types:
                                        val = np.where(val == self.empty_float, np.nan, val)
                                    if val.dtype in self.complex_types:
                                        val = np.where(val == self.empty_complex, np.nan, val)
                                dims = []
                                components = key.split('&')
                                if 'profiles_1d[]' in components:
                                    dims.append('time_cp')
                                if 'ion[]' in components:
                                    dims.append('ion_cp')
                                if 'neutral[]' in components:
                                    dims.append('neut_cp')
                                if 'global_quantities' in components:
                                    dims.append('time_cp')
                                if 'element[]' in components:
                                    val = val.squeeze(axis=2) if val.ndim == 3 else None
                                if f'{key}_SHAPE' in data:
                                    dims.append('rho_cp')
                                if isinstance(val, bytes):
                                    cp_attrs[f'core_profiles&{key}'] = val.decode('utf-8')
                                elif key == 'ids_properties&homogeneous_time':
                                    cp_attrs[f'core_profiles&{key}'] = val
                                elif key == 'vacuum_toroidal_field&b0':
                                    cp_data_vars[f'core_profiles&{key}'] = (['time_cp'], val)
                                elif key == 'vacuum_toroidal_field&r0':
                                    cp_data_vars[f'core_profiles&{key}'] = (['time_cp'], np.repeat(np.atleast_1d([val]), len(timevec), axis=0))
                                elif isinstance(val, np.ndarray):
                                    cp_data_vars[f'core_profiles&{key}'] = (dims, val)
                    if cp_data_vars:
                        cp_ds = xr.Dataset(coords=cp_coords, data_vars=cp_data_vars, attrs=cp_attrs)
                        dsvec.append(cp_ds)

                core_sources_path = ipath / 'core_sources.h5'
                core_sources_data = {}
                if core_sources_path.exists():
                    core_sources_data = h5py.File(core_sources_path)
                if 'core_sources' in core_sources_data:
                    data = {k: v[()] for k, v in core_sources_data['core_sources'].items()}
                    cs_coords = {}
                    cs_data_vars = {}
                    cs_attrs = {}
                    srctags = {}
                    if 'time' in data:
                        cs_coords['time_cs'] = np.atleast_1d(data.pop('time')).flatten()
                    for key in list(data.keys()):
                        if key.endswith('&AOS_SHAPE'):
                            val = data.pop(key)
                            if key == 'source[]&AOS_SHAPE' and 'source[]&identifier&name' in data:
                                srctags.update({name.decode('utf-8'): ii for ii, name in enumerate(data.pop('source[]&identifier&name'))})
                                data.pop('source[]&identifier&index')
                                data.pop('source[]&identifier&description')
                            if key == 'source[]&profiles_1d[]&ion[]&AOS_SHAPE' and 'source[]&profiles_1d[]&ion[]&label' in data:
                                cp_coords['ion_cs'] = [label.decode('utf-8') for label in data['source[]&profiles_1d[]&ion[]&label'][0, 0, :]]
                                data.pop('source[]&profiles_1d[]&ion[]&label')
                            if key == 'source[]&profiles_1d[]&neutral[]&AOS_SHAPE' and 'source[]&profiles_1d[]&neutral[]&label' in data:
                                cp_coords['neut_cs'] = [label.decode('utf-8') for label in data['source[]&profiles_1d[]&neutral[]&label'][0]]
                                data.pop('source[]&profiles_1d[]&neutral[]&label')
                        elif 'state[]' in key:
                            data.pop(key)
                        elif key == 'source[]&profiles_1d[]&grid&rho_tor_norm':  # Assumes rho_tor_norm is the constant radial coordinate
                            cs_coords['rho_cs'] = data.pop(key)[0, 0].flatten()
                            data.pop(f'{key}_SHAPE')
                    if 'time_cs' in cs_coords and 'rho_cs' in cs_coords:
                        timevec = cs_coords['time_cs']
                        cs_attrs['sources'] = list(srctags.keys())
                        for key, val in data.items():
                            if not key.endswith('_SHAPE'):
                                if isinstance(val, np.ndarray):
                                    if val.dtype in self.int_types:
                                        val = np.where(val == self.empty_int, np.nan, val)
                                    if val.dtype in self.float_types:
                                        val = np.where(val == self.empty_float, np.nan, val)
                                    if val.dtype in self.complex_types:
                                        val = np.where(val == self.empty_complex, np.nan, val)
                                dims = []
                                components = key.split('&')
                                if components[0] == 'source[]':
                                    if 'element[]' in components:
                                        val = val.squeeze(axis=2) if val.ndim == 3 else None
                                    for srctag, ii in srctags.items():
                                        dims = []
                                        if 'profiles_1d[]' in components:
                                            dims.append('time_cs')
                                        if 'ion[]' in components:
                                            dims.append('ion_cs')
                                        if 'neutral[]' in components:
                                            dims.append('neut_cs')
                                        if f'{key}_SHAPE' in data:
                                            dims.append('rho_cs')
                                        newkey = '&'.join(components[1:])
                                        if isinstance(val, np.ndarray):
                                            cs_data_vars[f'core_sources&{srctag}&{newkey}'] = (dims, val[ii, ...])
                                else:
                                    if isinstance(val, bytes):
                                        cs_attrs[f'core_sources&{key}'] = val.decode('utf-8')
                                    elif key == 'ids_properties&homogeneous_time':
                                        cs_attrs[f'core_sources&{key}'] = val
                                    elif key == 'vacuum_toroidal_field&b0':
                                        cs_data_vars[f'core_sources&{key}'] = (['time_cs'], val)
                                    elif key == 'vacuum_toroidal_field&r0':
                                        cs_data_vars[f'core_sources&{key}'] = (['time_cs'], np.repeat(np.atleast_1d([val]), len(timevec), axis=0))
                                    elif isinstance(val, np.ndarray):
                                        cs_data_vars[f'core_sources&{key}'] = (dims, val)
                    if cs_data_vars:
                        cs_ds = xr.Dataset(coords=cs_coords, data_vars=cs_data_vars, attrs=cs_attrs)
                        dsvec.append(cs_ds)

                equilibrium_path = ipath / 'equilibrium.h5'
                equilibrium_data = {}
                if equilibrium_path.exists():
                    equilibrium_data = h5py.File(equilibrium_path)
                if 'equilibrium' in equilibrium_data:
                    data = {k: v[()] for k, v in equilibrium_data['equilibrium'].items()}
                    eq_coords = {}
                    eq_data_vars = {}
                    eq_attrs = {}
                    if 'time' in data:
                        eq_coords['time_eq'] = np.atleast_1d(data.pop('time')).flatten()
                    rect2d_idx = None
                    for key in list(data.keys()):
                        if key.endswith('&AOS_SHAPE'):
                            val = data.pop(key)
                            if key == 'time_slice[]&profiles_2d[]&AOS_SHAPE' and 'time_slice[]&profiles_2d[]&grid_type&name' in data:
                                for ii, label in enumerate(data.pop('time_slice[]&profiles_2d[]&grid_type&name')[0]):
                                    if label.decode('utf-8') == 'rectangular':
                                        rect2d_idx = ii
                                data.pop('time_slice[]&profiles_2d[]&grid_type&index')
                                data.pop('time_slice[]&profiles_2d[]&grid_type&description')
                                if rect2d_idx is not None:
                                    eq_coords['r_eq'] = data.pop('time_slice[]&profiles_2d[]&grid&dim1')[0, rect2d_idx].flatten()
                                    data.pop('time_slice[]&profiles_2d[]&grid&dim1_SHAPE')
                                    eq_coords['z_eq'] = data.pop('time_slice[]&profiles_2d[]&grid&dim2')[0, rect2d_idx].flatten()
                                    data.pop('time_slice[]&profiles_2d[]&grid&dim2_SHAPE')
                        elif key == 'time_slice[]&profiles_1d&psi_norm':
                            eq_coords['psin_eq'] = data.pop('time_slice[]&profiles_1d&psi_norm')[0].flatten()
                        elif key == 'time_slice[]&profiles_1d&rho_tor_norm' and 'time_slice[]&profiles_1d&psi_norm' not in data:
                            eq_coords['rho_eq'] = data.pop('time_slice[]&profiles_1d&rho_tor_norm')[0].flatten()
                        elif key == 'time_slice[]&boundary&outline&r':
                            if 'i_bdry_eq' not in eq_coords:
                                eq_coords['i_bdry_eq'] = [j for j in range(len(data['time_slice[]&boundary&outline&r'][0]))]
                        elif key == 'time_slice[]&boundary&outline&z':
                            if 'i_bdry_eq' not in eq_coords:
                                eq_coords['i_bdry_eq'] = [j for j in range(len(data['time_slice[]&boundary&outline&z'][0]))]
                    if 'time_eq' in eq_coords and ('psin_eq' in eq_coords or 'rho_eq' in eq_coords):
                        timevec = eq_coords['time_eq']
                        for key, val in data.items():
                            if not key.endswith('_SHAPE'):
                                if isinstance(val, np.ndarray):
                                    if val.dtype in self.int_types:
                                        val = np.where(val == self.empty_int, np.nan, val)
                                    if val.dtype in self.float_types:
                                        val = np.where(val == self.empty_float, np.nan, val)
                                    if val.dtype in self.complex_types:
                                        val = np.where(val == self.empty_complex, np.nan, val)
                                dims = []
                                components = key.split('&')
                                if 'time_slice[]' in components:
                                    dims.append('time_eq')
                                if 'profiles_1d' in components:
                                    dims.extend(['psin_eq' if 'psin_eq' in eq_coords else 'rho_eq'])
                                if 'profiles_2d[]' in components:
                                    dims.extend(['r_eq', 'z_eq'])
                                    if rect2d_idx is not None and isinstance(val, np.ndarray):
                                        components[components.index('profiles_2d[]')] = 'profiles_2d'
                                        newkey = '&'.join(components)
                                        eq_data_vars[f'equilibrium&{newkey}'] = (dims, val[:, rect2d_idx, ...])
                                else:
                                    if isinstance(val, bytes):
                                        eq_attrs[f'equilibrium&{key}'] = val.decode('utf-8')
                                    elif key == 'ids_properties&homogeneous_time':
                                        eq_attrs[f'equilibrium&{key}'] = val
                                    elif key == 'vacuum_toroidal_field&b0':
                                        eq_data_vars[f'equilibrium&{key}'] = (['time_eq'], val)
                                    elif key == 'vacuum_toroidal_field&r0':
                                        eq_data_vars[f'equilibrium&{key}'] = (['time_eq'], np.repeat(np.atleast_1d([val]), len(timevec), axis=0))
                                    elif key == 'time_slice[]&boundary&outline&r' or key == 'time_slice[]&boundary&outline&z':
                                        dims.append('i_bdry_eq')
                                        eq_data_vars[f'equilibrium&{key}'] = (dims, val)
                                    elif isinstance(val, np.ndarray):
                                        eq_data_vars[f'equilibrium&{key}'] = (dims, val)
                    if eq_data_vars:
                        eq_ds = xr.Dataset(coords=eq_coords, data_vars=eq_data_vars, attrs=eq_attrs)
                        dsvec.append(eq_ds)

                summary_path = ipath / 'summary.h5'
                summary_data = {}
                if summary_path.exists():
                    summary_data = h5py.File(summary_path)
                if 'summary' in summary_data:
                    data = {k: v[()] for k, v in summary_data['summary'].items()}
                    sm_coords = {}
                    sm_data_vars = {}
                    sm_attrs = {}
                    hcdtags = []
                    if 'time' in data:
                        sm_coords['time_sm'] = np.atleast_1d(data.pop('time')).flatten()
                    for key in list(data.keys()):
                        if key.endswith('&AOS_SHAPE'):
                            val = data.pop(key)
                            components = key.split('&')
                            if components[0] == 'heating_current_drive' and components[1].endswith('[]'):
                                hcdtag = components[1][:-2]
                                sm_coords[f'i_{hcdtag}_sm'] = [j for j in range(val[0])]
                                hcdtags.append(hcdtag)
                    if 'time_sm' in sm_coords:
                        timevec = sm_coords['time_sm']
                        for key, val in data.items():
                            if not key.endswith('_SHAPE'):
                                if isinstance(val, np.ndarray):
                                    if val.dtype in self.int_types:
                                        val = np.where(val == self.empty_int, np.nan, val)
                                    if val.dtype in self.float_types:
                                        val = np.where(val == self.empty_float, np.nan, val)
                                    if val.dtype in self.complex_types:
                                        val = np.where(val == self.empty_complex, np.nan, val)
                                dims = []
                                components = key.split('&')
                                if components[0] == 'heating_current_drive' and components[1][:-2] in hcdtags:
                                    dims.append(f'i_{components[1][:-2]}_sm')
                                if isinstance(val, bytes):
                                    sm_attrs[f'summary&{key}'] = val.decode('utf-8')
                                elif key == 'ids_properties&homogeneous_time':
                                    sm_attrs[f'summary&{key}'] = val
                                elif key == 'vacuum_toroidal_field&b0':
                                    sm_data_vars[f'summary&{key}'] = (['time_sm'], val)
                                elif key == 'vacuum_toroidal_field&r0':
                                    sm_data_vars[f'summary&{key}'] = (['time_sm'], np.repeat(np.atleast_1d([val]), len(timevec), axis=0))
                                elif isinstance(val, np.ndarray):
                                    dims.append('time_sm')
                                    sm_data_vars[f'summary&{key}'] = (dims, val)
                    if sm_data_vars:
                        sm_ds = xr.Dataset(coords=sm_coords, data_vars=sm_data_vars, attrs=sm_attrs)
                        dsvec.append(sm_ds)

        ds = xr.Dataset()
        for dss in dsvec:
            ds = ds.assign_coords(dss.coords).assign(dss.data_vars).assign_attrs(**dss.attrs)

        return ds


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


    def to_eqdsk(self, time_index=-1, side='output'):
        eqdata = {}
        data = self.input.to_dataset().isel(time_eq=time_index) if side == 'input' else self.output.to_dataset().isel(time_eq=time_index)
        psinvec = data['psin_eq'].to_numpy().flatten() if 'psin_eq' in data else None
        conversion = None
        ikwargs = {'fill_value': 'extrapolate'}
        if psinvec is None:
            conversion = ((data['equilibrium&time_slice[]&profiles_1d&psi'] - data['equilibrium&time_slice[]&global_quantities&psi_axis']) / (data['equilibrium&time_slice[]&global_quantities&psi_boundary'] - data['equilibrium&time_slice[]&global_quantities&psi_axis'])).to_numpy().flatten()
        tag = 'r_eq'
        if tag in data:
            rvec = data[tag].to_numpy().flatten()
            eqdata['nr'] = rvec.size
            eqdata['rdim'] = float(np.nanmax(rvec) - np.nanmin(rvec))
            eqdata['rleft'] = float(np.nanmin(rvec))
            psinvec = np.linspace(0.0, 1.0, len(rvec))
        tag = 'z_eq'
        if tag in data:
            zvec = data[tag].to_numpy().flatten()
            eqdata['nz'] = zvec.size
            eqdata['zdim'] = float(np.nanmax(zvec) - np.nanmin(zvec))
            eqdata['zmid'] = float(np.nanmax(zvec) + np.nanmin(zvec)) / 2.0
        tag = 'equilibrium&vacuum_toroidal_field&r0'
        if tag in data:
            eqdata['rcentr'] = float(data.get(tag).to_numpy().flatten())
        tag = 'equilibrium&vacuum_toroidal_field&b0'
        if tag in data:
            eqdata['bcentr'] = float(data.get(tag).to_numpy().flatten())
        tag = 'equilibrium&time_slice[]&global_quantities&magnetic_axis&r'
        if tag in data:
            eqdata['rmagx'] = float(data.get(tag).to_numpy().flatten())
        tag = 'equilibrium&time_slice[]&global_quantities&magnetic_axis&z'
        if tag in data:
            eqdata['zmagx'] = float(data.get(tag).to_numpy().flatten())
        tag = 'equilibrium&time_slice[]&global_quantities&psi_axis'
        if tag in data:
            eqdata['simagx'] = float(data.get(tag).to_numpy().flatten())
        tag = 'equilibrium&time_slice[]&global_quantities&psi_boundary'
        if tag in data:
            eqdata['sibdry'] = float(data.get(tag).to_numpy().flatten())
        tag = 'equilibrium&time_slice[]&global_quantities&ip'
        if tag in data:
            eqdata['cpasma'] = float(data.get(tag).to_numpy().flatten())
        tag = 'equilibrium&time_slice[]&profiles_1d&f'
        if tag in data:
            if conversion is None:
                eqdata['fpol'] = data.drop_duplicates('psin_eq').get(tag).interp(psin_eq=psinvec).to_numpy().flatten()
            else:
                ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data.get(tag).to_numpy().flatten())})
                eqdata['fpol'] = ndata.drop_duplicates('psin_interp').get(tag).interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
        tag = 'equilibrium&time_slice[]&profiles_1d&pressure'
        if tag in data:
            if conversion is None:
                eqdata['pres'] = data.drop_duplicates('psin_eq').get(tag).interp(psin_eq=psinvec).to_numpy().flatten()
            else:
                ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data.get(tag).to_numpy().flatten())})
                eqdata['pres'] = ndata.drop_duplicates('psin_interp').get(tag).interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
        tag = 'equilibrium&time_slice[]&profiles_1d&f_df_dpsi'
        if tag in data:
            if conversion is None:
                eqdata['ffprime'] = data.drop_duplicates('psin_eq').get(tag).interp(psin_eq=psinvec).to_numpy().flatten()
            else:
                ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data.get(tag).to_numpy().flatten())})
                eqdata['ffprime'] = ndata.drop_duplicates('psin_interp').get(tag).interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
        tag = 'equilibrium&time_slice[]&profiles_1d&dpressure_dpsi'
        if tag in data:
            if conversion is None:
                eqdata['pprime'] = data.drop_duplicates('psin_eq').get(tag).interp(psin_eq=psinvec).to_numpy().flatten()
            else:
                ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data.get(tag).to_numpy().flatten())})
                eqdata['pprime'] = ndata.drop_duplicates('psin_interp').get(tag).interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
        tag = 'equilibrium&time_slice[]&profiles_2d&psi'
        if tag in data:
            eqdata['psi'] = data.get(tag).to_numpy().T
        tag = 'equilibrium&time_slice[]&profiles_1d&q'
        if tag in data:
            if conversion is None:
                eqdata['qpsi'] = data.drop_duplicates('psin_eq').get(tag).interp(psin_eq=psinvec).to_numpy().flatten()
            else:
                ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data.get(tag).to_numpy().flatten())})
                eqdata['qpsi'] = ndata.drop_duplicates('psin_interp').get(tag).interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
        rtag = 'equilibrium&time_slice[]&boundary&outline&r'
        ztag = 'equilibrium&time_slice[]&boundary&outline&z'
        if rtag in data and ztag in data:
            rdata = data.get(rtag).dropna('i_bdry_eq').to_numpy().flatten()
            zdata = data.get(ztag).dropna('i_bdry_eq').to_numpy().flatten()
            if len(rdata) == len(zdata):
                eqdata['nbdry'] = len(rdata)
                eqdata['rbdry'] = rdata
                eqdata['zbdry'] = zdata
        return eqdata


    def generate_eqdsk_file(self, path, time_index=-1, side='output'):
        eqpath = None
        if isinstance(path, (str, Path)):
            eqpath = Path(path)
        assert isinstance(eqpath, Path)
        eqdata = self.to_eqdsk(time_index=time_index, side=side)
        write_eqdsk(eqdata, eqpath)
        logger.info('Successfully generated g-eqdsk file, {path}')


    def generate_all_eqdsk_files(self, basepath, side='output'):
        path = None
        if isinstance(basepath, (str, Path)):
            path = Path(basepath)
        assert isinstance(path, Path)
        data = self.input if side == 'input' else self.output
        if 'time_eq' in data.coords:
            for ii, time in enumerate(data['time_eq'].to_numpy().flatten()):
                stem = f'{path.stem}'
                if stem.endswith('_input'):
                    stem = stem[:-6]
                time_tag = int(np.rint(time * 1000))
                eqpath = path.parent / f'{stem}_{time_tag:06d}ms_input{path.suffix}'
                self.generate_eqdsk_file(eqpath, time_index=ii, side=side)


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

