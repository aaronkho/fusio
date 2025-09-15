import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

from packaging.version import Version
import h5py  # type: ignore[import-untyped]
import imas  # type: ignore[import-untyped]
from imas.ids_base import IDSBase  # type: ignore[import-untyped]
from imas.ids_structure import IDSStructure  # type: ignore[import-untyped]
from imas.ids_struct_array import IDSStructArray  # type: ignore[import-untyped]
from .io import io
from ..utils.eqdsk_tools import (
    convert_cocos,
    write_eqdsk,
)

logger = logging.getLogger('fusio')


class imas_io(io):

    ids_top_levels: Final[Sequence[str]] = [
        'amns_data',
        'barometry',
        'b_field_non_axisymmetric',
        'bolometer',
        'bremsstrahlung_visible',
        'camera_ir',
        'camera_visible',
        'camera_x_rays',
        'charge_exchange',
        'coils_non_axisymmetric',
        'controllers',
        'core_instant_changes',
        'core_profiles',
        'core_sources',
        'core_transport',
        'cryostat',
        'dataset_description',
        'dataset_fair',
        'disruption',
        'distributions_sources',
        'distributions',
        'divertors',
        'ec_launchers',
        'ece',
        'edge_profiles',
        'edge_sources',
        'edge_transport',
        'em_coupling',
        'equilibrium',
        'ferritic',
        'focs',
        'gas_injection',
        'gas_pumping',
        'gyrokinetics_local',
        'hard_x_rays',
        'ic_antennas',
        'interferometer',
        'iron_core',
        'langmuir_probes',
        'lh_antennas',
        'magnetics',
        'operational_instrumentation',
        'mhd',
        'mhd_linear',
        'mse',
        'nbi',
        'neutron_diagnostic',
        'ntms',
        'pellets',
        'pf_active',
        'pf_passive',
        'pf_plasma',
        'plasma_initiation',
        'plasma_profiles',
        'plasma_sources',
        'plasma_transport',
        'polarimeter',
        'pulse_schedule',
        'radiation',
        'real_time_data',
        'reflectometer_profile',
        'reflectometer_fluctuation',
        'refractometer',
        'runaway_electrons',
        'sawteeth',
        'soft_x_rays',
        'spectrometer_mass',
        'spectrometer_uv',
        'spectrometer_visible',
        'spectrometer_x_ray_crystal',
        'spi',
        'summary',
        'temporary',
        'thomson_scattering',
        'tf',
        'transport_solver_numerics',
        'turbulence',
        'wall',
        'waves',
        'workflow',
    ]
    source_names: Final[Sequence[str]] = [
        'total',
        'nbi',
        'ec',
        'lh',
        'ic',
        'fusion',
        'ohmic',
        'bremsstrahlung',
        'synchrotron_radiation',
        'line_radiation',
        'collisional_equipartition',
        'cold_neutrals',
        'bootstrap_current',
        'pellet',
        'auxiliary',
        'ic_nbi',
        'ic_fusion',
        'ic_nbi_fusion',
        'ec_lh',
        'ec_ic',
        'lh_ic',
        'ec_lh_ic',
        'gas_puff',
        'killer_gas_puff',
        'radiation',
        'cyclotron_radiation',
        'cyclotron_synchrotron_radiation',
        'impurity_radiation',
        'particles_to_wall',
        'particles_to_pump',
        'charge_exchange',
        'transport',
        'neoclassical',
        'equipartition',
        'turbulent_equipartition',
        'runaways',
        'ionisation',
        'recombination',
        'excitation',
        'database',
        'gaussian',
    ]
    default_cocos_3: Final[int] = 11
    default_cocos_4: Final[int] = 17

    empty_int: Final[int] = imas.ids_defs.EMPTY_INT
    empty_float: Final[float] = imas.ids_defs.EMPTY_FLOAT
    #empty_complex: Final[complex] = imas.ids_defs.EMPTY_COMPLEX  # Removed since complex type cannot be JSON serialized
    int_types: Final[Sequence[Any]] = (int, np.int8, np.int16, np.int32, np.int64)
    float_types: Final[Sequence[Any]] = (float, np.float16, np.float32, np.float64, np.float128)
    #complex_types: Final[Sequence[Any]] = (complex, np.complex64, np.complex128, np.complex256)

    last_index_fields: Final[Sequence[str]] = [
        'core_profiles.profiles_1d.grid.rho_tor_norm',
        'core_sources.source.profiles_1d.grid.rho_tor_norm',
        'core_transport.model.profiles_1d.grid_flux.rho_tor_norm',
        'core_transport.model.profiles_1d.grid_d.rho_tor_norm',
        'core_transport.model.profiles_1d.grid_v.rho_tor_norm',
        'equilibrium.time_slice.profiles_1d.psi',
        'equilibrium.time_slice.profiles_2d.grid.dim1',
        'equilibrium.time_slice.profiles_2d.grid.dim2',
        'equilibrium.time_slice.boundary.outline.r',
        'wall.description_2d.limiter.unit.outline.r',
    ]


    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.has_imas: bool = imas.backends.imas_core.imas_interface.has_imas
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
            self.input = self._read_imas_directory(path)
        else:
            self.output = self._read_imas_directory(path)


    def write(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False,
    ) -> None:
        if side == 'input':
            self._write_imas_directory(path, self.input, overwrite=overwrite)
        else:
            self._write_imas_directory(path, self.output, overwrite=overwrite)


    def _convert_to_ids_structure(
        self,
        ids_name: str,
        data: MutableMapping[str, Any],
        delimiter: str,
        version: str | None = None,
    ) -> IDSStructure:

        def _recursive_resize_struct_array(
            ids: IDSBase,
            components: list[str],
            size: list[Any],
        ) -> None:
            if len(components) > 0:
                if isinstance(ids, IDSStructArray) and len(components) > 1:
                    for ii in range(ids.size):
                        if isinstance(size, np.ndarray) and ii < size.shape[0]:
                            _recursive_resize_struct_array(ids[ii], components, size[ii])
                elif isinstance(ids, IDSStructArray) and components[0] == 'AOS_SHAPE':
                    ids.resize(size[0])
                else:
                    _recursive_resize_struct_array(ids[f'{components[0]}'], components[1:], size)

        def _expanded_data_insertion(
            ids: IDSBase,
            components: list[str],
            data: Any,
        ) -> None:
            if len(components) > 0:
                if isinstance(ids, IDSStructArray):
                    for ii in range(ids.size):
                        if isinstance(data, np.ndarray) and ii < data.shape[0]:
                            _expanded_data_insertion(ids[ii], components, data[ii])
                        elif not isinstance(data, np.ndarray):
                            _expanded_data_insertion(ids[ii], components, data)
                elif len(components) == 1:
                    val = data if not isinstance(data, bytes) else data.decode('utf-8')
                    if isinstance(val, np.ndarray):
                        if val.dtype in self.int_types:
                            val = np.where(val == self.empty_int, np.nan, val)
                        if val.dtype in self.float_types:
                            val = np.where(val == self.empty_float, np.nan, val)
                        #if val.dtype in self.complex_types:
                        #    val = np.where(val == self.empty_complex, np.nan, val)
                        if val.ndim == 0:
                            val = val.item()
                    ids[f'{components[0]}'] = val
                else:
                    _expanded_data_insertion(ids[f'{components[0]}'], components[1:], data)

        dd_version: Any = None
        if f'ids_properties{delimiter}version_put{delimiter}data_dictionary' in data:
            dd_version = data[f'ids_properties{delimiter}version_put{delimiter}data_dictionary']
            if isinstance(dd_version, bytes):
                dd_version = dd_version.decode('utf-8')
            elif isinstance(dd_version, np.ndarray):
                dd_version = dd_version.item()
        if dd_version is None and isinstance(version, str):
            dd_version = version
        ids_struct = getattr(imas.IDSFactory(version=dd_version), f'{ids_name}')()
        index_data = {}
        for key in list(data.keys()):
            if key.endswith(':i'):
                vector = data.pop(key)
                index_data[f'{key[:-2]}'] = vector.size
        for key in sorted(index_data.keys(), key=len):
            zeros = np.array([0])
            prev_key = delimiter.join(key.split(delimiter)[:-1]) if delimiter in key else ''
            if prev_key in index_data and f'{prev_key}{delimiter}AOS_SHAPE' in data:
                zeros = np.repeat(np.expand_dims(np.zeros(np.array(data[f'{prev_key}{delimiter}AOS_SHAPE']).shape), axis=-1), index_data[prev_key], axis=-1)
            data[f'{key}{delimiter}AOS_SHAPE'] = zeros.astype(int) + index_data[key]
        shape_data = {}
        for key in list(data.keys()):
            if key.endswith(f'{delimiter}AOS_SHAPE'):
                shape_data[key] = data.pop(key)
            elif key.endswith('_SHAPE'):
                data.pop(key)
        for key in sorted(shape_data.keys(), key=len):
            _recursive_resize_struct_array(ids_struct, key.replace('[]', '').split(delimiter), shape_data[key])
        for key in data:
            _expanded_data_insertion(ids_struct, key.replace('[]', '').split(delimiter), data[key])

        return ids_struct


    def _read_imas_directory(
        self,
        path: str | Path,
        version: str | None = None,
    ) -> xr.Dataset:
        if isinstance(path, (str, Path)):
            ipath = Path(path)
            if ipath.is_dir():
                interface = 'netcdf'
                if (ipath / 'master.h5').is_file():
                    interface = 'hdf5'
                if interface == 'netcdf':
                    return self._read_imas_netcdf_files(ipath, version=version)
                if interface == 'hdf5':
                    if self.has_imas:
                        return self._read_imas_hdf5_files_with_core(ipath, version=version)
                    else:
                        return self._read_imas_hdf5_files_without_core(ipath, version=version)
            elif ipath.is_file() and ipath.suffix.lower() in ['.nc', '.ncdf', '.cdf']:
                return self._read_imas_netcdf_file(ipath, version=version)
        return xr.Dataset()


    def _read_imas_netcdf_file(
        self,
        path: str | Path,
        version: str | None = None,
    ) -> xr.Dataset:

        dsvec = []
        attrs: MutableMapping[str, Any] = {}

        if isinstance(path, (str, Path)):
            ipath = Path(path)  # TODO: Add consideration for db paths
            if ipath.is_file():
                idsmap = {}
                root = xr.load_dataset(ipath)
                dd_version = root.attrs.get('data_dictionary_version', None)
                if isinstance(dd_version, str) and 'data_dictionary_version' not in attrs:
                    attrs['data_dictionary_version'] = dd_version
                for ids in self.ids_top_levels:
                    try:
                        with imas.DBEntry(ipath, 'r', dd_version=dd_version) as netcdf_entry:
                            idsmap[f'{ids}'] = netcdf_entry.get(f'{ids}')
                    except Exception:
                        idsmap.pop(f'{ids}', None)
                for ids, ids_struct in idsmap.items():
                    if ids_struct.has_value:
                        ids_struct.validate()
                        ds_ids = imas.util.to_xarray(ids_struct)
                        unique_names = list(set(
                            [k for k in ds_ids.dims] +
                            [k for k in ds_ids.coords] +
                            [k for k in ds_ids.data_vars] +
                            [k for k in ds_ids.attrs]
                        ))
                        newcoords = {}
                        if ids == 'core_profiles' and 'profiles_1d:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.profiles_1d:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'core_sources' and 'source.profiles_1d:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.source.profiles_1d:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'core_transport' and 'model.profiles_1d:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.model.profiles_1d:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'equilibrium' and 'time_slice:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.time_slice:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'ntms' and 'time_slice:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.time_slice:i'] = np.arange(ds_ids['time'].size).astype(int)
                        dsvec.append(ds_ids.rename({k: f'{ids}.{k}' for k in unique_names}).assign_coords(newcoords))

        ds = xr.Dataset(attrs=attrs)
        for dss in dsvec:
            ds = ds.assign_coords(dss.coords).assign(dss.data_vars).assign_attrs(**dss.attrs)

        return ds


    def _read_imas_netcdf_files(
        self,
        path: str | Path,
        version: str | None = None,
    ) -> xr.Dataset:

        dsvec = []
        attrs: MutableMapping[str, Any] = {}

        if isinstance(path, (str, Path)):
            ipath = Path(path)  # TODO: Add consideration for db paths
            if ipath.is_dir():
                idsmap = {}
                for ids in self.ids_top_levels:
                    top_level_path = ipath / f'{ids}.nc'
                    if top_level_path.is_file():
                        root = xr.load_dataset(ipath / f'{ids}.nc')
                        dd_version = root.attrs.get('data_dictionary_version', None)
                        if isinstance(dd_version, str) and 'data_dictionary_version' not in attrs:
                            attrs['data_dictionary_version'] = dd_version
                        with imas.DBEntry(ipath / f'{ids}.nc', 'r', dd_version=dd_version) as netcdf_entry:
                            idsmap[f'{ids}'] = netcdf_entry.get(f'{ids}')
                for ids, ids_struct in idsmap.items():
                    if ids_struct.has_value:
                        ids_struct.validate()
                        ds_ids = imas.util.to_xarray(ids_struct)
                        unique_names = list(set(
                            [k for k in ds_ids.dims] +
                            [k for k in ds_ids.coords] +
                            [k for k in ds_ids.data_vars] +
                            [k for k in ds_ids.attrs]
                        ))
                        newcoords = {}
                        if ids == 'core_profiles' and 'profiles_1d:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.profiles_1d:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'core_sources' and 'source.profiles_1d:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.source.profiles_1d:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'core_transport' and 'model.profiles_1d:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.model.profiles_1d:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'equilibrium' and 'time_slice:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.time_slice:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'ntms' and 'time_slice:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.time_slice:i'] = np.arange(ds_ids['time'].size).astype(int)
                        dsvec.append(ds_ids.rename({k: f'{ids}.{k}' for k in unique_names}).assign_coords(newcoords))

        ds = xr.Dataset(attrs=attrs)
        for dss in dsvec:
            ds = ds.assign_coords(dss.coords).assign(dss.data_vars).assign_attrs(**dss.attrs)

        return ds


    def _read_imas_hdf5_files_with_core(
        self,
        path: str | Path,
        version: str | None = None,
    ) -> xr.Dataset:

        #dsvec = []

        ds = xr.Dataset()
        #for dss in dsvec:
        #    ds = ds.assign_coords(dss.coords).assign(dss.data_vars).assign_attrs(**dss.attrs)

        return ds


    def _read_imas_hdf5_files_without_core(
        self,
        path: str | Path,
        version: str | None = None,
    ) -> xr.Dataset:

        dsvec = []
        attrs: MutableMapping[str, Any] = {}

        if isinstance(path, (str, Path)):
            data: MutableMapping[str, Any] = {}
            ipath = Path(path)
            if ipath.is_dir():

                idsmap = {}
                for ids in self.ids_top_levels:
                    dd_version_tag = 'ids_properties&verions_put&data_dictionary'
                    top_level_path = ipath / f'{ids}.h5'
                    if top_level_path.is_file():
                        h5_data = h5py.File(top_level_path)
                        if f'{ids}' in h5_data:
                            idsmap[f'{ids}'] = {k: v[()] for k, v in h5_data[f'{ids}'].items()}
                            if isinstance(idsmap[f'{ids}'].get(dd_version_tag, None), bytes) and 'data_dictionary_version' not in attrs:
                                attrs['data_dictionary_version'] = idsmap[f'{ids}'][dd_version_tag].decode('utf-8')
                for ids, idsdata in idsmap.items():
                    ids_struct = self._convert_to_ids_structure(f'{ids}', idsdata, delimiter='&', version=attrs.get('data_dictionary_version', None))
                    if ids_struct.has_value:
                        ids_struct.validate()
                        ds_ids = imas.util.to_xarray(ids_struct)
                        unique_names = list(set(
                            [k for k in ds_ids.dims] +
                            [k for k in ds_ids.coords] +
                            [k for k in ds_ids.data_vars] +
                            [k for k in ds_ids.attrs]
                        ))
                        newcoords = {}
                        if ids == 'core_profiles' and 'profiles_1d:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.profiles_1d:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'core_sources' and 'source.profiles_1d:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.source.profiles_1d:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'core_transport' and 'model.profiles_1d:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.model.profiles_1d:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'equilibrium' and 'time_slice:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.time_slice:i'] = np.arange(ds_ids['time'].size).astype(int)
                        if ids == 'ntms' and 'time_slice:i' not in unique_names and 'time' in unique_names:
                            newcoords[f'{ids}.time_slice:i'] = np.arange(ds_ids['time'].size).astype(int)
                        dsvec.append(ds_ids.rename({k: f'{ids}.{k}' for k in unique_names}).assign_coords(newcoords))

        ds = xr.Dataset()
        for dss in dsvec:
            ds = ds.assign_coords(dss.coords).assign(dss.data_vars).assign_attrs(**dss.attrs)

        return ds


    def _write_imas_directory(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False,
        window: ArrayLike | None = None,
    ) -> None:
        if isinstance(path, (str, Path)):
            opath = Path(path)
            if opath.suffix.lower() in ['.nc', '.ncdf', '.cdf']:
                logger.warning(f'Writing multiple IDS structures into a single netcdf file not supported by imas-python. Aborting write...')
                #self._write_imas_netcdf_file(opath, data, overwrite=overwrite, window=window)
            else:
                interface = 'netcdf'
                if interface == 'netcdf':
                    self._write_imas_netcdf_files(opath, data, overwrite=overwrite, window=window)
                if interface == 'hdf5':
                    if self.has_imas:
                        self._write_imas_hdf5_files_with_core(opath, data, overwrite=overwrite, window=window)
                    else:
                        self._write_imas_hdf5_files_without_core(opath, data, overwrite=overwrite, window=window)


    def _write_imas_netcdf_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False,
        window: ArrayLike | None = None,
    ) -> None:
        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            opath = Path(path)
            if not (opath.exists() and not overwrite):
                opath.parent.mkdir(parents=True, exist_ok=True)
                datadict = {}
                datadict.update({k: np.arange(v).astype(int) for k, v in data.sizes.items()})
                datadict.update({k: v.values for k, v in data.coords.items()})
                datadict.update({k: v.values for k, v in data.data_vars.items()})
                for field_name in self.last_index_fields:
                    datadict.pop(f'{field_name}:i', None)
                idsmap = {}
                dd_version = data.attrs.get('data_dictionary_version', None)
                for ids in self.ids_top_levels:
                    idsdata = {f'{k}'[len(ids) + 1:]: v for k, v in datadict.items() if f'{k}'.startswith(f'{ids}.')}
                    if idsdata:
                        ids_struct = self._convert_to_ids_structure(f'{ids}', idsdata, delimiter='.', version=dd_version)
                        if ids_struct.has_value:
                            ids_struct.validate()
                            idsmap[f'{ids}'] = ids_struct
                            if dd_version is None:
                                dd_version = str(ids_struct['ids_properties']['version_put']['data_dictionary'])
                for ids, ids_struct in idsmap.items():
                    with imas.DBEntry(opath, 'w', dd_version=dd_version) as netcdf_entry:
                        netcdf_entry.put(ids_struct)
                logger.info(f'Saved {self.format} data into {opath.resolve()}')
            else:
                logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    def _write_imas_netcdf_files(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False,
        window: ArrayLike | None = None,
    ) -> None:
        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            opath = Path(path)
            if not (opath.exists() and not overwrite):
                opath.mkdir(parents=True, exist_ok=True)
                datadict = {}
                datadict.update({k: np.arange(v).astype(int) for k, v in data.sizes.items()})
                datadict.update({k: v.values for k, v in data.coords.items()})
                datadict.update({k: v.values for k, v in data.data_vars.items()})
                for field_name in self.last_index_fields:
                    datadict.pop(f'{field_name}:i', None)
                idsmap = {}
                dd_version = data.attrs.get('data_dictionary_version', None)
                for ids in self.ids_top_levels:
                    idsdata = {f'{k}'[len(ids) + 1:]: v for k, v in datadict.items() if f'{k}'.startswith(f'{ids}.')}
                    if idsdata:
                        ids_struct = self._convert_to_ids_structure(f'{ids}', idsdata, delimiter='.', version=dd_version)
                        if ids_struct.has_value:
                            ids_struct.validate()
                            idsmap[f'{ids}'] = ids_struct
                            if dd_version is None:
                                dd_version = str(ids_struct['ids_properties']['version_put']['data_dictionary'])
                for ids, ids_struct in idsmap.items():
                    with imas.DBEntry(opath / f'{ids}.nc', 'w', dd_version=dd_version) as netcdf_entry:
                        netcdf_entry.put(ids_struct)
                logger.info(f'Saved {self.format} data into {opath.resolve()}')
            else:
                logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    def _write_imas_hdf5_files_with_core(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False,
        window: ArrayLike | None = None,
    ) -> None:
        pass


    def _write_imas_hdf5_files_without_core(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False,
        window: ArrayLike | None = None,
    ) -> None:
        pass


    @property
    def input_cocos(
        self,
    ) -> int:
        version = self.input.attrs.get('data_dictionary_version', imas.dd_zip.latest_dd_version())
        return self.default_cocos_3 if Version(version) < Version('4') else self.default_cocos_4


    @property
    def output_cocos(
        self,
    ) -> int:
        version = self.output.attrs.get('data_dictionary_version', imas.dd_zip.latest_dd_version())
        return self.default_cocos_3 if Version(version) < Version('4') else self.default_cocos_4


    def to_eqdsk(
        self,
        time_index: int = -1,
        side: str = 'output',
        cocos: int | None = None,
        transpose: bool = False,
    ) -> MutableMapping[str, Any]:
        eqdata: MutableMapping[str, Any] = {}
        time_eq = 'equilibrium.time'
        data = (
            self.input.isel({time_eq: time_index})
            if side == 'input' else
            self.output.isel({time_eq: time_index})
        )
        default_cocos = self.input_cocos if side == 'input' else self.output_cocos
        if cocos is None:
            cocos = default_cocos
        rectangular_index = []
        tag = 'equilibrium.time_slice.profiles_2d.grid_type.name'
        if tag in data:
            rectangular_index = [i for i, name in enumerate(data[tag]) if name == 'rectangular']
        if len(rectangular_index) > 0:
            data = data.isel({'equilibrium.time_slice.profiles_2d:i': rectangular_index[0]})
            psin_eq = 'equilibrium.time_slice.profiles_1d.psi_norm'
            psinvec = data[psin_eq].to_numpy().flatten() if psin_eq in data else None
            conversion = None
            ikwargs = {'fill_value': 'extrapolate'}
            if psinvec is None:
                conversion = (
                    (data['equilibrium.time_slice.profiles_1d.psi'] - data['equilibrium.time_slice.global_quantities.psi_axis']) /
                    (data['equilibrium.time_slice.global_quantities.psi_boundary'] - data['equilibrium.time_slice.global_quantities.psi_axis'])
                ).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_2d.grid.dim1'
            if tag in data:
                rvec = data[tag].to_numpy().flatten()
                eqdata['nr'] = rvec.size
                eqdata['rdim'] = float(np.nanmax(rvec) - np.nanmin(rvec))
                eqdata['rleft'] = float(np.nanmin(rvec))
                if psinvec is None:
                    psinvec = np.linspace(0.0, 1.0, len(rvec)).flatten()
            tag = 'equilibrium.time_slice.profiles_2d.grid.dim2'
            if tag in data:
                zvec = data[tag].to_numpy().flatten()
                eqdata['nz'] = zvec.size
                eqdata['zdim'] = float(np.nanmax(zvec) - np.nanmin(zvec))
                eqdata['zmid'] = float(np.nanmax(zvec) + np.nanmin(zvec)) / 2.0
            tag = 'equilibrium.vacuum_toroidal_field.r0'
            if tag in data:
                eqdata['rcentr'] = float(data[tag].to_numpy().flatten())
            tag = 'equilibrium.vacuum_toroidal_field.b0'
            if tag in data:
                eqdata['bcentr'] = float(data[tag].to_numpy().flatten())
            tag = 'equilibrium.time_slice.global_quantities.magnetic_axis.r'
            if tag in data:
                eqdata['rmagx'] = float(data[tag].to_numpy().flatten())
            tag = 'equilibrium.time_slice.global_quantities.magnetic_axis.z'
            if tag in data:
                eqdata['zmagx'] = float(data[tag].to_numpy().flatten())
            tag = 'equilibrium.time_slice.global_quantities.psi_axis'
            if tag in data:
                eqdata['simagx'] = float(data[tag].to_numpy().flatten())
            tag = 'equilibrium.time_slice.global_quantities.psi_boundary'
            if tag in data:
                eqdata['sibdry'] = float(data[tag].to_numpy().flatten())
            tag = 'equilibrium.time_slice.global_quantities.ip'
            if tag in data:
                eqdata['cpasma'] = float(data[tag].to_numpy().flatten())
            tag = 'equilibrium.time_slice.profiles_1d.f'
            if tag in data:
                if conversion is None:
                    eqdata['fpol'] = data.drop_duplicates(psin_eq)[tag].interp({psin_eq: psinvec}).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['fpol'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.pressure'
            if tag in data:
                if conversion is None:
                    eqdata['pres'] = data.drop_duplicates(psin_eq)[tag].interp({psin_eq: psinvec}).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['pres'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.f_df_dpsi'
            if tag in data:
                if conversion is None:
                    eqdata['ffprime'] = data.drop_duplicates(psin_eq)[tag].interp({psin_eq: psinvec}).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['ffprime'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.dpressure_dpsi'
            if tag in data:
                if conversion is None:
                    eqdata['pprime'] = data.drop_duplicates(psin_eq)[tag].interp({psin_eq: psinvec}).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['pprime'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_2d.psi'
            if tag in data:
                dims = data[tag].dims
                dim1_tag = [dim for dim in dims if 'dim1' in f'{dim}'][0]
                dim2_tag = [dim for dim in dims if 'dim2' in f'{dim}'][0]
                do_transpose = bool(dims.index(dim1_tag) < dims.index(dim2_tag))
                if transpose:
                    do_transpose = bool(not do_transpose)
                eqdata['psi'] = data[tag].to_numpy().T if do_transpose else data[tag].to_numpy()
            tag = 'equilibrium.time_slice.profiles_1d.q'
            if tag in data:
                if conversion is None:
                    eqdata['qpsi'] = data.drop_duplicates(psin_eq)[tag].interp({psin_eq: psinvec}).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['qpsi'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            rtag = 'equilibrium.time_slice.boundary.outline.r'
            ztag = 'equilibrium.time_slice.boundary.outline.z'
            if rtag in data and ztag in data:
                rdata = data[rtag].dropna('equilibrium.time_slice.boundary.outline.r:i').to_numpy().flatten()
                zdata = data[ztag].dropna('equilibrium.time_slice.boundary.outline.r:i').to_numpy().flatten()
                if len(rdata) == len(zdata):
                    eqdata['nbdry'] = len(rdata)
                    eqdata['rbdry'] = rdata
                    eqdata['zbdry'] = zdata
            eqdata = convert_cocos(eqdata, cocos_in=default_cocos, cocos_out=cocos, bt_sign_out=None, ip_sign_out=None)
        return eqdata


    def generate_eqdsk_file(
        self,
        path: str | Path,
        time_index: int = -1,
        side: str = 'output',
        cocos: int | None = None,
        transpose: bool = False,
    ) -> None:
        eqpath = None
        if isinstance(path, (str, Path)):
            eqpath = Path(path)
        assert isinstance(eqpath, Path)
        eqdata = self.to_eqdsk(time_index=time_index, side=side, cocos=cocos, transpose=transpose)
        write_eqdsk(eqdata, eqpath)
        logger.info('Successfully generated g-eqdsk file, {path}')


    def generate_all_eqdsk_files(
        self,
        basepath: str | Path,
        side: str = 'output',
        cocos: int | None = None,
        transpose: bool = False,
    ) -> None:
        path = None
        if isinstance(basepath, (str, Path)):
            path = Path(basepath)
        assert isinstance(path, Path)
        data = self.input if side == 'input' else self.output
        time_eq = 'equilibrium.time'
        if time_eq in data:
            for ii, time in enumerate(data[time_eq].to_numpy().flatten()):
                stem = f'{path.stem}'
                if stem.endswith('_input'):
                    stem = stem[:-6]
                time_tag = int(np.rint(time * 1000))
                eqpath = path.parent / f'{stem}_{time_tag:06d}ms_input{path.suffix}'
                self.generate_eqdsk_file(eqpath, time_index=ii, side=side, cocos=cocos, transpose=transpose)


    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        input: str | Path | None = None,
        output: str | Path | None = None,
    ) -> Self:
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


    @classmethod
    def from_imas(
        cls,
        obj: io,
        side: str = 'output',
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            newobj.input = obj.input if side == 'input' else obj.output
        return newobj


    @classmethod
    def from_omas(
        cls,
        obj: io,
        side: str = 'output',
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input if side == 'input' else obj.output
            # TODO: Should compress down last_index_fields to true coordinates and set rho values as actual dimensions
            top_levels = {}
            for key in data.coords:
                components = f'{key}'.split('.')
                if components[0] not in top_levels:
                    top_levels[f'{components[0]}'] = 1
            for level in top_levels:
                n_time_coords = 0
                for key in data.coords:
                    components = f'{key}'.split('.')
                    if len(components) > 1 and components[0] == level and components[-1] == 'time':
                        n_time_coords += 1
                if n_time_coords > 1:
                    top_levels[level] = 0
            data = data.assign({f'{k}.ids_properties.homogeneous_time': ([], np.array(v)) for k, v in top_levels.items()})
            newobj.input = data
        return newobj


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
            data = obj.input if side == 'input' else obj.output
        return newobj

