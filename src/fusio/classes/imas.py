import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr
from scipy.integrate import cumulative_simpson

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
                        h5_data = h5py.File(top_level_path, 'r')
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


    @classmethod
    def from_gacode(
        cls,
        obj: io,
        side: str = 'output',
        time: float = 0.0,
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input if side == 'input' else obj.output

            if 'polflux' not in data:
                logger.warning('No polflux found in gacode data. Aborting from_gacode...')
                return newobj

            d = data.isel(n=0)

            rcentr = float(d['rcentr'].to_numpy().flatten()[0])
            bcentr = float(d['bcentr'].to_numpy().flatten()[0])
            current_MA = float(d['current'].to_numpy().flatten()[0])
            ip_A = current_MA * 1.0e6

            polflux = d['polflux'].to_numpy().flatten()
            q = d['q'].to_numpy().flatten()
            rmin = d['rmin'].to_numpy().flatten()
            rmaj = d['rmaj'].to_numpy().flatten()
            zmag = d['zmag'].to_numpy().flatten()
            kappa = d['kappa'].to_numpy().flatten()
            delta = d['delta'].to_numpy().flatten()
            nrho = polflux.size



            dpsi = np.gradient(polflux, rmin)
            phi = np.zeros(nrho)
            phi[1:] = cumulative_simpson(y=q * dpsi, x=rmin)[: nrho - 1] if nrho > 2 else np.cumsum(q[1:] * np.diff(polflux))
            phi = np.abs(phi)

            if 'b_unit' in d:
                b_unit = d['b_unit'].to_numpy().flatten()
            else:
                torflux = phi
                b_unit = np.ones(nrho)
                dtf = np.diff(torflux)
                drmin_diff = np.diff(rmin)
                bu_mask = (drmin_diff > 0) & (rmin[1:] > 0)
                b_unit[1:] = np.where(bu_mask, np.abs(dtf) / (2 * np.pi * rmin[1:] * drmin_diff), 1.0)
                b_unit[0] = b_unit[1] if nrho > 1 else 1.0

            rho_tor = np.sqrt(phi / (np.pi * np.abs(bcentr))) if np.abs(bcentr) > 0 else rmin
            rho_tor_a = rho_tor[-1] if rho_tor[-1] > 0.0 else 1.0
            rho_tor_norm = rho_tor / rho_tor_a

            dpsi_drho_tor = np.gradient(polflux, rho_tor)
            drho_tor_drmin = np.gradient(rho_tor, rmin)

            if 'fpol' in d:
                F = np.abs(d['fpol'].to_numpy().flatten())
            else:
                F = np.full(nrho, np.abs(bcentr * rcentr))

            r_in = d['r_in'].to_numpy().flatten() if 'r_in' in d else rmaj - rmin
            r_out = d['r_out'].to_numpy().flatten() if 'r_out' in d else rmaj + rmin

            volp_miller = d['volp_miller'].to_numpy().flatten() if 'volp_miller' in d else np.zeros(nrho)
            volume = np.zeros(nrho)
            if nrho > 2 and np.any(volp_miller > 0):
                volume[1:] = cumulative_simpson(y=volp_miller, x=rmin)[: nrho - 1]

            dvolume_dpsi = np.zeros(nrho)
            dpsi_drmin = np.gradient(polflux, rmin)
            mask = np.abs(dpsi_drmin) > 1.0e-30
            dvolume_dpsi[mask] = volp_miller[mask] / dpsi_drmin[mask]
            dvolume_dpsi = np.abs(dvolume_dpsi)

            fsa_1_over_R = d['fsa_1_over_R'].to_numpy().flatten() if 'fsa_1_over_R' in d else 1.0 / rmaj
            fsa_1_over_R2 = d['fsa_1_over_R2'].to_numpy().flatten() if 'fsa_1_over_R2' in d else 1.0 / rmaj ** 2

            if 'fsa_b_phys2' in d:
                fsa_B2 = d['fsa_b_phys2'].to_numpy().flatten()
                fsa_1_over_B2 = d['fsa_1_over_b_phys2'].to_numpy().flatten()
            elif 'fsa_B2' in d and 'bt2_miller' in d:
                bt2_miller = d['bt2_miller'].to_numpy().flatten()
                bp2_miller = d['bp2_miller'].to_numpy().flatten() if 'bp2_miller' in d else np.zeros(nrho)
                fsa_B2_miller = d['fsa_B2'].to_numpy().flatten()
                fsa_1_over_B2_miller = d['fsa_1_over_B2'].to_numpy().flatten()
                bt2_corrected = F ** 2 * fsa_1_over_R2
                bp2_from_miller = np.maximum(fsa_B2_miller - bt2_miller, 0.0)
                fsa_B2 = bt2_corrected + bp2_from_miller
                B_sq_norm = np.where(fsa_B2_miller > 1e-30, fsa_B2 / fsa_B2_miller, 1.0)
                fsa_1_over_B2 = np.where(B_sq_norm > 1e-30, fsa_1_over_B2_miller / B_sq_norm, fsa_1_over_B2_miller)
            else:
                fsa_B2 = F ** 2 * fsa_1_over_R2
                fsa_1_over_B2 = np.where(
                    fsa_B2 > 1e-30,
                    1.0 / fsa_B2,
                    1.0 / (bcentr ** 2),
                )
            gradr_miller = d['gradr_miller'].to_numpy().flatten() if 'gradr_miller' in d else np.ones(nrho)
            fsa_gradr2 = d['fsa_gradr2'].to_numpy().flatten() if 'fsa_gradr2' in d else gradr_miller ** 2
            fsa_gradr2_over_R2 = d['fsa_gradr2_over_R2'].to_numpy().flatten() if 'fsa_gradr2_over_R2' in d else gradr_miller ** 2 / rmaj ** 2

            mask_rho = np.abs(drho_tor_drmin) > 1.0e-30
            drho_tor_drmin_safe = np.where(mask_rho, drho_tor_drmin, 1.0)

            gm1 = fsa_1_over_R2
            dpsi_drho_tor_safe = np.where(np.abs(dpsi_drho_tor) > 1.0e-30, dpsi_drho_tor, 1.0)
            if 'fsa_grad_psi2_over_R2' in d:
                gm2 = d['fsa_grad_psi2_over_R2'].to_numpy().flatten() / dpsi_drho_tor_safe ** 2
            else:
                gm2 = fsa_gradr2_over_R2 * drho_tor_drmin_safe ** 2
            if 'fsa_grad_psi2' in d:
                gm3 = d['fsa_grad_psi2'].to_numpy().flatten() / dpsi_drho_tor_safe ** 2
            else:
                gm3 = fsa_gradr2 * drho_tor_drmin_safe ** 2
            gm4 = fsa_1_over_B2
            gm5 = fsa_B2
            if 'fsa_grad_psi' in d:
                gm7 = d['fsa_grad_psi'].to_numpy().flatten() / np.abs(dpsi_drho_tor_safe)
            else:
                gm7 = gradr_miller * drho_tor_drmin_safe
            gm9 = fsa_1_over_R

            jtor_fields = ['johm', 'jbs', 'jbstor', 'jrf', 'jnb']
            j_phi = np.zeros(nrho)
            for jfield in jtor_fields:
                if jfield in d:
                    j_phi += d[jfield].to_numpy().flatten()

            if np.all(j_phi == 0) and np.abs(ip_A) > 0 and np.any(volume > 0):
                if 'Ip_profile_miller' in d:
                    Ip_enc = d['Ip_profile_miller'].to_numpy().flatten()
                else:
                    Ip_enc = ip_A * rho_tor_norm ** 2
                rho_tor_a = rho_tor[-1] if rho_tor[-1] > 0.0 else 1.0
                drho_norm_drmin = drho_tor_drmin / rho_tor_a
                drho_norm_drmin_safe = np.where(np.abs(drho_norm_drmin) > 1e-30, drho_norm_drmin, 1.0)
                vpr = volp_miller / drho_norm_drmin_safe
                spr = vpr * fsa_1_over_R / (2.0 * np.pi)
                dIp_drhon = np.gradient(Ip_enc, rho_tor_norm)
                mask_s = np.abs(spr) > 1.0e-30
                j_phi[mask_s] = dIp_drhon[mask_s] / spr[mask_s]

            ds_vars = {}
            ds_coords = {}

            ds_vars['equilibrium.ids_properties.homogeneous_time'] = ([], np.int32(1))
            ds_vars['equilibrium.vacuum_toroidal_field.r0'] = ([], np.float64(rcentr))
            ds_vars['equilibrium.vacuum_toroidal_field.b0'] = (['equilibrium.time'], np.array([bcentr]))

            ts_i = 'equilibrium.time_slice:i'
            p1d = 'equilibrium.time_slice.profiles_1d'
            gq = 'equilibrium.time_slice.global_quantities'
            bdry = 'equilibrium.time_slice.boundary'
            rho_dim = f'{p1d}.psi:i'

            ds_coords['equilibrium.time'] = (['equilibrium.time'], np.array([time]))
            ds_coords[ts_i] = ([ts_i], np.array([0]))
            ds_coords[rho_dim] = ([rho_dim], np.arange(nrho))

            ds_vars[f'{p1d}.psi'] = ([ts_i, rho_dim], np.expand_dims(polflux, axis=0))
            ds_vars[f'{p1d}.phi'] = ([ts_i, rho_dim], np.expand_dims(phi, axis=0))
            ds_vars[f'{p1d}.rho_tor_norm'] = ([ts_i, rho_dim], np.expand_dims(rho_tor_norm, axis=0))
            ds_vars[f'{p1d}.rho_tor'] = ([ts_i, rho_dim], np.expand_dims(rho_tor, axis=0))
            ds_vars[f'{p1d}.f'] = ([ts_i, rho_dim], np.expand_dims(F, axis=0))
            ds_vars[f'{p1d}.r_inboard'] = ([ts_i, rho_dim], np.expand_dims(r_in, axis=0))
            ds_vars[f'{p1d}.r_outboard'] = ([ts_i, rho_dim], np.expand_dims(r_out, axis=0))
            ds_vars[f'{p1d}.q'] = ([ts_i, rho_dim], np.expand_dims(q, axis=0))
            ds_vars[f'{p1d}.dpsi_drho_tor'] = ([ts_i, rho_dim], np.expand_dims(dpsi_drho_tor, axis=0))
            ds_vars[f'{p1d}.dvolume_dpsi'] = ([ts_i, rho_dim], np.expand_dims(dvolume_dpsi, axis=0))
            ds_vars[f'{p1d}.volume'] = ([ts_i, rho_dim], np.expand_dims(volume, axis=0))
            ds_vars[f'{p1d}.elongation'] = ([ts_i, rho_dim], np.expand_dims(kappa, axis=0))
            ds_vars[f'{p1d}.triangularity_upper'] = ([ts_i, rho_dim], np.expand_dims(delta, axis=0))
            ds_vars[f'{p1d}.triangularity_lower'] = ([ts_i, rho_dim], np.expand_dims(delta, axis=0))
            ds_vars[f'{p1d}.j_phi'] = ([ts_i, rho_dim], np.expand_dims(j_phi, axis=0))

            ds_vars[f'{p1d}.gm1'] = ([ts_i, rho_dim], np.expand_dims(gm1, axis=0))
            ds_vars[f'{p1d}.gm2'] = ([ts_i, rho_dim], np.expand_dims(gm2, axis=0))
            ds_vars[f'{p1d}.gm3'] = ([ts_i, rho_dim], np.expand_dims(gm3, axis=0))
            ds_vars[f'{p1d}.gm4'] = ([ts_i, rho_dim], np.expand_dims(gm4, axis=0))
            ds_vars[f'{p1d}.gm5'] = ([ts_i, rho_dim], np.expand_dims(gm5, axis=0))
            ds_vars[f'{p1d}.gm7'] = ([ts_i, rho_dim], np.expand_dims(gm7, axis=0))
            ds_vars[f'{p1d}.gm9'] = ([ts_i, rho_dim], np.expand_dims(gm9, axis=0))

            ds_vars[f'{gq}.ip'] = ([ts_i], np.array([ip_A]))
            ds_vars[f'{gq}.magnetic_axis.r'] = ([ts_i], np.array([rmaj[0]]))
            ds_vars[f'{gq}.magnetic_axis.z'] = ([ts_i], np.array([zmag[0]]))
            ds_vars[f'{gq}.psi_axis'] = ([ts_i], np.array([polflux[0]]))
            ds_vars[f'{gq}.psi_boundary'] = ([ts_i], np.array([polflux[-1]]))

            ds_vars[f'{bdry}.minor_radius'] = ([ts_i], np.array([rmin[-1]]))
            ds_vars[f'{bdry}.type'] = ([ts_i], np.array([0]))

            newobj.input = xr.Dataset(data_vars=ds_vars, coords=ds_coords)

        return newobj

