import copy
import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr
from scipy.integrate import cumulative_simpson  # type: ignore[import-untyped]
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from packaging.version import Version
import h5py  # type: ignore[import-untyped]
import imas  # type: ignore[import-untyped]
from imas.ids_base import IDSBase  # type: ignore[import-untyped]
from imas.ids_structure import IDSStructure  # type: ignore[import-untyped]
from imas.ids_struct_array import IDSStructArray  # type: ignore[import-untyped]
from .io import io
from ..utils.eqdsk_tools import (
    calculate_mxh_coefficients_from_eqdsk_dict,
    convert_cocos,
    define_cocos_converter,
    write_eqdsk,
)
from ..utils import plasma_tools

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
    default_version: Final[str] = imas.dd_zip.latest_dd_version()
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
        # This was only available for imas-python < 2.2
        self.has_imas: bool = getattr(
            imas.backends.imas_core.imas_interface, 'has_imas', True
        )
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
            # One entry per element of every enclosing array of structures, not just the immediate parent
            parts = key.split(delimiter)
            ancestor_sizes = tuple(index_data[delimiter.join(parts[:n])] for n in range(1, len(parts)) if delimiter.join(parts[:n]) in index_data)
            data[f'{key}{delimiter}AOS_SHAPE'] = np.full(ancestor_sizes + (1, ), index_data[key]).astype(int)
        shape_data = {}
        for key in list(data.keys()):
            if key.endswith(f'{delimiter}AOS_SHAPE'):
                shape_data[key] = data.pop(key)
            elif key.endswith('_SHAPE'):
                data.pop(key)
        for key in sorted(shape_data.keys(), key=len):
            _recursive_resize_struct_array(ids_struct, key.replace('[]', '').split(delimiter), shape_data[key])
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].dtype.kind == 'S' and data[key].size == 1 and data[key].item() == b'':
                continue  # imas-python >= 2.1 to_xarray() emits empty placeholder variables for structure nodes
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
            psin_data = xr.Dataset()
            if psinvec is None:
                conversion = (
                    (data['equilibrium.time_slice.profiles_1d.psi'] - data['equilibrium.time_slice.global_quantities.psi_axis']) /
                    (data['equilibrium.time_slice.global_quantities.psi_boundary'] - data['equilibrium.time_slice.global_quantities.psi_axis'])
                ).to_numpy().flatten()
            else:
                psin_dim = data[psin_eq].dims[0]
                psin_data = data.swap_dims({psin_dim: psin_eq}).drop_duplicates(psin_eq)
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
                eqdata['rcentr'] = float(data[tag].to_numpy().item())
            tag = 'equilibrium.vacuum_toroidal_field.b0'
            if tag in data:
                eqdata['bcentr'] = float(data[tag].to_numpy().item())
            tag = 'equilibrium.time_slice.global_quantities.magnetic_axis.r'
            if tag in data:
                eqdata['rmagx'] = float(data[tag].to_numpy().item())
            tag = 'equilibrium.time_slice.global_quantities.magnetic_axis.z'
            if tag in data:
                eqdata['zmagx'] = float(data[tag].to_numpy().item())
            tag = 'equilibrium.time_slice.global_quantities.psi_axis'
            if tag in data:
                eqdata['simagx'] = float(data[tag].to_numpy().item())
            tag = 'equilibrium.time_slice.global_quantities.psi_boundary'
            if tag in data:
                eqdata['sibdry'] = float(data[tag].to_numpy().item())
            tag = 'equilibrium.time_slice.global_quantities.ip'
            if tag in data:
                eqdata['cpasma'] = float(data[tag].to_numpy().item())
            tag = 'equilibrium.time_slice.profiles_1d.f'
            if tag in data:
                if conversion is None:
                    eqdata['fpol'] = psin_data[tag].interp({psin_eq: psinvec}, kwargs=ikwargs).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['fpol'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.pressure'
            if tag in data:
                if conversion is None:
                    eqdata['pres'] = psin_data[tag].interp({psin_eq: psinvec}, kwargs=ikwargs).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['pres'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.f_df_dpsi'
            if tag in data:
                if conversion is None:
                    eqdata['ffprime'] = psin_data[tag].interp({psin_eq: psinvec}, kwargs=ikwargs).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['ffprime'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.dpressure_dpsi'
            if tag in data:
                if conversion is None:
                    eqdata['pprime'] = psin_data[tag].interp({psin_eq: psinvec}, kwargs=ikwargs).to_numpy().flatten()
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
                    eqdata['qpsi'] = psin_data[tag].interp({psin_eq: psinvec}, kwargs=ikwargs).to_numpy().flatten()
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


    def to_cgyro_parameters(
        self,
        time: float | Sequence[float] | NDArray | None = None,
        rho: float | Sequence[float] | NDArray | None = None,
        side: str = 'output',
        full_impurities: bool = False,
        n_mxh_moments: int = 5,
        n_fine: int = 201,
    ) -> xr.Dataset:
        """Compute local dimensionless CGYRO input parameters directly from the IMAS structure.

        Mirrors the torax_io.to_cgyro_parameters() output interface (species-suffixed
        variables with ions first and electrons last, dims (time, rho)). Physics is
        delegated to fusio.utils.plasma_tools and geometry to the MXH contour tracing
        in fusio.utils.eqdsk_tools via to_eqdsk().

        - time=None processes every core_profiles time slice; a scalar or sequence
          selects the nearest slice per requested time (each paired with the nearest
          equilibrium slice).
        - rho=None returns the internal fine grid (n_fine points); otherwise results
          are interpolated onto the requested rho_tor_norm values.
        - full_impurities=False lumps all non-main ions into a single effective species
          preserving quasineutrality and Zeff; True keeps every ion separately.
        - Q is returned with the sign convention of the source data (COCOS of the IDS).

        Radial derivatives use shape-preserving PCHIP interpolants evaluated
        analytically, which converge at the plasma edge where finite differences on
        linearly interpolated profiles do not.
        """
        time_cp = 'core_profiles.time'
        time_eq = 'equilibrium.time'
        data = self.input if side == 'input' else self.output
        required = (
            time_cp,
            time_eq,
            'core_profiles.profiles_1d.grid.rho_tor_norm',
            'equilibrium.time_slice.profiles_1d.rho_tor_norm',
            'equilibrium.time_slice.profiles_1d.psi',
            'equilibrium.time_slice.profiles_1d.phi',
            'equilibrium.time_slice.profiles_1d.q',
            'core_profiles.profiles_1d.electrons.temperature',
        )
        missing = [tag for tag in required if tag not in data]
        if missing:
            logger.error(f'Missing required fields for CGYRO parameter computation: {missing}')
            return xr.Dataset()

        cp_times = np.atleast_1d(data[time_cp].to_numpy()).flatten()
        eq_times = np.atleast_1d(data[time_eq].to_numpy()).flatten()
        if time is None:
            cp_indices = list(range(len(cp_times)))
        else:
            cp_indices = []
            for t in np.atleast_1d(np.asarray(time, dtype=float)).flatten():
                idx = int(np.argmin(np.abs(cp_times - t)))
                if idx not in cp_indices:
                    cp_indices.append(idx)

        rho_out = None if rho is None else np.atleast_1d(np.asarray(rho, dtype=float)).flatten()
        slice_vars: MutableSequence[MutableMapping[str, NDArray]] = []
        slice_attrs: MutableSequence[MutableMapping[str, NDArray]] = []
        rho_grid = None
        for i in cp_indices:
            j = int(np.argmin(np.abs(eq_times - cp_times[i])))
            svars, sattrs = self._compute_cgyro_parameters_slice(
                data,
                i,
                j,
                side=side,
                full_impurities=full_impurities,
                n_mxh_moments=n_mxh_moments,
                n_fine=n_fine,
            )
            rho_fine = svars.pop('rho')
            if rho_out is not None:
                svars = {key: np.interp(rho_out, rho_fine, val) for key, val in svars.items()}
                sattrs = {key: np.interp(rho_out, rho_fine, val) for key, val in sattrs.items()}
                rho_grid = rho_out
            else:
                rho_grid = rho_fine
            slice_vars.append(svars)
            slice_attrs.append(sattrs)

        coords: MutableMapping[str, Any] = {
            'time': cp_times[cp_indices],
            'rho': rho_grid,
        }
        data_vars: MutableMapping[str, Any] = {}
        for key in slice_vars[0]:
            data_vars[key] = (['time', 'rho'], np.stack([svars[key] for svars in slice_vars], axis=0))
        attrs: MutableMapping[str, Any] = {}
        for key in slice_attrs[0]:
            attrs[key] = np.stack([sattrs[key] for sattrs in slice_attrs], axis=0)
        return xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)


    def _compute_cgyro_parameters_slice(
        self,
        data: xr.Dataset,
        cp_index: int,
        eq_index: int,
        side: str = 'output',
        full_impurities: bool = False,
        n_mxh_moments: int = 5,
        n_fine: int = 201,
    ) -> tuple[MutableMapping[str, NDArray], MutableMapping[str, NDArray]]:
        """Compute CGYRO parameters for one paired core_profiles / equilibrium time slice."""
        c = plasma_tools.constants_si()
        twopi = 2.0 * np.pi
        cpd = data.isel({'core_profiles.time': cp_index})
        eqd = data.isel({'equilibrium.time': eq_index})

        def profile_interpolator(x, y):
            srt = np.argsort(x)
            return PchipInterpolator(x[srt], y[srt], extrapolate=True)

        # --- Equilibrium 1D profiles as functions of rho_tor_norm ---
        rho_eq = np.abs(eqd['equilibrium.time_slice.profiles_1d.rho_tor_norm'].to_numpy().flatten())
        psi_interp = profile_interpolator(rho_eq, eqd['equilibrium.time_slice.profiles_1d.psi'].to_numpy().flatten())
        phi_interp = profile_interpolator(rho_eq, eqd['equilibrium.time_slice.profiles_1d.phi'].to_numpy().flatten())
        q_interp = profile_interpolator(rho_eq, eqd['equilibrium.time_slice.profiles_1d.q'].to_numpy().flatten())

        # --- Flux-surface geometry from MXH contour tracing on the fine grid ---
        rho_f = np.linspace(0.0, 1.0, n_fine)
        psi_f = psi_interp(rho_f)
        eqdsk = self.to_eqdsk(time_index=eq_index, side=side)
        mxh = calculate_mxh_coefficients_from_eqdsk_dict(copy.deepcopy(eqdsk), psi_f.copy())
        rmin = np.asarray(mxh['rmin'], dtype=float)
        rmaj = np.asarray(mxh['rmaj'], dtype=float)
        zmag = np.asarray(mxh['zmag'], dtype=float)
        kappa = np.asarray(mxh['kappa'], dtype=float)
        a = float(rmin[-1])
        rmin_interp = PchipInterpolator(rho_f, rmin)
        drmin_drho = rmin_interp.derivative()(rho_f)
        drmin_drho = np.where(np.abs(drmin_drho) > 1.0e-10, drmin_drho, 1.0e-10)

        def ddr(vals):
            # radial derivative d/dr on the fine grid via chain rule through rho
            return PchipInterpolator(rho_f, vals).derivative()(rho_f) / drmin_drho

        out: MutableMapping[str, NDArray] = {}
        out['rho'] = rho_f
        out[r'#RHO'] = rho_f
        out['RMIN'] = rmin / a
        out['RMAJ'] = rmaj / a
        out['ZMAG'] = zmag / a
        out['SHIFT'] = ddr(rmaj)
        out['DZMAG'] = ddr(zmag)
        out['KAPPA'] = kappa
        out['S_KAPPA'] = rmin * ddr(kappa) / kappa
        for mxh_key, name in (('delta', 'DELTA'), ('zeta', 'ZETA')):
            vals = np.asarray(mxh[mxh_key], dtype=float)
            out[name] = vals
            out[f'S_{name}'] = rmin * ddr(vals)
        for nn in range(3, min(6, n_mxh_moments) + 1):
            vals = np.asarray(mxh[f'sin{nn:d}'], dtype=float)
            out[f'SHAPE_SIN{nn:d}'] = vals
            out[f'SHAPE_S_SIN{nn:d}'] = rmin * ddr(vals)
        for nn in range(0, min(6, n_mxh_moments) + 1):
            vals = np.asarray(mxh[f'cos{nn:d}'], dtype=float)
            out[f'SHAPE_COS{nn:d}'] = vals
            out[f'SHAPE_S_COS{nn:d}'] = rmin * ddr(vals)

        # --- Reference field B_unit = (1 / r) d(phi / 2pi)/dr ---
        rmin_safe = np.where(rmin > 1.0e-6, rmin, 1.0e-6)
        b_unit = np.abs(phi_interp.derivative()(rho_f) / (twopi * rmin_safe * drmin_drho))
        b_unit[0] = 2.0 * b_unit[1] - b_unit[2]

        # --- Safety factor and magnetic shear ---
        q_f = q_interp(rho_f)
        grad_q = q_interp.derivative()(rho_f) / drmin_drho
        out['Q'] = q_f
        out['S'] = plasma_tools.calc_s_from_q_and_grad_q(q_f, grad_q, rmin)

        # --- Kinetic profiles (IMAS units: eV, m^-3) ---
        rho_cp = np.abs(cpd['core_profiles.profiles_1d.grid.rho_tor_norm'].to_numpy().flatten())
        te = profile_interpolator(rho_cp, cpd['core_profiles.profiles_1d.electrons.temperature'].to_numpy().flatten())(rho_f)
        ne_tag = 'core_profiles.profiles_1d.electrons.density_thermal'
        if ne_tag not in cpd:
            ne_tag = 'core_profiles.profiles_1d.electrons.density'
        ne = profile_interpolator(rho_cp, cpd[ne_tag].to_numpy().flatten())(rho_f)

        # --- Ion species ---
        name_tag = 'core_profiles.profiles_1d.ion.name'
        if name_tag not in cpd:
            name_tag = 'core_profiles.profiles_1d.ion.label'
        ion_names = [str(name) for name in np.atleast_1d(cpd[name_tag].to_numpy()).flatten()] if name_tag in cpd else []
        n_ion = len(ion_names)
        ni_tag = 'core_profiles.profiles_1d.ion.density_thermal'
        if ni_tag not in cpd:
            ni_tag = 'core_profiles.profiles_1d.ion.density'
        ni_all = np.atleast_2d(cpd[ni_tag].to_numpy()) if ni_tag in cpd else np.zeros((n_ion, len(rho_cp)))
        ti_tag = 'core_profiles.profiles_1d.ion.temperature'
        if ti_tag in cpd:
            ti_all = np.atleast_2d(cpd[ti_tag].to_numpy())
        else:
            ti_all = np.repeat(np.atleast_2d(cpd['core_profiles.profiles_1d.t_i_average'].to_numpy()), n_ion, axis=0)
        mass_tag = 'core_profiles.profiles_1d.ion.element.a'
        mass_all = cpd[mass_tag].to_numpy().reshape(n_ion, -1)[:, 0] if mass_tag in cpd else np.full(n_ion, 2.0)
        z1d_tag = 'core_profiles.profiles_1d.ion.z_ion_1d'
        zion_tag = 'core_profiles.profiles_1d.ion.z_ion'
        zn_tag = 'core_profiles.profiles_1d.ion.element.z_n'

        ni_fine = np.zeros((n_ion, n_fine))
        ti_fine = np.zeros((n_ion, n_fine))
        zi_fine = np.zeros((n_ion, n_fine))
        for k in range(n_ion):
            ni_fine[k] = np.maximum(profile_interpolator(rho_cp, ni_all[k])(rho_f), 0.0)
            ti_fine[k] = np.maximum(profile_interpolator(rho_cp, ti_all[k])(rho_f), 0.0)
            if z1d_tag in cpd:
                zi_fine[k] = profile_interpolator(rho_cp, np.atleast_2d(cpd[z1d_tag].to_numpy())[k])(rho_f)
            elif zion_tag in cpd:
                zi_fine[k] = float(np.atleast_1d(cpd[zion_tag].to_numpy()).flatten()[k])
            elif zn_tag in cpd:
                zi_fine[k] = float(cpd[zn_tag].to_numpy().reshape(n_ion, -1)[k, 0])

        # assemble output species list: (label, mass_amu, z, n, t), electrons appended last
        ntiny = 1.0e-12 * float(np.nanmax(ne))
        species: MutableSequence[tuple[str, NDArray, NDArray, NDArray, NDArray]] = []
        active = [k for k in range(n_ion) if np.nanmax(ni_fine[k]) > ntiny]
        if full_impurities:
            for k in active:
                species.append((ion_names[k], np.full(n_fine, mass_all[k]), zi_fine[k], ni_fine[k], ti_fine[k]))
        else:
            main_ions = [k for k in active if np.nanmean(zi_fine[k]) < 1.5]
            impurities = [k for k in active if k not in main_ions]
            for k in main_ions:
                species.append((ion_names[k], np.full(n_fine, mass_all[k]), zi_fine[k], ni_fine[k], ti_fine[k]))
            if impurities:
                # lumped impurity preserving quasineutrality and Zeff, with
                # charge-density-weighted mass and temperature
                s1 = np.sum([ni_fine[k] * zi_fine[k] for k in impurities], axis=0)
                s2 = np.sum([ni_fine[k] * zi_fine[k] ** 2 for k in impurities], axis=0)
                s1 = np.where(s1 > 0.0, s1, 1.0e-20)
                s2 = np.where(s2 > 0.0, s2, 1.0e-20)
                z_lump = s2 / s1
                n_lump = s1 ** 2 / s2
                m_lump = np.sum([ni_fine[k] * zi_fine[k] * mass_all[k] for k in impurities], axis=0) / s1
                t_lump = np.sum([ni_fine[k] * zi_fine[k] * ti_fine[k] for k in impurities], axis=0) / s1
                species.append(('LUMPED', m_lump, z_lump, n_lump, t_lump))

        ns = 0
        for label, mass_amu, zs, dens, temp in species:
            ns += 1
            dens_safe = np.maximum(dens, ntiny)
            temp_safe = np.maximum(temp, 1.0e-3 * float(np.nanmax(te)))
            out[f'MASS_{ns:d}'] = mass_amu * c['u'] / c['md']
            out[f'Z_{ns:d}'] = zs
            out[f'DENS_{ns:d}'] = dens / ne
            out[f'TEMP_{ns:d}'] = temp / te
            out[f'DLNNDR_{ns:d}'] = plasma_tools.calc_ak_from_grad_k(ddr(dens_safe), dens_safe, a)
            out[f'DLNTDR_{ns:d}'] = plasma_tools.calc_ak_from_grad_k(ddr(temp_safe), temp_safe, a)
            out[f'SDLNNDR_{ns:d}'] = np.zeros(n_fine)
            out[f'SDLNTDR_{ns:d}'] = np.zeros(n_fine)
            out[r'#' + f'N_{ns:d}'] = 1.0e-19 * dens
            out[r'#' + f'T_{ns:d}'] = 1.0e-3 * temp
        ns += 1
        out[f'MASS_{ns:d}'] = np.full(n_fine, c['me'] / c['md'])
        out[f'Z_{ns:d}'] = np.full(n_fine, -1.0)
        out[f'DENS_{ns:d}'] = np.ones(n_fine)
        out[f'TEMP_{ns:d}'] = np.ones(n_fine)
        out[f'DLNNDR_{ns:d}'] = plasma_tools.calc_ak_from_grad_k(ddr(ne), ne, a)
        out[f'DLNTDR_{ns:d}'] = plasma_tools.calc_ak_from_grad_k(ddr(te), te, a)
        out[f'SDLNNDR_{ns:d}'] = np.zeros(n_fine)
        out[f'SDLNTDR_{ns:d}'] = np.zeros(n_fine)
        out[r'#' + f'N_{ns:d}'] = 1.0e-19 * ne
        out[r'#' + f'T_{ns:d}'] = 1.0e-3 * te
        out['N_SPECIES'] = np.full(n_fine, float(ns))

        # --- Reference quantities and dimensionless local parameters ---
        cs = plasma_tools.calc_vref_from_te_and_aref(te, 2.0)
        gref = cs / a
        rhoref = plasma_tools.calc_rhoref_from_te_and_btot(te, b_unit, 2.0)
        out['BETAE_UNIT'] = plasma_tools.calc_beta_from_p(c['e'] * ne * te, b_unit)
        out['NU_EE'] = plasma_tools.calc_nueenorm_from_ne_and_te(ne, te, gref)
        out['LAMBDA_STAR'] = plasma_tools.calc_ldenorm_from_ne_te_and_rhoref(ne, te, rhoref)

        # --- Rotation (zeros when absent from core_profiles) ---
        omega_tag = 'core_profiles.profiles_1d.rotation_frequency_tor_sonic'
        if omega_tag in cpd:
            omega_interp = profile_interpolator(rho_cp, cpd[omega_tag].to_numpy().flatten())
            omega = omega_interp(rho_f)
            grad_omega = omega_interp.derivative()(rho_f) / drmin_drho
            out['MACH'] = plasma_tools.calc_mach_from_u(omega * rmaj, cs)
            out['GAMMA_P'] = -1.0 * rmaj * grad_omega / gref
            out['GAMMA_E'] = -1.0 * (rmin / q_f) * grad_omega / gref
        else:
            out['MACH'] = np.zeros(n_fine)
            out['GAMMA_P'] = np.zeros(n_fine)
            out['GAMMA_E'] = np.zeros(n_fine)

        # --- Dimensional back-conversion variables and attrs ---
        zeff_tag = 'core_profiles.profiles_1d.zeff'
        if zeff_tag in cpd:
            zeff = profile_interpolator(rho_cp, cpd[zeff_tag].to_numpy().flatten())(rho_f)
        else:
            zeff = np.sum(ni_fine * zi_fine ** 2, axis=0) / ne
        out[r'#ZEFF'] = zeff
        roa_safe = np.where(out['RMIN'] > 1.0e-5, out['RMIN'], 1.0e-5)
        out[r'#Q_PRIME'] = (q_f ** 2 / roa_safe ** 2) * out['S']
        ptot = c['e'] * (ne * te + np.sum(ni_fine * ti_fine, axis=0))
        grad_ptot = PchipInterpolator(rho_f, ptot).derivative()(rho_f) / drmin_drho
        out[r'#P_PRIME'] = q_f * (c['mu'] / (4.0 * np.pi)) * (a / roa_safe) * grad_ptot / (b_unit ** 2)
        b_zero_tag = 'equilibrium.vacuum_toroidal_field.b0'
        b_zero = float(np.abs(np.atleast_1d(eqd[b_zero_tag].to_numpy()).flatten()[0])) if b_zero_tag in eqd else np.nan
        out[r'#BUNIT_BY_BREF'] = b_unit / b_zero

        attrs: MutableMapping[str, NDArray] = {
            'b_unit': b_unit,
            'b_zero': np.full(n_fine, b_zero),
        }
        return out, attrs


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
    def from_plasma(
        cls,
        obj: io,
        side: str = 'output',
        window: Sequence[int | float] | None = None,
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            source_mapping = {
                'ohmic': 'ohmic',
                'neutral_beam': 'nbi',
                'ion_cyclotron': 'ic',
                'electron_cyclotron': 'ec',
                'synchrotron': 'synchrotron_radiation',
                'bremsstrahlung': 'bremsstrahlung',
                'line_radiation': 'line_radiation',
                'ionization': 'ionisation',
                'charge_exchange': 'charge_exchange',
                'bootstrap': 'bootstrap_current',
                'fusion': 'fusion',
            }
            data = obj.input if side == 'input' else obj.output
            dsvec = []
            attrs: MutableMapping[str, Any] = {}
            dd_version = newobj.default_version
            if isinstance(dd_version, str) and 'data_dictionary_version' not in attrs:
                attrs['data_dictionary_version'] = dd_version
            factory = imas.IDSFactory(version=dd_version)
            ion_field = 'name' if Version(dd_version) >= Version('4.0.0') else 'label'
            # Explicit conversion from the COCOS convention of the plasma state to that of the data dictionary
            imas_cocos = newobj.default_cocos_3 if Version(dd_version) < Version('4') else newobj.default_cocos_4
            plasma_cocos = getattr(obj, 'input_cocos' if side == 'input' else 'output_cocos', 1)
            cocos = define_cocos_converter(plasma_cocos, imas_cocos)
            psi_scale = np.power(2.0 * np.pi, cocos['eBp']) * cocos['sBp'] * cocos['scyl']  # plasma magnetic_flux is in Wb/radian
            phi_scale = 2.0 * np.pi * cocos['scyl']  # IMAS toroidal flux is always in Wb, never per radian
            s_tor = cocos['scyl']  # Toroidal components, field and current
            s_pol = cocos['spol'] * cocos['scyl']  # Poloidal components
            s_q = cocos['spol']
            mu0 = 4.0e-7 * np.pi
            idsmap = {}
            if 'time' in data and 'radius' in data:
                cp = factory.core_profiles()
                cs = factory.core_sources()
                eq = factory.equilibrium()
                sm = factory.summary()
                time_orig = data['time'].to_numpy()
                time_window = [time_orig[-1]]
                if window is not None and len(window) >= 2:
                    window_mask = (time_orig >= window[0]) & (time_orig <= window[-1])
                    if np.any(window_mask):
                        time_window = [t for t in time_orig[window_mask]]
                data = data.sel(time=time_window, method='nearest').drop_duplicates('time')  # Fine because data is a copy
                time = data['time'].to_numpy().astype(float)
                rho = data['radius'].to_numpy()
                ions = data['ion'].to_numpy() if 'ion' in data.coords else np.array([])
                sources = [s for s in data['source'].to_numpy() if s in source_mapping] if 'source' in data.coords else []

                def _get(key, i, **sel):
                    da = data[key].isel(time=i, drop=True)
                    if sel:
                        da = da.sel(sel, drop=True)
                    return da.to_numpy()

                def _set_ion_identity(ion, i, s):
                    setattr(ion, ion_field, str(s))
                    if 'atomic_number_i' in data or 'mass_i' in data:
                        ion.element.resize(1)
                        ion.element[0].atoms_n = 1
                        if 'atomic_number_i' in data:
                            ion.element[0].z_n = float(_get('atomic_number_i', i, ion=s).item())
                        if 'mass_i' in data:
                            ion.element[0].a = float(_get('mass_i', i, ion=s).item())

                r0 = data.attrs.get('rcentr', None)
                if r0 is not None:
                    r0 = float(np.mean(np.atleast_1d(r0)))  # Stored per time slice by from_gacode, scalar when read from file
                if r0 is None and 'r_geometric' in data:
                    r0 = float(data['r_geometric'].isel(radius=0).mean('time').to_numpy())
                b0 = s_tor * data['field_axis'].to_numpy() if 'field_axis' in data else None
                for ids_struct in [cp, cs, eq]:
                    ids_struct.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
                    ids_struct.time = time
                    if r0 is not None and b0 is not None:
                        ids_struct.vacuum_toroidal_field.r0 = float(r0)
                        ids_struct.vacuum_toroidal_field.b0 = b0

                # Fill core profiles IDS
                cp.profiles_1d.resize(len(time))
                for i, t in enumerate(time):
                    p1d = cp.profiles_1d[i]
                    p1d.time = t
                    p1d.grid.rho_tor_norm = rho
                    if 'magnetic_flux' in data:
                        psi = psi_scale * _get('magnetic_flux', i, direction='poloidal')
                        phi = phi_scale * _get('magnetic_flux', i, direction='toroidal')
                        p1d.grid.psi = psi
                        p1d.grid.psi_magnetic_axis = psi[0]
                        p1d.grid.psi_boundary = psi[-1]
                        p1d.grid.rho_pol_norm = np.sqrt(np.clip((psi - psi[0]) / (psi[-1] - psi[0]), 0.0, None))
                        if b0 is not None:
                            p1d.grid.rho_tor = np.sqrt(np.abs(phi) / (np.pi * np.abs(b0[i])))
                    if 'volume' in data:
                        p1d.grid.volume = _get('volume', i)
                    if 'cross_sectional_area' in data:
                        p1d.grid.area = _get('cross_sectional_area', i)
                    if 'surface_area' in data:
                        p1d.grid.surface = _get('surface_area', i)
                    if 'safety_factor' in data:
                        p1d.q = s_q * _get('safety_factor', i)
                    if 'magnetic_shear' in data:
                        p1d.magnetic_shear = _get('magnetic_shear', i)
                    if 'effective_charge' in data:
                        p1d.zeff = _get('effective_charge', i)
                    if 'pressure_thermal_total' in data:
                        p1d.pressure_thermal = _get('pressure_thermal_total', i)
                    if 'density_e' in data:
                        p1d.electrons.density_thermal = _get('density_e', i)
                        p1d.electrons.density = _get('density_e', i)
                    if 'temperature_e' in data:
                        p1d.electrons.temperature = _get('temperature_e', i)
                    if 'pressure_e' in data:
                        p1d.electrons.pressure_thermal = _get('pressure_e', i)
                    if len(ions) > 0:
                        p1d.ion.resize(len(ions))
                        ni_thermal = np.zeros_like(rho)
                        niti_thermal = np.zeros_like(rho)
                        for j, s in enumerate(ions):
                            ion = p1d.ion[j]
                            _set_ion_identity(ion, i, s)
                            thermal = ('type_i' not in data) or (str(_get('type_i', i, ion=s).item()) == 'thermal')
                            if 'density_i' in data:
                                ni = _get('density_i', i, ion=s)
                                ion.density = ni
                                # Both always filled, ragged arrays of structures are not round-trippable
                                ion.density_thermal = ni if thermal else np.zeros_like(ni)
                                ion.density_fast = np.zeros_like(ni) if thermal else ni
                                if thermal:
                                    ni_thermal += ni
                            if 'temperature_i' in data:
                                ti = _get('temperature_i', i, ion=s)
                                ion.temperature = ti
                                if thermal and 'density_i' in data:
                                    niti_thermal += _get('density_i', i, ion=s) * ti
                            if 'pressure_i' in data:
                                ion.pressure = _get('pressure_i', i, ion=s)
                            if 'velocity_i' in data:
                                ion.velocity.toroidal = s_tor * _get('velocity_i', i, ion=s, direction='toroidal')
                                ion.velocity.poloidal = s_pol * _get('velocity_i', i, ion=s, direction='poloidal')
                            if 'charge_i' in data:
                                ion.z_ion_1d = _get('charge_i', i, ion=s)
                                ion.z_ion = float(ion.z_ion_1d[0])
                        if np.any(ni_thermal > 0.0):
                            p1d.n_i_thermal_total = ni_thermal
                            p1d.t_i_average = np.where(ni_thermal > 0.0, niti_thermal / np.where(ni_thermal > 0.0, ni_thermal, 1.0), 0.0)
                    if 'rotation_frequency_sonic' in data:
                        p1d.rotation_frequency_tor_sonic = s_tor * _get('rotation_frequency_sonic', i)
                    if 'current_source' in data:
                        jsrc = s_tor * data['current_source'].isel(time=i, drop=True)
                        p1d.j_ohmic = jsrc.sel(source='ohmic', drop=True).to_numpy()
                        p1d.j_bootstrap = jsrc.sel(source='bootstrap', drop=True).to_numpy()
                        p1d.j_non_inductive = jsrc.sel(source=['neutral_beam', 'ion_cyclotron', 'electron_cyclotron', 'bootstrap']).sum('source').to_numpy()
                        p1d.j_total = jsrc.sum('source').to_numpy()
                if 'current' in data:
                    cp.global_quantities.ip = s_tor * data['current'].to_numpy()
                idsmap['core_profiles'] = cp

                # Fill core sources IDS
                entries = [(source_mapping[s], s) for s in sources]
                if 'heat_exchange_ei' in data:
                    entries.append(('collisional_equipartition', None))
                cs.source.resize(len(entries))
                for k, (imas_name, src) in enumerate(entries):
                    cs.source[k].identifier = imas.identifiers.core_source_identifier[imas_name]
                    cs.source[k].profiles_1d.resize(len(time))
                    for i, t in enumerate(time):
                        sp = cs.source[k].profiles_1d[i]
                        sp.time = t
                        sp.grid.rho_tor_norm = rho
                        if 'volume' in data:
                            sp.grid.volume = _get('volume', i)
                        if src is None:
                            # Same field layout as the other sources, ragged arrays of structures are not round-trippable
                            qei = _get('heat_exchange_ei', i)
                            zeros = np.zeros_like(rho)
                            sp.electrons.energy = -qei  # heat_exchange_ei > 0 means electrons heat ions, mirrors plasma_io.from_imas
                            sp.electrons.particles = zeros
                            sp.j_parallel = zeros
                            sp.momentum_phi = zeros
                            sp.total_ion_energy = qei  # Per-species split not available, only the total is stored
                            sp.ion.resize(len(ions))
                            for j, s in enumerate(ions):
                                _set_ion_identity(sp.ion[j], i, s)
                                sp.ion[j].energy = zeros
                                sp.ion[j].particles = zeros
                                sp.ion[j].momentum.toroidal = zeros
                            continue
                        if 'heat_source_e' in data:
                            sp.electrons.energy = _get('heat_source_e', i, source=src)
                        if 'particle_source_e' in data:
                            sp.electrons.particles = _get('particle_source_e', i, source=src)
                        if 'current_source' in data:
                            sp.j_parallel = s_tor * _get('current_source', i, source=src)
                        if len(ions) > 0 and any(key in data for key in ['heat_source_i', 'particle_source_i', 'momentum_source_i']):
                            sp.ion.resize(len(ions))
                            for j, s in enumerate(ions):
                                _set_ion_identity(sp.ion[j], i, s)
                                if 'heat_source_i' in data:
                                    sp.ion[j].energy = _get('heat_source_i', i, ion=s, source=src)
                                if 'particle_source_i' in data:
                                    sp.ion[j].particles = _get('particle_source_i', i, ion=s, source=src)
                                if 'momentum_source_i' in data:
                                    sp.ion[j].momentum.toroidal = s_tor * _get('momentum_source_i', i, ion=s, source=src, direction='toroidal')
                            if 'heat_source_i' in data:
                                sp.total_ion_energy = data['heat_source_i'].isel(time=i, drop=True).sel(source=src, drop=True).sum('ion').to_numpy()
                        if 'momentum_source_i' in data:
                            sp.momentum_phi = s_tor * data['momentum_source_i'].isel(time=i, drop=True).sel(source=src, direction='toroidal', drop=True).sum('ion').to_numpy()
                idsmap['core_sources'] = cs

                # Fill equilibrium IDS
                eq.time_slice.resize(len(time))
                for i, t in enumerate(time):
                    ts = eq.time_slice[i]
                    ts.time = t
                    ts.profiles_1d.rho_tor_norm = rho
                    if 'magnetic_flux' in data:
                        psi = psi_scale * _get('magnetic_flux', i, direction='poloidal')
                        phi = phi_scale * _get('magnetic_flux', i, direction='toroidal')
                        ts.profiles_1d.psi = psi
                        ts.profiles_1d.psi_norm = (psi - psi[0]) / (psi[-1] - psi[0])
                        ts.profiles_1d.phi = phi
                        ts.global_quantities.psi_axis = psi[0]
                        ts.global_quantities.psi_boundary = psi[-1]
                        if b0 is not None:
                            ts.profiles_1d.rho_tor = np.sqrt(np.abs(phi) / (np.pi * np.abs(b0[i])))
                    if 'current' in data:
                        ts.global_quantities.ip = s_tor * float(_get('current', i))
                    if 'volume' in data:
                        ts.profiles_1d.volume = _get('volume', i)
                    if 'cross_sectional_area' in data:
                        ts.profiles_1d.area = _get('cross_sectional_area', i)
                    if 'surface_area' in data:
                        ts.profiles_1d.surface = _get('surface_area', i)
                    if 'safety_factor' in data:
                        ts.profiles_1d.q = s_q * _get('safety_factor', i)
                    if 'magnetic_shear' in data:
                        ts.profiles_1d.magnetic_shear = _get('magnetic_shear', i)
                    # Per-surface shape, standard IMAS definitions
                    if 'r_geometric' in data and 'r_minor' in data:
                        rgeo = _get('r_geometric', i)
                        rmin = _get('r_minor', i)
                        ts.profiles_1d.geometric_axis.r = rgeo
                        ts.profiles_1d.r_inboard = rgeo - rmin
                        ts.profiles_1d.r_outboard = rgeo + rmin
                        if 'z_geometric' in data:
                            ts.profiles_1d.geometric_axis.z = _get('z_geometric', i)
                    if 'mxh_kappa' in data:
                        ts.profiles_1d.elongation = _get('mxh_kappa', i)
                    if 'contour' in data and 'r_geometric' in data and 'r_minor' in data:
                        rc = _get('contour', i, grid='r')
                        zc = _get('contour', i, grid='z')
                        iu = np.argmax(zc, axis=-1)
                        il = np.argmin(zc, axis=-1)
                        r_upper = np.take_along_axis(rc, iu[:, np.newaxis], axis=-1)[:, 0]
                        r_lower = np.take_along_axis(rc, il[:, np.newaxis], axis=-1)[:, 0]
                        rmin_safe = np.where(rmin > 0.0, rmin, 1.0)
                        ts.profiles_1d.triangularity_upper = np.where(rmin > 0.0, (rgeo - r_upper) / rmin_safe, 0.0)
                        ts.profiles_1d.triangularity_lower = np.where(rmin > 0.0, (rgeo - r_lower) / rmin_safe, 0.0)
                        # Full flux-surface contours on an inverse (rho_tor_norm, poloidal index) grid
                        theta = np.arctan2(zc - zc[0, 0], rc - rc[0, 0])  # Polar angle about the magnetic axis (axis surface is a point)
                        theta = np.unwrap(theta, axis=-1)
                        theta[0, :] = np.linspace(0.0, 2.0 * np.pi, theta.shape[-1])
                        ts.profiles_2d.resize(1)
                        p2d = ts.profiles_2d[0]
                        p2d.grid_type = imas.identifiers.poloidal_plane_coordinates_identifier.inverse_rhotornorm_polar
                        p2d.grid.dim1 = rho
                        p2d.grid.dim2 = np.linspace(0.0, 2.0 * np.pi, rc.shape[-1])
                        p2d.r = rc
                        p2d.z = zc
                        p2d.theta = theta
                        if 'magnetic_flux' in data:
                            p2d.psi = np.repeat(psi[:, np.newaxis], rc.shape[-1], axis=-1)
                        ts.boundary.outline.r = rc[-1]
                        ts.boundary.outline.z = zc[-1]
                        ts.boundary.geometric_axis.r = float(rgeo[-1])
                        ts.boundary.minor_radius = float(rmin[-1])
                        ts.boundary.triangularity_upper = float(ts.profiles_1d.triangularity_upper[-1])
                        ts.boundary.triangularity_lower = float(ts.profiles_1d.triangularity_lower[-1])
                        ts.boundary.triangularity = 0.5 * (ts.boundary.triangularity_upper + ts.boundary.triangularity_lower)
                        if 'z_geometric' in data:
                            ts.boundary.geometric_axis.z = float(_get('z_geometric', i)[-1])
                        if 'mxh_kappa' in data:
                            ts.boundary.elongation = float(_get('mxh_kappa', i)[-1])
                        ts.global_quantities.magnetic_axis.r = float(rc[0, 0])
                        ts.global_quantities.magnetic_axis.z = float(zc[0, 0])
                    elif 'mxh_delta' in data:
                        ts.profiles_1d.triangularity_upper = _get('mxh_delta', i)
                        ts.profiles_1d.triangularity_lower = _get('mxh_delta', i)
                idsmap['equilibrium'] = eq

                # Fill summary IDS
                sm.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
                sm.time = time
                if 'current' in data:
                    sm.global_quantities.ip.value = s_tor * data['current'].to_numpy()
                    if 'pressure_total_volume_average' in data and 'volume' in data and r0 is not None:
                        volume_lcfs = data['volume'].isel(radius=-1).to_numpy()
                        sm.global_quantities.beta_pol.value = 4.0 * data['pressure_total_volume_average'].to_numpy() * volume_lcfs / (mu0 * float(r0) * data['current'].to_numpy() ** 2)
                idsmap['summary'] = sm
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
            newobj.input = ds
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

            # GACODE polflux is ψ/(2π) in Wb/radian; IMAS psi is in Wb.
            # Keep polflux_gacode in GACODE units for internal geometry
            # computations (fsa_grad_psi* are in these units).
            # polflux_imas = polflux_gacode * 2π for IMAS output fields.
            polflux_gacode = d['polflux'].to_numpy().flatten()
            polflux_imas = polflux_gacode * 2.0 * np.pi
            q = d['q'].to_numpy().flatten()
            rmin = d['rmin'].to_numpy().flatten()
            rmaj = d['rmaj'].to_numpy().flatten()
            zmag = d['zmag'].to_numpy().flatten()
            kappa = d['kappa'].to_numpy().flatten()
            delta = d['delta'].to_numpy().flatten()
            nrho = polflux_gacode.size

            # Use IMAS-scaled polflux for phi calculation:
            # q = dΦ/dψ_Wb, so ∫ q·dψ_Wb gives Φ in Wb.
            dpsi_imas = np.gradient(polflux_imas, rmin)
            phi = np.zeros(nrho)
            phi[1:] = cumulative_simpson(y=q * dpsi_imas, x=rmin)[: nrho - 1] if nrho > 2 else np.cumsum(q[1:] * np.diff(polflux_imas))
            phi = np.abs(phi)  # type: ignore[assignment]

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

            # dpsi_drho_tor in IMAS units (Wb) for IMAS output.
            dpsi_drho_tor_imas = np.gradient(polflux_imas, rho_tor)
            # dpsi_drho_tor in GACODE units (Wb/rad) for gm metric conversion
            # (consistent with fsa_grad_psi* from GACODE geometry).
            dpsi_drho_tor_gacode = np.gradient(polflux_gacode, rho_tor)
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
            dpsi_drmin_imas = np.gradient(polflux_imas, rmin)
            mask = np.abs(dpsi_drmin_imas) > 1.0e-30
            dvolume_dpsi[mask] = volp_miller[mask] / dpsi_drmin_imas[mask]
            dvolume_dpsi = np.abs(dvolume_dpsi)  # type: ignore[assignment]

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
                fsa_B2 = bt2_corrected + bp2_from_miller  # type: ignore[assignment]
                B_sq_norm = np.where(fsa_B2_miller > 1e-30, fsa_B2 / fsa_B2_miller, 1.0)
                fsa_1_over_B2 = np.where(B_sq_norm > 1e-30, fsa_1_over_B2_miller / B_sq_norm, fsa_1_over_B2_miller)  # type: ignore[assignment]
            else:
                fsa_B2 = F ** 2 * fsa_1_over_R2  # type: ignore[assignment]
                fsa_1_over_B2 = np.where(  # type: ignore[assignment]
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
            # Use GACODE-unit dpsi_drho_tor for gm metrics, since
            # fsa_grad_psi* from GACODE are in Wb/rad units.
            dpsi_drho_tor_gac_safe = np.where(
                np.abs(dpsi_drho_tor_gacode) > 1.0e-30,
                dpsi_drho_tor_gacode, 1.0,
            )
            if 'fsa_grad_psi2_over_R2' in d:
                gm2 = d['fsa_grad_psi2_over_R2'].to_numpy().flatten() / dpsi_drho_tor_gac_safe ** 2
            else:
                gm2 = fsa_gradr2_over_R2 * drho_tor_drmin_safe ** 2
            if 'fsa_grad_psi2' in d:
                gm3 = d['fsa_grad_psi2'].to_numpy().flatten() / dpsi_drho_tor_gac_safe ** 2
            else:
                gm3 = fsa_gradr2 * drho_tor_drmin_safe ** 2
            gm4 = fsa_1_over_B2
            gm5 = fsa_B2
            if 'fsa_grad_psi' in d:
                gm7 = d['fsa_grad_psi'].to_numpy().flatten() / np.abs(dpsi_drho_tor_gac_safe)
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
                    Ip_enc = ip_A * rho_tor_norm ** 2  # type: ignore[assignment]
                rho_tor_a = rho_tor[-1] if rho_tor[-1] > 0.0 else 1.0
                drho_norm_drmin = drho_tor_drmin / rho_tor_a
                drho_norm_drmin_safe = np.where(np.abs(drho_norm_drmin) > 1e-30, drho_norm_drmin, 1.0)
                vpr = volp_miller / drho_norm_drmin_safe
                spr = vpr * fsa_1_over_R / (2.0 * np.pi)
                dIp_drhon = np.gradient(Ip_enc, rho_tor_norm)
                mask_s = np.abs(spr) > 1.0e-30
                j_phi[mask_s] = dIp_drhon[mask_s] / spr[mask_s]

            ds_vars: dict[str, Any] = {}
            ds_coords: dict[str, Any] = {}

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

            # Write IMAS-unit polflux (Wb) to psi and dpsi_drho_tor.
            ds_vars[f'{p1d}.psi'] = ([ts_i, rho_dim], np.expand_dims(polflux_imas, axis=0))
            ds_vars[f'{p1d}.phi'] = ([ts_i, rho_dim], np.expand_dims(phi, axis=0))
            ds_vars[f'{p1d}.rho_tor_norm'] = ([ts_i, rho_dim], np.expand_dims(rho_tor_norm, axis=0))
            ds_vars[f'{p1d}.rho_tor'] = ([ts_i, rho_dim], np.expand_dims(rho_tor, axis=0))
            ds_vars[f'{p1d}.f'] = ([ts_i, rho_dim], np.expand_dims(F, axis=0))
            ds_vars[f'{p1d}.r_inboard'] = ([ts_i, rho_dim], np.expand_dims(r_in, axis=0))
            ds_vars[f'{p1d}.r_outboard'] = ([ts_i, rho_dim], np.expand_dims(r_out, axis=0))
            ds_vars[f'{p1d}.q'] = ([ts_i, rho_dim], np.expand_dims(q, axis=0))
            ds_vars[f'{p1d}.dpsi_drho_tor'] = ([ts_i, rho_dim], np.expand_dims(dpsi_drho_tor_imas, axis=0))
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
            ds_vars[f'{gq}.psi_axis'] = ([ts_i], np.array([polflux_imas[0]]))
            ds_vars[f'{gq}.psi_boundary'] = ([ts_i], np.array([polflux_imas[-1]]))

            ds_vars[f'{bdry}.minor_radius'] = ([ts_i], np.array([rmin[-1]]))
            ds_vars[f'{bdry}.type'] = ([ts_i], np.array([0]))

            newobj.input = xr.Dataset(data_vars=ds_vars, coords=ds_coords)

        return newobj

