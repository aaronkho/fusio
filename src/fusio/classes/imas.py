import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

import h5py  # type: ignore[import-untyped]
import imas
from imas.ids_base import IDSBase
from imas.ids_structure import IDSStructure
from imas.ids_struct_array import IDSStructArray
from .io import io
from ..utils.eqdsk_tools import write_eqdsk

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

    empty_int: Final[int] = -999999999
    empty_float: Final[float] = -9.0e40
    #empty_complex: Final[complex] = -9.0e40-9.0e40j  # Removed since complex type cannot be JSON serialized
    int_types: Final[Sequence[Any]] = (int, np.int8, np.int16, np.int32, np.int64)
    float_types: Final[Sequence[Any]] = (float, np.float16, np.float32, np.float64, np.float128)
    #complex_types: Final[Sequence[Any]] = (complex, np.complex64, np.complex128, np.complex256)


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
            self._write_imas_file(path, self.input.to_dataset(), overwrite=overwrite)
        else:
            self._write_imas_file(path, self.output.to_dataset(), overwrite=overwrite)


    def _read_imas_directory(
        self,
        path: str | Path,
    ) -> xr.Dataset:
        if self.has_imas:
            return self._read_imas_directory_from_netcdf(path)
        else:
            return self._read_imas_directory_from_hdf5_without_core(path)


    def _read_imas_directory_from_netcdf(
        self,
        path: str | Path,
        version: str | None = None,
    ) -> xr.Dataset:

        dsvec = []

        if isinstance(path, (str, Path)):
            ipath = Path(path)
            if ipath.is_dir():
                for ids in self.ids_top_levels:
                    top_level_path = ipath / f'{ids}.nc'
                    if top_level_path.is_file():
                        ds_ids = xr.load_dataset(top_level_path, group=f'{ids}/0')
                        unique_names = list(set(
                            [k for k in ds_ids.dims] +
                            [k for k in ds_ids.coords] +
                            [k for k in ds_ids.data_vars] +
                            [k for k in ds_ids.attrs]
                        ))
                        dsvec.append(ds_ids.rename({k: f'{ids}.{k}' for k in unique_names}))

        ds = xr.Dataset()
        for dss in dsvec:
            ds = ds.assign_coords(dss.coords).assign(dss.data_vars).assign_attrs(**dss.attrs)

        return ds


    def _read_imas_directory_from_hdf5_without_core(
        self,
        path: str | Path,
        version: str | None = None,
    ) -> xr.Dataset:

        def _recursive_resize_struct_array(
            ids: IDSBase,
            tag: str,
            size: list[Any],
        ) -> None:
            components = tag.split('&')
            if len(components) > 0:
                if isinstance(ids, IDSStructArray) and len(components) > 1:
                    for ii in range(ids.size):
                        if isinstance(size, np.ndarray) and ii < size.shape[0]:
                            _recursive_resize_struct_array(ids[ii], '&'.join(components), size[ii])
                elif isinstance(ids, IDSStructArray) and components[0] == 'AOS_SHAPE':
                    ids.resize(size[0])
                else:
                    _recursive_resize_struct_array(ids[f'{components[0]}'], '&'.join(components[1:]), size)

        def _expanded_data_insertion(
            ids: IDSBase,
            tag: str,
            data: Any,
        ) -> None:
            components = tag.split('&')
            if len(components) > 0:
                if isinstance(ids, IDSStructArray):
                    for ii in range(ids.size):
                        if isinstance(data, np.ndarray) and ii < data.shape[0]:
                            _expanded_data_insertion(ids[ii], '&'.join(components), data[ii])
                        elif not isinstance(data, np.ndarray):
                            _expanded_data_insertion(ids[ii], '&'.join(components), data)
                elif len(components) == 1:
                    val = data if not isinstance(data, bytes) else data.decode('utf-8')
                    if isinstance(val, np.ndarray):
                        if val.dtype in self.int_types:
                            val = np.where(val == self.empty_int, np.nan, val)
                        if val.dtype in self.float_types:
                            val = np.where(val == self.empty_float, np.nan, val)
                        #if val.dtype in self.complex_types:
                        #    val = np.where(val == self.empty_complex, np.nan, val)
                    ids[f'{components[0]}'] = val
                else:
                    _expanded_data_insertion(ids[f'{components[0]}'], '&'.join(components[1:]), data)

        dsvec = []

        if isinstance(path, (str, Path)):
            data: MutableMapping[str, Any] = {}
            ipath = Path(path)
            if ipath.is_dir():

                for ids in self.ids_top_levels:
                    top_level_path = ipath / f'{ids}.h5'
                    if top_level_path.is_file():
                        idsdata = h5py.File(top_level_path)
                        if f'{ids}' in idsdata:
                            data = {k: v[()] for k, v in idsdata[f'{ids}'].items()}
                            dd_version = None
                            if 'ids_properties&version_put&data_dictionary' in data:
                                dd_version = data['ids_properties&version_put&data_dictionary'].decode('utf-8')
                            ids_struct = getattr(imas.IDSFactory(version=dd_version), f'{ids}')()
                            shape_data = {}
                            for key in list(data.keys()):
                                if key.endswith('&AOS_SHAPE'):
                                    shape_data[key] = data.pop(key)
                                elif key.endswith('_SHAPE'):
                                    data.pop(key)
                            for key in sorted(shape_data.keys(), key=len):
                                _recursive_resize_struct_array(ids_struct, key.replace('[]', ''), shape_data[key])
                            for key in data:
                                _expanded_data_insertion(ids_struct, key.replace('[]', ''), data[key])
                            if ids_struct.has_value:
                                ids_struct.validate()
                                ds_ids = imas.util.to_xarray(ids_struct)
                                unique_names = list(set(
                                    [k for k in ds_ids.dims] +
                                    [k for k in ds_ids.coords] +
                                    [k for k in ds_ids.data_vars] +
                                    [k for k in ds_ids.attrs]
                                ))
                                dsvec.append(ds_ids.rename({k: f'{ids}.{k}' for k in unique_names}))

        ds = xr.Dataset()
        for dss in dsvec:
            ds = ds.assign_coords(dss.coords).assign(dss.data_vars).assign_attrs(**dss.attrs)

        return ds


    def _write_imas_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        window: ArrayLike | None = None,
        overwrite: bool = False,
    ) -> None:

        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            wdata = data.sel(time=-1)
            opath = Path(path)
            logger.info(f'Saved {self.format} data into {opath.resolve()}')
            #else:
            #    logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    def to_eqdsk(
        self,
        time_index: int = -1,
        side: str = 'output',
        transpose: bool = False,
    ) -> MutableMapping[str, Any]:
        eqdata: MutableMapping[str, Any] = {}
        data = (
            self.input.to_dataset().isel({'equilibrium.time': time_index})
            if side == 'input' else
            self.output.to_dataset().isel({'equilibrium.time': time_index})
        )
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
                    eqdata['fpol'] = data.drop_duplicates(psin_eq)[tag].interp(psin_eq=psinvec).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['fpol'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.pressure'
            if tag in data:
                if conversion is None:
                    eqdata['pres'] = data.drop_duplicates(psin_eq)[tag].interp(psin_eq=psinvec).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['pres'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.f_df_dpsi'
            if tag in data:
                if conversion is None:
                    eqdata['ffprime'] = data.drop_duplicates(psin_eq)[tag].interp(psin_eq=psinvec).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['ffprime'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.dpressure_dpsi'
            if tag in data:
                if conversion is None:
                    eqdata['pprime'] = data.drop_duplicates(psin_eq)[tag].interp(psin_eq=psinvec).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['pprime'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_2d.psi'
            if tag in data:
                eqdata['psi'] = data[tag].to_numpy() if transpose else data[tag].to_numpy().T
            tag = 'equilibrium.time_slice.profiles_1d.q'
            if tag in data:
                if conversion is None:
                    eqdata['qpsi'] = data.drop_duplicates(psin_eq)[tag].interp(psin_eq=psinvec).to_numpy().flatten()
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
        return eqdata


    def generate_eqdsk_file(
        self,
        path: str | Path,
        time_index: int = -1,
        side: str = 'output',
        transpose: bool = False,
    ) -> None:
        eqpath = None
        if isinstance(path, (str, Path)):
            eqpath = Path(path)
        assert isinstance(eqpath, Path)
        eqdata = self.to_eqdsk(time_index=time_index, side=side, transpose=transpose)
        write_eqdsk(eqdata, eqpath)
        logger.info('Successfully generated g-eqdsk file, {path}')


    def generate_all_eqdsk_files(
        self,
        basepath: str | Path,
        side: str = 'output',
        transpose: bool = False,
    ) -> None:
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
                self.generate_eqdsk_file(eqpath, time_index=ii, side=side, transpose=transpose)


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
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            newobj.input = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
        return newobj


    # Assumed that the self creation method transfers output to input
    @classmethod
    def from_gacode(
        cls,
        obj: io,
        side: str = 'output',
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
        return newobj

