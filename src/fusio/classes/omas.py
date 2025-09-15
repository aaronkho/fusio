import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

from packaging.version import Version
import copy
import json
import omas  # type: ignore[import-untyped]
from omas.omas_core import ODS  # type: ignore[import-untyped]
from .io import io
from ..utils.eqdsk_tools import (
    convert_cocos,
    write_eqdsk,
)

logger = logging.getLogger('fusio')


class omas_io(io):


    default_cocos_3: Final[int] = 11
    default_cocos_4: Final[int] = 17

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
    time_equivalent_dimensions: Final[Mapping[str, str]] = {
        'core_profiles.time': 'core_profiles.profiles_1d',
        'core_sources.time': 'core_sources.source.profiles_1d',
        'core_transport.time': 'core_transport.model.profiles_1d',
        'equilibrium.time': 'equilibrium.time_slice',
    }


    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
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


    def read(
        self,
        path: str | Path,
        side: str = 'output',
    ) -> None:
        if side == 'input':
            self.input = self._read_omas_json_file(path)
        else:
            self.output = self._read_omas_json_file(path)


    def write(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False
    ) -> None:
        if side == 'input':
            self._write_omas_json_file(path, self.input, overwrite=overwrite)
        else:
            self._write_omas_json_file(path, self.output, overwrite=overwrite)


    def _convert_to_imas_like_dataset(
        self,
        data: ODS,
    ) -> xr.Dataset:

        def _apply_special_coordinate_rules(
            segments: Sequence[str],
            dimensions: MutableSequence[str],
        ) -> MutableSequence[str]:
            # Uses field segments to manually append data array dimensions added after discovery
            newdims = copy.deepcopy(dimensions)
            if segments[0] == 'core_profiles' and 'profiles_1d' in segments:
                newdims.append('core_profiles.profiles_1d.grid.rho_tor_norm:i')
            elif segments[0] == 'core_profiles' and 'global_quantities' in segments:
                newdims.append('core_profiles.profiles_1d:i')
            elif segments[0] == 'core_sources' and 'profiles_1d' in segments:
                newdims.append('core_sources.source.profiles_1d.grid.rho_tor_norm:i')
            elif segments[0] == 'core_sources' and 'global_quantities' in segments:
                newdims.append('core_sources.source.profiles_1d:i')
            elif segments[0] == 'core_transport' and 'profiles_1d' in segments and ('flux' in segments or 'grid_flux' in segments):
                newdims.append('core_transport.model.profiles_1d.grid_flux.rho_tor_norm:i')
            elif segments[0] == 'core_transport' and 'profiles_1d' in segments and ('d' in segments or 'grid_d' in segments):
                newdims.append('core_transport.model.profiles_1d.grid_d.rho_tor_norm:i')
            elif segments[0] == 'core_transport' and 'profiles_1d' in segments and ('v' in segments or 'grid_v' in segments):
                newdims.append('core_transport.model.profiles_1d.grid_v.rho_tor_norm:i')
            elif segments[0] == 'equilibrium' and 'vacuum_toroidal_field' in segments and 'b0' in segments:
                newdims.append('equilibrium.time_slice:i')
            elif segments[0] == 'equilibrium' and 'profiles_1d' in segments:
                newdims.append('equilibrium.time_slice.profiles_1d.psi:i')
            elif segments[0] == 'equilibrium' and 'profiles_2d' in segments and 'grid' not in segments:
                newdims.extend(['equilibrium.time_slice.profiles_2d.grid.dim2:i', 'equilibrium.time_slice.profiles_2d.grid.dim1:i'])
            elif segments[0] == 'equilibrium' and 'boundary' in segments and 'outline' in segments:
                newdims.append('equilibrium.time_slice.boundary.outline.r:i')
            elif segments[0] == 'pulse_schedule' and 'reference' in segments:
                dims = list(segments[:-1])
                dims.append('time')
                newdims.append('.'.join(dims))
            elif segments[0] == 'summary' and 'global_quantities' in segments:
                newdims.append('summary.time')
            elif segments[0] == 'wall' and 'description_2d' in segments and 'outline' in segments:
                newdims.append('wall.description_2d.limiter.unit.outline.r:i')
            return newdims

        def _recursive_array_structure_search(
            ods: ODS,
        ) -> Sequence[MutableMapping[str, Any]]:
            struct_names: MutableMapping[str, Any] = {}
            field_names: MutableMapping[str, Any] = {}
            for key, val in ods.items():
                if isinstance(val, ODS):
                    if 0 in val:
                        struct_names[f'{key}'] = len(val)
                    next_struct_names, next_field_names = _recursive_array_structure_search(val)
                    if isinstance(key, int):
                        add_struct_names = {}
                        # This construction allows setting dimension sizes to maximum value found
                        for name, length in next_struct_names.items():
                            if f'{name}' not in struct_names:
                                add_struct_names[f'{name}'] = length
                            elif struct_names[f'{name}'] is None and length is not None:
                                add_struct_names[f'{name}'] = length
                            elif struct_names[f'{name}'] is not None and length is not None:
                                if isinstance(length, int) and length > struct_names[f'{name}']:
                                    add_struct_names[f'{name}'] = length
                                elif isinstance(length, list) and np.all([length[i] > struct_names[f'{name}'][i] for i in range(len(length))]):
                                    add_struct_names[f'{name}'] = length
                        struct_names.update(add_struct_names)
                        add_field_names = {}
                        # This construction allows setting dimension sizes to maximum value found
                        for name, length in next_field_names.items():
                            if f'{name}' not in field_names:
                                add_field_names[f'{name}'] = length
                            elif field_names[f'{name}'] is None and length is not None:
                                add_field_names[f'{name}'] = length
                            elif field_names[f'{name}'] is not None and length is not None:
                                if isinstance(length, int) and length > field_names[f'{name}']:
                                    add_field_names[f'{name}'] = length
                                elif isinstance(length, list) and np.all([length[i] > field_names[f'{name}'][i] for i in range(len(length))]):
                                    add_field_names[f'{name}'] = length
                        field_names.update(add_field_names)
                    else:
                        add_struct_names = {}
                        # This construction allows setting dimension sizes to maximum value found
                        for name, length in next_struct_names.items():
                            if f'{key}.{name}' not in struct_names:
                                add_struct_names[f'{key}.{name}'] = length
                            elif struct_names[f'{key}.{name}'] is None and length is not None:
                                add_struct_names[f'{key}.{name}'] = length
                            elif struct_names[f'{key}.{name}'] is not None and length is not None:
                                if isinstance(length, int) and length > struct_names[f'{key}.{name}']:
                                    add_struct_names[f'{key}.{name}'] = length
                                elif isinstance(length, list) and np.all([length[i] > struct_names[f'{key}.{name}'][i] for i in range(len(length))]):
                                    add_struct_names[f'{key}.{name}'] = length
                        struct_names.update(add_struct_names)
                        add_field_names = {}
                        # This construction allows setting dimension sizes to maximum value found
                        for name, length in next_field_names.items():
                            if f'{key}.{name}' not in field_names:
                                add_field_names[f'{key}.{name}'] = length
                            elif field_names[f'{key}.{name}'] is None and length is not None:
                                add_field_names[f'{key}.{name}'] = length
                            elif field_names[f'{key}.{name}'] is not None and length is not None:
                                if isinstance(length, int) and length > field_names[f'{key}.{name}']:
                                    add_field_names[f'{key}.{name}'] = length
                                elif isinstance(length, list) and np.all([length[i] > field_names[f'{key}.{name}'][i] for i in range(len(length))]):
                                    add_field_names[f'{key}.{name}'] = length
                        field_names.update(add_field_names)
                elif isinstance(val, np.ndarray) and val.size > 0:
                    field_names[f'{key}'] = [i for i in val.shape]
                else:
                    field_names[f'{key}'] = None
            return struct_names, field_names

        def _recursive_array_field_stack(
            ods: ODS,
            field: str,
        ) -> NDArray:
            out = np.array([])
            components = field.split('.')
            if len(components) > 0:
                if f'{components[0]}' in ods:
                    if len(components) > 1 and '0' in ods[f'{components[0]}']:
                        dvec = []
                        ndim = 0
                        for ii in range(len(ods[f'{components[0]}'])):
                            dvec.append(_recursive_array_field_stack(ods[f'{components[0]}'][ii], '.'.join(components[1:])))
                            if ndim < dvec[-1].ndim:
                                ndim = dvec[-1].ndim
                        max_shape = [0] * ndim
                        for ii in range(len(dvec)):
                            while dvec[ii].ndim < ndim:
                                dvec[ii] = np.expand_dims(dvec[ii], axis=-1)
                            for jj in range(len(max_shape)):
                                if max_shape[jj] < dvec[ii].shape[jj]:
                                    max_shape[jj] = dvec[ii].shape[jj]
                        for ii in range(len(dvec)):
                            if dvec[ii].shape != tuple(max_shape):
                                shape_pad = [max_shape[jj] - dvec[ii].shape[jj] for jj in range(len(max_shape))]
                                if len(shape_pad) > 0:
                                    dvec[ii] = np.pad(dvec[ii], [(0, pad) for pad in shape_pad], mode='constant', constant_values=np.nan)
                        out = np.stack(dvec, axis=0)
                    elif len(components) > 1:
                        out = _recursive_array_field_stack(ods[f'{components[0]}'], '.'.join(components[1:]))
                    else:
                        out = ods[f'{components[0]}'] if isinstance(ods[f'{components[0]}'], np.ndarray) else np.array(ods[f'{components[0]}'])
            return out

        coords = {}
        data_vars = {}
        attrs: MutableMapping[str, Any] = {}

        aos_sizes, data_sizes = _recursive_array_structure_search(data)
        for key, size in aos_sizes.items():
            coords[f'{key}:i'] = np.arange(size).astype(int)
        for ctag in self.last_index_fields:
            if ctag in data_sizes and isinstance(data_sizes[ctag], (list, tuple)):
                coords[f'{ctag}:i'] = np.arange(data_sizes[ctag][0]).astype(int)

        for key, size in data_sizes.items():
            dims: MutableSequence[str] = []
            components = key.split('.')
            long_key = ''
            for ii in range(len(components)):
                long_key = '.'.join([comp for i, comp in enumerate(components) if i <= ii])
                if f'{long_key}:i' in coords:
                    dims.append(f'{long_key}:i')
            if size is not None and key not in self.last_index_fields:
                dims = _apply_special_coordinate_rules(components, dims)
            val = _recursive_array_field_stack(data, key)
            if val.dtype.type is np.str_ and len(val.shape) > len(dims) and val.shape[-1] == 1:
                val = np.squeeze(val, axis=-1)
            if f'{key}' in self.time_equivalent_dimensions and self.time_equivalent_dimensions[f'{key}'] in aos_sizes:
                long_key = self.time_equivalent_dimensions[f'{key}']
                dims.append(f'{long_key}:i')
                data_vars[f'{key}'] = (dims, val)
            elif len(components) == 2 and components[-1] == 'time':
                coords[f'{key}'] = val
            elif components[0] == 'pulse_schedule' and components[-1] == 'time':
                coords[f'{key}'] = val
            else:
                data_vars[f'{key}'] = (dims, val)
                if components[-1] == 'data_dictionary' and components[-2] == 'version_put' and 'data_dictionary_version' not in attrs:
                    attrs['data_dictionary_version'] = val

        # Does this belong here, or is it better in imas conversion function?
        tag = 'core_sources.time'
        if tag not in coords:
            if 'core_sources.source.profiles_1d.time' in data_vars:
                data_vars[f'{tag}'] = (
                    ['core_sources.source.profiles_1d:i'],
                    np.nanmean(data_vars['core_sources.source.profiles_1d.time'][1], axis=0).flatten()
                )
            elif 'core_sources.source.global_quantities.time' in data_vars:
                data_vars[f'{tag}'] = (
                    ['core_sources.source.global_quantities:i'],
                    np.nanmean(data_vars['core_sources.source.global_quantities.time'][1], axis=0).flatten()
                )

        if hasattr(data, 'cocos'):
            attrs['cocos'] = data.cocos
        if 'data_dictionary_version' not in attrs and hasattr(data, 'imas_version'):
            attrs['data_dictionary_version'] = data.imas_version

        return xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)


    def _read_omas_json_file(
        self,
        path: str | Path,
    ) -> xr.Dataset:

        dsvec = []
        if isinstance(path, (str, Path)):
            ipath = Path(path)
            if ipath.exists():
                data = omas.load_omas_json(str(ipath.resolve()))
                ds = self._convert_to_imas_like_dataset(data)
                dsvec.append(ds)

        ds = xr.Dataset()
        for dss in dsvec:
            ds = ds.assign_coords(dss.coords).assign(dss.data_vars).assign_attrs(**dss.attrs)

        return ds


    def _write_omas_json_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        window: ArrayLike | None = None,
        overwrite: bool = False
    ) -> None:
        pass


    @property
    def input_cocos(
        self,
    ) -> int:
        cocos = self.input.attrs.get('cocos', None)
        if cocos is None:
            version = self.input.attrs.get('data_dictionary_version', '4.0.0')
            cocos = self.default_cocos_3 if Version(version) < Version('4') else self.default_cocos_4
        return cocos


    @property
    def output_cocos(
        self,
    ) -> int:
        cocos = self.input.attrs.get('cocos', None)
        if cocos is None:
            version = self.output.attrs.get('data_dictionary_version', '4.0.0')
            cocos = self.default_cocos_3 if Version(version) < Version('4') else self.default_cocos_4
        return cocos


    def to_eqdsk(
        self,
        time_index: int = -1,
        side: str = 'output',
        cocos: int | None = None,
        transpose: bool = False,
    ) -> MutableMapping[str, Any]:
        eqdata: MutableMapping[str, Any] = {}
        time_eq = 'equilibrium.time_slice:i'
        data = (
            self.input.isel({time_eq: time_index})
            if side == 'input' else
            self.output.isel({time_eq: time_index})
        )
        default_cocos = self.input_cocos if side == 'input' else self.output_cocos
        if cocos is None:
            cocos = default_cocos
        tag = 'equilibrium.time_slice.profiles_2d.grid_type.name'
        if tag in data:
            rectangular_index = [i for i, name in enumerate(data[tag]) if name == 'rectangular']
        if len(rectangular_index) > 0:
            data = data.isel({'equilibrium.time_slice.profiles_2d:i': rectangular_index[0]})
            psi_eq_i = 'equilibrium.time_slice.profiles_1d.psi:i'
            psin_eq = 'equilibrium.time_slice.profiles_1d.psi_norm'
            psinvec = None
            if psin_eq in data:
                data = data.swap_dims({psi_eq_i: psin_eq}).drop_duplicates(psin_eq)
                #psinvec = data[psin_eq].to_numpy().flatten()
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
                #eqdata['fpol'] = data.drop_duplicates('psin_eq')[tag].interp(psin_eq=psinvec).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.pressure'
            if tag in data:
                if conversion is None:
                    eqdata['pres'] = data.drop_duplicates(psin_eq)[tag].interp({psin_eq: psinvec}).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['pres'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
                #eqdata['pres'] = data.drop_duplicates('psin_eq')[tag].interp(psin_eq=psinvec).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.f_df_dpsi'
            if tag in data:
                if conversion is None:
                    eqdata['ffprime'] = data.drop_duplicates(psin_eq)[tag].interp({psin_eq: psinvec}).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['ffprime'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
                #eqdata['ffprime'] = data.drop_duplicates('psin_eq')[tag].interp(psin_eq=psinvec).to_numpy().flatten()
            tag = 'equilibrium.time_slice.profiles_1d.dpressure_dpsi'
            if tag in data:
                if conversion is None:
                    eqdata['pprime'] = data.drop_duplicates(psin_eq)[tag].interp({psin_eq: psinvec}).to_numpy().flatten()
                else:
                    ndata = xr.Dataset(coords={'psin_interp': conversion}, data_vars={tag: (['psin_interp'], data[tag].to_numpy().flatten())})
                    eqdata['pprime'] = ndata.drop_duplicates('psin_interp')[tag].interp(psin_interp=psinvec, kwargs=ikwargs).to_numpy().flatten()
                #eqdata['pprime'] = data.drop_duplicates('psin_eq')[tag].interp(psin_eq=psinvec).to_numpy().flatten()
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
                #eqdata['qpsi'] = data.drop_duplicates('psin_eq')[tag].interp(psin_eq=psinvec).to_numpy().flatten()
            rtag = 'equilibrium.time_slice.boundary.outline.r'
            ztag = 'equilibrium.time_slice.boundary.outline.z'
            if rtag in data and ztag in data:
                rdata = data[rtag].dropna(f'{rtag}:i').to_numpy().flatten()
                zdata = data[ztag].dropna(f'{rtag}:i').to_numpy().flatten()
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
    def from_omas(
        cls,
        obj: io,
        side: str = 'output',
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            newobj.input = obj.input if side == 'input' else obj.output
        return newobj
