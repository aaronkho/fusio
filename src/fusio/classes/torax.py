import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

import copy
import json
from .io import io
from ..utils.plasma_tools import define_ion_species
from ..utils.eqdsk_tools import (
    define_cocos_converter,
)

logger = logging.getLogger('fusio')


class torax_io(io):

    basevars: Final[Mapping[str, Sequence[str]]] = {
        'plasma_composition': [
            'main_ion',
            'impurity',
            'Z_eff',
            'Z_i_override',
            'A_i_override',
            'Z_impurity_override',
            'A_impurity_override',
        ],
        'profile_conditions': [
            'Ip',
            'use_v_loop_lcfs_boundary_condition',
            'v_loop_lcfs',
            'T_i',
            'T_i_right_bc',
            'T_e',
            'T_e_right_bc',
            'psi',
            'n_e',
            'normalize_n_e_to_nbar',
            'nbar',
            'n_e_nbar_is_fGW',
            'n_e_right_bc',
            'n_e_right_bc_is_fGW',
            'set_pedestal',
            'current_profile_nu',
            'initial_j_is_total_current',
            'initial_psi_from_j',
        ],
        'numerics': [
            't_initial',
            't_final',
            'exact_t_final',
            'evolve_ion_heat',
            'evolve_electron_heat',
            'evolve_current',
            'evolve_density',
            'resistivity_multiplier',
            'max_dt',
            'min_dt',
            'chi_timestep_prefactor',
            'fixed_dt',
            'dt_reduction_factor',
            'adaptive_T_source_prefactor',
            'adaptive_n_source_prefactor',
        ],
        'geometry': [
            'geometry_type',
            'n_rho',
            'hires_factor',
        ],
        'pedestal': [
            'model_name',
            'set_pedestal',
        ],
        'transport': [
            'model_name',
            'chi_min',
            'chi_max',
            'D_e_min',
            'D_e_max',
            'V_e_min',
            'V_e_max',
            'apply_inner_patch',
            'D_e_inner',
            'V_e_inner',
            'chi_i_inner',
            'chi_e_inner',
            'rho_inner',
            'apply_outer_patch',
            'D_e_outer',
            'V_e_outer',
            'chi_i_outer',
            'chi_e_outer',
            'rho_outer',
            'smoothing_width',
            'smooth_everywhere',
        ],
        'sources': [
        ],
        'mhd': [
        ],
        'neoclassical': [
        ],
        'solver': [
            'solver_type',
            'theta_implicit',
            'use_predictor_corrector',
            'n_corrector_steps',
            'use_pereverzev',
            'chi_pereverzev',
            'D_pereverzev',
        ],
        'time_step_calculator': [
            'calculator_type',
            'tolerance',
        ],
    }
    restartvars: Final[Sequence[str]] = [
        'filename',
        'time',
        'do_restart',
        'stitch',
    ]
    specvars: Final[Mapping[str, Any]] = {
        'geometry': {
            'circular': [
                'R_major',
                'a_minor',
                'B_0',
                'elongation_LCFS',
            ],
            'chease': [
                'geometry_file',
                'geometry_directory',
                'Ip_from_parameters',
                'R_major',
                'a_minor',
                'B_0',
            ],
            'fbt': [
                'geometry_file',
                'geometry_directory',
                'Ip_from_parameters',
                'LY_object',
                'LY_bundle_object',
                'LY_to_torax_times',
                'L_object',
            ],
            'eqdsk': [
                'geometry_file',
                'geometry_directory',
                'Ip_from_parameters',
                'n_surfaces',
                'last_surface_factor',
            ],
        },
        'pedestal': {
            'set_T_ped_n_ped': [
                'n_e_ped',
                'n_e_ped_is_fGW',
                'T_i_ped',
                'T_e_ped',
                'rho_norm_ped_top',
            ],
            'set_P_ped_n_ped': [
                'P_ped',
                'n_e_ped',
                'n_e_ped_is_fGW',
                'T_i_T_e_ratio',
                'rho_norm_ped_top',
            ],
        },
        'neoclassical': {
            'bootstrap_current': [
                'model_name',
                'bootstrap_multiplier',
            ],
            'conductivity': [
                'model_name',
            ],
        },
        'mhd': {
            'sawtooth': [
                'model_name',
                'crash_step_duration',
                's_critical',
                'minimum_radius',
                'flattening_factor',
                'mixing_radius_multiplier',
            ],
        },
        'sources': {
            'generic_heat': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'gaussian_location',
                'gaussian_width',
                'P_total',
                'electron_heat_fraction',
                'absorption_fraction',
            ],
            'generic_particle': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'deposition_location',
                'particle_width',
                'S_total',
            ],
            'generic_current': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'gaussian_location',
                'gaussian_width',
                'I_generic',
                'fraction_of_total_current',
                'use_absolute_current',
            ],
            'ei_exchange': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'Qei_multiplier',
            ],
            'ohmic': [
                'prescribed_values',
                'mode',
                'is_explicit',
            ],
            'fusion': [
                'prescribed_values',
                'mode',
                'is_explicit',
            ],
            'gas_puff': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'puff_decay_length',
                'S_total',
            ],
            'pellet': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'pellet_deposition_location',
                'pellet_width',
                'S_total',
            ],
            'bremsstrahlung': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'use_relativistic_correction',
            ],
            'impurity_radiation': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'model_name',
            ],
            'cyclotron_radiation': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'wall_reflection_coeff',
                'beta_min',
                'beta_max',
                'beta_grid_size',
            ],
            'ecrh': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'extra_prescribed_power_density',
                'gaussian_location',
                'gaussian_width',
                'P_total',
                'current_drive_efficiency',
            ],
            'icrh': [
                'prescribed_values',
                'mode',
                'is_explicit',
                'model_path',
                'wall_inner',
                'wall_outer',
                'frequency',
                'minority_concentration',
                'P_total',
            ],
        },
        'transport': {
            'constant': [
                'chi_i',
                'chi_e',
                'D_e',
                'V_e',
            ],
            'CGM': [
                'alpha',
                'chi_stiff',
                'chi_e_i_ratio',
                'chi_D_ratio',
                'VR_D_ratio',
            ],
            'Bohm-GyroBohm': [
                'chi_e_bohm_coeff',
                'chi_e_gyrobohm_coeff',
                'chi_i_bohm_coeff',
                'chi_i_gyrobohm_coeff',
                'chi_e_bohm_multiplier',
                'chi_e_gyrobohm_multiplier',
                'chi_i_bohm_multiplier',
                'chi_i_gyrobohm_multiplier',
                'D_face_c1',
                'D_face_c2',
                'V_face_coeff',
            ],
            'qlknn': [
                'model_path',
                'qlknn_model_name',
                'include_ITG',
                'include_TEM',
                'include_ETG',
                'ITG_flux_ratio_correction',
                'ETG_correction_factor',
                'clip_inputs',
                'clip_margin',
                'collisionality_multiplier',
                'DV_effective',
                'An_min',
                'avoid_big_negative_s',
                'smag_alpha_correction',
                'q_sawtooth_proxy',
            ],
            'qualikiz': [
                'n_max_runs',
                'n_processes',
                'collisionality_multiplier',
                'DV_effective',
                'An_min',
                'avoid_big_negative_s',
                'smag_alpha_correction',
                'q_sawtooth_proxy',
            ],
        },
        'solver': {
            'linear': [
            ],
            'newton_raphson': [
                'log_iterations',
                'initial_guess_mode',
                'residual_tol',
                'residual_coarse_tol',
                'n_max_iterations',
                'delta_reduction_factor',
                'tau_min',
            ],
            'optimizer': [
                'initial_guess_mode',
                'loss_tol',
                'n_max_iterations',
            ],
        },
    }
    allowed_radiation_species: Final[Sequence[str]] = [
        'H',
        'D',
        'T',
        'He3',
        'He4',
        'Li',
        'Be',
        'C',
        'N',
        'O',
        'N',
        'O',
        'Ne',
        'Ar',
        'Kr',
        'Xe',
        'W',
    ]


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


    def _unflatten(
        self,
        datadict: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        odict: MutableMapping[str, Any] = {}
        udict: MutableMapping[str, Any] = {}
        for key in datadict:
            klist = key.split('.')
            if len(klist) > 1:
                nkey = '.'.join(klist[1:])
                if klist[0] not in udict:
                    udict[klist[0]] = []
                udict[klist[0]].append(nkey)
            else:
                odict[klist[0]] = datadict[f'{key}']
        if udict:
            for key in udict:
                gdict = {}
                for lkey in udict[key]:
                    gdict[lkey] = datadict[f'{key}.{lkey}']
                odict[key] = self._unflatten(gdict)
        else:
            odict = datadict
        return odict


    def _clean(
        self,
    ) -> None:
        data = self.input
        use_psi = data.attrs.get('use_psi', True)
        if not use_psi:
            self.reset_psi()
        use_generic_heat = data.attrs.get('use_generic_heat', True)
        if not use_generic_heat:
            self.reset_generic_heat_source()
        use_generic_particle = data.attrs.get('use_generic_particle', True)
        if not use_generic_particle:
            self.reset_generic_particle_source()
        use_generic_current = data.attrs.get('use_generic_current', True)
        if not use_generic_current:
            self.reset_generic_current_source()
        use_fusion = data.attrs.get('use_fusion', True)
        if not use_fusion:
            self.reset_fusion_source()


    def read(
        self,
        path: str | Path,
        side: str = 'output',
    ) -> None:
        if side == 'input':
            self.input = self._read_torax_file(path)
        else:
            self.output = self._read_torax_file(path)
        #logger.warning(f'{self.format} reading function not defined yet...')


    def write(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False,
    ) -> None:
        if side == 'input':
            self._write_torax_file(path, self.input, overwrite=overwrite)
        else:
            self._write_torax_file(path, self.output, overwrite=overwrite)


    def _read_torax_file(
        self,
        path: str | Path,
    ) -> xr.Dataset:
        ds = xr.Dataset()
        if isinstance(path, (str, Path)):
            ipath = Path(path)
            if ipath.exists():
                dt = xr.open_datatree(ipath)
                ds_temp = xr.combine_by_coords([dt[key].to_dataset() for key in dt.groups], compat="override")
                ds = ds_temp if isinstance(ds_temp, xr.Dataset) else ds_temp.to_dataset()
                newattrs: MutableMapping[str, Any] = {}
                for attr in ds.attrs:
                    if isinstance(ds.attrs[attr], str):
                        if ds.attrs[attr].startswith('dict'):
                            newattrs[attr] = json.loads(ds.attrs[attr][4:])
                        if ds.attrs[attr] == 'true':
                            newattrs[attr] = True
                        if ds.attrs[attr] == 'false':
                            newattrs[attr] = False
                        if attr == 'config':
                            newattrs[attr] = json.loads(ds.attrs[attr])
                ds.attrs.update(newattrs)
        return ds


    def _write_torax_file(
        self,
        path: str | Path,
        data: xr.Dataset,
        overwrite: bool = False,
    ) -> None:
        if isinstance(path, (str, Path)):
            opath = Path(path)
            if overwrite or not opath.exists():
                if isinstance(data, (xr.Dataset, xr.DataTree)):
                    newattrs: MutableMapping[str, Any] = {}
                    for attr in data.attrs:
                        if isinstance(data.attrs[attr], dict):
                            newattrs[attr] = 'dict' + json.dumps(data.attrs[attr])
                        if isinstance(data.attrs[attr], bool):
                            newattrs[attr] = str(data.attrs[attr])
                    data.attrs.update(newattrs)
                    data.to_netcdf(opath)
                    logger.info(f'Saved {self.format} data into {opath.resolve()}')
            else:
                logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    def time_coordinate(
        self,
        time: float | ArrayLike = 0.0,
    ) -> xr.DataArray:
        if 'time' not in self.input.coords:
            newcoords: MutableMapping[str, Any] = {}
            newcoords['time'] = np.array([time]).flatten()
            self.update_input_coords(newcoords)
        return self.input.get('time', xr.DataArray())


    def radial_coordinate(
        self,
        rho: float | ArrayLike = [0.0, 1.0],
    ) -> xr.DataArray:
        if 'rho' not in self.input.coords:
            newcoords: MutableMapping[str, Any] = {}
            newcoords['rho'] = np.array([rho]).flatten()
            self.update_input_coords(newcoords)
            if 'geometry.n_rho' not in self.input.attrs:
                newattrs: MutableMapping[str, Any] = {}
                newattrs['geometry.n_rho'] = len(newcoords['rho'])
                self.update_input_attrs(newattrs)
        return self.input.get('rho', xr.DataArray())


    def set_flat_initial_and_constant_kinetic_boundary_condition(
        self,
        tebc: float,
        tibc: float,
        nebc: float,
    ) -> None:
        time = self.time_coordinate().to_numpy().flatten()
        rho = self.radial_coordinate().to_numpy().flatten()
        tref = 1.0e3
        newvars: MutableMapping[str, Any] = {}
        newattrs: MutableMapping[str, Any] = {}
        newvars['profile_conditions.T_e'] = (['time', 'rho'], np.full((len(time), len(rho)), tebc / tref))
        newvars['profile_conditions.T_i'] = (['time', 'rho'], np.full((len(time), len(rho)), tibc / tref))
        newvars['profile_conditions.n_e'] = (['time', 'rho'], np.full((len(time), len(rho)), nebc))
        newattrs['profile_conditions.normalize_n_e_to_nbar'] = False
        newattrs['profile_conditions.n_e_nbar_is_fGW'] = False
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def set_cosine_initial_and_constant_kinetic_boundary_condition(
        self,
        tebc: float,
        tibc: float,
        nebc: float,
        tecore: float,
        ticore: float,
        necore: float,
    ) -> None:
        time = self.time_coordinate().to_numpy().flatten()
        rho = self.radial_coordinate().to_numpy().flatten()
        tref = 1.0e3
        newvars: MutableMapping[str, Any] = {}
        newattrs: MutableMapping[str, Any] = {}
        te = (tecore - tebc) * np.cos(0.5 * np.pi * rho) + tebc
        ti = (ticore - tibc) * np.cos(0.5 * np.pi * rho) + tibc
        ne = (necore - nebc) * np.cos(0.5 * np.pi * rho) + nebc
        newvars['profile_conditions.T_e'] = (['time', 'rho'], np.repeat(np.expand_dims(te / tref, axis=0), len(time), axis=0))
        newvars['profile_conditions.T_i'] = (['time', 'rho'], np.repeat(np.expand_dims(ti / tref, axis=0), len(time), axis=0))
        newvars['profile_conditions.n_e'] = (['time', 'rho'], np.repeat(np.expand_dims(ne, axis=0), len(time), axis=0))
        newattrs['profile_conditions.normalize_n_e_to_nbar'] = False
        newattrs['profile_conditions.n_e_nbar_is_fGW'] = False
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def set_constant_flat_main_ion_composition(
        self,
        ions: MutableMapping[str, float],
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        rho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        if 'main_ion' in data.coords:
            dropvars = [var for var in data.data_vars if 'main_ion' in data[var].dims and var != 'main_ion']
            self.delete_input_data_vars(dropvars)
            self.delete_input_data_vars(['main_ion'])
        newcoords: MutableMapping[str, Any] = {'main_ion': [key for key in ions]}
        self.update_input_coords(newcoords)
        newvars: MutableMapping[str, Any] = {}
        total = np.sum([ions[key] for key in ions])
        composition = [np.expand_dims(np.full((len(time), len(rho)), float(ions[key] / total)), axis=0) for key in ions]
        newvars['plasma_composition.main_ion'] = (['main_ion', 'time', 'rho'], np.concatenate(composition, axis=0))
        self.update_input_data_vars(newvars)


    def add_hydrogenic_minority_species(
        self,
        sname: str,
        sfrac: float,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        rho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        if sname in ['H', 'D', 'T'] and sname not in data.get('main_ion', xr.DataArray()):
            coords: MutableMapping[str, Any] = {'main_ion': np.array([sname]), 'time': time, 'rho': rho}
            data_vars: MutableMapping[str, Any] = {}
            if 'plasma_composition.main_ion' in data:
                total = np.atleast_1d(data['plasma_composition.main_ion'].sum('main_ion').to_numpy())
                data_vars['plasma_composition.main_ion'] = (['main_ion', 'time', 'rho'], np.expand_dims(sfrac / (total - sfrac), axis=0))
            newdata = xr.Dataset(coords=coords, data_vars=data_vars)
            self.input = xr.concat([data, newdata], dim='main_ion', data_vars='minimal', coords='different', join='outer')
            if 'plasma_composition.main_ion' in self.input:
                val = self.input['plasma_composition.main_ion']
                newvars: MutableMapping[str, Any] = {}
                newvars['plasma_composition.main_ion'] = (['main_ion', 'time', 'rho'], (val / val.sum('main_ion')).to_numpy())
                self.update_input_data_vars(newvars)


    def set_constant_flat_effective_charge(
        self,
        zeff: float,
    ) -> None:
        data = self.input
        shape = (data.get('time', xr.DataArray()).size, data.get('rho', xr.DataArray()).size)
        newvars: MutableMapping[str, Any] = {}
        newvars['plasma_composition.Z_eff'] = (['time', 'rho'], np.full(shape, float(zeff)))
        self.update_input_data_vars(newvars)


    def set_constant_flat_impurity_composition(
        self,
        impurities: MutableMapping[str, float],
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        rho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        if 'impurity' in data.coords:
            dropvars = [var for var in data.data_vars if 'impurity' in data[var].dims and var != 'impurity']
            self.delete_input_data_vars(dropvars)
            self.delete_input_data_vars(['impurity'])
        newcoords: MutableMapping[str, Any] = {'impurity': [key for key in impurities]}
        self.update_input_coords(newcoords)
        newvars: MutableMapping[str, Any] = {}
        total = np.sum([impurities[key] for key in impurities])
        composition = [np.expand_dims(np.full((len(time), len(rho)), float(impurities[key] / total)), axis=0) for key in impurities]
        newvars['plasma_composition.impurity.species'] = (['impurity', 'time', 'rho'], np.concatenate(composition, axis=0))
        self.update_input_data_vars(newvars)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['plasma_composition.impurity.impurity_mode'] = 'fractions'
        self.update_input_attrs(newattrs)


    def add_geometry(
        self,
        geotype: str,
        geofiles: str | Mapping[str, str],
        geodir: str | None = None,
    ) -> None:
        data = self.input
        newattrs: MutableMapping[str, Any] = {}
        newattrs['use_psi'] = False
        #newattrs['geometry.hires_factor'] = 4
        newattrs['geometry.Ip_from_parameters'] = bool(data.attrs.get('profile_conditions.Ip_tot', False))
        newattrs['geometry.geometry_type'] = f'{geotype}'
        if geodir is not None:
            newattrs['geometry.geometry_directory'] = f'{geodir}'
        if isinstance(geofiles, dict):
            geoconfig = {}
            for time, geofile in geofiles.items():
                geotime: MutableMapping[str, Any] = {}
                geotime['geometry_file'] = f'{geofile}'
                if geotype == 'eqdsk':
                    geotime['n_surfaces'] = 251
                    geotime['last_surface_factor'] = 0.9999
                geoconfig[time] = geotime
            newattrs['geometry.geometry_configs'] = geoconfig
        else:
            newattrs['geometry.geometry_file'] = f'{geofiles}'
            if geotype == 'eqdsk':
                newattrs['geometry.n_surfaces'] = 251
                newattrs['geometry.last_surface_factor'] = 0.9999
        self.update_input_attrs(newattrs)


    def reset_psi(
        self,
    ) -> None:
        delvars = [
            'profile_conditions.psi',
        ]
        self.delete_input_data_vars(delvars)


    def set_defined_current_psi(
        self,
        nu: float = 2.0,
        use_total: bool = False
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['use_psi'] = False
        newattrs['profile_conditions.initial_psi_from_j'] = True
        newattrs['profile_conditions.current_profile_nu'] = float(nu)
        newattrs['profile_conditions.initial_j_is_total_current'] = use_total
        self.update_input_attrs(newattrs)


    def add_pedestal_by_pressure(
        self,
        pped: float,
        nped: float,
        tpedratio: float,
        wrho: float,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newvars: MutableMapping[str, Any] = {}
        newvars['pedestal.P_ped'] = (['time'], np.zeros_like(time) + pped)
        newvars['pedestal.n_e_ped'] = (['time'], np.zeros_like(time) + nped)
        newvars['pedestal.T_i_T_e_ratio'] = (['time'], np.zeros_like(time) + tpedratio)
        newvars['pedestal.rho_norm_ped_top'] = (['time'], np.zeros_like(time) + wrho)
        self.update_input_data_vars(newvars)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['pedestal.set_pedestal'] = True
        newattrs['pedestal.model_name'] = 'set_P_ped_n_ped'
        newattrs['pedestal.n_e_ped_is_fGW'] = False
        newattrs['transport.smooth_everywhere'] = False
        newattrs['numerics.adaptive_T_source_prefactor'] = 1.0e10
        newattrs['numerics.adaptive_n_source_prefactor'] = 1.0e8
        self.update_input_attrs(newattrs)


    def add_pedestal_by_temperature(
        self,
        nped: float,
        tped: float,
        wrho: float,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        tref = 1.0e3
        newvars: MutableMapping[str, Any] = {}
        newvars['pedestal.n_e_ped'] = (['time'], np.zeros_like(time) + nped)
        newvars['pedestal.T_e_ped'] = (['time'], np.zeros_like(time) + (tped / tref))
        newvars['pedestal.T_i_ped'] = (['time'], np.zeros_like(time) + (tped / tref))
        newvars['pedestal.rho_norm_ped_top'] = (['time'], np.zeros_like(time) + wrho)
        self.update_input_data_vars(newvars)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['pedestal.set_pedestal'] = True
        newattrs['pedestal.model_name'] = 'set_T_ped_n_ped'
        newattrs['pedestal.n_e_ped_is_fGW'] = False
        newattrs['transport.smooth_everywhere'] = False
        newattrs['numerics.adaptive_T_source_prefactor'] = 1.0e10
        newattrs['numerics.adaptive_n_source_prefactor'] = 1.0e8
        self.update_input_attrs(newattrs)


    def add_pedestal_exponential_transport(
        self,
        chiscale: float,
        chidecay: float,
        dscale: float,
        ddecay: float,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newcoords: MutableMapping[str, Any] = {}
        newvars: MutableMapping[str, Any] = {}
        newattrs: MutableMapping[str, Any] = {}
        wrho_array = self.input.get('pedestal.rho_norm_ped_top', None)
        if self.input.attrs.get('transport.model_name', '') == 'combined' and wrho_array is not None:
            models = self.input.attrs.get('map_combined_models', [])
            prefix = f'transport.transport_models.{len(models):d}'
            #newvars[f'{prefix}.rho_min'] = (['time'], wrho_array.to_numpy())
            wrho = float(wrho_array.mean().to_numpy())
            xrho = np.linspace(wrho, 1.0, 25)
            factor = np.abs((xrho - wrho) / (1.0 - wrho))
            chirho = chiscale * np.exp(-factor / chidecay)
            drho = dscale * np.exp(-factor / ddecay)
            vrho = np.zeros_like(factor)
            newcoords['rho_ped_exp'] = xrho.flatten()
            newvars[f'{prefix}.chi_i'] = (['time', 'rho_ped_exp'], np.repeat(np.expand_dims(chirho, axis=0), len(time), axis=0))
            newvars[f'{prefix}.chi_e'] = (['time', 'rho_ped_exp'], np.repeat(np.expand_dims(chirho, axis=0), len(time), axis=0))
            newvars[f'{prefix}.D_e'] = (['time', 'rho_ped_exp'], np.repeat(np.expand_dims(drho, axis=0), len(time), axis=0))
            newvars[f'{prefix}.V_e'] = (['time', 'rho_ped_exp'], np.repeat(np.expand_dims(vrho, axis=0), len(time), axis=0))
            newattrs[f'{prefix}.model_name'] = 'constant'
            newattrs[f'{prefix}.rho_min'] = float(np.mean(wrho_array.to_numpy()))
            models.append('constant')
            newattrs['map_combined_models'] = models
        self.update_input_coords(newcoords)
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def add_internal_boundary(
        self,
        rho: float,
    ) -> None:
        data = self.input
        ne_bc = data['profile_conditions.n_e'].interp(rho=rho).to_numpy().flatten()
        te_bc = data['profile_conditions.T_e'].interp(rho=rho).to_numpy().flatten()
        ti_bc = data['profile_conditions.T_i'].interp(rho=rho).to_numpy().flatten()
        newvars: MutableMapping[str, Any] = {}
        newvars['pedestal.n_e_ped'] = (['time'], ne_bc)
        newvars['pedestal.T_e_ped'] = (['time'], te_bc)
        newvars['pedestal.T_i_ped'] = (['time'], ti_bc)
        newvars['pedestal.rho_norm_ped_top'] = (['time'], np.zeros_like(ne_bc) + rho)
        self.update_input_data_vars(newvars)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['pedestal.set_pedestal'] = True
        newattrs['pedestal.model_name'] = 'set_T_ped_n_ped'
        newattrs['pedestal.n_e_ped_is_fGW'] = False
        newattrs['transport.smooth_everywhere'] = False
        newattrs['numerics.adaptive_T_source_prefactor'] = 1.0e10
        newattrs['numerics.adaptive_n_source_prefactor'] = 1.0e8
        self.update_input_attrs(newattrs)


    def add_neoclassical_transport(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['neoclassical.conductivity.model_name'] = 'sauter'
        newattrs['neoclassical.transport.model_name'] = 'angioni_sauter'
        newattrs['neoclassical.transport.chi_min'] = 0.0
        newattrs['neoclassical.transport.chi_max'] = 100.0
        newattrs['neoclassical.transport.D_e_min'] = 0.0
        newattrs['neoclassical.transport.D_e_max'] = 100.0
        newattrs['neoclassical.transport.V_e_min'] = -50.0
        newattrs['neoclassical.transport.V_e_max'] = 50.0
        self.update_input_attrs(newattrs)
        self.add_neoclassical_bootstrap_current()


    def add_neoclassical_bootstrap_current(
        self,
    ) -> None:
        data = self.input
        newattrs: MutableMapping[str, Any] = {}
        newattrs['neoclassical.bootstrap_current.model_name'] = 'sauter'
        newattrs['neoclassical.bootstrap_current.bootstrap_multiplier'] = 1.0
        self.update_input_attrs(newattrs)
        if 'sources.generic_current.prescribed_values' in data and 'profile_conditions.j_bootstrap' in data:
            self.input['sources.generic_current.prescribed_values'] = data['sources.generic_current.prescribed_values'] - data['profile_conditions.j_bootstrap']


    def add_combined_transport(
        self,
    ) -> None:
        data = self.input
        newattrs: MutableMapping[str, Any] = {}
        newattrs['transport.model_name'] = 'combined'
        newattrs['transport.chi_min'] = 0.05
        newattrs['transport.chi_max'] = 100.0
        newattrs['transport.D_e_min'] = 0.05
        newattrs['transport.D_e_max'] = 100.0
        newattrs['transport.V_e_min'] = -50.0
        newattrs['transport.V_e_max'] = 50.0
        newattrs['transport.smoothing_width'] = 0.1
        newattrs['transport.smooth_everywhere'] = (not data.attrs.get('pedestal.set_pedestal', False))
        newattrs['map_combined_models'] = data.attrs.get('map_combined_models', [])
        self.update_input_attrs(newattrs)


    def add_constant_transport(
        self,
        chi_i: float = 0.0,
        chi_e: float = 0.0,
        D_e: float = 0.0,
        V_e: float = 0.0,
        rho_min: float | None = None,
        rho_max: float | None = None,
        n_rho: int | None = None,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newcoords: MutableMapping[str, Any] = {}
        newvars: MutableMapping[str, Any] = {}
        newattrs: MutableMapping[str, Any] = {}
        prefix = 'transport'
        if data.attrs.get('transport.model_name', '') == 'combined':
            models = data.attrs.get('map_combined_models', [])
            prefix = f'transport.transport_models.{len(models):d}'
            if 'pedestal.rho_norm_ped_top' in data:
                #newvars[f'{prefix}.rho_max'] = (['time'], data['pedestal.rho_norm_ped_top'].to_numpy())
                newattrs[f'{prefix}.rho_max'] = float(np.mean(data['pedestal.rho_norm_ped_top'].to_numpy()))
            newattrs[f'{prefix}.apply_inner_patch'] = False
            newattrs[f'{prefix}.apply_outer_patch'] = False
            models.append('constant')
            newattrs['map_combined_models'] = models
        if rho_min is not None:
            #newvars[f'{prefix}.rho_min'] = (['time'], np.zeros_like(time) + rho_min)
            newattrs[f'{prefix}.rho_min'] = float(rho_min)
        if rho_max is not None:
            #newvars[f'{prefix}.rho_max'] = (['time'], np.zeros_like(time) + rho_max)
            newattrs[f'{prefix}.rho_max'] = float(rho_max)
        newattrs[f'{prefix}.model_name'] = 'constant'
        nrho = n_rho if isinstance(n_rho, int) else 2
        xrho = np.linspace(rho_min if rho_min is not None else 0.0, rho_max if rho_max is not None else 1.0, nrho)
        #factor = np.abs((xrho - np.nanmin(xrho)) / (1.0 - np.nanmin(xrho)))
        #chirho = 0.01 * np.exp(-factor / 0.2)
        #drho = 0.01 * np.exp(-factor / 0.2)
        #vrho = np.zeros_like(factor)
        chiirho = np.zeros_like(xrho) + chi_i
        chierho = np.zeros_like(xrho) + chi_e
        derho = np.zeros_like(xrho) + D_e
        verho = np.zeros_like(xrho) + V_e
        newcoords['rho_const'] = xrho.flatten()
        newvars[f'{prefix}.chi_i'] = (['time', 'rho_const'], np.repeat(np.expand_dims(chiirho, axis=0), len(time), axis=0))
        newvars[f'{prefix}.chi_e'] = (['time', 'rho_const'], np.repeat(np.expand_dims(chierho, axis=0), len(time), axis=0))
        newvars[f'{prefix}.D_e'] = (['time', 'rho_const'], np.repeat(np.expand_dims(derho, axis=0), len(time), axis=0))
        newvars[f'{prefix}.V_e'] = (['time', 'rho_const'], np.repeat(np.expand_dims(verho, axis=0), len(time), axis=0))
        newattrs['transport.chi_min'] = 0.05
        newattrs['transport.chi_max'] = 100.0
        newattrs['transport.D_e_min'] = 0.05
        newattrs['transport.D_e_max'] = 100.0
        newattrs['transport.V_e_min'] = -50.0
        newattrs['transport.V_e_max'] = 50.0
        newattrs['transport.smoothing_width'] = 0.1
        newattrs['transport.smooth_everywhere'] = (not data.attrs.get('pedestal.set_pedestal', False))
        self.update_input_coords(newcoords)
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def add_critical_gradient_transport(
        self,
        alpha: float = 2.0,
        chi_grad: float = 2.0,
        ei_ratio: float = 2.0,
        D_ratio: float = 5.0,
        peaking: float = 0.0,
        rho_min: float | None = None,
        rho_max: float | None = None,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newcoords: MutableMapping[str, Any] = {}
        newvars: MutableMapping[str, Any] = {}
        newattrs: MutableMapping[str, Any] = {}
        prefix = 'transport'
        if data.attrs.get('transport.model_name', '') == 'combined':
            models = data.attrs.get('map_combined_models', [])
            prefix = f'transport.transport_models.{len(models):d}'
            if 'pedestal.rho_norm_ped_top' in data:
                #newvars[f'{prefix}.rho_max'] = (['time'], data['pedestal.rho_norm_ped_top'].to_numpy())
                newattrs[f'{prefix}.rho_max'] = float(np.mean(data['pedestal.rho_norm_ped_top'].to_numpy()))
            newattrs[f'{prefix}.apply_inner_patch'] = False
            newattrs[f'{prefix}.apply_outer_patch'] = False
            models.append('CGM')
            newattrs['map_combined_models'] = models
        if rho_min is not None:
            #newvars[f'{prefix}.rho_min'] = (['time'], np.zeros_like(time) + rho_min)
            newattrs[f'{prefix}.rho_min'] = float(rho_min)
        if rho_max is not None:
            #newvars[f'{prefix}.rho_max'] = (['time'], np.zeros_like(time) + rho_max)
            newattrs[f'{prefix}.rho_max'] = float(rho_max)
        newattrs[f'{prefix}.model_name'] = 'CGM'
        newvars[f'{prefix}.chi_e_i_ratio'] = (['time'], np.zeros_like(time) + ei_ratio)
        newvars[f'{prefix}.chi_D_ratio'] = (['time'], np.zeros_like(time) + D_ratio)
        newvars[f'{prefix}.VR_D_ratio'] = (['time'], np.zeros_like(time) + peaking)
        newattrs[f'{prefix}.alpha'] = float(alpha)
        newattrs[f'{prefix}.chi_stiff'] = float(chi_grad)
        newattrs['transport.chi_min'] = 0.05
        newattrs['transport.chi_max'] = 100.0
        newattrs['transport.D_e_min'] = 0.05
        newattrs['transport.D_e_max'] = 100.0
        newattrs['transport.V_e_min'] = -50.0
        newattrs['transport.V_e_max'] = 50.0
        newattrs['transport.smoothing_width'] = 0.1
        newattrs['transport.smooth_everywhere'] = (not data.attrs.get('pedestal.set_pedestal', False))
        self.update_input_coords(newcoords)
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def add_qualikiz_transport(
        self,
        rho_min: float | None = None,
        rho_max: float | None = None,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newvars: MutableMapping[str, Any] = {}
        newattrs: MutableMapping[str, Any] = {}
        prefix = 'transport'
        if data.attrs.get('transport.model_name', '') == 'combined':
            models = data.attrs.get('map_combined_models', [])
            prefix = f'transport.transport_models.{len(models):d}'
            if 'pedestal.rho_norm_ped_top' in data:
                newvars[f'{prefix}.rho_max'] = (['time'], data['pedestal.rho_norm_ped_top'].to_numpy())
            newattrs[f'{prefix}.apply_inner_patch'] = False
            newattrs[f'{prefix}.apply_outer_patch'] = False
            models.append('qualikiz')
            newattrs['map_combined_models'] = models
        if rho_min is not None:
            #newvars[f'{prefix}.rho_min'] = (['time'], np.zeros_like(time) + rho_min)
            newattrs[f'{prefix}.rho_min'] = float(rho_min)
        if rho_max is not None:
            #newvars[f'{prefix}.rho_max'] = (['time'], np.zeros_like(time) + rho_max)
            newattrs[f'{prefix}.rho_max'] = float(rho_max)
        newattrs[f'{prefix}.model_name'] = 'qualikiz'
        newattrs[f'{prefix}.n_max_runs'] = 1
        newattrs[f'{prefix}.n_processes'] = 60
        newattrs[f'{prefix}.collisionality_multiplier'] = 1.0
        newattrs[f'{prefix}.DV_effective'] = True
        newattrs[f'{prefix}.An_min'] = 0.05
        newattrs[f'{prefix}.avoid_big_negative_s'] = True
        newattrs[f'{prefix}.smag_alpha_correction'] = False
        newattrs[f'{prefix}.q_sawtooth_proxy'] = True
        newattrs['transport.chi_min'] = 0.05
        newattrs['transport.chi_max'] = 100.0
        newattrs['transport.D_e_min'] = 0.05
        newattrs['transport.D_e_max'] = 100.0
        newattrs['transport.V_e_min'] = -50.0
        newattrs['transport.V_e_max'] = 50.0
        newattrs['transport.smoothing_width'] = 0.1
        newattrs['transport.smooth_everywhere'] = (not data.attrs.get('pedestal.set_pedestal', False))
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def set_qualikiz_model_path(
        self,
        path: str | Path,
    ) -> None:
        data = self.input
        newattrs: MutableMapping[str, Any] = {}
        if data.attrs.get('transport.model_name', '') == 'combined':
            models = data.attrs.get('map_combined_models', [])
            for n in range(len(models)):
                if data.attrs.get(f'transport.transport_models.{n:d}.model_name', '') == 'qualikiz':
                    newattrs['TORAX_QLK_EXEC_PATH'] = f'{path}'  # Is this still necessary?
        elif data.attrs.get('transport.model_name', '') == 'qualikiz':
            newattrs['TORAX_QLK_EXEC_PATH'] = f'{path}'  # Is this still necessary?
        self.update_input_attrs(newattrs)


    def add_qlknn_transport(
        self,
        rho_min: float | None = None,
        rho_max: float | None = None,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newvars: MutableMapping[str, Any] = {}
        newattrs: MutableMapping[str, Any] = {}
        prefix = 'transport'
        if data.attrs.get('transport.model_name', '') == 'combined':
            models = data.attrs.get('map_combined_models', [])
            prefix = f'transport.transport_models.{len(models):d}'
            #newattrs[f'{prefix}.rho_min'] = {0.0: 0.15}
            if 'pedestal.rho_norm_ped_top' in data:
                #newvars[f'{prefix}.rho_max'] = (['time'], data['pedestal.rho_norm_ped_top'].to_numpy())
                newattrs[f'{prefix}.rho_max'] = float(np.mean(data['pedestal.rho_norm_ped_top'].to_numpy()))
            newattrs[f'{prefix}.apply_inner_patch'] = False
            newattrs[f'{prefix}.apply_outer_patch'] = False
            models.append('qlknn')
            newattrs['map_combined_models'] = models
        if rho_min is not None:
            #newvars[f'{prefix}.rho_min'] = (['time'], np.zeros_like(time) + rho_min)
            newattrs[f'{prefix}.rho_min'] = float(rho_min)
        if rho_max is not None:
            #newvars[f'{prefix}.rho_max'] = (['time'], np.zeros_like(time) + rho_max)
            newattrs[f'{prefix}.rho_max'] = float(rho_max)
        newattrs[f'{prefix}.model_name'] = 'qlknn'
        #newattrs[f'{prefix}.model_path'] = ''
        newattrs[f'{prefix}.include_ITG'] = True
        newattrs[f'{prefix}.include_TEM'] = True
        newattrs[f'{prefix}.include_ETG'] = True
        newattrs[f'{prefix}.ITG_flux_ratio_correction'] = 1.0
        newattrs[f'{prefix}.ETG_correction_factor'] = 1.0 / 3.0
        newattrs[f'{prefix}.clip_inputs'] = False
        newattrs[f'{prefix}.clip_margin'] = 0.95
        newattrs[f'{prefix}.collisionality_multiplier'] = 1.0
        newattrs[f'{prefix}.DV_effective'] = True
        newattrs[f'{prefix}.An_min'] = 0.05
        newattrs[f'{prefix}.avoid_big_negative_s'] = True
        newattrs[f'{prefix}.smag_alpha_correction'] = True
        newattrs[f'{prefix}.q_sawtooth_proxy'] = True
        newattrs['transport.chi_min'] = 0.05
        newattrs['transport.chi_max'] = 100.0
        newattrs['transport.D_e_min'] = 0.05
        newattrs['transport.D_e_max'] = 100.0
        newattrs['transport.V_e_min'] = -50.0
        newattrs['transport.V_e_max'] = 50.0
        newattrs['transport.smoothing_width'] = 0.0
        newattrs['transport.smooth_everywhere'] = (not data.attrs.get('pedestal.set_pedestal', False))
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def set_qlknn_model_path(
        self,
        path: str | Path,
    ) -> None:
        data = self.input
        newattrs: MutableMapping[str, Any] = {}
        if data.attrs.get('transport.model_name', '') == 'combined':
            models = data.attrs.get('map_combined_models', [])
            for n in range(len(models)):
                if data.attrs.get(f'transport.transport_models.{n:d}.model_name', '') == 'qlknn':
                    newattrs[f'transport.transport_models.{n:d}.model_path'] = f'{path}'
        if data.attrs.get('transport.model_name', '') == 'qlknn':
            newattrs['transport.model_path'] = f'{path}'
        self.update_input_attrs(newattrs)


    def add_transport_inner_patch(
        self,
        de: float,
        ve: float,
        chii: float,
        chie: float,
        rho: float,
        tstart: float | None = None,
        tend: float | None = None,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        trigger = np.isfinite(time)
        if isinstance(tstart, (float, int)):
            trigger &= (time >= tstart)
        if isinstance(tend, (float, int)):
            trigger &= (time <= tend)
        newvars: MutableMapping[str, Any] = {}
        newattrs: MutableMapping[str, Any] = {}
        if data.attrs.get('transport.model_name', '') == 'combined':
            self.add_constant_transport(chii, chie, de, ve, rho_max=rho)
        else:
            newvars['transport.apply_inner_patch'] = (['time'], trigger)
            newvars['transport.D_e_inner'] = (['time'], np.zeros_like(time) + de)
            newvars['transport.V_e_inner'] = (['time'], np.zeros_like(time) + ve)
            newvars['transport.chi_i_inner'] = (['time'], np.zeros_like(time) + chii)
            newvars['transport.chi_e_inner'] = (['time'], np.zeros_like(time) + chie)
            newattrs['transport.rho_inner'] = float(rho)
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def add_transport_outer_patch(
        self, 
        de: float,
        ve: float,
        chii: float,
        chie: float,
        rho: float,
        tstart: float | None = None,
        tend: float | None = None,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        trigger = np.isfinite(time)
        if isinstance(tstart, (float, int)):
            trigger &= (time >= tstart)
        if isinstance(tend, (float, int)):
            trigger &= (time <= tend)
        newvars: MutableMapping[str, Any] = {}
        newattrs: MutableMapping[str, Any] = {}
        if data.attrs.get('transport.model_name', '') == 'combined':
            self.add_constant_transport(chii, chie, de, ve, rho_min=rho)
        else:
            newvars['transport.apply_outer_patch'] = (['time'], trigger)
            newvars['transport.D_e_outer'] = (['time'], np.zeros_like(time) + de)
            newvars['transport.V_e_outer'] = (['time'], np.zeros_like(time) + ve)
            newvars['transport.chi_i_outer'] = (['time'], np.zeros_like(time) + chii)
            newvars['transport.chi_e_outer'] = (['time'], np.zeros_like(time) + chie)
            newattrs['transport.rho_outer'] = float(rho)
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def reset_mhd_sawtooth_trigger(
        self,
    ) -> None:
        delvars = [
            'mhd.sawtooth.trigger_model.minimum_radius',
            'mhd.sawtooth.trigger_model.s_critical',
            'mhd.sawtooth.redistribution_model.flattening_factor',
            'mhd.sawtooth.redistribution_model.mixing_radius_multiplier',
        ]
        self.delete_input_data_vars(delvars)
        delattrs = [
            'mhd.sawtooth.crash_step_duration',
            'mhd.sawtooth.trigger_model.model_name',
            'mhd.sawtooth.redistribution_model.model_name',
        ]
        self.delete_input_attrs(delattrs)


    def set_mhd_sawtooth_trigger(
        self,
        rmin: float,
        scrit: float,
        flat: float = 1.01,
        rmult: float = 1.1,
        deltat: float = 1.0e-3,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newvars: MutableMapping[str, Any] = {}
        newvars['mhd.sawtooth.trigger_model.minimum_radius'] = (['time'], np.zeros_like(time) + rmin)
        newvars['mhd.sawtooth.trigger_model.s_critical'] = (['time'], np.zeros_like(time) + scrit)
        newvars['mhd.sawtooth.redistribution_model.flattening_factor'] = (['time'], np.zeros_like(time) + flat)
        newvars['mhd.sawtooth.redistribution_model.mixing_radius_multiplier'] = (['time'], np.zeros_like(time) + rmult)
        self.update_input_data_vars(newvars)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['mhd.sawtooth.crash_step_duration'] = float(deltat)
        newattrs['mhd.sawtooth.trigger_model.model_name'] = 'simple'
        newattrs['mhd.sawtooth.redistribution_model.model_name'] = 'simple'
        newattrs['mhd.sawtooth.crash_step_duration'] = 1.0e-3
        self.update_input_attrs(newattrs)


    def reset_exchange_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.ei_exchange.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delattrs = [
            'sources.ei_exchange.Qei_multiplier'
        ]
        self.delete_input_attrs(delattrs)
        delvars = [
            'sources.ei_exchange.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)


    def set_default_exchange_source(
        self,
    ) -> None:
        self.reset_exchange_source()
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.ei_exchange.mode'] = 'MODEL_BASED'
        newattrs['sources.ei_exchange.Qei_multiplier'] = 1.0
        self.update_input_attrs(newattrs)


    def set_prescribed_exchange_source(
        self,
        rho: NDArray,
        values: NDArray,
    ) -> None:
        # TODO: Should make 2D version
        self.reset_exchange_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newrho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        newvals = np.interp(newrho, rho, values)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.ei_exchange.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.ei_exchange.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newvals), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def reset_ohmic_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.ohmic.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.ohmic.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)


    def set_default_ohmic_source(
        self,
    ) -> None:
        self.reset_ohmic_source()
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.ohmic.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)
        if 'sources.generic_current.prescribed_values' in self.input and 'profile_conditions.j_ohmic' in self.input:
            self.input['sources.generic_current.prescribed_values'] = self.input['sources.generic_current.prescribed_values'] - self.input['profile_conditions.j_ohmic']


    def set_prescribed_ohmic_source(
        self,
        rho: NDArray,
        values: NDArray,
    ) -> None:
        # TODO: Should make 2D version
        self.reset_ohmic_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newrho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        newvals = np.interp(newrho, rho, values)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.ohmic.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.ohmic.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newvals), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def reset_fusion_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.fusion.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.fusion.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)


    def set_default_fusion_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.fusion.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)


    def set_prescribed_fusion_source(
        self,
        rho: NDArray,
        values: NDArray,
    ) -> None:
        # TODO: Should make 2D version
        self.reset_fusion_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newrho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        newvals = np.interp(newrho, rho, values)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.fusion.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.fusion.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newvals), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def reset_gas_puff_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.gas_puff.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.gas_puff.puff_decay_length',
            'sources.gas_puff.S_total',
        ]
        self.delete_input_data_vars(delvars)


    def set_default_gas_puff_source(
        self, 
        length: float,
        total: float,
    ) -> None:
        self.reset_gas_puff_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.gas_puff.puff_decay_length'] = (['time'], np.zeros_like(time) + length)
        newvars['sources.gas_puff.S_total'] = (['time'], np.zeros_like(time) + total)
        self.update_input_data_vars(newvars)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.gas_puff.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)


    def set_prescribed_gas_puff_source(
        self,
        rho: NDArray,
        values: NDArray,
    ) -> None:
        # TODO: Should make 2D version
        self.reset_gas_puff_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newrho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        newvals = np.interp(newrho, rho, values)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.gas_puff.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.gas_puff.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newvals), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def set_default_bootstrap_current_source(
        self,
    ) -> None:
        self.add_neoclassical_bootstrap_current()


    def reset_bremsstrahlung_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.bremsstrahlung.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.bremsstrahlung.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)
        delattrs = [
            'sources.bremsstrahlung.use_relativistic_correction',
        ]
        self.delete_input_attrs(delattrs)


    def set_default_bremsstrahlung_source(
        self,
    ) -> None:
        self.reset_bremsstrahlung_source()
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.bremsstrahlung.mode'] = 'MODEL_BASED'
        newattrs['sources.bremsstrahlung.use_relativistic_correction'] = True
        self.update_input_attrs(newattrs)


    def set_prescribed_bremsstrahlung_source(
        self,
        rho: NDArray,
        values: NDArray,
    ) -> None:
        # TODO: Should make 2D version
        self.reset_bremsstrahlung_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newrho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        newvals = np.interp(newrho, rho, values)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.bremsstrahlung.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.bremsstrahlung.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newvals), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def reset_line_radiation_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.impurity_radiation.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.impurity_radiation.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)
        delattrs = [
            'sources.impurity_radiation.model_name',
            'sources.impurity_radiation.radiation_multiplier',
        ]
        self.delete_input_attrs(delattrs)


    def set_mavrin_line_radiation_source(
        self,
    ) -> None:
        self.reset_line_radiation_source()
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.impurity_radiation.mode'] = 'MODEL_BASED'
        newattrs['sources.impurity_radiation.model_name'] = 'mavrin_fit'
        newattrs['sources.impurity_radiation.radiation_multiplier'] = 1.0
        self.update_input_attrs(newattrs)
        # Mavrin polynomial model includes Bremsstrahlung so zero that out as well
        self.reset_bremsstrahlung_source()


    def set_prescribed_line_radiation_source(
        self,
        rho: NDArray,
        values: NDArray,
    ) -> None:
        # TODO: Should make 2D version
        self.reset_line_radiation_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newrho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        newvals = np.interp(newrho, rho, values)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.impurity_radiation.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.impurity_radiation.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newvals), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def reset_synchrotron_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.cyclotron_radiation.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.cyclotron_radiation.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)
        delattrs = [
            'sources.cyclotron_radiation.wall_reflection_coeff',
            'sources.cyclotron_radiation.beta_min',
            'sources.cyclotron_radiation.beta_max',
            'sources.cyclotron_radiation.beta_grid_size',
        ]
        self.delete_input_attrs(delattrs)


    def set_default_synchrotron_source(
        self,
    ) -> None:
        self.reset_synchrotron_source()
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.cyclotron_radiation.mode'] = 'MODEL_BASED'
        newattrs['sources.cyclotron_radiation.wall_reflection_coeff'] = 0.9
        newattrs['sources.cyclotron_radiation.beta_min'] = 0.5
        newattrs['sources.cyclotron_radiation.beta_max'] = 8.0
        newattrs['sources.cyclotron_radiation.beta_grid_size'] = 32
        self.update_input_attrs(newattrs)


    def set_prescribed_synchrotron_source(
        self,
        rho: NDArray,
        values: NDArray,
    ) -> None:
        # TODO: Should make 2D version
        self.reset_synchrotron_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newrho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        newvals = np.interp(newrho, rho, values)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.cyclotron_radiation.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.cyclotron_radiation.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newvals), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def add_toricnn_icrh_source(
        self,
        freq: float,
        mfrac: float,
        total: float,
        iwall: float = 1.24,
        owall: float = 2.43,
    ) -> None:
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.icrh.frequency'] = (['time'], np.zeros_like(time) + freq)
        newvars['sources.icrh.minority_concentration'] = (['time'], np.zeros_like(time) + mfrac)
        newvars['sources.icrh.P_total'] = (['time'], np.zeros_like(time) + total)
        self.update_input_data_vars(newvars)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.icrh.mode'] = 'MODEL_BASED'
        newattrs['sources.icrh.model_path'] = ''
        newattrs['sources.icrh.wall_inner'] = iwall
        newattrs['sources.icrh.wall_outer'] = owall
        self.update_input_attrs(newattrs)


    def set_toricnn_model_path(
        self,
        path: str | Path
    ) -> None:
        data = self.input
        newattrs: MutableMapping[str, Any] = {}
        if self.input.attrs.get('sources.icrh.mode', 'ZERO') == 'MODEL_BASED':
            newattrs['sources.icrh.model_path'] = f'{path}'
        self.update_input_attrs(newattrs)


    def reset_generic_heat_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.generic_heat.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.generic_heat.prescribed_values',
            'sources.generic_heat.prescribed_values_el',
            'sources.generic_heat.prescribed_values_ion',
        ]
        self.delete_input_data_vars(delvars)
        delattrs = [
            'sources.generic_heat.gaussian_location',
            'sources.generic_heat.gaussian_width',
            'sources.generic_heat.P_total',
            'sources.generic_heat.electron_heat_fraction',
            'sources.generic_heat.absorption_fraction',
        ]
        self.delete_input_attrs(delattrs)


    def reset_generic_particle_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.generic_particle.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.generic_particle.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)
        delattrs = [
            'sources.generic_particle.deposition_location',
            'sources.generic_particle.particle_width',
            'sources.generic_particle.S_total',
        ]
        self.delete_input_attrs(delattrs)


    def reset_generic_current_source(
        self,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.generic_current.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.generic_current.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)
        delattrs = [
            'sources.generic_current.gaussian_location',
            'sources.generic_current.gaussian_width',
            'sources.generic_current.I_generic',
            'sources.generic_current.fraction_of_total_current',
            'sources.generic_current.use_absolute_current',
        ]
        self.delete_input_attrs(delattrs)


    def set_gaussian_generic_heat_source(
        self,
        mu: float,
        sigma: float,
        total: float,
        efrac: float = 0.5,
        afrac: float = 1.0,
    ) -> None:
        self.reset_generic_heat_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.generic_heat.gaussian_location'] = (['time'], np.zeros_like(time) + mu)
        newvars['sources.generic_heat.gaussian_width'] = (['time'], np.zeros_like(time) + sigma)
        newvars['sources.generic_heat.P_total'] = (['time'], np.zeros_like(time) + total)
        newvars['sources.generic_heat.electron_heat_fraction'] = (['time'], np.zeros_like(time) + efrac)
        newvars['sources.generic_heat.absorption_fraction'] = (['time'], np.zeros_like(time) + afrac)
        self.update_input_data_vars(newvars)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.generic_heat.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)


    def set_gaussian_generic_particle_source(
        self,
        mu: float,
        sigma: float,
        total: float,
    ) -> None:
        self.reset_generic_particle_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.generic_particle.deposition_location'] = (['time'], np.zeros_like(time) + mu)
        newvars['sources.generic_particle.particle_width'] = (['time'], np.zeros_like(time) + sigma)
        newvars['sources.generic_particle.S_total'] = (['time'], np.zeros_like(time) + total)
        self.update_input_data_vars(newvars)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.generic_particle.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)


    def set_gaussian_generic_current_source(
        self,
        mu: float,
        sigma: float,
        total: float,
    ) -> None:
        self.reset_generic_current_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.generic_current.mode'] = 'MODEL_BASED'
        newattrs['sources.generic_current.gaussian_location'] = (['time'], np.zeros_like(time) + mu)
        newattrs['sources.generic_current.gaussian_width'] = (['time'], np.zeros_like(time) + sigma)
        newattrs['sources.generic_current.I_generic'] = (['time'], np.zeros_like(time) + total)
        newattrs['sources.generic_current.fraction_of_total_current'] = (['time'], np.ones_like(time))
        newattrs['sources.generic_current.use_absolute_current'] = True
        self.update_input_attrs(newattrs)


    def set_prescribed_generic_heat_source(
        self,
        rho: NDArray,
        eheat: NDArray,
        iheat: NDArray,
    ) -> None:
        self.reset_generic_heat_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newrho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        neweheat = np.interp(newrho, rho, eheat)
        newiheat = np.interp(newrho, rho, iheat)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.generic_heat.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.generic_heat.prescribed_values_el'] = (['time', 'rho'], np.repeat(np.atleast_2d(neweheat), len(time), axis=0))
        newvars['sources.generic_heat.prescribed_values_ion'] = (['time', 'rho'], np.repeat(np.atleast_2d(newiheat), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def set_prescribed_generic_particle_source(
        self,
        rho: NDArray,
        particle: NDArray,
    ) -> None:
        self.reset_generic_particle_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newrho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        newparticle = np.interp(newrho, rho, particle)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.generic_particle.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.generic_particle.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newparticle), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def set_prescribed_generic_current_source(
        self,
        rho: NDArray,
        current: NDArray,
    ) -> None:
        self.reset_generic_current_source()
        data = self.input
        time = data.get('time', xr.DataArray()).to_numpy().flatten()
        newrho = data.get('rho', xr.DataArray()).to_numpy().flatten()
        newcurrent = np.interp(newrho, rho, current)
        newattrs: MutableMapping[str, Any] = {}
        newattrs['sources.generic_current.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars: MutableMapping[str, Any] = {}
        newvars['sources.generic_current.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newcurrent), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def add_fixed_linear_solver(
        self,
        dt_fixed: float | None = None,
        single: bool = False,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['solver.solver_type'] = 'linear'
        newattrs['solver.theta_implicit'] = 1.0
        newattrs['solver.use_predictor_corrector'] = True
        newattrs['solver.n_corrector_steps'] = 10
        newattrs['solver.use_pereverzev'] = True
        newattrs['solver.chi_pereverzev'] = 30.0
        newattrs['solver.D_pereverzev'] = 15.0
        newattrs['time_step_calculator.calculator_type'] = 'fixed'
        newattrs['time_step_calculator.tolerance'] = 1.0e-7 if not single else 1.0e-5
        newattrs['numerics.fixed_dt'] = float(dt_fixed) if isinstance(dt_fixed, (float, int)) else 1.0e-1
        self.update_input_attrs(newattrs)


    def add_adaptive_linear_solver(
        self,
        dt_mult: float | None = None,
        single: bool = False,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['solver.solver_type'] = 'linear'
        newattrs['solver.theta_implicit'] = 1.0
        newattrs['solver.use_predictor_corrector'] = True
        newattrs['solver.n_corrector_steps'] = 10
        newattrs['solver.use_pereverzev'] = True
        newattrs['solver.chi_pereverzev'] = 30.0
        newattrs['solver.D_pereverzev'] = 15.0
        newattrs['time_step_calculator.calculator_type'] = 'chi'
        newattrs['time_step_calculator.tolerance'] = 1.0e-7 if not single else 1.0e-5
        newattrs['numerics.chi_timestep_prefactor'] = float(dt_mult) if isinstance(dt_mult, (float, int)) else 50.0
        self.update_input_attrs(newattrs)


    def add_fixed_newton_raphson_solver(
        self,
        dt_fixed: float | None = None,
        single: bool = False,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['solver.solver_type'] = 'newton_raphson'
        newattrs['solver.theta_implicit'] = 1.0
        newattrs['solver.use_predictor_corrector'] = True
        newattrs['solver.n_corrector_steps'] = 10
        newattrs['solver.use_pereverzev'] = True
        newattrs['solver.chi_pereverzev'] = 30.0
        newattrs['solver.D_pereverzev'] = 15.0
        newattrs['solver.log_iterations'] = False
        newattrs['solver.initial_guess_mode'] = 'linear'
        newattrs['solver.residual_tol'] = 1.0e-5 if not single else 1.0e-3
        newattrs['solver.residual_coarse_tol'] = 1.0e-2 if not single else 1.0e-1
        newattrs['solver.delta_reduction_factor'] = 0.5
        newattrs['solver.n_max_iterations'] = 30
        newattrs['solver.tau_min'] = 0.01
        newattrs['time_step_calculator.calculator_type'] = 'fixed'
        newattrs['time_step_calculator.tolerance'] = 1.0e-7 if not single else 1.0e-5
        newattrs['numerics.fixed_dt'] = float(dt_fixed) if isinstance(dt_fixed, (float, int)) else 1.0e-1
        self.update_input_attrs(newattrs)


    def add_adaptive_newton_raphson_solver(
        self,
        dt_mult: float | None = None,
        single: bool = False,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['solver.solver_type'] = 'newton_raphson'
        newattrs['solver.theta_implicit'] = 1.0
        newattrs['solver.use_predictor_corrector'] = True
        newattrs['solver.n_corrector_steps'] = 10
        newattrs['solver.use_pereverzev'] = True
        newattrs['solver.chi_pereverzev'] = 30.0
        newattrs['solver.D_pereverzev'] = 15.0
        newattrs['solver.log_iterations'] = False
        newattrs['solver.initial_guess_mode'] = 'linear'
        newattrs['solver.residual_tol'] = 1.0e-5 if not single else 1.0e-3
        newattrs['solver.residual_coarse_tol'] = 1.0e-2 if not single else 1.0e-1
        newattrs['solver.delta_reduction_factor'] = 0.5
        newattrs['solver.n_max_iterations'] = 30
        newattrs['solver.tau_min'] = 0.01
        newattrs['time_step_calculator.calculator_type'] = 'chi'
        newattrs['time_step_calculator.tolerance'] = 1.0e-7 if not single else 1.0e-5
        newattrs['numerics.chi_timestep_prefactor'] = float(dt_mult) if isinstance(dt_mult, (float, int)) else 50.0
        self.update_input_attrs(newattrs)


    def set_numerics(
        self,
        t_initial: float,
        t_final: float,
        eqs: Sequence[str] = ['te', 'ti', 'ne', 'j'],
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['geometry.n_rho'] = 25 if 'geometry.n_rho' not in self.input.attrs else int(self.input.attrs['geometry.n_rho'])
        newattrs['numerics.t_initial'] = float(t_initial)
        newattrs['numerics.t_final'] = float(t_final)
        newattrs['numerics.exact_t_final'] = True
        newattrs['numerics.max_dt'] = 1.0e-1
        newattrs['numerics.min_dt'] = 1.0e-8
        newattrs['numerics.evolve_electron_heat'] = (isinstance(eqs, (list, tuple)) and 'te' in eqs)
        newattrs['numerics.evolve_ion_heat'] = (isinstance(eqs, (list, tuple)) and 'ti' in eqs)
        newattrs['numerics.evolve_density'] = (isinstance(eqs, (list, tuple)) and 'ne' in eqs)
        newattrs['numerics.evolve_current'] = (isinstance(eqs, (list, tuple)) and 'j' in eqs)
        newattrs['numerics.resistivity_multiplier'] = 1.0
        self.update_input_attrs(newattrs)


    def add_restart(
        self,
        restart_path: str | Path,
        restart_time: float,
    ) -> None:
        newattrs: MutableMapping[str, Any] = {}
        newattrs['restart.filename'] = f'{restart_path}'
        newattrs['restart.time'] = float(restart_time)
        newattrs['restart.do_restart'] = True
        newattrs['restart.stitch'] = False
        self.update_input_attrs(newattrs)


    def print_summary(
        self,
    ) -> None:
        if self.has_output:
            fields = {
                'Bt': ('B_0', 1.0, 'T'),
                'Ip': ('Ip', 1.0e6, 'MA'),
                'q95': ('q95', 1.0, ''),
                'R': ('R_major', 1.0, 'm'),
                'a': ('a_minor', 1.0, 'm'),
                'Q': ('Q_fusion', 1.0, ''),
                'Pfus': ('P_alpha_total', 2.0e5, 'MW'),
                'Pin': ('P_external_injected', 1.0e6, 'MW'),
                'H98y2': ('H98', 1.0, ''),
                'H89p': ('H89P', 1.0, ''),
                '<ne>': ('n_e_volume_avg', 1.0e20, '10^20 m^-3'),
                '<Te>': ('T_e_volume_avg', 1.0, 'keV'),
                '<Ti>': ('T_i_volume_avg', 1.0, 'keV'),
                'betaN': ('beta_N', 1.0, ''),
                'Prad': ('P_radiation_e', -1.0e6, 'MW'),
                'Psol': ('P_SOL_total', 1.0e6, 'MW'),
                'fG': ('fgw_n_e_volume_avg', 1.0, ''),
                'We': ('W_thermal_e', 1.0e6, 'MJ'),
                'Wi': ('W_thermal_i', 1.0e6, 'MJ'),
                'W_thr': ('W_thermal_total', 1.0e6, 'MJ'),
                'tauE': ('tau_E', 1.0, ''),
            }
            radial_fields = {
                #'p_vol': ('pressure_thermal_total', -1, 1.0e-3, 'kPa'),
                'nu_ne': ('n_e', 0, 'n_e_volume_avg', ''),
                'nu_Te': ('T_e', 0, 'T_e_volume_avg', ''),
                'nu_Ti': ('T_i', 0, 'T_i_volume_avg', ''),
            }
            data = self.output.isel(time=-1)
            for key, sspecs in fields.items():
                var, scale, units = sspecs
                val = data[var] / scale
                print(f'{key:10}: {val:.2f} {units}')
            for key, rspecs in radial_fields.items():
                var, idx, rscale, units = rspecs
                if isinstance(rscale, str):
                    val = data.isel(rho_norm=idx)[var] / data[rscale]
                else:
                    val = data.isel(rho_norm=idx)[var] / rscale
                print(f'{key:10}: {val:.2f} {units}')


    def to_dict(
        self,
    ) -> MutableMapping[str, Any]:
        datadict: MutableMapping[str, Any] = {}
        self._clean()
        ds = self.input
        datadict.update(ds.attrs)
        for key in ds.data_vars:
            dims = ds[key].dims
            ttag: str | None = 'time' if 'time' in dims else None
            if ttag is None:
                for dim in dims:
                    if str(dim).startswith('time_'):
                        ttag = str(dim)
                        break
            rtag: str | None = 'rho' if 'rho' in dims else None
            if rtag is None:
                for dim in dims:
                    if str(dim).startswith('rho_'):
                        rtag = str(dim)
                        break
            if ttag is not None and ttag in dims:
                time = ds[ttag].to_numpy().flatten()
                if 'main_ion' in dims:
                    for species in ds['main_ion'].to_numpy().flatten():
                        da = ds[key].dropna(ttag).sel(main_ion=species)
                        if rtag is not None and rtag in da.dims:
                            da = da.rename({rtag: 'rho_norm'}).dropna('rho_norm').isel(rho_norm=0)
                        if da.size > 0:
                            datadict[f'{key}.{species}'] = da
                elif 'impurity' in dims:
                    for species in ds['impurity'].to_numpy().flatten():
                        da = ds[key].dropna(ttag).sel(impurity=species)
                        if rtag is not None and rtag in da.dims:
                            da = da.rename({rtag: 'rho_norm'}).dropna('rho_norm')
                        if da.size > 0:
                            datadict[f'{key}.{species}'] = da
                elif rtag is not None and rtag in dims:
                    da = ds[key].dropna(ttag).rename({rtag: 'rho_norm'}).dropna('rho_norm')
                    if da.size > 0:
                        datadict[f'{key}'] = da
                else:
                    da = ds[key].dropna(ttag)
                    if da.size > 0:
                        datadict[f'{key}'] = da
        models = datadict.pop('map_combined_models', {})
        if datadict.get('transport.model_name', '') == 'combined':
            datadict['transport.transport_models'] = []
            for nn in range(len(models)):
                modeldict = {key.replace(f'transport.transport_models.{nn:d}.', ''): val for key, val in datadict.items() if key.startswith(f'transport.transport_models.{nn:d}.')}
                for key in modeldict:
                    datadict.pop(f'transport.transport_models.{nn:d}.{key}', None)
                datadict['transport.transport_models'].append(self._unflatten(modeldict))
        srctags = [
            'sources.ei_exchange',
            'sources.ohmic',
            'sources.fusion',
            'sources.gas_puff',
            'sources.bremsstrahlung',
            'sources.impurity_radiation',
            'sources.cyclotron_radiation',
            'sources.generic_heat',
            'sources.generic_particle',
            'sources.generic_current',
        ]
        for srctag in srctags:
            if datadict.get(f'{srctag}.mode', 'MODEL_BASED') != 'PRESCRIBED':
                src = datadict.pop(f'{srctag}.prescribed_values', None)
                if srctag in ['sources.bremsstrahlung']:
                    datadict.pop('sources.bremsstrahlung.use_relativistic_correction', None)
                if srctag in ['sources.generic_heat']:
                    datadict.pop(f'{srctag}.prescribed_values_el', None)
                    datadict.pop(f'{srctag}.prescribed_values_ion', None)
        if (
            datadict.get('sources.generic_heat.mode', 'MODEL_BASED') == 'PRESCRIBED' and
            'sources.generic_heat.prescribed_values_el' in datadict and
            'sources.generic_heat.prescribed_values_ion' in datadict
        ):
            e_source = datadict.pop('sources.generic_heat.prescribed_values_el')
            i_source = datadict.pop('sources.generic_heat.prescribed_values_ion')
            datadict['sources.generic_heat.prescribed_values'] = (i_source, e_source)
        if (
            datadict.get('sources.generic_particle.mode', 'MODEL_BASED') == 'PRESCRIBED' and
            'sources.generic_particle.prescribed_values' in datadict
        ):
            datadict['sources.generic_particle.prescribed_values'] = (datadict.pop('sources.generic_particle.prescribed_values'), )
        if (
            datadict.get('sources.generic_current.mode', 'MODEL_BASED') == 'PRESCRIBED' and
            'sources.generic_current.prescribed_values' in datadict
        ):
            datadict['sources.generic_current.prescribed_values'] = (datadict.pop('sources.generic_current.prescribed_values'), )
        datadict.pop('use_psi', None)
        datadict.pop('use_generic_heat', None)
        datadict.pop('use_generic_particle', None)
        datadict.pop('use_generic_current', None)
        datadict.pop('use_fusion', None)
        datadict.pop('profile_conditions.n_i', None)
        datadict.pop('profile_conditions.q', None)
        datadict.pop('profile_conditions.j_ohmic', None)
        datadict.pop('profile_conditions.j_bootstrap', None)
        datadict.pop('TORAX_QLK_EXEC_PATH', None)
        if 'pedestal.set_pedestal' not in datadict:
            datadict['pedestal.set_pedestal'] = False
        return self._unflatten(datadict)


    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        input: str | Path | None = None,
        output: str | Path | None = None,
    ) -> Self:
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


    @classmethod
    def from_gacode(
        cls,
        obj: io,
        side: str = 'output',
        item: int = 0,
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input if side == 'input' else obj.output
            if 'n' in data.coords and 'rho' in data.coords:
                coords = {}
                data_vars = {}
                attrs: MutableMapping[str, Any] = {}
                data = data.sel(n=item)
                time = data.get('time', 0.0)
                attrs['numerics.t_initial'] = float(time)
                coords['time'] = np.array([time]).flatten()
                coords['rho'] = data['rho'].to_numpy().flatten()
                if 'z' in data and 'name' in data and 'type' in data and 'ni' in data:
                    species = []
                    density = []
                    nfilt = (np.isclose(data['z'], 1.0) & (['fast' not in v for v in data['type'].to_numpy().flatten()]))
                    if np.any(nfilt):
                        namelist = data['name'].to_numpy()[nfilt].tolist()
                        nfuelsum = data['ni'].sel(name=namelist).sum('name')
                        for ii in range(len(namelist)):
                            species.append(namelist[ii])
                            density.append(np.expand_dims((data['ni'].sel(name=namelist[ii]) / nfuelsum).mean('rho').to_numpy().flatten(), axis=0))
                        coords['main_ion'] = np.array(species).flatten()
                    else:
                        species = ['D']
                        density = [np.atleast_2d([1.0])]
                    coords['main_ion'] = np.array(species).flatten()
                    data_vars['plasma_composition.main_ion'] = (['main_ion', 'time'], np.concatenate(density, axis=0))
                if 'z' in data and 'mass' in data and 'ni' in data and 'ne' in data:
                    nfilt = (~np.isclose(data['z'], 1.0))
                    zeff = xr.ones_like(data['ne'])
                    if np.any(nfilt):
                        namelist = data['name'].to_numpy()[nfilt].tolist()
                        implist = []
                        for ii in range(len(namelist)):
                            if namelist[ii] not in ['H', 'D', 'T']:
                                implist.append(ii)
                        impcomp = {}
                        zeff = xr.zeros_like(data['ne'])
                        nsum = xr.zeros_like(data['ne'])
                        for ii in range(len(data['name'])):
                            nz = copy.deepcopy(data['ni'].isel(name=ii))
                            if str(data['name'].isel(name=ii).to_numpy()) in namelist and 'therm' in str(data['type'].isel(name=ii).to_numpy()):
                                sname = str(data['name'].isel(name=ii).to_numpy())
                                if sname not in newobj.allowed_radiation_species:
                                    sn, sa, sz = define_ion_species(short_name=sname)
                                    if sz > 2.0:
                                        sname = 'C'
                                    if sz > 8.0:
                                        sname = 'Ne'
                                    if sz > 14.0:
                                        sname = 'Ar'
                                    if sz > 24.0:
                                        sname = 'Kr'
                                    if sz > 45.0:
                                        sname = 'Xe'
                                    if sz > 64.0:
                                        sname = 'W'
                                    newsn, newsa, newsz = define_ion_species(short_name=sname)
                                    nz = nz * sz / newsz
                                    if sn == 'He':
                                        sname = 'He4'
                                if ii in implist:
                                    # Intentional mismatch between composition and Zeff densities to handle species changes for radiation calculation
                                    impcomp[sname] = copy.deepcopy(nz)
                                nsum += nz
                            zeff += (data['ni'] * data['z'] ** 2.0 / data['ne']).isel(name=ii)
                        total = 0.0
                        impcoord = []
                        impfracs = []
                        for key in impcomp:
                            impcomp[key] = (impcomp[key] / nsum).mean('rho')
                            total += float(impcomp[key].to_numpy())
                        for key in impcomp:
                            impval = (impcomp[key] / total).to_numpy().flatten()
                            impcoord.append(key)
                            impfracs.append(np.expand_dims(impval, axis=0))
                        if len(impcoord) == 0:
                            impcoord = ['Ne']
                            impfracs = [np.atleast_2d([1.0])]
                        if 'z_eff' in data:
                            zeff = data['z_eff']
                        coords['impurity'] = np.array(impcoord).flatten()
                        data_vars['plasma_composition.impurity'] = (['impurity', 'time'], np.concatenate(impfracs, axis=0))
                    data_vars['plasma_composition.Z_eff'] = (['time', 'rho'], np.expand_dims(zeff.to_numpy().flatten(), axis=0))
                if 'current' in data:
                    data_vars['profile_conditions.Ip'] = (['time'], 1.0e6 * np.expand_dims(data['current'].mean(), axis=0))
                if 'ne' in data:
                    data_vars['profile_conditions.n_e'] = (['time', 'rho'], np.expand_dims(1.0e19 * data['ne'].to_numpy().flatten(), axis=0))
                    attrs['profile_conditions.normalize_n_e_to_nbar'] = False
                    attrs['profile_conditions.n_e_nbar_is_fGW'] = False
                if 'te' in data:
                    data_vars['profile_conditions.T_e'] = (['time', 'rho'], np.expand_dims(data['te'].to_numpy().flatten(), axis=0))
                if 'ti' in data and 'z' in data:
                    nfilt = (np.isclose(data['z'], 1.0) & (['fast' not in v for v in data['type'].to_numpy().flatten()]))
                    tfuel = data['ti'].mean('name')
                    if np.any(nfilt):
                        namelist = data['name'].to_numpy()[nfilt].tolist()
                        tfuel = data['ti'].sel(name=namelist).mean('name')
                    data_vars['profile_conditions.T_i'] = (['time', 'rho'], np.expand_dims(tfuel.to_numpy().flatten(), axis=0))
                if 'polflux' in data:
                    attrs['use_psi'] = True
                    data_vars['profile_conditions.psi'] = (['time', 'rho'], np.expand_dims(data['polflux'].to_numpy().flatten(), axis=0))
                if 'q' in data:
                    data_vars['profile_conditions.q'] = (['time', 'rho'], np.expand_dims(data['q'].to_numpy().flatten(), axis=0))
                # Place the sources
                external_el_heat_source = None
                external_ion_heat_source = None
                external_particle_source = None
                external_current_source = None
                fusion_source = None
                if 'qohme' in data and np.abs(data['qohme']).sum() != 0.0:
                    attrs['sources.ohmic.mode'] = 'PRESCRIBED'
                    data_vars['sources.ohmic.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qohme'].to_numpy().flatten(), axis=0))
                if 'qbeame' in data and np.abs(data['qbeame']).sum() != 0.0:
                    if external_el_heat_source is None:
                        external_el_heat_source = np.zeros_like(data['qbeame'].to_numpy().flatten())
                    external_el_heat_source += 1.0e6 * data['qbeame'].to_numpy().flatten()
                if 'qbeami' in data and np.abs(data['qbeami']).sum() != 0.0:
                    if external_ion_heat_source is None:
                        external_ion_heat_source = np.zeros_like(data['qbeami'].to_numpy().flatten())
                    external_ion_heat_source += 1.0e6 * data['qbeami'].to_numpy().flatten()
                if 'qrfe' in data and np.abs(data['qrfe']).sum() != 0.0:
                    if external_el_heat_source is None:
                        external_el_heat_source = np.zeros_like(data['qrfe'].to_numpy().flatten())
                    external_el_heat_source += 1.0e6 * data['qrfe'].to_numpy().flatten()
                if 'qrfi' in data and np.abs(data['qrfi']).sum() != 0.0:
                    if external_ion_heat_source is None:
                        external_ion_heat_source = np.zeros_like(data['qrfi'].to_numpy().flatten())
                    external_ion_heat_source += 1.0e6 * data['qrfi'].to_numpy().flatten()
                if 'qsync' in data and np.abs(data['qsync']).sum() != 0.0:
                    attrs['sources.cyclotron_radiation.mode'] = 'PRESCRIBED'
                    data_vars['sources.cyclotron_radiation.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qsync'].to_numpy().flatten(), axis=0))
                if 'qbrem' in data and np.abs(data['qbrem']).sum() != 0.0:
                    attrs['sources.bremsstrahlung.mode'] = 'PRESCRIBED'
                    data_vars['sources.bremsstrahlung.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qbrem'].to_numpy().flatten(), axis=0))
                if 'qline' in data and np.abs(data['qline']).sum() != 0.0:
                    attrs['sources.impurity_radiation.mode'] = 'PRESCRIBED'
                    data_vars['sources.impurity_radiation.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qline'].to_numpy().flatten(), axis=0))
                if 'qfuse' in data and np.abs(data['qfuse']).sum() != 0.0:
                    if fusion_source is None:
                        fusion_source = np.zeros_like(data['qfuse'].to_numpy().flatten())
                    fusion_source += 1.0e6 * data['qfuse'].to_numpy().flatten()
                if 'qfusi' in data and np.abs(data['qfusi']).sum() != 0.0:
                    if fusion_source is None:
                        fusion_source = np.zeros_like(data['qfuse'].to_numpy().flatten())
                    fusion_source += 1.0e6 * data['qfuse'].to_numpy().flatten()
                if 'qei' in data and np.abs(data['qei']).sum() != 0.0:
                    attrs['sources.ei_exchange.mode'] = 'PRESCRIBED'
                    data_vars['sources.ei_exchange.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qei'].to_numpy().flatten(), axis=0))
                #if 'qione' in data and np.abs(data['qione']).sum() != 0.0:
                #    pass
                #if 'qioni' in data and np.abs(data['qioni']).sum() != 0.0:
                #    pass
                #if 'qcxi' in data and np.abs(data['qcxi']).sum() != 0.0:
                #    pass
                if 'jbs' in data and np.abs(data['jbs']).sum() != 0.0:
                    data_vars['profile_conditions.j_bootstrap'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['jbs'].to_numpy().flatten(), axis=0))
                    if external_current_source is None:
                        external_current_source = np.zeros_like(data['jbs'].to_numpy().flatten())
                    external_current_source += 1.0e6 * data['jbs'].to_numpy().flatten()
                #if 'jbstor' in data and data['jbstor'].sum() != 0.0:
                #    pass
                if 'johm' in data and np.abs(data['johm']).sum() != 0.0:
                    data_vars['profile_conditions.j_ohmic'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['johm'].to_numpy().flatten(), axis=0))
                    if external_current_source is None:
                        external_current_source = np.zeros_like(data['johm'].to_numpy().flatten())
                    external_current_source += 1.0e6 * data['johm'].to_numpy().flatten()
                if 'jrf' in data and np.abs(data['jrf']).sum() != 0.0:
                    if external_current_source is None:
                        external_current_source = np.zeros_like(data['jrf'].to_numpy().flatten())
                    external_current_source += 1.0e6 * data['jrf'].to_numpy().flatten()
                if 'jnb' in data and np.abs(data['jnb']).sum() != 0.0:
                    if external_current_source is None:
                        external_current_source = np.zeros_like(data['jnb'].to_numpy().flatten())
                    external_current_source += 1.0e6 * data['jnb'].to_numpy().flatten()
                if 'qpar_beam' in data and np.abs(data['qpar_beam']).sum() != 0.0:
                    if external_particle_source is None:
                        external_particle_source = np.zeros_like(data['qpar_beam'].to_numpy().flatten())
                    external_particle_source += data['qpar_beam'].to_numpy().flatten()
                if 'qpar_wall' in data and np.abs(data['qpar_wall']).sum() != 0.0:
                    if external_particle_source is None:
                        external_particle_source = np.zeros_like(data['qpar_wall'].to_numpy().flatten())
                    external_particle_source += data['qpar_wall'].to_numpy().flatten()
                #if 'qmom' in data and np.abs(data['qmom']).sum() != 0.0:
                #    pass
                if external_el_heat_source is not None:
                    attrs['use_generic_heat'] = True
                    attrs['sources.generic_heat.mode'] = 'PRESCRIBED'
                    data_vars['sources.generic_heat.prescribed_values_el'] = (['time', 'rho'], np.expand_dims(external_el_heat_source, axis=0))
                if external_ion_heat_source is not None:
                    attrs['use_generic_heat'] = True
                    attrs['sources.generic_heat.mode'] = 'PRESCRIBED'
                    data_vars['sources.generic_heat.prescribed_values_ion'] = (['time', 'rho'], np.expand_dims(external_ion_heat_source, axis=0))
                if external_particle_source is not None:
                    attrs['use_generic_particle'] = True
                    attrs['sources.generic_particle.mode'] = 'PRESCRIBED'
                    data_vars['sources.generic_particle.prescribed_values'] = (['time', 'rho'], np.expand_dims(external_particle_source, axis=0))
                if external_current_source is not None:
                    attrs['use_generic_current'] = True
                    attrs['sources.generic_current.mode'] = 'PRESCRIBED'
                    data_vars['sources.generic_current.prescribed_values'] = (['time', 'rho'], np.expand_dims(external_current_source, axis=0))
                    attrs['sources.generic_current.use_absolute_current'] = True
                if fusion_source is not None:
                    attrs['use_fusion'] = True
                    attrs['sources.fusion.mode'] = 'PRESCRIBED'
                    data_vars['sources.fusion.prescribed_values'] = (['time', 'rho'], np.expand_dims(fusion_source, axis=0))
                newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj


    @classmethod
    def from_omas(
        cls,
        obj,
        side: str = 'output',
        window: float | None = None,
        **kwargs: Any,
    ) -> Self:

        newobj = cls()
        if isinstance(obj, io):

            data = obj.input if side == 'input' else obj.output
            time_cp_i = 'core_profiles.profiles_1d:i'
            time_cp = 'core_profiles.time'
            rho_cp_i = 'core_profiles.profiles_1d.grid.rho_tor_norm:i'
            rho_cp = 'core_profiles.profiles_1d.grid.rho_tor_norm'
            ion_cp_i = 'core_profiles.profiles_1d.ion:i'
            ion_cp = 'core_profiles.profiles_1d.ion.label'
            time_eq_i = 'equilibrium.time_slice:i'
            time_eq = 'equilibrium.time'
            psi_eq_i = 'equilibrium.time_slice.profiles_1d.psi:i'
            psi_eq = 'equilibrium.time_slice.profiles_1d.psi'
            rho_eq = 'equilibrium.time_slice.profiles_1d.rho_tor_norm'
            time_cs_i = 'core_sources.source.profiles_1d:i'
            time_cs = 'core_sources.time'
            src_cs_i = 'core_sources.source:i'
            src_cs = 'core_sources.source.identifier.name'
            rho_cs_i = 'core_sources.source.profiles_1d.grid.rho_tor_norm:i'
            rho_cs = 'core_sources.source.profiles_1d.grid.rho_tor_norm'
            ion_cs_i = 'core_sources.source.profiles_1d.ion:i'
            ion_cs = 'core_sources.source.profiles_1d.ion.label'
            cocos = define_cocos_converter(17, 2)  # Assumed IMAS=17 -> GACODE=2
            ikwargs = {'fill_value': 'extrapolate'}

            dsmap = {}

            if time_cp_i in data.coords and rho_cp_i in data.coords:
                cp_coords = {}
                cp_data_vars = {}
                cp_attrs: MutableMapping[str, Any] = {}
                data = data.swap_dims({time_cp_i: time_cp})
                cp_coords['time'] = data.get(time_cp, xr.DataArray()).to_numpy().flatten()
                cp_coords['rho'] = data.isel({time_cp: 0}).get(rho_cp, xr.DataArray()).to_numpy().flatten()
                cp_attrs['numerics.t_initial'] = cp_coords['time'][0]
                omas_tag = 'core_profiles.profiles_1d.ion.density'
                if omas_tag in data and ion_cp in data:
                    namelist = data.isel({time_cp: 0}).get(ion_cp, xr.DataArray()).to_numpy().tolist()
                    mainlist = []
                    for ii in range(len(namelist)):
                        if namelist[ii] in ['H', 'D', 'T']:
                            mainlist.append(ii)
                    maincoord = []
                    mainfracs = []
                    total_main = data.isel({ion_cp_i: mainlist})[omas_tag].sum(ion_cp_i)
                    for ii in mainlist:
                        maincoord.append(namelist[ii])
                        mainfracs.append(np.expand_dims(np.atleast_1d((data.isel({ion_cp_i: ii})[omas_tag] / total_main).mean(rho_cp_i).to_numpy()), axis=0))
                    if len(mainlist) == 0:
                        maincoord = ['D']
                        mainfracs = [np.atleast_2d([1.0])]
                    cp_coords['main_ion'] = np.array(maincoord).flatten()
                    cp_data_vars['plasma_composition.main_ion'] = (['main_ion', 'time'], np.concatenate(mainfracs, axis=0))
                if omas_tag in data and 'core_profiles.profiles_1d.electrons.density' in data:
                    namelist = data.isel({time_cp: 0}).get(ion_cp, xr.DataArray()).to_numpy().tolist()
                    implist = []
                    #zeff = xr.zeros_like(data['core_profiles.profiles_1d.electrons.density'])
                    #nqn = xr.zeros_like(data['core_profiles.profiles_1d.electrons.density'])
                    for ii in range(len(namelist)):
                        if namelist[ii] not in ['H', 'D', 'T']:
                            implist.append(ii)
                    nzt = np.zeros_like(data[omas_tag].to_numpy()) # time, ion, rho
                    szt = np.zeros_like(data[omas_tag].to_numpy())
                    for ii in range(len(namelist)):
                        sname = namelist[ii]
                        sn, sa, sz = define_ion_species(short_name=sname)
                        nzi = data.isel({ion_cp_i: ii})[omas_tag].to_numpy()
                        if sname not in newobj.allowed_radiation_species:
                            if sz > 2.0:
                                sname = 'C'
                            if sz > 8.0:
                                sname = 'Ne'
                            if sz > 14.0:
                                sname = 'Ar'
                            if sz > 24.0:
                                sname = 'Kr'
                            if sz > 45.0:
                                sname = 'Xe'
                            if sz > 64.0:
                                sname = 'W'
                            newsn, newsa, newsz = define_ion_species(short_name=sname)
                            nzi = nzi * sz / newsz
                            if sn == 'He':
                                sname = 'He4'
                        szi = np.zeros_like(nzi) + sz
                        if 'core_profiles.profiles_1d.electrons.temperature' in data:
                            if sname == 'Kr':
                                szi = 13.0 * (np.log10(data['core_profiles.profiles_1d.electrons.temperature'].to_numpy()) - 2.0) + 12.0
                                szi = np.where(szi > 36.0, 36.0, szi)
                            if sname == 'Xe':
                                szi = 18.0 * (np.log10(data['core_profiles.profiles_1d.electrons.temperature'].to_numpy()) - 2.0) + 12.0
                                szi = float(np.mean(np.where(szi > 54.0, 54.0, szi)))
                            if sname == 'W':
                                szi = 21.5 * (np.log10(data['core_profiles.profiles_1d.electrons.temperature'].to_numpy()) - 2.0) + 12.0
                                szi = float(np.mean(np.where(szi > 74.0, 74.0, szi)))
                        nzt[:, ii, :] = nzi
                        szt[:, ii, :] = szi
                    nzz = xr.DataArray(data=nzt, coords=data[omas_tag].coords)
                    szz = xr.DataArray(data=szt, coords=data[omas_tag].coords)
                    nqn = (nzz * szz).sum(ion_cp_i) / data['core_profiles.profiles_1d.electrons.density']
                    nzz = nzz / nqn
                    zeff = (nzz * (szz ** 2.0)).sum(ion_cp_i) / data['core_profiles.profiles_1d.electrons.density']
                    impcomp = {}
                    for ii in range(len(namelist)):
                        if ii in implist:
                            # Intentional mismatch between composition and Zeff densities to handle species changes for radiation calculation
                            impcomp[sname] = copy.deepcopy(nzz.isel({ion_cp_i: ii}))
                    total = xr.zeros_like(nqn.mean(rho_cp_i))
                    for key in impcomp:
                        #impcomp[key] = (impcomp[key] / nsum).mean(rho_cp_i)
                        total += impcomp[key].mean(rho_cp_i)
                    impcoord = []
                    impfracs = []
                    for key in impcomp:
                        impcoord.append(key)
                        impfracs.append(np.expand_dims(np.atleast_1d((impcomp[key].mean(rho_cp_i) / total).to_numpy()), axis=0))
                    if len(impcoord) == 0:
                        impcoord = ['Ne']
                        impfracs = [np.atleast_2d([1.0])]
                    cp_coords['impurity'] = np.array(impcoord).flatten()
                    cp_data_vars['plasma_composition.impurity'] = (['impurity', 'time'], np.concatenate(impfracs, axis=0))
                    cp_data_vars['plasma_composition.Z_eff'] = (['time', 'rho'], zeff.to_numpy())
                omas_tag = 'core_profiles.global_quantities.ip'
                if omas_tag in data:
                    cp_data_vars['profile_conditions.Ip'] = (['time'], data[omas_tag].to_numpy())
                omas_tag = 'core_profiles.profiles_1d.electrons.density'
                if omas_tag in data:
                    cp_data_vars['profile_conditions.n_e'] = (['time', 'rho'], data[omas_tag].to_numpy())
                    cp_attrs['profile_conditions.normalize_n_e_to_nbar'] = False
                    cp_attrs['profile_conditions.n_e_nbar_is_fGW'] = False
                omas_tag = 'core_profiles.profiles_1d.electrons.temperature'
                if omas_tag in data:
                    cp_data_vars['profile_conditions.T_e'] = (['time', 'rho'], 1.0e-3 * data[omas_tag].to_numpy())
                omas_tag = 'core_profiles.profiles_1d.ion.temperature'
                if omas_tag in data:
                    cp_data_vars['profile_conditions.T_i'] = (['time', 'rho'], 1.0e-3 * data[omas_tag].mean(ion_cp_i).to_numpy()) if ion_cp_i in data.dims else (['time', 'rho'], data[omas_tag].to_numpy())
                omas_tag = 'core_profiles.profiles_1d.grid.psi'
                if omas_tag in data:
                    cp_data_vars['profile_conditions.psi'] = (['time', 'rho'], data[omas_tag].to_numpy())
                omas_tag = 'core_profiles.profiles_1d.q'
                if omas_tag in data:
                    cp_data_vars['profile_conditions.q'] = (['time', 'rho'], data[omas_tag].to_numpy())
                omas_tag = 'core_profiles.global_quantities.v_loop'
                if omas_tag in data:
                    cp_data_vars['profile_conditions.v_loop_lcfs'] = (['time'], data[omas_tag].to_numpy())
                    cp_attrs['profile_conditions.use_v_loop_lcfs_boundary_condition'] = False
                #core_profiles.profiles_1d.conductivity_parallel              (time_cp, rho_cp)
                #core_profiles.profiles_1d.current_parallel_inside            (time_cp, rho_cp)
                #core_profiles.profiles_1d.j_tor                              (time_cp, rho_cp)
                #core_profiles.profiles_1d.j_total                            (time_cp, rho_cp)
                dsmap['core_profiles'] = xr.Dataset(coords=cp_coords, data_vars=cp_data_vars, attrs=cp_attrs).drop_duplicates(list(cp_coords.keys()), keep='first')

            if time_cs_i in data.coords and rho_cs_i in data.coords:
                cs_coords = {}
                cs_data_vars = {}
                cs_attrs: MutableMapping[str, Any] = {}
                data = data.swap_dims({time_cs_i: time_cs, src_cs_i: src_cs})
                cs_coords['time'] = data.get(time_cs, xr.DataArray()).to_numpy().flatten()
                cs_coords['rho'] = np.nanmean(data.isel({time_cs: 0}).get(rho_cs, xr.DataArray()).to_numpy(), axis=0).flatten()
                srclist = data[src_cs].to_numpy().tolist()
                external_el_heat_source = None
                external_ion_heat_source = None
                external_particle_source = None
                external_current_source = None
                fusion_source = None
                omas_tag = 'core_sources.source.profiles_1d.electrons.energy'
                if omas_tag in data:
                    srctag = 'ohmic'
                    if srctag in srclist and np.abs(data.sel({src_cs: srctag})[omas_tag]).sum() != 0.0:
                        cs_attrs['sources.ohmic.mode'] = 'PRESCRIBED'
                        cs_data_vars['sources.ohmic.prescribed_values'] = (['time', 'rho'], data.sel({src_cs: srctag})[omas_tag].to_numpy())
                    srctag = 'line_radiation'
                    if srctag in srclist and np.abs(data.sel({src_cs: srctag})[omas_tag]).sum() != 0.0:
                        cs_attrs['sources.impurity_radiation.mode'] = 'PRESCRIBED'
                        cs_data_vars['sources.impurity_radiation.prescribed_values'] = (['time', 'rho'], data.sel({src_cs: srctag})[omas_tag].to_numpy())
                    srctag = 'bremsstrahlung'
                    if srctag in srclist and np.abs(data.sel({src_cs: srctag})[omas_tag]).sum() != 0.0:
                        cs_attrs['sources.bremsstrahlung.mode'] = 'PRESCRIBED'
                        cs_data_vars['sources.bremsstrahlung.prescribed_values'] = (['time', 'rho'], data.sel({src_cs: srctag})[omas_tag].to_numpy())
                    #srctag = 'synchrotron'
                    srctag = 'ic'
                    if srctag in srclist and np.abs(data.sel({src_cs: srctag})[omas_tag]).sum() != 0.0:
                        if external_el_heat_source is None:
                            external_el_heat_source = np.zeros_like(data.sel({src_cs: srctag})[omas_tag].to_numpy())
                        external_el_heat_source += 1.0e6 * data.sel({src_cs: srctag})[omas_tag].to_numpy()
                    srctag = 'fusion'
                    if srctag in srclist and np.abs(data.sel({src_cs: srctag})[omas_tag]).sum() != 0.0:
                        if fusion_source is None:
                            fusion_source = np.zeros_like(data.sel({src_cs: srctag})[omas_tag].to_numpy())
                        fusion_source += data.sel({src_cs: srctag})[omas_tag].to_numpy()
                omas_tag = 'core_sources.source.profiles_1d.total_ion_energy'
                if omas_tag in data:
                    srctag = 'ic'
                    if srctag in srclist and np.abs(data.sel({src_cs: srctag})[omas_tag]).sum() != 0.0:
                        if external_ion_heat_source is None:
                            external_ion_heat_source = np.zeros_like(data.sel({src_cs: srctag})[omas_tag].to_numpy())
                        external_ion_heat_source += 1.0e6 * data.sel({src_cs: srctag})[omas_tag].to_numpy()
                #omas_tag = 'core_sources.source.global_quantities.power'
                #if omas_tag in data:
                #    srctag = 'ic'
                #    if srctag in srclist and np.abs(data.sel({src_cs: srctag})[omas_tag]).sum() != 0.0:
                #        data_vars['sources.icrh.P_total'] = (['time'], data[omas_tag].to_numpy().flatten())
                omas_tag = 'core_sources.source.profiles_1d.ion.energy'
                if omas_tag in data:
                    srctag = 'fusion'
                    if srctag in srclist and np.abs(data.sel({src_cs: srctag})[omas_tag]).sum() != 0.0:
                        if fusion_source is None:
                            fusion_source = np.zeros_like(data.sel({src_cs: srctag})[omas_tag].sum(ion_cp_i).to_numpy())
                        fusion_source += data.sel({src_cs: srctag})[omas_tag].sum(ion_cp_i).to_numpy()
                if external_el_heat_source is not None:
                    cs_attrs['use_generic_heat'] = True
                    cs_attrs['sources.generic_heat.mode'] = 'PRESCRIBED'
                    cs_data_vars['sources.generic_heat.prescribed_values_el'] = (['time', 'rho'], external_el_heat_source)
                    if external_ion_heat_source is not None:
                        cs_data_vars['sources.generic_heat.prescribed_values_ion'] = (['time', 'rho'], external_ion_heat_source)
                if external_particle_source is not None:
                    cs_attrs['use_generic_particle'] = True
                    cs_attrs['sources.generic_particle.mode'] = 'PRESCRIBED'
                    cs_data_vars['sources.generic_particle.prescribed_values'] = (['time', 'rho'], external_particle_source)
                if external_current_source is not None:
                    cs_attrs['use_generic_current'] = True
                    cs_attrs['sources.generic_current.mode'] = 'PRESCRIBED'
                    cs_data_vars['sources.generic_current.prescribed_values'] = (['time', 'rho'], external_current_source)
                    cs_attrs['sources.generic_current.use_absolute_current'] = True
                #if fusion_source is not None:
                #    cs_attrs['use_fusion'] = True
                #    cs_attrs['sources.fusion.mode'] = 'PRESCRIBED'
                #    cs_data_vars['sources.fusion.prescribed_values'] = (['time', 'rho'], fusion_source)
                dsmap['core_sources'] = xr.Dataset(coords=cs_coords, data_vars=cs_data_vars, attrs=cs_attrs).drop_duplicates(list(cs_coords.keys()), keep='first')

            # Require EQDSK?
            #if time_eq in data.coords and psi_eq in data.coords and rho_eq in data.coords:
            #    eq_coords = {}
            #    eq_data_vars = {}
            #    eq_attrs: MutableMapping[str, Any] = {}
            #    data = data.swap_dims({time_eq_i: time_eq})
            #    eq_coords['time'] = data.get(time_eq, xr.DataArray()).to_numpy().flatten()
            #    eq_coords['rho'] = data.get(rho_eq, xr.DataArray()).to_numpy().flatten()
            #    omas_tag = 'equilibrium.time_slice.profiles_1d.psi'
            #    if omas_tag in data:
            #        eq_data_vars['profile_conditions.psi'] = (['time', 'rho'], data[omas_tag].to_numpy())
            #    omas_tag = 'equilibrium.time_slice.profiles_1d.q'
            #    if omas_tag in data:
            #        eq_data_vars['profile_conditions.q'] = (['time', 'rho'], data[omas_tag].to_numpy())
            #    dsmap['equilibrium'] = xr.Dataset(coords=eq_coords, data_vars=eq_data_vars, attrs=eq_attrs).drop_duplicates(list(coords.keys()), keep='first')

            if len(dsmap) > 0:
                if 'equilibrium' in dsmap:
                    drop = [
                        'profile_conditions.q',
                    ]
                    dsmap['equilibrium'] = dsmap['equilibrium'].drop_vars(drop, errors='ignore')
                if 'core_profiles' in dsmap:
                    drop = [
                        'profile_conditions.psi',
                    ]
                    dsmap['core_profiles'] = dsmap['core_profiles'].drop_vars(drop, errors='ignore')
                newobj.input = xr.merge([v for k, v in dsmap.items()], join='outer')

        return newobj
