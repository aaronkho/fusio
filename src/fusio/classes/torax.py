import copy
import json
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from .io import io
from ..utils.json_tools import serialize, deserialize
from ..utils.plasma_tools import define_ion_species

logger = logging.getLogger('fusio')


class torax_io(io):

    basevars = {
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
    restartvars = [
        'filename',
        'time',
        'do_restart',
        'stitch',
    ]
    specvars = {
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
    allowed_radiation_species = [
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


    def _unflatten(self, datadict):
        odict = {}
        udict = {}
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


    def read(self, path, side='output'):
        if side == 'input':
            self.input = self._read_torax_file(path)
        else:
            self.output = self._read_torax_file(path)
        #logger.warning(f'{self.format} reading function not defined yet...')


    def write(self, path, side='input', overwrite=False):
        if side == 'input':
            self._write_torax_file(path, self.input, overwrite=overwrite)
        else:
            self._write_torax_file(path, self.output, overwrite=overwrite)
        #if not self.empty:
        #    odict = self._uncompress(self.input)
        #    with open(opath, 'w') as jf:
        #        json.dump(odict, jf, indent=4, default=serialize)
        #    logger.info(f'Saved {self.format} data into {opath.resolve()}')
        #else:
        #    logger.error(f'Attempting to write empty {self.format} class instance... Failed!')


    def _read_torax_file(self, path):
        ds = xr.Dataset()
        if isinstance(path, (str, Path)):
            ipath = Path(path)
            if ipath.exists():
                dt = xr.open_datatree(ipath)
                ds = xr.combine_by_coords([dt[key].to_dataset() for key in dt.groups], compat="override")
                newattrs = {}
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


    def _write_torax_file(self, path, data, overwrite=False):
        if isinstance(path, (str, Path)):
            opath = Path(path)
            if overwrite or not opath.exists():
                if isinstance(data, (xr.Dataset, xr.DataTree)):
                    newattrs = {}
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


    def add_hydrogenic_minority_species(self, sname, sfrac):
        if sname in ['H', 'D', 'T'] and sname not in self.input.get('main_ion', []):
            data = self.input.to_dataset()
            coords = {'main_ion': [sname], 'time': np.atleast_1d(data.get('time').to_numpy())}
            data_vars = {}
            if 'plasma_composition.main_ion' in data:
                total = np.atleast_1d(data.get('plasma_composition.main_ion').sum('main_ion').to_numpy())
                data_vars['plasma_composition.main_ion'] = (['main_ion', 'time'], np.expand_dims(sfrac / (total - sfrac), axis=0))
            newdata = xr.Dataset(coords=coords, data_vars=data_vars)
            self.input = xr.concat([data, newdata], dim='main_ion', data_vars='minimal', coords='different', join='outer')
            if 'plasma_composition.main_ion' in self.input:
                val = self.input['plasma_composition.main_ion']
                newvars = {}
                newvars['plasma_composition.main_ion'] = (['main_ion', 'time'], (val / val.sum('main_ion')).to_numpy())
                self.update_input_data_vars(newvars)


    def set_constant_flat_effective_charge(self, zeff):
        shape = (self.input.get('time').size, self.input.get('rho').size)
        newvars = {}
        newvars['plasma_composition.Z_eff'] = (['time', 'rho'], np.full(shape, float(zeff)))
        self.update_input_data_vars(newvars)


    def add_geometry(self, geotype, geofiles, geodir=None):
        newattrs = {}
        newattrs['use_psi'] = False
        #newattrs['geometry.hires_factor'] = 4
        #newattrs['profile_conditions.initial_psi_from_j'] = True
        #newattrs['profile_conditions.initial_j_is_total_current'] = True
        newattrs['geometry.Ip_from_parameters'] = bool(self.input.attrs.get('profile_conditions.Ip_tot', False))
        newattrs['geometry.geometry_type'] = f'{geotype}'
        if geodir is not None:
            newattrs['geometry.geometry_directory'] = f'{geodir}'
        if isinstance(geofiles, dict):
            geoconfig = {}
            for time, geofile in geofiles.items():
                geotime = {}
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


    def add_pedestal_by_pressure(self, pped, nped, tpedratio, wrho):
        time = self.input.get('time').to_numpy().flatten()
        newvars = {}
        newvars['pedestal.P_ped'] = (['time'], np.zeros_like(time) + pped)
        newvars['pedestal.n_e_ped'] = (['time'], np.zeros_like(time) + nped)
        newvars['pedestal.T_i_T_e_ratio'] = (['time'], np.zeros_like(time) + tpedratio)
        newvars['pedestal.rho_norm_ped_top'] = (['time'], np.zeros_like(time) + wrho)
        self.update_input_data_vars(newvars)
        newattrs = {}
        newattrs['pedestal.set_pedestal'] = True
        newattrs['pedestal.model_name'] = 'set_P_ped_n_ped'
        newattrs['pedestal.n_e_ped_is_fGW'] = False
        newattrs['transport.smooth_everywhere'] = False
        newattrs['numerics.adaptive_T_source_prefactor'] = 1.0e10
        newattrs['numerics.adaptive_n_source_prefactor'] = 1.0e8
        self.update_input_attrs(newattrs)


    def add_pedestal_by_temperature(self, nped, tped, wrho):
        time = self.input.get('time').to_numpy().flatten()
        tref = 1.0e3
        newvars = {}
        newvars['pedestal.n_e_ped'] = (['time'], np.zeros_like(time) + nped)
        newvars['pedestal.T_e_ped'] = (['time'], np.zeros_like(time) + (tped / tref))
        newvars['pedestal.T_i_ped'] = (['time'], np.zeros_like(time) + (tped / tref))
        newvars['pedestal.rho_norm_ped_top'] = (['time'], np.zeros_like(time) + wrho)
        self.update_input_data_vars(newvars)
        newattrs = {}
        newattrs['pedestal.set_pedestal'] = True
        newattrs['pedestal.model_name'] = 'set_T_ped_n_ped'
        newattrs['pedestal.n_e_ped_is_fGW'] = False
        newattrs['transport.smooth_everywhere'] = False
        newattrs['numerics.adaptive_T_source_prefactor'] = 1.0e10
        newattrs['numerics.adaptive_n_source_prefactor'] = 1.0e8
        self.update_input_attrs(newattrs)


    def add_pedestal_exponential_transport(self, chiscale, chidecay, dscale, ddecay):
        time = self.input.get('time').to_numpy().flatten()
        newcoords = {}
        newvars = {}
        newattrs = {}
        wrho_array = self.input.get('pedestal.rho_norm_ped_top', None)
        if self.input.attrs.get('transport.model_name', '') == 'combined' and wrho_array is not None:
            models = self.input.attrs.get('map_combined_models', {})
            prefix = f'transport.transport_models.{len(models):d}'
            newvars[f'{prefix}.rho_min'] = (['time'], wrho_array.to_numpy())
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
            # Improper form given base xarray structure, but necessary for now due to disjointed rho grid
            #newattrs[f'{prefix}.chi_i'] = {0.0: (xrho, chirho)}
            #newattrs[f'{prefix}.chi_e'] = {0.0: (xrho, chirho)}
            #newattrs[f'{prefix}.D_e'] = {0.0: (xrho, drho)}
            #newattrs[f'{prefix}.V_e'] = {0.0: (xrho, np.zeros_like(xrho))}
            models.update({'constant_ped': len(models)})
            newattrs['map_combined_models'] = models
        self.update_input_coords(newcoords)
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def add_neoclassical_transport(self):
        newattrs = {}
        newattrs['neoclassical.conductivity.model_name'] = 'sauter'
        self.update_input_attrs(newattrs)
        self.add_neoclassical_bootstrap_current()


    def add_neoclassical_bootstrap_current(self):
        newattrs = {}
        newattrs['neoclassical.bootstrap_current.model_name'] = 'sauter'
        newattrs['neoclassical.bootstrap_current.bootstrap_multiplier'] = 1.0
        self.update_input_attrs(newattrs)
        if 'sources.generic_current.prescribed_values' in self.input and 'profile_conditions.j_bootstrap' in self.input:
            self.input['sources.generic_current.prescribed_values'] = self.input['sources.generic_current.prescribed_values'] - self.input['profile_conditions.j_bootstrap']


    def add_combined_transport(self):
        newattrs = {}
        newattrs['transport.model_name'] = 'combined'
        newattrs['transport.chi_min'] = 0.05
        newattrs['transport.chi_max'] = 100.0
        newattrs['transport.D_e_min'] = 0.05
        newattrs['transport.D_e_max'] = 100.0
        newattrs['transport.V_e_min'] = -50.0
        newattrs['transport.V_e_max'] = 50.0
        newattrs['transport.smoothing_width'] = 0.1
        newattrs['transport.smooth_everywhere'] = (not self.input.attrs.get('pedestal.set_pedestal', False))
        #newattrs['transport.predict_pedestal'] = (self.input.attrs.get('pedestal.set_pedestal', False))
        #if 'transport.apply_inner_patch' not in self.input.attrs:
        #    newattrs['transport.apply_inner_patch'] = {0.0: False}
        #if 'transport.apply_outer_patch' not in self.input.attrs:
        #    newattrs['transport.apply_outer_patch'] = {0.0: False}
        newattrs['map_combined_models'] = self.input.attrs.get('map_combined_models', {})
        self.update_input_attrs(newattrs)


    def add_qualikiz_transport(self):
        newvars = {}
        newattrs = {}
        prefix = 'transport'
        if self.input.attrs.get('transport.model_name', '') == 'combined':
            time = self.input.get('time').to_numpy().flatten()
            models = self.input.attrs.get('map_combined_models', {})
            prefix = f'transport.transport_models.{len(models):d}'
            #newattrs[f'{prefix}.rho_min'] = {0.0: 0.15}
            if self.input.get('pedestal.rho_norm_ped_top', None) is not None:
                newvars[f'{prefix}.rho_max'] = (['time'], self.input['pedestal.rho_norm_ped_top'].to_numpy())
            newvars[f'{prefix}.rho_min'] = (['time'], np.zeros_like(time) + 0.45)
            newattrs[f'{prefix}.apply_inner_patch'] = False
            newattrs[f'{prefix}.apply_outer_patch'] = False
            models.update({'qualikiz': len(models)})
            newattrs['map_combined_models'] = models
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
        newattrs['transport.smooth_everywhere'] = (not self.input.attrs.get('pedestal.set_pedestal', False))
        #if 'transport.apply_inner_patch' not in self.input.attrs:
        #    newattrs['transport.apply_inner_patch'] = {0.0: False}
        #if 'transport.apply_outer_patch' not in self.input.attrs:
        #    newattrs['transport.apply_outer_patch'] = {0.0: False}
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def set_qualikiz_model_path(self, path):
        newattrs = {}
        if self.input.attrs.get('transport.model_name', '') == 'combined':
            models = self.input.attrs.get('map_combined_models', {})
            for n in range(len(models)):
                if self.input.attrs.get(f'transport.transport_models.{n:d}.model_name', '') == 'qualikiz':
                    newattrs['TORAX_QLK_EXEC_PATH'] = f'{path}'  # Is this still necessary?
        elif self.input.attrs.get('transport.model_name', '') == 'qualikiz':
            newattrs['TORAX_QLK_EXEC_PATH'] = f'{path}'  # Is this still necessary?
        self.update_input_attrs(newattrs)


    def add_qlknn_transport(self):
        newvars = {}
        newattrs = {}
        prefix = 'transport'
        if self.input.attrs.get('transport.model_name', '') == 'combined':
            models = self.input.attrs.get('map_combined_models', {})
            prefix = f'transport.transport_models.{len(models):d}'
            #newattrs[f'{prefix}.rho_min'] = {0.0: 0.15}
            if self.input.get('pedestal.rho_norm_ped_top', None) is not None:
                newvars[f'{prefix}.rho_max'] = (['time'], self.input['pedestal.rho_norm_ped_top'].to_numpy())
            newattrs[f'{prefix}.apply_inner_patch'] = False
            newattrs[f'{prefix}.apply_outer_patch'] = False
            models.update({'qlknn': len(models)})
            newattrs['map_combined_models'] = models
        newattrs[f'{prefix}.model_name'] = 'qlknn'
        newattrs[f'{prefix}.model_path'] = ''
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
        newattrs['transport.smooth_everywhere'] = (not self.input.attrs.get('pedestal.set_pedestal', False))
        #if 'transport.apply_inner_patch' not in self.input.attrs:
        #    newattrs['transport.apply_inner_patch'] = {0.0: False}
        #if 'transport.apply_outer_patch' not in self.input.attrs:
        #    newattrs['transport.apply_outer_patch'] = {0.0: False}
        self.update_input_data_vars(newvars)
        self.update_input_attrs(newattrs)


    def set_qlknn_model_path(self, path):
        newattrs = {}
        if self.input.attrs.get('transport.model_name', '') == 'combined':
            models = self.input.attrs.get('map_combined_models', {})
            for n in range(len(models)):
                if self.input.attrs.get(f'transport.transport_models.{n:d}.model_name', '') == 'qlknn':
                    newattrs[f'transport.transport_models.{n:d}.model_path'] = f'{path}'
        if self.input.attrs.get('transport.model_name', '') == 'qlknn':
            newattrs['transport.model_path'] = f'{path}'
        self.update_input_attrs(newattrs)


    def add_transport_inner_patch(self, de, ve, chii, chie, rho, tstart=None, tend=None):
        time = self.input.get('time').to_numpy().flatten()
        trigger = np.isfinite(time)
        if isinstance(tstart, (float, int)):
            trigger &= (time >= tstart)
        if isinstance(tend, (float, int)):
            trigger &= (time <= tend)
        newvars = {}
        newvars['transport.apply_inner_patch'] = (['time'], trigger)
        newvars['transport.D_e_inner'] = (['time'], np.zeros_like(time) + de)
        newvars['transport.V_e_inner'] = (['time'], np.zeros_like(time) + ve)
        newvars['transport.chi_i_inner'] = (['time'], np.zeros_like(time) + chii)
        newvars['transport.chi_e_inner'] = (['time'], np.zeros_like(time) + chie)
        self.update_input_data_vars(newvars)
        newattrs = {}
        newattrs['transport.rho_inner'] = float(rho)
        self.update_input_attrs(newattrs)


    def add_transport_outer_patch(self, de, ve, chii, chie, rho, tstart=None, tend=None):
        time = self.input.get('time').to_numpy().flatten()
        trigger = np.isfinite(time)
        if isinstance(tstart, (float, int)):
            trigger &= (time >= tstart)
        if isinstance(tend, (float, int)):
            trigger &= (time <= tend)
        newvars = {}
        newvars['transport.apply_outer_patch'] = (['time'], trigger)
        newvars['transport.D_e_outer'] = (['time'], np.zeros_like(time) + de)
        newvars['transport.V_e_outer'] = (['time'], np.zeros_like(time) + ve)
        newvars['transport.chi_i_outer'] = (['time'], np.zeros_like(time) + chii)
        newvars['transport.chi_e_outer'] = (['time'], np.zeros_like(time) + chie)
        self.update_input_data_vars(newvars)
        newattrs = {}
        newattrs['transport.rho_outer'] = float(rho)
        self.update_input_attrs(newattrs)


    def reset_mhd_sawtooth_trigger(self):
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


    def set_mhd_sawtooth_trigger(self, rmin, scrit, flat=1.01, rmult=1.1, deltat=1.0e-3):
        time = self.input.get('time').to_numpy().flatten()
        newvars = {}
        newvars['mhd.sawtooth.trigger_model.minimum_radius'] = (['time'], np.zeros_like(time) + rmin)
        newvars['mhd.sawtooth.trigger_model.s_critical'] = (['time'], np.zeros_like(time) + scrit)
        newvars['mhd.sawtooth.redistribution_model.flattening_factor'] = (['time'], np.zeros_like(time) + flat)
        newvars['mhd.sawtooth.redistribution_model.mixing_radius_multiplier'] = (['time'], np.zeros_like(time) + rmult)
        self.update_input_data_vars(newvars)
        newattrs = {}
        newattrs['mhd.sawtooth.crash_step_duration'] = float(deltat)
        newattrs['mhd.sawtooth.trigger_model.model_name'] = 'simple'
        newattrs['mhd.sawtooth.redistribution_model.model_name'] = 'simple'
        self.update_input_attrs(newattrs)


    def set_exchange_source(self):
        newattrs = {}
        newattrs['sources.ei_exchange.mode'] = 'MODEL_BASED'
        newattrs['sources.ei_exchange.Qei_multiplier'] = 1.0
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.ei_exchange.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)


    def set_ohmic_source(self):
        newattrs = {}
        newattrs['sources.ohmic.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.ohmic.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)


    def set_fusion_source(self):
        newattrs = {}
        newattrs['sources.fusion.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.fusion.prescribed_values',
        ]
        self.delete_input_data_vars(delvars)


    def reset_gas_puff_source(self):
        newattrs = {}
        newattrs['sources.gas_puff.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delvars = [
            'sources.gas_puff.puff_decay_length',
            'sources.gas_puff.S_total',
        ]
        self.delete_input_data_vars(delvars)


    def set_gas_puff_source(self, length, total):
        time = self.input.get('time').to_numpy().flatten()
        newvars = {}
        newvars['sources.gas_puff.puff_decay_length'] = (['time'], np.zeros_like(time) + length)
        newvars['sources.gas_puff.S_total'] = (['time'], np.zeros_like(time) + total)
        self.update_input_data_vars(newvars)
        newattrs = {}
        newattrs['sources.gas_puff.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)


    def set_bootstrap_current_source(self):
        self.add_neoclassical_bootstrap_current()


    def reset_bremsstrahlung_source(self):
        newattrs = {}
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


    def set_bremsstrahlung_source(self):
        self.reset_bremsstrahlung_source()
        newattrs = {}
        newattrs['sources.bremsstrahlung.mode'] = 'MODEL_BASED'
        newattrs['sources.bremsstrahlung.use_relativistic_correction'] = True
        self.update_input_attrs(newattrs)


    def reset_line_radiation_source(self):
        newattrs = {}
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


    def set_mavrin_line_radiation_source(self):
        self.reset_line_radiation_source()
        newattrs = {}
        newattrs['sources.impurity_radiation.mode'] = 'MODEL_BASED'
        newattrs['sources.impurity_radiation.model_name'] = 'mavrin_fit'
        newattrs['sources.impurity_radiation.radiation_multiplier'] = 1.0
        self.update_input_attrs(newattrs)
        # Mavrin polynomial model includes Bremsstrahlung so zero that out as well
        self.reset_bremsstrahlung_source()


    def reset_synchrotron_source(self):
        newattrs = {}
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


    def set_synchrotron_source(self):
        self.reset_synchrotron_source()
        newattrs = {}
        newattrs['sources.cyclotron_radiation.mode'] = 'MODEL_BASED'
        newattrs['sources.cyclotron_radiation.wall_reflection_coeff'] = 0.9
        newattrs['sources.cyclotron_radiation.beta_min'] = 0.5
        newattrs['sources.cyclotron_radiation.beta_max'] = 8.0
        newattrs['sources.cyclotron_radiation.beta_grid_size'] = 32
        self.update_input_attrs(newattrs)


    def add_toricnn_icrh_source(self, freq, mfrac, total, iwall=1.24, owall=2.43):
        time = self.input.get('time').to_numpy().flatten()
        newvars = {}
        newvars['sources.icrh.frequency'] = (['time'], np.zeros_like(time) + freq)
        newvars['sources.icrh.minority_concentration'] = (['time'], np.zeros_like(time) + mfrac)
        newvars['sources.icrh.P_total'] = (['time'], np.zeros_like(time) + total)
        self.update_input_data_vars(newvars)
        newattrs = {}
        newattrs['sources.icrh.mode'] = 'MODEL_BASED'
        newattrs['sources.icrh.model_path'] = ''
        newattrs['sources.icrh.wall_inner'] = iwall
        newattrs['sources.icrh.wall_outer'] = owall
        self.update_input_attrs(newattrs)


    def set_toricnn_model_path(self, path):
        newattrs = {}
        if self.input.attrs.get('sources.icrh.mode', 'ZERO') == 'MODEL_BASED':
            newattrs['sources.icrh.model_path'] = f'{path}'
        self.update_input_attrs(newattrs)


    def reset_generic_heat_source(self):
        newattrs = {}
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


    def reset_generic_particle_source(self):
        newattrs = {}
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


    def reset_generic_current_source(self):
        newattrs = {}
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


    def set_gaussian_generic_heat_source(self, mu, sigma, total, efrac=0.5, afrac=1.0):
        self.reset_generic_heat_source()
        time = self.input.get('time').to_numpy().flatten()
        newvars = {}
        newvars['sources.generic_heat.gaussian_location'] = (['time'], np.zeros_like(time) + mu)
        newvars['sources.generic_heat.gaussian_width'] = (['time'], np.zeros_like(time) + sigma)
        newvars['sources.generic_heat.P_total'] = (['time'], np.zeros_like(time) + total)
        newvars['sources.generic_heat.electron_heat_fraction'] = (['time'], np.zeros_like(time) + efrac)
        newvars['sources.generic_heat.absorption_fraction'] = (['time'], np.zeros_like(time) + afrac)
        self.update_input_data_vars(newvars)
        newattrs = {}
        newattrs['sources.generic_heat.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)


    def set_gaussian_generic_particle_source(self, mu, sigma, total):
        self.reset_generic_particle_source()
        time = self.input.get('time').to_numpy().flatten()
        newvars = {}
        newvars['sources.generic_particle.deposition_location'] = (['time'], np.zeros_like(time) + mu)
        newvars['sources.generic_particle.particle_width'] = (['time'], np.zeros_like(time) + sigma)
        newvars['sources.generic_particle.S_total'] = (['time'], np.zeros_like(time) + total)
        self.update_input_data_vars(newvars)
        newattrs = {}
        newattrs['sources.generic_particle.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)


    def set_gaussian_generic_current_source(self, mu, sigma, total):
        self.reset_generic_current_source()
        time = self.input.get('time').to_numpy().flatten()
        newattrs = {}
        newattrs['sources.generic_current.mode'] = 'MODEL_BASED'
        newattrs['sources.generic_current.gaussian_location'] = (['time'], np.zeros_like(time) + mu)
        newattrs['sources.generic_current.gaussian_width'] = (['time'], np.zeros_like(time) + sigma)
        newattrs['sources.generic_current.I_generic'] = (['time'], np.zeros_like(time) + total)
        newattrs['sources.generic_current.fraction_of_total_current'] = (['time'], np.ones_like(time))
        newattrs['sources.generic_current.use_absolute_current'] = True
        self.update_input_attrs(newattrs)


    def set_prescribed_generic_heat_source(self, rho, eheat, iheat):
        self.reset_generic_heat_source()
        time = self.input.get('time').to_numpy().flatten()
        newrho = self.input.get('rho').to_numpy().flatten()
        neweheat = np.interp(newrho, rho, eheat)
        newiheat = np.interp(newrho, rho, iheat)
        newattrs = {}
        newattrs['sources.generic_heat.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars = {}
        newvars['sources.generic_heat.prescribed_values_el'] = (['time', 'rho'], np.repeat(np.atleast_2d(neweheat), len(time), axis=0))
        newvars['sources.generic_heat.prescribed_values_ion'] = (['time', 'rho'], np.repeat(np.atleast_2d(newiheat), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def set_prescribed_generic_particle_source(self, rho, particle):
        self.reset_generic_particle_source()
        time = self.input.get('time').to_numpy().flatten()
        newrho = self.input.get('rho').to_numpy().flatten()
        newparticle = np.interp(newrho, rho, particle)
        newattrs = {}
        newattrs['sources.generic_particle.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars = {}
        newvars['sources.generic_particle.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newparticle), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def set_prescribed_generic_current_source(self, rho, current):
        self.reset_generic_current_source()
        newrho = self.input.get('rho').to_numpy().flatten()
        newcurrent = np.interp(newrho, rho, current)
        newattrs = {}
        newattrs['sources.generic_current.mode'] = 'PRESCRIBED'
        self.update_input_attrs(newattrs)
        newvars = {}
        newvars['sources.generic_current.prescribed_values'] = (['time', 'rho'], np.repeat(np.atleast_2d(newcurrent), len(time), axis=0))
        self.update_input_data_vars(newvars)


    def add_fixed_linear_solver(self, dt_fixed=None, single=False):
        newattrs = {}
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


    def add_adaptive_linear_solver(self, dt_mult=None, single=False):
        newattrs = {}
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


    def add_fixed_newton_raphson_solver(self, dt_fixed=None, single=False):
        newattrs = {}
        newattrs['solver.solver_type'] = 'newton_raphson'
        newattrs['solver.theta_implicit'] = 1.0
        newattrs['solver.use_predictor_corrector'] = True
        newattrs['solver.n_corrector_steps'] = 10
        newattrs['solver.use_pereverzev'] = True
        newattrs['solver.chi_pereverzev'] = 30.0
        newattrs['solver.D_pereverzev'] = 15.0
        newattrs['solver.log_iterations'] = False
        newattrs['solver.initial_guess_mode'] = 1  # 0 = x_old, 1 = linear
        newattrs['solver.residual_tol'] = 1.0e-5 if not single else 1.0e-3
        newattrs['solver.residual_coarse_tol'] = 1.0e-2 if not single else 1.0e-1
        newattrs['solver.delta_reduction_factor'] = 0.5
        newattrs['solver.n_max_iterations'] = 30
        newattrs['solver.tau_min'] = 0.01
        newattrs['time_step_calculator.calculator_type'] = 'fixed'
        newattrs['time_step_calculator.tolerance'] = 1.0e-7 if not single else 1.0e-5
        newattrs['numerics.fixed_dt'] = float(dt_fixed) if isinstance(dt_fixed, (float, int)) else 1.0e-1
        self.update_input_attrs(newattrs)


    def add_adaptive_newton_raphson_solver(self, dt_mult=None, single=False):
        newattrs = {}
        newattrs['solver.solver_type'] = 'newton_raphson'
        newattrs['solver.theta_implicit'] = 1.0
        newattrs['solver.use_predictor_corrector'] = True
        newattrs['solver.n_corrector_steps'] = 10
        newattrs['solver.use_pereverzev'] = True
        newattrs['solver.chi_pereverzev'] = 30.0
        newattrs['solver.D_pereverzev'] = 15.0
        newattrs['solver.log_iterations'] = False
        newattrs['solver.initial_guess_mode'] = 1  # 0 = x_old, 1 = linear
        newattrs['solver.residual_tol'] = 1.0e-5 if not single else 1.0e-3
        newattrs['solver.residual_coarse_tol'] = 1.0e-2 if not single else 1.0e-1
        newattrs['solver.delta_reduction_factor'] = 0.5
        newattrs['solver.n_max_iterations'] = 30
        newattrs['solver.tau_min'] = 0.01
        newattrs['time_step_calculator.calculator_type'] = 'chi'
        newattrs['time_step_calculator.tolerance'] = 1.0e-7 if not single else 1.0e-5
        newattrs['numerics.chi_timestep_prefactor'] = float(dt_mult) if isinstance(dt_mult, (float, int)) else 50.0
        self.update_input_attrs(newattrs)


    def set_numerics(self, t_initial, t_final, eqs=['te', 'ti', 'ne', 'j']):
        newattrs = {}
        newattrs['geometry.n_rho'] = 25
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


    def print_summary(self):
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
            for key, specs in fields.items():
                var, scale, units = specs
                val = data[var] / scale
                print(f'{key:10}: {val:.2f} {units}')
            for key, specs in radial_fields.items():
                var, idx, scale, units = specs
                if isinstance(scale, str):
                    val = data.isel(rho_norm=idx)[var] / data[scale]
                else:
                    val = data.isel(rho_norm=idx)[var] / scale
                print(f'{key:10}: {val:.2f} {units}')


    def to_dict(self):
        datadict = {}
        ds = self.input
        datadict.update(ds.attrs)
        for key in ds.data_vars:
            dims = ds[key].dims
            ttag = 'time' if 'time' in dims else None
            if ttag is None:
                for dim in dims:
                    if dim.startswith('time_'):
                        ttag = dim
                        break
            rtag = 'rho' if 'rho' in dims else None
            if rtag is None:
                for dim in dims:
                    if dim.startswith('rho_'):
                        rtag = dim
                        break
            if ttag is not None and ttag in ds[key].dims:
                time = ds[ttag].to_numpy().flatten()
                time_dependent_var = None
                if 'main_ion' in ds[key].dims:
                    time_dependent_var = {}
                    for species in ds['main_ion'].to_numpy().flatten():
                        da = ds[key].dropna(ttag).sel(main_ion=species)
                        if rtag in da:
                            da = da.rename({rtag: 'rho_norm'})
                        if da.size > 0:
                            #time_dependent_var[str(species)] = {float(t): v for t, v in zip(da[ttag].to_numpy().flatten(), da.to_numpy().flatten())}
                            time_dependent_var[str(species)] = da
                elif 'impurity' in ds[key].dims:
                    time_dependent_var = {}
                    for species in ds['impurity'].to_numpy().flatten():
                        da = ds[key].dropna(ttag).sel(impurity=species)
                        if rtag in da:
                            da = da.rename({rtag: 'rho_norm'})
                        if da.size > 0:
                            time_dependent_var[str(species)] = da
                elif rtag is not None and rtag in ds[key].dims:
                    #for ii in range(len(time)):
                    #    da = ds[key].isel(time=ii).dropna(rtag)
                    #    if da.size > 0:
                    #        time_dependent_var[float(time[ii])] = (da[rtag].to_numpy().flatten(), da.to_numpy().flatten())
                    da = ds[key].dropna(ttag).rename({rtag: 'rho_norm'})
                    if da.size > 0:
                        time_dependent_var = da
                else:
                    #for ii in range(len(time)):
                    #    da = ds[key].isel(time=ii)
                    #    if np.all(np.isfinite(da.values)):
                    #        time_dependent_var[float(time[ii])] = float(da.to_numpy())
                    da = ds[key].dropna(ttag)
                    if da.size > 0:
                        time_dependent_var = da
                if time_dependent_var is not None:
                    datadict[key] = time_dependent_var
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
        use_psi = datadict.pop('use_psi', True)
        if not use_psi:
            datadict.pop('profile_conditions.psi', None)
        use_generic_heat = datadict.pop('use_generic_heat', True)
        if not use_generic_heat:
            self.reset_generic_heat_source()
        use_generic_particle = datadict.pop('use_generic_particle', True)
        if not use_generic_particle:
            self.reset_generic_particle_source()
        use_generic_current = datadict.pop('use_generic_current', True)
        if not use_generic_current:
            self.reset_generic_current_source()
        use_fusion = datadict.pop('use_fusion', True)
        if not use_fusion:
            self.reset_fusion_source()
        datadict.pop('profile_conditions.n_i', None)
        datadict.pop('profile_conditions.q', None)
        datadict.pop('profile_conditions.j_ohmic', None)
        datadict.pop('profile_conditions.j_bootstrap', None)
        datadict.pop('TORAX_QLK_EXEC_PATH', None)
        if 'pedestal.set_pedestal' not in datadict:
            datadict['pedestal.set_pedestal'] = False
        return self._unflatten(datadict)


    @classmethod
    def from_file(cls, path=None, input=None, output=None):
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


    @classmethod
    def from_gacode(cls, obj, side='output', n=0):
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            if 'n' in data.coords and 'rho' in data.coords:
                coords = {}
                data_vars = {}
                attrs = {}
                data = data.sel(n=n)
                time = data.get('time', 0.0)
                attrs['numerics.t_initial'] = float(time)
                coords['time'] = np.array([time])
                coords['rho'] = data['rho'].to_numpy().flatten()
                if 'z' in data and 'name' in data and 'type' in data and 'ni' in data:
                    species = []
                    density = []
                    nfilt = (np.isclose(data['z'], 1.0) & (['fast' not in v for v in data['type'].to_numpy().flatten()]))
                    if np.any(nfilt):
                        namelist = data['name'].to_numpy()[nfilt].tolist()
                        nfuelsum = data['ni'].sel(name=namelist).sum('name')
                        for ii in range(len(namelist)):
                            sdata = data['ni'].sel(name=namelist[ii])
                            species.append(namelist[ii])
                            density.append(np.expand_dims((sdata / nfuelsum).mean('rho').to_numpy().flatten(), axis=0))
                        coords['main_ion'] = species
                    else:
                        species = ['D']
                        density = [np.atleast_2d([1.0])]
                    coords['main_ion'] = species
                    data_vars['plasma_composition.main_ion'] = (['main_ion', 'time'], np.concatenate(density, axis=0))
                if 'z' in data and 'mass' in data and 'ni' in data and 'ne' in data:
                    nfilt = (~np.isclose(data['z'], 1.0))
                    zeff = xr.ones_like(data['ne'])
                    if np.any(nfilt):
                        namelist = data['name'].to_numpy()[nfilt].tolist()
                        impcomp = {}
                        zeff = xr.zeros_like(data['ne'])
                        nsum = xr.zeros_like(data['ne'])
                        for ii in range(len(data['name'])):
                            sdata = data.isel(name=ii)
                            nz = sdata['ni']
                            if sdata['name'] in namelist and 'therm' in str(sdata['type'].to_numpy()):
                                sname = str(sdata['name'].to_numpy())
                                scharge = float(sdata['z'].to_numpy())
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
                                    nz = nz * scharge / newsz
                                    if sn == 'He':
                                        sname = 'He4'
                                # Intentional mismatch between composition and Zeff densities to handle species changes for radiation calculation
                                impcomp[sname] = nz
                                nsum += nz
                            zeff += sdata['ni'] * sdata['z'] ** 2.0 / data['ne']
                        total = 0.0
                        impcoord = []
                        impfracs = []
                        for key in impcomp:
                            impcomp[key] = (impcomp[key] / nsum).mean('rho')
                            total += impcomp[key]
                        for key in impcomp:
                            impval = (impcomp[key] / total).to_numpy().flatten()
                            impcoord.append(key)
                            impfracs.append(np.expand_dims(impval, axis=0))
                        if impfracs is None:
                            impcoord = ['Ne']
                            impfracs = [np.atleast_2d([1.0])]
                        if 'z_eff' in data:
                            zeff = data['z_eff']
                        coords['impurity'] = impcoord
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
                    data_vars['sources.generic_heat.prescribed_values_el'] = (['time', 'rho'], np.expand_dims(external_ion_heat_source, axis=0))
                    data_vars['sources.generic_heat.prescribed_values_ion'] = (['time', 'rho'], np.expand_dims(external_el_heat_source, axis=0))
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
    def from_omas(cls, obj, side='output', time=None):
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            dsvec = []
            struct_index = {}
            if 'time_cp' in data.coords and 'rho_cp' in data.coords:
                coords = {}
                data_vars = {}
                attrs = {}
                coords['time'] = data.get('time_cp').to_numpy().flatten()
                coords['rho'] = data.get('rho_cp').to_numpy().flatten()
                #if 'ion_cp' in data:
                #    #sn, sa, sz = define_ion_species(short_name=sname)
                #    coords['main_ion'] = data.get('ion_cp').to_numpy().flatten()
                attrs['numerics.t_initial'] = coords['time'][0]
                omas_tag = 'core_profiles.profiles_1d.ion.density'
                if omas_tag in data and 'ion_cp' in data:
                    namelist = data.get('ion_cp').to_numpy().tolist()
                    mainlist = []
                    for ii in range(len(namelist)):
                        if namelist[ii] in ['H', 'D', 'T']:
                            mainlist.append(ii)
                    maincoord = []
                    mainfracs = []
                    total_main = data.isel(ion_cp=mainlist)[omas_tag].sum('ion_cp')
                    for ii in mainlist:
                        maincoord.append(namelist[ii])
                        mainfracs.append(np.expand_dims(np.atleast_1d((data.isel(ion_cp=ii)[omas_tag] / total_main).mean('rho_cp').to_numpy()), axis=0))
                    if len(mainlist) == 0:
                        maincoord = ['D']
                        mainfracs = [np.atleast_2d([1.0])]
                    coords['main_ion'] = maincoord
                    data_vars['plasma_composition.main_ion'] = (['main_ion', 'time'], np.concatenate(mainfracs, axis=0))
                if omas_tag in data and 'core_profiles.profiles_1d.electrons.density' in data:
                    namelist = data.get('ion_cp').to_numpy().tolist()
                    implist = []
                    zeff = xr.zeros_like(data['core_profiles.profiles_1d.electrons.density'])
                    nsum = xr.zeros_like(data['core_profiles.profiles_1d.electrons.density'])
                    for ii in range(len(namelist)):
                        if namelist[ii] not in ['H', 'D', 'T']:
                            implist.append(ii)
                    impcomp = {}
                    for ii in range(len(namelist)):
                        sname = namelist[ii]
                        sn, sa, sz = define_ion_species(short_name=sname)
                        nz = data.isel(ion_cp=ii)[omas_tag]
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
                            nz = nz * sz / newsz
                            if sn == 'He':
                                sname = 'He4'
                        # Intentional mismatch between composition and Zeff densities to handle species changes for radiation calculation
                        impcomp[sname] = nz
                        nsum += nz
                        zeff += data.isel(ion_cp=ii)[omas_tag] * sz ** 2.0 / data['core_profiles.profiles_1d.electrons.density']
                    total = 0.0
                    for key in impcomp:
                        impcomp[key] = (impcomp[key] / nsum).mean('rho_cp')
                        total += impcomp[key]
                    impcoord = []
                    impfracs = []
                    for key in impcomp:
                        impcoord.append(key)
                        impfracs.append(np.expand_dims(np.atleast_1d((impcomp[key] / total).to_numpy()), axis=0))
                    if not impcomp:
                        impcoord = ['Ne']
                        impfracs = [np.atleast_2d([1.0])]
                    coords['impurity'] = impcoord
                    data_vars['plasma_composition.impurity'] = (['impurity', 'time'], np.concatenate(impfracs, axis=0))
                    data_vars['plasma_composition.Z_eff'] = (['time', 'rho'], zeff.to_numpy())
                omas_tag = 'core_profiles.global_quantities.ip'
                if omas_tag in data:
                    data_vars['profile_conditions.Ip'] = (['time'], data[omas_tag].to_numpy())
                omas_tag = 'core_profiles.profiles_1d.electrons.density'
                if omas_tag in data:
                    data_vars['profile_conditions.n_e'] = (['time', 'rho'], data[omas_tag].to_numpy())
                    attrs['profile_conditions.normalize_n_e_to_nbar'] = False
                    attrs['profile_conditions.n_e_nbar_is_fGW'] = False
                omas_tag = 'core_profiles.profiles_1d.electrons.temperature'
                if omas_tag in data:
                    data_vars['profile_conditions.T_e'] = (['time', 'rho'], 1.0e-3 * data[omas_tag].to_numpy())
                omas_tag = 'core_profiles.profiles_1d.ion.temperature'
                if omas_tag in data:
                    data_vars['profile_conditions.T_i'] = (['time', 'rho'], 1.0e-3 * data[omas_tag].mean('ion_cp').to_numpy()) if 'ion_cp' in data else (['time', 'rho'], data[omas_tag].to_numpy())
                omas_tag = 'core_profiles.profiles_1d.grid.psi'
                if omas_tag in data:
                    data_vars['profile_conditions.psi'] = (['time', 'rho'], data[omas_tag].to_numpy())
                omas_tag = 'core_profiles.profiles_1d.q'
                if omas_tag in data:
                    data_vars['profile_conditions.q'] = (['time', 'rho'], data[omas_tag].to_numpy())
                omas_tag = 'core_profiles.global_quantities.v_loop'
                if omas_tag in data:
                    data_vars['profile_conditions.v_loop_lcfs'] = (['time'], data[omas_tag].to_numpy())
                    attrs['profile_conditions.use_v_loop_lcfs_boundary_condition'] = False
                #core_profiles.profiles_1d.conductivity_parallel              (time_cp, rho_cp)
                #core_profiles.profiles_1d.current_parallel_inside            (time_cp, rho_cp)
                #core_profiles.profiles_1d.j_tor                              (time_cp, rho_cp)
                #core_profiles.profiles_1d.j_total                            (time_cp, rho_cp)
                dsvec.append(xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs).drop_duplicates(list(coords.keys()), keep='first'))
                struct_index['core_profiles'] = len(dsvec) - 1
            if 'time_cs' in data.coords and 'rho_cs' in data.coords:
                coords = {}
                data_vars = {}
                attrs = {}
                coords['time'] = data.get('time_cs').to_numpy().flatten()
                coords['rho'] = data.get('rho_cs').to_numpy().flatten()
                # Place the sources
                external_el_heat_source = None
                external_ion_heat_source = None
                external_particle_source = None
                external_current_source = None
                fusion_source = None
                omas_tag = 'core_sources.ohmic.profiles_1d.electrons.energy'
                if omas_tag in data and np.abs(data[omas_tag]).sum() != 0.0:
                    attrs['sources.ohmic.mode'] = 'PRESCRIBED'
                    data_vars['sources.ohmic.prescribed_values'] = (['time', 'rho'], data[omas_tag].to_numpy())
                omas_tag = 'core_sources.line_radiation.profiles_1d.electrons.energy'
                if omas_tag in data and np.abs(data[omas_tag]).sum() != 0.0:
                    attrs['sources.impurity_radiation.mode'] = 'PRESCRIBED'
                    data_vars['sources.impurity_radiation.prescribed_values'] = (['time', 'rho'], data[omas_tag].to_numpy())
                omas_tag = 'core_sources.bremsstrahlung.profiles_1d.electrons.energy'
                if omas_tag in data and np.abs(data[omas_tag]).sum() != 0.0:
                    attrs['sources.bremsstrahlung.mode'] = 'PRESCRIBED'
                    data_vars['sources.bremsstrahlung.prescribed_values'] = (['time', 'rho'], data[omas_tag].to_numpy())
                omas_tag = 'core_sources.synchrotron.profiles_1d.electrons.energy'
                omas_tag = 'core_sources.ic.profiles_1d.electrons.energy'
                if omas_tag in data and np.abs(data[omas_tag]).sum() != 0.0:
                    if external_el_heat_source is None:
                        external_el_heat_source = np.zeros_like(data[omas_tag].to_numpy())
                    external_el_heat_source += 1.0e6 * data[omas_tag].to_numpy()
                omas_tag = 'core_sources.ic.profiles_1d.total_ion_energy'
                if omas_tag in data and np.abs(data[omas_tag]).sum() != 0.0:
                    if external_ion_heat_source is None:
                        external_ion_heat_source = np.zeros_like(data[omas_tag].to_numpy())
                    external_ion_heat_source += 1.0e6 * data[omas_tag].to_numpy()
                omas_tag = 'core_sources.ic.global_quantities.power'
                #if omas_tag in data and np.abs(data[omas_tag]).sum() != 0.0:
                #    data_vars['sources.icrh.P_total'] = (['time'], data[omas_tag].to_numpy().flatten())
                omas_tag = 'core_sources.fusion.profiles_1d.electrons.energy'
                if omas_tag in data and np.abs(data[omas_tag]).sum() != 0.0:
                    if fusion_source is None:
                        fusion_source = np.zeros_like(data[omas_tag].to_numpy())
                    fusion_source += data[omas_tag].to_numpy()
                omas_tag = 'core_sources.fusion.profiles_1d.ion.energy'
                if omas_tag in data and np.abs(data[omas_tag]).sum() != 0.0:
                    if fusion_source is None:
                        fusion_source = np.zeros_like(data[omas_tag].sum('ion_cp').to_numpy())
                    fusion_source += data[omas_tag].sum('ion_cp').to_numpy()
                if external_el_heat_source is not None:
                    attrs['use_generic_heat'] = True
                    attrs['sources.generic_heat.mode'] = 'PRESCRIBED'
                    data_vars['sources.generic_heat.prescribed_values_el'] = (['time', 'rho'], external_ion_heat_source)
                    data_vars['sources.generic_heat.prescribed_values_ion'] = (['time', 'rho'], external_el_heat_source)
                if external_particle_source is not None:
                    attrs['use_generic_particle'] = True
                    attrs['sources.generic_particle.mode'] = 'PRESCRIBED'
                    data_vars['sources.generic_particle.prescribed_values'] = (['time', 'rho'], external_particle_source)
                if external_current_source is not None:
                    attrs['use_generic_current'] = True
                    attrs['sources.generic_current.mode'] = 'PRESCRIBED'
                    data_vars['sources.generic_current.prescribed_values'] = (['time', 'rho'], external_current_source)
                    attrs['sources.generic_current.use_absolute_current'] = True
                dsvec.append(xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs).drop_duplicates(list(coords.keys()), keep='first'))
                struct_index['core_sources'] = len(dsvec) - 1
            if 'time_eq' in data.coords and 'rho_eq' in data.coords:
                coords = {}
                data_vars = {}
                attrs = {}
                coords['time'] = data.get('time_eq').to_numpy().flatten()
                coords['rho'] = data.get('rho_eq').to_numpy().flatten()
                omas_tag = 'equilibrium.time_slice.profiles_1d.psi'
                if omas_tag in data:
                    data_vars['profile_conditions.psi'] = (['time', 'rho'], data[omas_tag].to_numpy())
                omas_tag = 'equilibrium.time_slice.profiles_1d.q'
                if omas_tag in data:
                    data_vars['profile_conditions.q'] = (['time', 'rho'], data[omas_tag].to_numpy())
                dsvec.append(xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs).drop_duplicates(list(coords.keys()), keep='first'))
                struct_index['equilibrium'] = len(dsvec) - 1
            if len(dsvec) > 0:
                idx = struct_index.get('equilibrium', None)
                if idx is not None:
                    drop = [
                        'profile_conditions.q',
                    ]
                    dsvec[idx] = dsvec[idx].drop_vars(drop, errors='ignore')
                idx = struct_index.get('core_profiles', None)
                if idx is not None:
                    drop = [
                        'profile_conditions.psi',
                    ]
                    dsvec[idx] = dsvec[idx].drop_vars(drop, errors='ignore')
                newobj.input = xr.merge(dsvec, join='outer')
        return newobj
