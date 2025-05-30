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


    #def add_output_dir(self, outdir):
    #    newattrs = {}
    #    if isinstance(outdir, (str, Path)):
    #        newattrs['runtime_params.output_dir'] = f'{outdir}'
    #    self.update_input_attrs(newattrs)


    def add_geometry(self, geotype, geofile, geodir=None):
        newattrs = {}
        newattrs['use_psi'] = False
        #newattrs['profile_conditions.initial_psi_from_j'] = True
        #newattrs['profile_conditions.initial_j_is_total_current'] = True
        newattrs['geometry.geometry_type'] = f'{geotype}'
        newattrs['geometry.hires_factor'] = 4
        newattrs['geometry.geometry_file'] = f'{geofile}'
        if geodir is not None:
            newattrs['geometry.geometry_directory'] = f'{geodir}'
        newattrs['geometry.Ip_from_parameters'] = bool(self.input.attrs.get('profile_conditions.Ip_tot', False))
        if geotype == 'eqdsk':
            newattrs['geometry.n_surfaces'] = 100
            newattrs['geometry.last_surface_factor'] = 0.99
        self.update_input_attrs(newattrs)


    def add_pedestal_by_pressure(self, pped, nped, tpedratio, wrho):
        newattrs = {}
        newattrs['pedestal.set_pedestal'] = True
        newattrs['pedestal.model_name'] = 'set_P_ped_n_ped'
        newattrs['pedestal.P_ped'] = {0.0: float(pped)}
        newattrs['pedestal.n_e_ped'] = {0.0: float(nped)}
        newattrs['pedestal.n_e_ped_is_fGW'] = False
        newattrs['pedestal.T_i_T_e_ratio'] = {0.0: float(tpedratio)}
        newattrs['pedestal.rho_norm_ped_top'] = {0.0: float(wrho)}
        newattrs['transport.smooth_everywhere'] = False
        newattrs['numerics.adaptive_T_source_prefactor'] = 1.0e10
        newattrs['numerics.adaptive_n_source_prefactor'] = 1.0e8
        self.update_input_attrs(newattrs)


    def add_pedestal_by_temperature(self, nped, tped, wrho):
        newattrs = {}
        tref = 1.0e3
        newattrs['pedestal.set_pedestal'] = True
        newattrs['pedestal.model_name'] = 'set_T_ped_n_ped'
        newattrs['pedestal.n_e_ped'] = {0.0: float(nped)}
        newattrs['pedestal.n_e_ped_is_fGW'] = False
        newattrs['pedestal.T_e_ped'] = {0.0: float(tped) / tref}
        newattrs['pedestal.T_i_ped'] = {0.0: float(tped) / tref}
        newattrs['pedestal.rho_norm_ped_top'] = {0.0: float(wrho)}
        newattrs['transport.smooth_everywhere'] = False
        newattrs['numerics.adaptive_T_source_prefactor'] = 1.0e10
        newattrs['numerics.adaptive_n_source_prefactor'] = 1.0e8
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


    def add_qualikiz_transport(self):
        newattrs = {}
        newattrs['transport.model_name'] = 'qualikiz'
        newattrs['transport.n_max_runs'] = 10
        newattrs['transport.n_processes'] = 60
        newattrs['transport.collisionality_multiplier'] = 1.0
        newattrs['transport.DV_effective'] = False
        newattrs['transport.An_min'] = 0.05
        newattrs['transport.avoid_big_negative_s'] = True
        newattrs['transport.smag_alpha_correction'] = True
        newattrs['transport.q_sawtooth_proxy'] = True
        newattrs['transport.chi_min'] = 0.05
        newattrs['transport.chi_max'] = 100.0
        newattrs['transport.D_e_min'] = 0.05
        newattrs['transport.D_e_max'] = 100.0
        newattrs['transport.V_e_min'] = -50.0
        newattrs['transport.V_e_max'] = 50.0
        newattrs['transport.smoothing_width'] = 0.1
        newattrs['transport.smooth_everywhere'] = (not self.input.attrs.get('pedestal.set_pedestal', False))
        self.update_input_attrs(newattrs)


    def set_qualikiz_model_path(self, path):
        newattrs = {}
        if self.input.attrs.get('transport.model_name', '') == 'qualikiz':
            newattrs['TORAX_QLK_EXEC_PATH'] = f'{path}'
        self.update_input_attrs(newattrs)


    def add_qlknn_transport(self):
        newattrs = {}
        newattrs['transport.model_name'] = 'qlknn'
        newattrs['transport.model_path'] = ''
        newattrs['transport.include_ITG'] = True
        newattrs['transport.include_TEM'] = True
        newattrs['transport.include_ETG'] = True
        newattrs['transport.ITG_flux_ratio_correction'] = 1.0
        newattrs['transport.ETG_correction_factor'] = 1.0
        newattrs['transport.clip_inputs'] = False
        newattrs['transport.clip_margin'] = 0.95
        newattrs['transport.collisionality_multiplier'] = 1.0
        newattrs['transport.DV_effective'] = True
        newattrs['transport.An_min'] = 0.05
        newattrs['transport.avoid_big_negative_s'] = True
        newattrs['transport.smag_alpha_correction'] = True
        newattrs['transport.q_sawtooth_proxy'] = True
        newattrs['transport.chi_min'] = 0.05
        newattrs['transport.chi_max'] = 100.0
        newattrs['transport.D_e_min'] = 0.05
        newattrs['transport.D_e_max'] = 100.0
        newattrs['transport.V_e_min'] = -50.0
        newattrs['transport.V_e_max'] = 50.0
        newattrs['transport.smoothing_width'] = 0.0
        newattrs['transport.smooth_everywhere'] = (not self.input.attrs.get('pedestal.set_pedestal', False))
        self.update_input_attrs(newattrs)


    def set_qlknn_model_path(self, path):
        newattrs = {}
        if self.input.attrs.get('transport.model_name', '') == 'qlknn':
            newattrs['transport.model_path'] = f'{path}'
        self.update_input_attrs(newattrs)


    def add_transport_inner_patch(self, de, ve, chii, chie, rho):
        newattrs = {}
        newattrs['transport.apply_inner_patch'] = {0.0: True}
        newattrs['transport.D_e_inner'] = {0.0: float(de)}
        newattrs['transport.V_e_inner'] = {0.0: float(ve)}
        newattrs['transport.chi_i_inner'] = {0.0: float(chii)}
        newattrs['transport.chi_e_inner'] = {0.0: float(chie)}
        newattrs['transport.rho_inner'] = float(rho)
        self.update_input_attrs(newattrs)


    def add_transport_outer_patch(self, de, ve, chii, chie, rho):
        newattrs = {}
        newattrs['transport.apply_outer_patch'] = {0.0: True}
        newattrs['transport.D_e_outer'] = {0.0: float(de)}
        newattrs['transport.V_e_outer'] = {0.0: float(ve)}
        newattrs['transport.chi_i_outer'] = {0.0: float(chii)}
        newattrs['transport.chi_e_outer'] = {0.0: float(chie)}
        newattrs['transport.rho_outer'] = float(rho)
        self.update_input_attrs(newattrs)


    def set_exchange_source(self):
        newattrs = {}
        newattrs['sources.ei_exchange.mode'] = 'MODEL_BASED'
        newattrs['sources.ei_exchange.Qei_multiplier'] = 1.0
        self.update_input_attrs(newattrs)


    def set_ohmic_source(self):
        newattrs = {}
        newattrs['sources.ohmic.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)


    def set_fusion_source(self):
        newattrs = {}
        newattrs['sources.fusion.mode'] = 'MODEL_BASED'
        self.update_input_attrs(newattrs)


    def reset_gas_puff_source(self):
        newattrs = {}
        newattrs['sources.gas_puff.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)


    def set_gas_puff_source(self, length, total):
        newattrs = {}
        newattrs['sources.gas_puff.mode'] = 'MODEL_BASED'
        newattrs['sources.gas_puff.puff_decay_length'] = {0.0: length}
        newattrs['sources.gas_puff.S_total'] = {0.0: total}
        self.update_input_attrs(newattrs)


    def set_bootstrap_current_source(self):
        self.add_neoclassical_bootstrap_current()


    def reset_bremsstrahlung_source(self):
        newattrs = {}
        newattrs['sources.bremsstrahlung.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delattrs = [
            'sources.bremsstrahlung.use_relativistic_correction',
        ]
        self.delete_input_attrs(delattrs)


    def set_bremsstrahlung_source(self):
        newattrs = {}
        newattrs['sources.bremsstrahlung.mode'] = 'MODEL_BASED'
        newattrs['sources.bremsstrahlung.use_relativistic_correction'] = True
        self.update_input_attrs(newattrs)


    def set_mavrin_line_radiation_source(self):
        newattrs = {}
        newattrs['sources.impurity_radiation.mode'] = 'MODEL_BASED'
        newattrs['sources.impurity_radiation.model_name'] = 'mavrin_fit'
        newattrs['sources.impurity_radiation.radiation_multiplier'] = 1.0
        self.update_input_attrs(newattrs)
        # Mavrin polynomial model includes Bremsstrahlung so zero that out as well
        self.reset_bremsstrahlung_source()


    def set_synchrotron_source(self):
        newattrs = {}
        newattrs['sources.cyclotron_radiation.mode'] = 'MODEL_BASED'
        newattrs['sources.cyclotron_radiation.wall_reflection_coeff'] = 0.9
        newattrs['sources.cyclotron_radiation.beta_min'] = 0.5
        newattrs['sources.cyclotron_radiation.beta_max'] = 8.0
        newattrs['sources.cyclotron_radiation.beta_grid_size'] = 32
        self.update_input_attrs(newattrs)


    def reset_generic_heat_source(self):
        newattrs = {}
        newattrs['sources.generic_heat.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delattrs = [
            'sources.generic_heat.prescribed_values',
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
        delattrs = [
            'sources.generic_particle.prescribed_values',
            'sources.generic_particle.deposition_location',
            'sources.generic_particle.particle_width',
            'sources.generic_particle.S_total',
        ]
        self.delete_input_attrs(delattrs)


    def reset_generic_current_source(self):
        newattrs = {}
        newattrs['sources.generic_current.mode'] = 'ZERO'
        self.update_input_attrs(newattrs)
        delattrs = [
            'sources.generic_current.prescribed_values',
            'sources.generic_current.gaussian_location',
            'sources.generic_current.gaussian_width',
            'sources.generic_current.I_generic',
            'sources.generic_current.fraction_of_total_current',
            'sources.generic_current.use_absolute_current',
        ]
        self.delete_input_attrs(delattrs)


    def set_gaussian_generic_heat_source(self, mu, sigma, total, efrac=0.5, afrac=1.0):
        newattrs = {}
        newattrs['sources.generic_heat.mode'] = 'MODEL_BASED'
        newattrs['sources.generic_heat.gaussian_location'] = {0.0: mu}
        newattrs['sources.generic_heat.gaussian_width'] = {0.0: sigma}
        newattrs['sources.generic_heat.P_total'] = {0.0: total}
        newattrs['sources.generic_heat.electron_heat_fraction'] = {0.0: efrac}
        newattrs['sources.generic_heat.absorption_fraction'] = {0.0: afrac}
        self.update_input_attrs(newattrs)
        delattrs = [
            'sources.generic_heat.prescribed_values',
        ]
        self.delete_input_attrs(delattrs)


    def set_gaussian_generic_particle_source(self, mu, sigma, total):
        newattrs = {}
        newattrs['sources.generic_particle.mode'] = 'MODEL_BASED'
        newattrs['sources.generic_particle.deposition_location'] = {0.0: mu}
        newattrs['sources.generic_particle.particle_width'] = {0.0: sigma}
        newattrs['sources.generic_particle.S_total'] = {0.0: total}
        self.update_input_attrs(newattrs)
        delattrs = [
            'sources.generic_particle.prescribed_values',
        ]
        self.delete_input_attrs(delattrs)


    def set_gaussian_generic_current_source(self, mu, sigma, total):
        newattrs = {}
        newattrs['sources.generic_current.mode'] = 'MODEL_BASED'
        newattrs['sources.generic_current.gaussian_location'] = {0.0: mu}
        newattrs['sources.generic_current.gaussian_width'] = {0.0: sigma}
        newattrs['sources.generic_current.I_generic'] = {0.0: total}
        newattrs['sources.generic_current.fraction_of_total_current'] = {0.0: 0.0}
        newattrs['sources.generic_current.use_absolute_current'] = True
        self.update_input_attrs(newattrs)
        delattrs = [
            'sources.generic_current.prescribed_values',
        ]
        self.delete_input_attrs(delattrs)


    def add_fixed_linear_solver(self, dt_fixed=None, single=False):
        newattrs = {}
        newattrs['solver.solver_type'] = 'linear'
        newattrs['solver.theta_implicit'] = 1.0
        newattrs['solver.use_predictor_corrector'] = True
        newattrs['solver.n_corrector_steps'] = 10
        newattrs['solver.use_pereverzev'] = True
        newattrs['solver.chi_pereverzev'] = 20.0
        newattrs['solver.D_pereverzev'] = 10.0
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
        newattrs['solver.chi_pereverzev'] = 20.0
        newattrs['solver.D_pereverzev'] = 10.0
        newattrs['time_step_calculator.calculator_type'] = 'chi'
        newattrs['time_step_calculator.tolerance'] = 1.0e-7 if not single else 1.0e-5
        newattrs['numerics.chi_timestep_prefactor'] = float(dt_mult) if isinstance(dtmult, (float, int)) else 50.0
        self.update_input_attrs(newattrs)


    def add_fixed_newton_raphson_solver(self, dt_fixed=None, single=False):
        newattrs = {}
        newattrs['solver.solver_type'] = 'newton_raphson'
        newattrs['solver.theta_implicit'] = 1.0
        newattrs['solver.use_predictor_corrector'] = True
        newattrs['solver.n_corrector_steps'] = 10
        newattrs['solver.use_pereverzev'] = True
        newattrs['solver.chi_pereverzev'] = 20.0
        newattrs['solver.D_pereverzev'] = 10.0
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
        newattrs['solver.chi_pereverzev'] = 20.0
        newattrs['solver.D_pereverzev'] = 10.0
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


    def to_dict(self):
        datadict = {}
        ds = self.input
        datadict.update(ds.attrs)
        for key in ds.data_vars:
            if 'time' in ds[key].dims:
                time = ds['time'].to_numpy().flatten()
                time_dependent_var = {}
                if 'rho' in ds[key].dims:
                    for ii in range(len(time)):
                        time_dependent_var[float(time[ii])] = (ds['rho'].to_numpy().flatten(), ds[key].isel(time=ii).to_numpy().flatten())
                elif 'main_ion' in ds[key].dims:
                    for species in ds['main_ion'].to_numpy().flatten():
                        time_dependent_var[str(species)] = {}
                        for ii in range(len(time)):
                            time_dependent_var[str(species)] = {float(t): v for t, v in zip(time, ds[key].sel(main_ion=species).to_numpy().flatten())}
                else:
                    for ii in range(len(time)):
                        time_dependent_var[float(time[ii])] = float(ds[key].isel(time=ii).to_numpy().flatten())
                datadict[key] = time_dependent_var
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
            datadict.get('sources.generic_heat.mode', 'MODEL') == 'PRESCRIBED' and
            'sources.generic_heat.prescribed_values_el' in datadict and
            'sources.generic_heat.prescribed_values_ion' in datadict
        ):
            e_source = datadict.pop('sources.generic_heat.prescribed_values_el')
            i_source = datadict.pop('sources.generic_heat.prescribed_values_ion')
            source = {t: (copy.deepcopy(e_source[t][0]), e_source[t][1] + i_source[t][1]) for t in e_source}
            fraction = {t: float((e_source[t][1] / (e_source[t][1] + i_source[t][1]))[0]) for t in e_source}
            datadict['sources.generic_heat.prescribed_values'] = source
            datadict['sources.generic_heat.electron_heat_fraction'] = fraction
            #datadict['sources.generic_heat.prescribed_values'] = ((e_source, i_source), {'time_interpolation_mode': 'PIECEWISE_LINEAR', 'rho_interpolation_mode': 'PIECEWISE_LINEAR'})
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
        datadict.pop('profile_conditions.q', None)
        datadict.pop('profile_conditions.j_ohmic', None)
        datadict.pop('profile_conditions.j_bootstrap', None)
        datadict.pop('TORAX_QLK_EXEC_PATH', None)
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
                        nfuelsum = data['ni'].sel(name=namelist).sum('name').to_numpy().flatten()
                        for ii in range(len(namelist)):
                            sdata = data['ni'].sel(name=namelist[ii])
                            species.append(namelist[ii])
                            density.append(float((sdata.to_numpy().flatten() / nfuelsum).mean()))
                        coords['main_ion'] = species
                    else:
                        species = ['D']
                        density = [1.0]
                    coords['main_ion'] = species
                    data_vars['plasma_composition.main_ion'] = (['main_ion', 'time'], np.expand_dims(density, axis=1))
                if 'z' in data and 'mass' in data and 'ni' in data and 'ne' in data:
                    nfilt = (~np.isclose(data['z'], 1.0))
                    zeff = np.ones_like(data['ne'].to_numpy().flatten())
                    if np.any(nfilt):
                        namelist = data['name'].to_numpy()[nfilt].tolist()
                        impcomp = {}
                        zeff = np.zeros_like(data['ne'].to_numpy().flatten())
                        nsum = np.zeros_like(data['ne'].to_numpy().flatten())
                        for ii in range(len(data['name'])):
                            sdata = data.isel(name=ii)
                            if sdata['name'] in namelist and 'therm' in str(sdata['type'].to_numpy()):
                                sname = str(sdata['name'].to_numpy())
                                if sname not in newobj.allowed_radiation_species:
                                    sn, sa, sz = define_ion_species(short_name=sname)
                                    if sz > 60.0:
                                        sname = 'W'
                                    if sz > 48.0:
                                        sname = 'Xe'
                                    if sz > 30.0:
                                        sname = 'Kr'
                                    if sz > 14.0:
                                        sname = 'Ar'
                                    if sz > 6.0:
                                        sname = 'Ne'
                                    if sz > 1.0:
                                        sname = 'C'
                                    if sn == 'He':
                                        sname = 'He4'
                                impcomp[sname] = sdata['ni'].to_numpy().flatten()
                                nsum += sdata['ni'].to_numpy().flatten()
                            zeff += sdata['ni'].to_numpy().flatten() * sdata['z'].to_numpy().flatten() ** 2.0 / data['ne'].to_numpy().flatten()
                        total = 0.0
                        for key in impcomp:
                            impcomp[key] = float(np.mean(impcomp[key] / nsum))
                            total += impcomp[key]
                        for key in impcomp:
                            impcomp[key] = impcomp[key] / total
                        if 'z_eff' in data:
                            zeff = data['z_eff'].to_numpy().flatten()
                        if not impcomp:
                            impcomp['Ne'] = 1.0
                        attrs['plasma_composition.impurity'] = impcomp
                    data_vars['plasma_composition.Z_eff'] = (['time', 'rho'], np.expand_dims(zeff, axis=0))
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
                if 'qohme' in data and data['qohme'].sum() != 0.0:
                    attrs['sources.ohmic.mode'] = 'PRESCRIBED'
                    data_vars['sources.ohmic.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qohme'].to_numpy().flatten(), axis=0))
                if 'qbeame' in data and data['qbeame'].sum() != 0.0:
                    if external_el_heat_source is None:
                        external_el_heat_source = np.zeros_like(data['qbeame'].to_numpy().flatten())
                    external_el_heat_source += 1.0e6 * data['qbeame'].to_numpy().flatten()
                if 'qbeami' in data and data['qbeami'].sum() != 0.0:
                    if external_ion_heat_source is None:
                        external_ion_heat_source = np.zeros_like(data['qbeami'].to_numpy().flatten())
                    external_ion_heat_source += 1.0e6 * data['qbeami'].to_numpy().flatten()
                if 'qrfe' in data and data['qrfe'].sum() != 0.0:
                    if external_el_heat_source is None:
                        external_el_heat_source = np.zeros_like(data['qrfe'].to_numpy().flatten())
                    external_el_heat_source += 1.0e6 * data['qrfe'].to_numpy().flatten()
                if 'qrfi' in data and data['qrfi'].sum() != 0.0:
                    if external_ion_heat_source is None:
                        external_ion_heat_source = np.zeros_like(data['qrfi'].to_numpy().flatten())
                    external_ion_heat_source += 1.0e6 * data['qrfi'].to_numpy().flatten()
                if 'qsync' in data and data['qsync'].sum() != 0.0:
                    attrs['sources.cyclotron_radiation.mode'] = 'PRESCRIBED'
                    data_vars['sources.cyclotron_radiation.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qsync'].to_numpy().flatten(), axis=0))
                if 'qbrem' in data and data['qbrem'].sum() != 0.0:
                    attrs['sources.bremsstrahlung.mode'] = 'PRESCRIBED'
                    data_vars['sources.bremsstrahlung.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qbrem'].to_numpy().flatten(), axis=0))
                if 'qline' in data and data['qline'].sum() != 0.0:
                    attrs['sources.impurity_radiation.mode'] = 'PRESCRIBED'
                    data_vars['sources.impurity_radiation.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qline'].to_numpy().flatten(), axis=0))
                if 'qfuse' in data and data['qfuse'].sum() != 0.0:
                    if fusion_source is None:
                        fusion_source = np.zeros_like(data['qfuse'].to_numpy().flatten())
                    fusion_source += 1.0e6 * data['qfuse'].to_numpy().flatten()
                if 'qfusi' in data and data['qfusi'].sum() != 0.0:
                    if fusion_source is None:
                        fusion_source = np.zeros_like(data['qfuse'].to_numpy().flatten())
                    fusion_source += 1.0e6 * data['qfuse'].to_numpy().flatten()
                if 'qei' in data and data['qei'].sum() != 0.0:
                    attrs['sources.ei_exchange.mode'] = 'PRESCRIBED'
                    data_vars['sources.ei_exchange.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qei'].to_numpy().flatten(), axis=0))
                #if 'qione' in data and data['qione'].sum() != 0.0:
                #    pass
                #if 'qioni' in data and data['qioni'].sum() != 0.0:
                #    pass
                #if 'qcxi' in data and data['qcxi'].sum() != 0.0:
                #    pass
                if 'jbs' in data and data['jbs'].sum() != 0.0:
                    data_vars['profile_conditions.j_bootstrap'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['jbs'].to_numpy().flatten(), axis=0))
                    if external_current_source is None:
                        external_current_source = np.zeros_like(data['jbs'].to_numpy().flatten())
                    external_current_source += 1.0e6 * data['jbs'].to_numpy().flatten()
                #if 'jbstor' in data and data['jbstor'].sum() != 0.0:
                #    pass
                if 'johm' in data and data['johm'].sum() != 0.0:
                    data_vars['profile_conditions.j_ohmic'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['johm'].to_numpy().flatten(), axis=0))
                    if external_current_source is None:
                        external_current_source = np.zeros_like(data['johm'].to_numpy().flatten())
                    external_current_source += 1.0e6 * data['johm'].to_numpy().flatten()
                if 'jrf' in data and data['jrf'].sum() != 0.0:
                    if external_current_source is None:
                        external_current_source = np.zeros_like(data['jrf'].to_numpy().flatten())
                    external_current_source += 1.0e6 * data['jrf'].to_numpy().flatten()
                if 'jnb' in data and data['jnb'].sum() != 0.0:
                    if external_current_source is None:
                        external_current_source = np.zeros_like(data['jnb'].to_numpy().flatten())
                    external_current_source += 1.0e6 * data['jnb'].to_numpy().flatten()
                if 'qpar_beam' in data and data['qpar_beam'].sum() != 0.0:
                    if external_particle_source is None:
                        external_particle_source = np.zeros_like(data['qpar_beam'].to_numpy().flatten())
                    external_particle_source += data['qpar_beam'].to_numpy().flatten()
                if 'qpar_wall' in data and data['qpar_wall'].sum() != 0.0:
                    if external_particle_source is None:
                        external_particle_source = np.zeros_like(data['qpar_wall'].to_numpy().flatten())
                    external_particle_source += data['qpar_wall'].to_numpy().flatten()
                #if 'qmom' in data and data['qmom'].sum() != 0.0:
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
                newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj
