import copy
import json
from pathlib import Path
import logging
import numpy as np
from fusio.classes.io import io
from fusio.utils.json_tools import serialize, deserialize

logger = logging.getLogger('fusio')


class torax(io):

    basevars = {
        'runtime_params': {
            'plasma_composition': [
                'main_ion',
                'impurity',
                'Zeff',
                'Zi_override',
                'Ai_override',
                'Zimp_override',
                'Aimp_override',
            ],
            'profile_conditions': [
                'Ip_tot',
                'Ti',
                'Ti_bound_right',
                'Te',
                'Te_bound_right',
                'psi',
                'ne',
                'ne_bound_right',
                'normalize_to_nbar',
                'ne_is_fGW',
                'ne_bound_right_is_fGW',
                'nbar',
                'set_pedestal',
                'nu',
                'initial_j_is_total_current',
                'initial_psi_from_j',
            ],
            'numerics': [
                't_initial',
                't_final',
                'exact_t_final',
                'ion_heat_eq',
                'el_heat_eq',
                'current_eq',
                'dens_eq',
                'resistivity_mult',
                'maxdt',
                'mindt',
                'dtmult',
                'fixed_dt',
                'dt_reduction_factor',
                'largeValue_T',
                'largeValue_n',
                'nref',
            ],
            'output_dir': [
                'output_dir',
            ],
        },
        'geometry': [
            'geometry_type',
            'n_rho',
            'hires_fac',
            'geometry_configs',
        ],
        'pedestal': [
            'pedestal_model',
        ],
        'transport': [
            'transport_model',
            'chimin',
            'chimax',
            'Demin',
            'Demax',
            'Vemin',
            'Vemax',
            'apply_inner_patch',
            'De_inner',
            'Ve_inner',
            'chii_inner',
            'chie_inner',
            'rho_inner',
            'apply_outer_patch',
            'De_outer',
            'Ve_outer',
            'chii_outer',
            'chie_outer',
            'rho_outer',
            'smoothing_sigma',
        ],
        'sources': [
        ],
        'stepper': [
            'stepper_type',
            'theta_imp',
            'adaptive_dt',
            'dt_reduction_factor',
            'predictor_corrector',
            'corrector_steps',
            'use_pereverzev',
            'chi_per',
            'd_per',
        ],
        'time_step_calculator': [
            'time_step_calculator_type',
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
                'Rmaj',
                'Rmin',
                'B0',
                'kappa',
            ],
            'chease': [
                'geometry_file',
                'geometry_dir',
                'Ip_from_parameters',
                'Rmaj',
                'Rmin',
                'B0',
            ],
            'fbt': [
                'geometry_file',
                'geometry_dir',
                'Ip_from_parameters',
                'LY_object',
                'LY_bundle_object',
                'LY_to_torax_times',
                'L_object',
            ],
            'eqdsk': [
                'geometry_file',
                'geometry_dir',
                'Ip_from_parameters',
                'n_surfaces',
                'last_surface_factor',
            ],
        },
        'pedestal': {
            'set_tped_nped': [
                'neped',
                'neped_is_fGW',
                'Tiped',
                'Teped',
                'rho_norm_ped_top',
            ],
            'set_pped_tpedratio_nped': [
                'Pped',
                'neped',
                'neped_is_fGW',
                'ion_electron_temperature_ratio',
                'rho_norm_ped_top',
            ],
        },
        'sources': {
            'generic_ion_el_heat_source': [
                'mode',
                'is_explicit',
                'rsource',
                'w',
                'Ptot',
                'el_heat_fraction',
            ],
            'qei_source': [
                'mode',
                'is_explicit',
                'Qei_mult',
            ],
            'ohmic_heat_source': [
                'mode',
                'is_explicit',
            ],
            'fusion_heat_source': [
                'mode',
                'is_explicit',
            ],
            'gas_puff_source': [
                'mode',
                'is_explicit',
                'puff_decay_length',
                'S_puff_tot',
            ],
            'pellet_source': [
                'mode',
                'is_explicit',
                'pellet_deposition_location',
                'pellet_width',
                'S_pellet_tot',
            ],
            'generic_particle_source': [
                'mode',
                'is_explicit',
                'deposition_location',
                'particle_width',
                'S_tot',
            ],
            'j_bootstrap': [
                'mode',
                'is_explicit',
                'bootstrap_mult',
            ],
            'generic_current_source': [
                'mode',
                'is_explicit',
                'rext',
                'wext',
                'Iext',
                'fext',
                'use_absolute_current',
            ],
            'bremsstrahlung_heat_sink': [
                'mode',
                'is_explicit',
                'use_relativistic_correction',
            ],
            'impurity_radiation_heat_sink': [
                'mode',
                'is_explicit',
                'model_func',
            ],
            'cyclotron_radiation_heat_sink': [
                'mode',
                'is_explicit',
                'wall_reflection_coeff',
                'beta_min',
                'beta_max',
                'beta_grid_size',
            ],
            'electron_cyclotron_source': [
                'mode',
                'is_explicit',
                'manual_ec_power_density',
                'gaussian_ec_power_density_width',
                'gaussian_ec_power_density_location',
                'gaussian_ec_total_power',
                'cd_efficiency',
            ],
            'ion_cyclotron_source': [
                'mode',
                'is_explicit',
                'wall_inner',
                'wall_outer',
                'frequency',
                'minority_concentration',
                'Ptot',
            ],
        },
        'transport': {
            'constant': [
                'chii_const',
                'chie_const',
                'De_const',
                'Ve_const',
            ],
            'CGM': [
                'alpha',
                'chistiff',
                'chiei_ratio',
                'chi_D_ratio',
                'VR_D_ratio',
            ],
            'bohm-gyrobohm': [
                'chi_e_bohm_coeff',
                'chi_e_gyrobohm_coeff',
                'chi_i_bohm_coeff',
                'chi_i_gyrobohm_coeff',
                'chi_e_bohm_multiplier',
                'chi_e_gyrobohm_multiplier',
                'chi_i_bohm_multiplier',
                'chi_i_gyrobohm_multiplier',
                'd_face_c1',
                'd_face_c2',
            ],
            'qlknn': [
                'model_path',
                'coll_mult',
                'include_ITG',
                'include_TEM',
                'include_ETG',
                'ITG_flux_ratio_correction',
                'DVeff',
                'An_min',
                'avoid_big_negative_s',
                'smag_alpha_correction',
                'q_sawtooth_proxy',
            ],
            'qualikiz': [
                'maxruns',
                'numprocs',
                'coll_mult',
                'DVeff',
                'An_min',
                'avoid_big_negative_s',
                'q_sawtooth_proxy',
            ],
        },
        'stepper': {
            'newton_raphson': [
                'log_iterations',
                'initial_guess_mode',
                'tol',
                'coarse_tol',
                'maxiter',
                'delta_reduction_factor',
                'tau_min',
            ],
            'optimizer': [
                'initial_guess_mode',
                'tol',
                'maxiter',
            ],
        },
    }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = None
        for arg in args:
            if path is None and isinstance(arg, (str, Path)):
                path = Path(arg)
        for key, kwarg in kwargs.items():
            if path is None and key in ['path', 'file', 'input'] and isinstance(kwarg, (str, Path)):
                path = Path(kwarg)
        if path is not None:
            self._data.update(self.read(path))


    def _uncompress(self, data):
        odict = {}
        udict = {}
        for key in data:
            klist = key.split('.')
            if len(klist) > 1:
                nkey = '.'.join(klist[1:])
                if klist[0] not in udict:
                    udict[klist[0]] = []
                udict[klist[0]].append(nkey)
            else:
                odict[klist[0]] = data[f'{key}']
        if udict:
            for key in udict:
                gdict = {}
                for lkey in udict[key]:
                    gdict[lkey] = data[f'{key}.{lkey}']
                odict[key] = self._uncompress(gdict)
        else:
            odict = data
        return odict


    def write(self, path):
        opath = Path(path) if isinstance(path, (str, Path)) else None
        if not self.empty:
            odict = self._uncompress(self.data)
            with open(opath, 'w') as jf:
                json.dump(odict, jf, indent=4, default=serialize)
            logger.info(f'Saved {self.format} data into {opath.resolve()}')
        else:
            logger.error(f'Attempting to write empty {self.format} class instance... Failed!')


    def add_output_dir(self, outdir):
        if isinstance(outdir, (str, Path)):
            self._data['runtime_params.output_dir'] = f'{outdir}'


    def add_geometry(self, geotype, geofile, geodir=None):
        if 'runtime_params.profile_conditions.psi' in self._data:
            del self._data['runtime_params.profile_conditions.psi']
        self._data['geometry.geometry_type'] = f'{geotype}'
        self._data['geometry.n_rho'] = 25
        self._data['geometry.hires_fac'] = 4
        self._data['geometry.geometry_file'] = f'{geofile}'
        if geodir is not None:
            self._data['geometry.geometry_dir'] = f'{geodir}'
        self._data['geometry.Ip_from_parameters'] = bool(self._data.get('runtime_params.profile_conditions.Ip_tot', False))
        if geotype == 'eqdsk':
            self._data['geometry.n_surfaces'] = 100
            self._data['geometry.last_surface_factor'] = 0.995


    def add_pedestal_by_pressure(self, pped, nped, tpedratio, wrho):
        nref = self._data.get('runtime_params.numerics.nref', 1.0e20)
        self._data['runtime_params.profile_conditions.set_pedestal'] = True
        self._data['transport.smooth_everywhere'] = False
        self._data['pedestal.pedestal_model'] = 'set_pped_tpedratio_nped'
        self._data['pedestal.Pped'] = {0.0: float(pped)}
        self._data['pedestal.neped'] = {0.0: float(nped) / nref}
        self._data['pedestal.neped_is_fGW'] = False
        self._data['pedestal.ion_electron_temperature_ratio'] = {0.0: float(tpedratio)}
        self._data['pedestal.rho_norm_ped_top'] = {0.0: float(wrho)}


    def add_pedestal_by_temperature(self, nped, tped, wrho):
        nref = self._data.get('runtime_params.numerics.nref', 1.0e20)
        tref = 1.0e3
        self._data['runtime_params.profile_conditions.set_pedestal'] = True
        self._data['transport.smooth_everywhere'] = False
        self._data['pedestal.pedestal_model'] = 'set_tped_nped'
        self._data['pedestal.neped'] = {0.0: float(nped) / nref}
        self._data['pedestal.neped_is_fGW'] = False
        self._data['pedestal.Teped'] = {0.0: float(tped) / tref}
        self._data['pedestal.Tiped'] = {0.0: float(tped) / tref}
        self._data['pedestal.rho_norm_ped_top'] = {0.0: float(wrho)}


    def add_default_transport(self):
        self._data['transport.transport_model'] = 'qlknn'
        self._data['transport.chimin'] = 0.05
        self._data['transport.chimax'] = 100.0
        self._data['transport.Demin'] = 0.01
        self._data['transport.Demax'] = 50.0
        self._data['transport.Vemin'] = -10.0
        self._data['transport.Vemax'] = 10.0
        self._data['transport.smoothing_sigma'] = 0.0
        self._data['transport.smooth_everywhere'] = (not self._data.get('runtime_params.profile_conditions.set_pedestal', False))
        self._data['transport.model_path'] = ''
        self._data['transport.coll_mult'] = 0.25
        self._data['transport.include_ITG'] = True
        self._data['transport.include_TEM'] = True
        self._data['transport.include_ETG'] = True
        self._data['transport.DVeff'] = True
        self._data['transport.An_min'] = 0.05
        self._data['transport.avoid_big_negative_s'] = True
        self._data['transport.smag_alpha_correction'] = True
        self._data['transport.q_sawtooth_proxy'] = True


    def set_qlknn_model_path(self, path):
        if 'transport.model_path' in self._data:
            self._data['transport.model_path'] = f'{path}'


    def add_transport_inner_patch(self, de, ve, chii, chie, rho):
        self._data['transport.apply_inner_patch'] = {0.0: True}
        self._data['transport.De_inner'] = {0.0: float(de)}
        self._data['transport.Ve_inner'] = {0.0: float(ve)}
        self._data['transport.chii_inner'] = {0.0: float(chii)}
        self._data['transport.chie_inner'] = {0.0: float(chie)}
        self._data['transport.rho_inner'] = float(rho)


    def add_transport_outer_patch(self, de, ve, chii, chie, rho):
        self._data['transport.apply_outer_patch'] = {0.0: True}
        self._data['transport.De_outer'] = {0.0: float(de)}
        self._data['transport.Ve_outer'] = {0.0: float(ve)}
        self._data['transport.chii_outer'] = {0.0: float(chii)}
        self._data['transport.chie_outer'] = {0.0: float(chie)}
        self._data['transport.rho_outer'] = float(rho)


    def reset_exchange_source(self):
        if 'sources.qei_source.prescribed_values' in self._data:
            del self._data['sources.qei_source.prescribed_values']
        self._data['sources.qei_source.mode'] = 'MODEL_BASED'
        self._data['sources.qei_source.Qei_mult'] = 1.0


    def reset_ohmic_source(self):
        if 'sources.ohmic_heat_source.prescribed_values' in self._data:
            del self._data['sources.ohmic_heat_source.prescribed_values']
        self._data['sources.ohmic_heat_source.mode'] = 'MODEL_BASED'


    def reset_fusion_source(self):
        if 'sources.fusion_heat_source.prescribed_values' in self._data:
            del self._data['sources.fusion_heat_source.prescribed_values']
        self._data['sources.fusion_heat_source.mode'] = 'MODEL_BASED'


    def reset_gas_puff_source(self):
        if 'sources.gas_puff_source.prescribed_values' in self._data:
            del self._data['sources.gas_puff_source.prescribed_values']
        self._data['sources.gas_puff_source.mode'] = 'MODEL_BASED'
        self._data['sources.gas_puff_source.puff_decay_length'] = {0.0: 0.05}
        self._data['sources.gas_puff_source.S_puff_tot'] = {0.0: 1.0e22}


    def reset_bootstrap_source(self):
        if 'sources.j_bootstrap.prescribed_values' in self._data:
            del self._data['sources.j_bootstrap.prescribed_values']
        self._data['sources.j_bootstrap.mode'] = 'MODEL_BASED'
        self._data['sources.j_bootstrap.bootstrap_mult'] = 1.0


    def reset_bremsstrahlung_source(self):
        if 'sources.bremsstrahlung_heat_sink.prescribed_values' in self._data:
            del self._data['sources.bremsstrahlung_heat_sink.prescribed_values']
        self._data['sources.bremsstrahlung_heat_sink.mode'] = 'MODEL_BASED'
        self._data['sources.bremsstrahlung_heat_sink.use_relativistic_correction'] = True


    def reset_line_radiation_source(self):
        if 'sources.impurity_radiation_heat_sink.prescribed_values' in self._data:
            del self._data['sources.impurity_radiation_heat_sink.prescribed_values']
        self._data['sources.impurity_radiation_heat_sink.mode'] = 'MODEL_BASED'
        self._data['sources.impurity_radiation_heat_sink.model_function_name'] = 'impurity_radiation_mavrin_fit'
        self._data['sources.impurity_radiation_heat_sink.radiation_multiplier'] = 1.0
        self._data['sources.bremsstrahlung_heat_sink.mode'] = 'ZERO'
        if 'sources.bremsstrahlung_heat_sink.prescribed_values' in self._data:
            del self._data['sources.bremsstrahlung_heat_sink.prescribed_values']
        if 'sources.bremsstrahlung_heat_sink.use_relativistic_correction' in self._data:
            del self._data['sources.bremsstrahlung_heat_sink.use_relativistic_correction']


    def reset_synchrotron_source(self):
        if 'sources.cyclotron_radiation_heat_sink.prescribed_values' in self._data:
            del self._data['sources.cyclotron_radiation_heat_sink.prescribed_values']
        self._data['sources.cyclotron_radiation_heat_sink.mode'] = 'MODEL_BASED'
        self._data['sources.cyclotron_radiation_heat_sink.wall_reflection_coeff'] = 0.9


    def reset_generic_heat_source(self):
        if 'sources.generic_ion_el_heat_source.prescribed_values' in self._data:
            del self._data['sources.generic_ion_el_heat_source.prescribed_values']
        self._data['sources.generic_ion_el_heat_source.mode'] = 'MODEL_BASED'
        self._data['sources.generic_ion_el_heat_source.rsource'] = {0.0: 0.0}
        self._data['sources.generic_ion_el_heat_source.w'] = {0.0: 0.1}
        self._data['sources.generic_ion_el_heat_source.Ptot'] = {0.0: 0.0}
        self._data['sources.generic_ion_el_heat_source.el_heat_fraction'] = {0.0: 0.5}


    def add_linear_stepper(self):
        self._data['stepper.stepper_type'] = 'linear'
        self._data['stepper.theta_imp'] = 1.0
        self._data['stepper.predictor_corrector'] = True
        self._data['stepper.corrector_steps'] = 10
        self._data['stepper.use_pereverzev'] = True
        self._data['stepper.chi_per'] = 20.0
        self._data['stepper.d_per'] = 10.0
        self._data['time_step_calculator.calculator_type'] = 'chi'


    def add_newton_raphson_stepper(self):
        self._data['stepper.stepper_type'] = 'newton_raphson'
        self._data['stepper.theta_imp'] = 1.0
        self._data['stepper.predictor_corrector'] = True
        self._data['stepper.corrector_steps'] = 10
        self._data['stepper.use_pereverzev'] = True
        self._data['stepper.chi_per'] = 20.0
        self._data['stepper.d_per'] = 10.0
        self._data['stepper.log_iterations'] = True
        self._data['stepper.initial_guess_mode'] = 1
        self._data['stepper.delta_reduction_factor'] = 0.5
        self._data['stepper.maxiter'] = 30
        self._data['stepper.tau_min'] = 0.01
        self._data['time_step_calculator.calculator_type'] = 'chi'


    def set_numerics(self, t_initial, t_final, eqs=['te', 'ti', 'ne', 'j'], dtmult=None):
        self._data['runtime_params.numerics.t_initial'] = float(t_initial)
        self._data['runtime_params.numerics.t_final'] = float(t_final)
        self._data['runtime_params.numerics.exact_t_final'] = True
        self._data['runtime_params.numerics.maxdt'] = 1.0e-1
        self._data['runtime_params.numerics.mindt'] = 1.0e-8
        self._data['runtime_params.numerics.dtmult'] = float(dtmult) if isinstance(dtmult, (float, int)) else 9.0
        self._data['runtime_params.numerics.resistivity_mult'] = 1.0
        self._data['runtime_params.numerics.el_heat_eq'] = (isinstance(eqs, (list, tuple)) and 'te' in eqs)
        self._data['runtime_params.numerics.ion_heat_eq'] = (isinstance(eqs, (list, tuple)) and 'ti' in eqs)
        self._data['runtime_params.numerics.dens_eq'] = (isinstance(eqs, (list, tuple)) and 'ne' in eqs)
        self._data['runtime_params.numerics.current_eq'] = (isinstance(eqs, (list, tuple)) and 'j' in eqs)


    @classmethod
    def from_gacode(cls, obj):
        newobj = cls()
        if isinstance(obj, io):
            data = copy.deepcopy(obj.data)
            if 'z' in data and 'name' in data and 'type' in data and 'ni' in data:
                species = {}
                zi = np.isclose(data['z'], 1.0) & np.array(['fast' not in sub for sub in data['type']])
                nfuel = data['ni'][:, zi]
                nfuel = nfuel.sum(axis=-1).flatten() if nfuel.ndim > 1 else nfuel.flatten()
                for ii in range(len(zi)):
                    if zi[ii]:
                        species[f'{data["name"][ii]}'] = {0.0: float((data['ni'][:, ii].flatten() / nfuel).mean())}
                newobj._data['runtime_params.plasma_composition.main_ion'] = species
            if 'z' in data and 'mass' in data and 'ni' in data and 'ne' in data:
                zi = ~np.isclose(data['z'], 1.0)
                if np.any(zi):
                    newobj._data['runtime_params.plasma_composition.impurity'] = 'Ne'
                zeff = np.zeros(data['ne'].shape)
                nzave = np.zeros(data['ne'].shape)
                for ii in range(len(zi)):
                    if zi[ii]:
                        nzave += data['ni'][:, ii] * data['z'][ii] / data['ne']
                    else:
                        zeff += data['ni'][:, ii] * data['z'][ii] ** 2.0 / data['ne']
                zeff += nzave * 10.0
                newobj._data['runtime_params.plasma_composition.Zeff'] = {0.0: (data['rho'].flatten(), zeff.flatten())}
            if 'current' in data:
                newobj._data['runtime_params.profile_conditions.Ip_tot'] = {0.0: float(data['current'].mean())}
            if 'ne' in data:
                newobj._data['runtime_params.numerics.nref'] = 1.0e20
                nref = newobj._data.get('runtime_params.numerics.nref', 1.0e20)
                newobj._data['runtime_params.profile_conditions.ne'] = {0.0: (data['rho'].flatten(), 1.0e19 * data['ne'].flatten() / nref)}
                newobj._data['runtime_params.profile_conditions.normalize_to_nbar'] = False
                newobj._data['runtime_params.profile_conditions.ne_is_fGW'] = False
            if 'te' in data:
                newobj._data['runtime_params.profile_conditions.Te'] = {0.0: (data['rho'].flatten(), data['te'].flatten())}
            if 'ti' in data and 'z' in data:
                zi = np.isclose(data['z'], 1.0) & np.array(['fast' not in sub for sub in data['type']])
                ti = data['ti'][:, zi]
                if ti.ndim > 1:
                    newobj._data['runtime_params.profile_conditions.Ti'] = {0.0: (data['rho'].flatten(), ti.mean(axis=-1).flatten())}
                else:
                    newobj._data['runtime_params.profile_conditions.Ti'] = {0.0: (data['rho'].flatten(), ti.flatten())}
            if 'polflux' in data:
                newobj._data['runtime_params.profile_conditions.psi'] = {0.0: (data['rho'].flatten(), data['polflux'].flatten())}
            # Place the sources
            external_el_heat_source = None
            external_ion_heat_source = None
            external_particle_source = None
            external_current_source = None
            fusion_source = None
            if 'qohme' in data and data['qohme'].sum() != 0.0:
                newobj._data['sources.ohmic_heat_source.mode'] = 'PRESCRIBED'
                newobj._data['sources.ohmic_heat_source.prescribed_values'] = {0.0: (data['rho'].flatten(), 1.0e6 * data['qohme'].flatten())}
            if 'qbeame' in data and data['qbeame'].sum() != 0.0:
                if external_el_heat_source is None:
                    external_el_heat_source = np.zeros(data['qbeame'].shape).flatten()
                external_el_heat_source += 1.0e6 * data['qbeame'].flatten()
            if 'qbeami' in data and data['qbeami'].sum() != 0.0:
                if external_ion_heat_source is None:
                    external_ion_heat_source = np.zeros(data['qbeami'].shape).flatten()
                external_ion_heat_source += 1.0e6 * data['qbeami'].flatten()
            if 'qrfe' in data and data['qrfe'].sum() != 0.0:
                if external_el_heat_source is None:
                    external_el_heat_source = np.zeros(data['qrfe'].shape).flatten()
                external_el_heat_source += 1.0e6 * data['qrfe'].flatten()
            if 'qrfi' in data and data['qrfi'].sum() != 0.0:
                if external_ion_heat_source is None:
                    external_ion_heat_source = np.zeros(data['qrfi'].shape).flatten()
                external_ion_heat_source += 1.0e6 * data['qrfi'].flatten()
            if 'qsync' in data and data['qsync'].sum() != 0.0:
                newobj._data['sources.cyclotron_radiation_heat_sink.mode'] = 'PRESCRIBED'
                newobj._data['sources.cyclotron_radiation_heat_sink.prescribed_values'] = {0.0: (data['rho'].flatten(), 1.0e6 * data['qsync'].flatten())}
            if 'qbrem' in data and data['qbrem'].sum() != 0.0:
                newobj._data['sources.bremsstrahlung_heat_sink.mode'] = 'PRESCRIBED'
                newobj._data['sources.bremsstrahlung_heat_sink.prescribed_values'] = {0.0: (data['rho'].flatten(), 1.0e6 * data['qbrem'].flatten())}
            if 'qline' in data and data['qline'].sum() != 0.0:
                newobj._data['sources.impurity_radiation_heat_sink.mode'] = 'PRESCRIBED'
                newobj._data['sources.impurity_radiation_heat_sink.prescribed_values'] = {0.0: (data['rho'].flatten(), 1.0e6 * data['qline'].flatten())}
            if 'qfuse' in data and data['qfuse'].sum() != 0.0:
                if fusion_source is None:
                    fusion_source = np.zeros(data['qfuse'].shape).flatten()
                fusion_source += 1.0e6 * data['qfuse'].flatten()
            if 'qfusi' in data and data['qfusi'].sum() != 0.0:
                if fusion_source is None:
                    fusion_source = np.zeros(data['qfuse'].shape).flatten()
                fusion_source += 1.0e6 * data['qfuse'].flatten()
            if 'qei' in data and data['qei'].sum() != 0.0:
                newobj._data['sources.qei_source.mode'] = 'PRESCRIBED'
                newobj._data['sources.qei_source.prescribed_values'] = {0.0: (data['rho'].flatten(), 1.0e6 * data['qei'].flatten())}
            #if 'qione' in data and data['qione'].sum() != 0.0:
            #    pass
            #if 'qioni' in data and data['qioni'].sum() != 0.0:
            #    pass
            #if 'qcxi' in data and data['qcxi'].sum() != 0.0:
            #    pass
            if 'jbs' in data and data['jbs'].sum() != 0.0:
                newobj._data['sources.j_bootstrap.mode'] = 'PRESCRIBED'
                newobj._data['sources.j_bootstrap.prescribed_values'] = {0.0: (data['rho'].flatten(), 1.0e6 * data['jbs'].flatten())}
            #if 'jbstor' in data and data['jbstor'].sum() != 0.0:
            #    pass
            if 'johm' in data and data['johm'].sum() != 0.0:
                if external_current_source is None:
                    external_current_source = np.zeros(data['johm'].shape).flatten()
                external_current_source += 1.0e6 * data['johm'].flatten()
            if 'jrf' in data and data['jrf'].sum() != 0.0:
                if external_current_source is None:
                    external_current_source = np.zeros(data['jrf'].shape).flatten()
                external_current_source += 1.0e6 * data['jrf'].flatten()
            if 'jnb' in data and data['jnb'].sum() != 0.0:
                if external_current_source is None:
                    external_current_source = np.zeros(data['jnb'].shape).flatten()
                external_current_source += 1.0e6 * data['jnb'].flatten()
            if 'qpar_beam' in data and data['qpar_beam'].sum() != 0.0:
                if external_particle_source is None:
                    external_particle_source = np.zeros(data['qpar_beam'].shape).flatten()
                external_particle_source += data['qpar_beam'].flatten()
            if 'qpar_wall' in data and data['qpar_wall'].sum() != 0.0:
                if external_particle_source is None:
                    external_particle_source = np.zeros(data['qpar_wall'].shape).flatten()
                external_particle_source += data['qpar_wall'].flatten()
            #if 'qmom' in data and data['qmom'].sum() != 0.0:
            #    pass
            if external_el_heat_source is not None:
                total_heat_source = copy.deepcopy(external_el_heat_source)
                if external_ion_heat_source is not None:
                    total_heat_source += external_ion_heat_source
                el_heat_fraction = (external_el_heat_source / total_heat_source).mean()
                newobj._data['sources.generic_ion_el_heat_source.mode'] = 'PRESCRIBED'
                newobj._data['sources.generic_ion_el_heat_source.prescribed_values'] = {0.0: (data['rho'].flatten(), total_heat_source)}
                newobj._data['sources.generic_ion_el_heat_source.el_heat_fraction'] = {0.0: float(el_heat_fraction)}
            if external_particle_source is not None:
                newobj._data['sources.generic_particle_source.mode'] = 'PRESCRIBED'
                newobj._data['sources.generic_particle_source.prescribed_values'] = {0.0: (data['rho'].flatten(), external_particle_source)}
            if external_current_source is not None:
                newobj._data['sources.generic_current_source.mode'] = 'PRESCRIBED'
                newobj._data['sources.generic_current_source.prescribed_values'] = {0.0: (data['rho'].flatten(), external_current_source)}
                newobj._data['sources.generic_cuurent_source.use_absolute_current'] = True
        return newobj
