import copy
import json
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from fusio.classes.io import io
from fusio.utils.json_tools import serialize, deserialize

logger = logging.getLogger('fusio')


class torax_io(io):

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


    def read(self, path, side='input'):
        logger.warning(f'{self.format} reading function not defined yet...')


    def write(self, path, side='input', overwrite=False):
        opath = Path(path) if isinstance(path, (str, Path)) else None
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


    def _write_torax_file(self, path, data, overwrite=False):
        if isinstance(path, (str, Path)):
            opath = Path(path)
            if overwrite or not opath.exists():
                if isinstance(data, (xr.Dataset, xr.DataTree)):
                    data.to_netcdf(dump_path)
            else:
                logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting dump...')
        else:
            logger.warning(f'Invalid path argument given to {self.format} write function! Aborting dump...')


    def add_output_dir(self, outdir):
        newattrs = {}
        if isinstance(outdir, (str, Path)):
            newattrs['runtime_params.output_dir'] = f'{outdir}'
        self._input.attrs.update(newattrs)


    def add_geometry(self, geotype, geofile, geodir=None):
        newattrs = {}
        #if 'runtime_params.profile_conditions.psi' in self._input:
            #del self._input['runtime_params.profile_conditions.psi']
        newattrs['use_psi'] = False
        newattrs['geometry.geometry_type'] = f'{geotype}'
        newattrs['geometry.n_rho'] = 25
        newattrs['geometry.hires_fac'] = 4
        newattrs['geometry.geometry_file'] = f'{geofile}'
        if geodir is not None:
            newattrs['geometry.geometry_dir'] = f'{geodir}'
        newattrs['geometry.Ip_from_parameters'] = bool(self.input.attrs.get('runtime_params.profile_conditions.Ip_tot', False))
        if geotype == 'eqdsk':
            newattrs['geometry.n_surfaces'] = 100
            newattrs['geometry.last_surface_factor'] = 0.995
        self._input.attrs.update(newattrs)


    def add_pedestal_by_pressure(self, pped, nped, tpedratio, wrho):
        newattrs = {}
        nref = self.input.attrs.get('runtime_params.numerics.nref', 1.0e20)
        newattrs['runtime_params.profile_conditions.set_pedestal'] = True
        newattrs['transport.smooth_everywhere'] = False
        newattrs['pedestal.pedestal_model'] = 'set_pped_tpedratio_nped'
        newattrs['pedestal.Pped'] = {0.0: float(pped)}
        newattrs['pedestal.neped'] = {0.0: float(nped) / nref}
        newattrs['pedestal.neped_is_fGW'] = False
        newattrs['pedestal.ion_electron_temperature_ratio'] = {0.0: float(tpedratio)}
        newattrs['pedestal.rho_norm_ped_top'] = {0.0: float(wrho)}
        self._input.attrs.update(newattrs)


    def add_pedestal_by_temperature(self, nped, tped, wrho):
        newattrs = {}
        nref = self.input.attrs.get('runtime_params.numerics.nref', 1.0e20)
        tref = 1.0e3
        newattrs['runtime_params.profile_conditions.set_pedestal'] = True
        newattrs['transport.smooth_everywhere'] = False
        newattrs['pedestal.pedestal_model'] = 'set_tped_nped'
        newattrs['pedestal.neped'] = {0.0: float(nped) / nref}
        newattrs['pedestal.neped_is_fGW'] = False
        newattrs['pedestal.Teped'] = {0.0: float(tped) / tref}
        newattrs['pedestal.Tiped'] = {0.0: float(tped) / tref}
        newattrs['pedestal.rho_norm_ped_top'] = {0.0: float(wrho)}
        self._input.attrs.update(newattrs)


    def add_default_transport(self):
        newattrs = {}
        newattrs['transport.transport_model'] = 'qlknn'
        newattrs['transport.chimin'] = 0.05
        newattrs['transport.chimax'] = 100.0
        newattrs['transport.Demin'] = 0.01
        newattrs['transport.Demax'] = 50.0
        newattrs['transport.Vemin'] = -10.0
        newattrs['transport.Vemax'] = 10.0
        newattrs['transport.smoothing_sigma'] = 0.0
        newattrs['transport.smooth_everywhere'] = (not self.input.attrs.get('runtime_params.profile_conditions.set_pedestal', False))
        newattrs['transport.model_path'] = ''
        newattrs['transport.coll_mult'] = 0.25
        newattrs['transport.include_ITG'] = True
        newattrs['transport.include_TEM'] = True
        newattrs['transport.include_ETG'] = True
        newattrs['transport.DVeff'] = True
        newattrs['transport.An_min'] = 0.05
        newattrs['transport.avoid_big_negative_s'] = True
        newattrs['transport.smag_alpha_correction'] = True
        newattrs['transport.q_sawtooth_proxy'] = True
        self._input.attrs.update(newattrs)


    def set_qlknn_model_path(self, path):
        newattrs = {}
        if self.input.attrs.get('transport.transport_model', '') == 'qlknn':
            newattrs['transport.model_path'] = f'{path}'
        self._input.attrs.update(newattrs)


    def add_transport_inner_patch(self, de, ve, chii, chie, rho):
        newattrs = {}
        newattrs['transport.apply_inner_patch'] = {0.0: True}
        newattrs['transport.De_inner'] = {0.0: float(de)}
        newattrs['transport.Ve_inner'] = {0.0: float(ve)}
        newattrs['transport.chii_inner'] = {0.0: float(chii)}
        newattrs['transport.chie_inner'] = {0.0: float(chie)}
        newattrs['transport.rho_inner'] = float(rho)
        self._input.attrs.update(newattrs)


    def add_transport_outer_patch(self, de, ve, chii, chie, rho):
        newattrs = {}
        newattrs['transport.apply_outer_patch'] = {0.0: True}
        newattrs['transport.De_outer'] = {0.0: float(de)}
        newattrs['transport.Ve_outer'] = {0.0: float(ve)}
        newattrs['transport.chii_outer'] = {0.0: float(chii)}
        newattrs['transport.chie_outer'] = {0.0: float(chie)}
        newattrs['transport.rho_outer'] = float(rho)
        self._input.attrs.update(newattrs)


    def reset_exchange_source(self):
        newattrs = {}
        newattrs['sources.qei_source.mode'] = 'MODEL_BASED'
        newattrs['sources.qei_source.Qei_mult'] = 1.0
        if newattrs:
            self._input.attrs.pop('sources.qei_source.prescribed_values', None)
        self._input.attrs.update(newattrs)


    def reset_ohmic_source(self):
        newattrs = {}
        newattrs['sources.ohmic_heat_source.mode'] = 'MODEL_BASED'
        if newattrs:
            self._input.attrs.pop('sources.ohmic_heat_source.prescribed_values', None)
        self._input.attrs.update(newattrs)


    def reset_fusion_source(self):
        newattrs = {}
        newattrs['sources.fusion_heat_source.mode'] = 'MODEL_BASED'
        if newattrs:
            self._input.attrs.pop('sources.fusion_heat_source.prescribed_values', None)
        self._input.attrs.update(newattrs)


    def reset_gas_puff_source(self):
        newattrs = {}
        newattrs['sources.gas_puff_source.mode'] = 'MODEL_BASED'
        newattrs['sources.gas_puff_source.puff_decay_length'] = {0.0: 0.05}
        newattrs['sources.gas_puff_source.S_puff_tot'] = {0.0: 1.0e22}
        if newattrs:
            self._input.attrs.pop('sources.gas_puff_source.prescribed_values', None)
        self._input.attrs.update(newattrs)


    def reset_bootstrap_source(self):
        newattrs = {}
        newattrs['sources.j_bootstrap.mode'] = 'MODEL_BASED'
        newattrs['sources.j_bootstrap.bootstrap_mult'] = 1.0
        if newattrs:
            self._input.attrs.pop('sources.j_bootstrap.prescribed_values', None)
        self._input.attrs.update(newattrs)


    def reset_bremsstrahlung_source(self):
        newattrs = {}
        newattrs['sources.bremsstrahlung_heat_sink.mode'] = 'MODEL_BASED'
        newattrs['sources.bremsstrahlung_heat_sink.use_relativistic_correction'] = True
        if newattrs:
            self._input.attrs.pop('sources.bremsstrahlung_heat_sink.prescribed_values', None)
        self._input.attrs.update(newattrs)


    def reset_line_radiation_source(self):
        newattrs = {}
        newattrs['sources.impurity_radiation_heat_sink.mode'] = 'MODEL_BASED'
        newattrs['sources.impurity_radiation_heat_sink.model_function_name'] = 'impurity_radiation_mavrin_fit'
        newattrs['sources.impurity_radiation_heat_sink.radiation_multiplier'] = 1.0
        newattrs['sources.bremsstrahlung_heat_sink.mode'] = 'ZERO'
        if newattrs:
            self._input.attrs.pop('sources.impurity_radiation_heat_sink.prescribed_values', None)
            # Mavrin polynomial model includes Bremsstrahlung so zero that out as well
            self._input.attrs.pop('sources.bremsstrahlung_heat_sink.prescribed_values', None)
            self._input.attrs.pop('sources.bremsstrahlung_heat_sink.use_relativistic_correction', None)
        self._input.attrs.update(newattrs)


    def reset_synchrotron_source(self):
        newattrs = {}
        newattrs['sources.cyclotron_radiation_heat_sink.mode'] = 'MODEL_BASED'
        newattrs['sources.cyclotron_radiation_heat_sink.wall_reflection_coeff'] = 0.9
        if newattrs:
            self._input.attrs.pop('sources.cyclotron_radiation_heat_sink.prescribed_values', None)
        self._input.attrs.update(newattrs)


    def reset_generic_heat_source(self):
        newattrs = {}
        newattrs['sources.generic_ion_el_heat_source.mode'] = 'MODEL_BASED'
        newattrs['sources.generic_ion_el_heat_source.rsource'] = {0.0: 0.0}
        newattrs['sources.generic_ion_el_heat_source.w'] = {0.0: 0.1}
        newattrs['sources.generic_ion_el_heat_source.Ptot'] = {0.0: 0.0}
        newattrs['sources.generic_ion_el_heat_source.el_heat_fraction'] = {0.0: 0.5}
        if newattrs:
            self._input.attrs.pop('sources.generic_ion_el_heat_source.prescribed_values', None)
        self._input.attrs.update(newattrs)


    def add_linear_stepper(self):
        newattrs = {}
        newattrs['stepper.stepper_type'] = 'linear'
        newattrs['stepper.theta_imp'] = 1.0
        newattrs['stepper.predictor_corrector'] = True
        newattrs['stepper.corrector_steps'] = 10
        newattrs['stepper.use_pereverzev'] = True
        newattrs['stepper.chi_per'] = 20.0
        newattrs['stepper.d_per'] = 10.0
        newattrs['time_step_calculator.calculator_type'] = 'chi'
        self._input.attrs.update(newattrs)


    def add_newton_raphson_stepper(self):
        newattrs = {}
        newattrs['stepper.stepper_type'] = 'newton_raphson'
        newattrs['stepper.theta_imp'] = 1.0
        newattrs['stepper.predictor_corrector'] = True
        newattrs['stepper.corrector_steps'] = 10
        newattrs['stepper.use_pereverzev'] = True
        newattrs['stepper.chi_per'] = 20.0
        newattrs['stepper.d_per'] = 10.0
        newattrs['stepper.log_iterations'] = True
        newattrs['stepper.initial_guess_mode'] = 1
        newattrs['stepper.delta_reduction_factor'] = 0.5
        newattrs['stepper.maxiter'] = 30
        newattrs['stepper.tau_min'] = 0.01
        newattrs['time_step_calculator.calculator_type'] = 'chi'
        self._input.attrs.update(newattrs)


    def set_numerics(self, t_initial, t_final, eqs=['te', 'ti', 'ne', 'j'], dtmult=None):
        newattrs = {}
        newattrs['runtime_params.numerics.t_initial'] = float(t_initial)
        newattrs['runtime_params.numerics.t_final'] = float(t_final)
        newattrs['runtime_params.numerics.exact_t_final'] = True
        newattrs['runtime_params.numerics.maxdt'] = 1.0e-1
        newattrs['runtime_params.numerics.mindt'] = 1.0e-8
        newattrs['runtime_params.numerics.dtmult'] = float(dtmult) if isinstance(dtmult, (float, int)) else 9.0
        newattrs['runtime_params.numerics.resistivity_mult'] = 1.0
        newattrs['runtime_params.numerics.el_heat_eq'] = (isinstance(eqs, (list, tuple)) and 'te' in eqs)
        newattrs['runtime_params.numerics.ion_heat_eq'] = (isinstance(eqs, (list, tuple)) and 'ti' in eqs)
        newattrs['runtime_params.numerics.dens_eq'] = (isinstance(eqs, (list, tuple)) and 'ne' in eqs)
        newattrs['runtime_params.numerics.current_eq'] = (isinstance(eqs, (list, tuple)) and 'j' in eqs)
        self._input.attrs.update(newattrs)


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
                    for ii in range(len(time)):
                        time_dependent_var[float(time[ii])] = {str(species): val for species, val in zip(ds['main_ion'].to_numpy().flatten(), ds[key].isel(time=ii).to_numpy().flatten())}
                else:
                    for ii in range(len(time)):
                        time_dependent_var[float(time[ii])] = float(ds[key].isel(time=ii).to_numpy().flatten())
                datadict[key] = time_dependent_var
        return self._unflatten(datadict)


    @classmethod
    def from_file(cls, obj, side='output'):
        raise NotImplementedError(f'{self.__class__.__name__}.from_file() not yet implemented since there is not standard TORAX config file format.')


    @classmethod
    def from_gacode(cls, obj, side='output', n=0):
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            if 'n' in data.coords and  'rho' in data.coords:
                coords = {}
                data_vars = {}
                attrs = {}
                data = data.sel(n=n)
                time = data.get('time', 0.0)
                attrs['runtime_params.numerics.t_initial'] = float(time)
                attrs['runtime_params.numerics.nref'] = 1.0e20
                coords['time'] = np.array([time])
                coords['rho'] = data['rho'].to_numpy().flatten()
                if 'z' in data and 'name' in data and 'type' in data and 'ni' in data:
                    species = []
                    density = []
                    nfilt = (np.isclose(data['z'], 1.0) & (['fast' not in v for v in data['type'].to_numpy().flatten()]))
                    if np.any(nfilt):
                        namelist = data['name'].to_numpy()[nfilt].tolist()
                        nfuelsum = data['ni'].sum('name').to_numpy().flatten()
                        for ii in range(len(namelist)):
                            sdata = data['ni'].sel(name=namelist[ii])
                            species.append(namelist[ii])
                            density.append(float((sdata.to_numpy().flatten() / nfuelsum).mean()))
                        coords['main_ion'] = species
                    else:
                        species = ['D']
                        density = [1.0]
                    coords['main_ion'] = species
                    data_vars['runtime_params.plasma_composition.main_ion'] = (['time', 'main_ion'], np.expand_dims(density, axis=0))
                if 'z' in data and 'mass' in data and 'ni' in data and 'ne' in data:
                    nfilt = (~np.isclose(data['z'], 1.0))
                    zeff = np.ones_like(data['ne'].to_numpy())
                    if np.any(nfilt):
                        namelist = data['name'].to_numpy()[nfilt].tolist()
                        attrs['runtime_params.plasma_composition.impurity'] = 'Ne'
                        zeff = np.zeros_like(data['ne'].to_numpy())
                        nzave = np.zeros_like(data['ne'].to_numpy())
                        for ii in range(len(data['name'])):
                            sdata = data.isel(name=ii)
                            if sdata['name'] in namelist:
                                nzave += sdata['ni'].to_numpy().flatten() * sdata['z'].to_numpy().flatten() / data['ne'].to_numpy().flatten()
                            else:
                                zeff += sdata['ni'].to_numpy().flatten() * sdata['z'].to_numpy().flatten() ** 2.0 / data['ne'].to_numpy().flatten()
                        zeff += nzave * 10.0
                    data_vars['runtime_params.plasma_composition.Zeff'] = (['time', 'rho'], np.expand_dims(zeff, axis=0))
                if 'current' in data:
                    data_vars['runtime_params.profile_conditions.Ip_tot'] = (['time'], np.expand_dims(data['current'].mean(), axis=0))
                if 'ne' in data:
                    nref = attrs.get('runtime_params.numerics.nref', 1.0e20)
                    data_vars['runtime_params.profile_conditions.ne'] = (['time', 'rho'], np.expand_dims(1.0e19 * data['ne'].to_numpy().flatten() / nref, axis=0))
                    attrs['runtime_params.profile_conditions.normalize_to_nbar'] = False
                    attrs['runtime_params.profile_conditions.ne_is_fGW'] = False
                if 'te' in data:
                    data_vars['runtime_params.profile_conditions.Te'] = (['time', 'rho'], np.expand_dims(data['te'].to_numpy().flatten(), axis=0))
                if 'ti' in data and 'z' in data:
                    nfilt = (np.isclose(data['z'], 1.0) & (['fast' not in v for v in data['type'].to_numpy().flatten()]))
                    tfuel = data['ti'].mean('name')
                    if np.any(nfilt):
                        namelist = data['name'].to_numpy()[nfilt].tolist()
                        tfuel = data['ti'].sel(name=namelist).mean('name')
                    data_vars['runtime_params.profile_conditions.Ti'] = (['time', 'rho'], np.expand_dims(tfuel.to_numpy().flatten(), axis=0))
                if 'polflux' in data:
                    data_vars['runtime_params.profile_conditions.psi'] = (['time', 'rho'], np.expand_dims(data['polflux'].to_numpy().flatten(), axis=0))
                # Place the sources
                external_el_heat_source = None
                external_ion_heat_source = None
                external_particle_source = None
                external_current_source = None
                fusion_source = None
                if 'qohme' in data and data['qohme'].sum() != 0.0:
                    attrs['sources.ohmic_heat_source.mode'] = 'PRESCRIBED'
                    data_vars['sources.ohmic_heat_source.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qohme'].to_numpy().flatten(), axis=0))
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
                    attrs['sources.cyclotron_radiation_heat_sink.mode'] = 'PRESCRIBED'
                    data_vars['sources.cyclotron_radiation_heat_sink.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qsync'].to_numpy().flatten(), axis=0))
                if 'qbrem' in data and data['qbrem'].sum() != 0.0:
                    attrs['sources.bremsstrahlung_heat_sink.mode'] = 'PRESCRIBED'
                    data_vars['sources.bremsstrahlung_heat_sink.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qbrem'].to_numpy().flatten(), axis=0))
                if 'qline' in data and data['qline'].sum() != 0.0:
                    attrs['sources.impurity_radiation_heat_sink.mode'] = 'PRESCRIBED'
                    data_vars['sources.impurity_radiation_heat_sink.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qline'].to_numpy().flatten(), axis=0))
                if 'qfuse' in data and data['qfuse'].sum() != 0.0:
                    if fusion_source is None:
                        fusion_source = np.zeros_like(data['qfuse'].to_numpy().flatten())
                    fusion_source += 1.0e6 * data['qfuse'].to_numpy().flatten()
                if 'qfusi' in data and data['qfusi'].sum() != 0.0:
                    if fusion_source is None:
                        fusion_source = np.zeros_like(data['qfuse'].to_numpy().flatten())
                    fusion_source += 1.0e6 * data['qfuse'].to_numpy().flatten()
                if 'qei' in data and data['qei'].sum() != 0.0:
                    attrs['sources.qei_source.mode'] = 'PRESCRIBED'
                    data_vars['sources.qei_source.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['qei'].to_numpy().flatten(), axis=0))
                #if 'qione' in data and data['qione'].sum() != 0.0:
                #    pass
                #if 'qioni' in data and data['qioni'].sum() != 0.0:
                #    pass
                #if 'qcxi' in data and data['qcxi'].sum() != 0.0:
                #    pass
                if 'jbs' in data and data['jbs'].sum() != 0.0:
                    attrs['sources.j_bootstrap.mode'] = 'PRESCRIBED'
                    data_vars['sources.j_bootstrap.prescribed_values'] = (['time', 'rho'], np.expand_dims(1.0e6 * data['jbs'].to_numpy().flatten(), axis=0))
                #if 'jbstor' in data and data['jbstor'].sum() != 0.0:
                #    pass
                if 'johm' in data and data['johm'].sum() != 0.0:
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
                    #total_heat_source = copy.deepcopy(external_el_heat_source)
                    #if external_ion_heat_source is not None:
                    #    total_heat_source += external_ion_heat_source
                    #el_heat_fraction = (external_el_heat_source / total_heat_source).mean()
                    attrs['sources.generic_ion_el_heat_source.mode'] = 'PRESCRIBED'
                    data_vars['sources.generic_ion_el_heat_source.prescribed_values_el'] = (['time', 'rho'], np.expand_dims(external_ion_heat_source, axis=0))
                    data_vars['sources.generic_ion_el_heat_source.prescribed_values_ion'] = (['time', 'rho'], np.expand_dims(external_el_heat_source, axis=0))
                    #data_vars['sources.generic_ion_el_heat_source.el_heat_fraction'] = (['time'], np.expand_dims([el_heat_fraction], axis=0))
                if external_particle_source is not None:
                    attrs['sources.generic_particle_source.mode'] = 'PRESCRIBED'
                    data_vars['sources.generic_particle_source.prescribed_values'] = (['time', 'rho'], np.expand_dims(external_particle_source, axis=0))
                if external_current_source is not None:
                    attrs['sources.generic_current_source.mode'] = 'PRESCRIBED'
                    data_vars['sources.generic_current_source.prescribed_values'] = (['time', 'rho'], np.expand_dims(external_current_source, axis=0))
                    attrs['sources.generic_cuurent_source.use_absolute_current'] = True
                newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj
