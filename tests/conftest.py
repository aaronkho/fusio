import pytest
from pathlib import Path
import numpy as np
import xarray as xr


@pytest.fixture(scope='session')
def gacode_file_path():
    return Path(__file__).parent / 'data' / 'test_input.gacode'

@pytest.fixture(scope='module')
def physical_2ion_plasma_state():
    coords = {
        'time': np.array([0.0]),
        'radius': np.array([0.5]),
        'ion': np.array(['D', 'Ne']),
        'direction': np.array(['toroidal', 'poloidal']),
    }
    data_vars = {
        'field_axis': (['time'], np.array([3.0])),
        'mass_e': (['time'], np.array([0.00054858])),
        'charge_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([1.0, 10.0]), axis=0)),
        'mass_i': (['time', 'ion'], np.atleast_2d([2.0, 20.0])),
        'density_e': (['time', 'radius'], np.atleast_2d([1.0e19])),
        'temperature_e': (['time', 'radius'], np.atleast_2d([2.0e3])),
        'grad_density_e': (['time', 'radius'], np.atleast_2d([-2.0e19])),
        'grad_temperature_e': (['time', 'radius'], np.atleast_2d([-6.0e3])),
        'density_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([9.0e18, 1.0e17]), axis=0)),
        'temperature_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([3.0e3, 3.0e3]), axis=0)),
        'grad_density_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([-1.71e19, -2.9e17]), axis=0)),
        'grad_temperature_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([-1.2e4, -1.2e4]), axis=0)),
        'gyroradius_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([0.00374711, 0.00118494]), axis=0)),
        'r_minor': (['time', 'radius'], np.atleast_2d([0.275])),
        'r_geometric': (['time', 'radius'], np.atleast_2d([2.1])),
        'r_minor_lcfs': (['time'], np.array([0.5])),
        'r_geometric_lcfs': (['time'], np.array([2.0])),
        'field_geometric': (['time', 'radius', 'direction'], np.expand_dims(np.atleast_2d([2.1, 0.1375]), axis=0)),
        'grad_field_geometric': (['time', 'radius', 'direction'], np.expand_dims(np.atleast_2d([7.35, 0.4725]), axis=0)),
        'velocity_i': (['time', 'radius', 'ion', 'direction'], np.expand_dims(np.expand_dims(np.atleast_2d([[1.0e5, 1.0e4], [1.0e5, 1.0e4]]), axis=0), axis=0)),
        'gyroradius_ref': (['time', 'radius'], np.atleast_2d([0.0030595])),
        'velocity_ref': (['time', 'radius'], np.atleast_2d([3.1062112e5])),
        'velocity_thermal_e': (['time', 'radius'], np.atleast_2d([2.652412e7]))
    }
    return xr.Dataset(coords=coords, data_vars=data_vars)

@pytest.fixture(scope='module')
def dimensionless_2ion_plasma_state():
    coords = {
        'time': np.array([0.0]),
        'radius': np.array([0.5]),
        'ion': np.array(['D', 'Ne']),
        'direction': np.array(['toroidal', 'poloidal']),
    }
    data_vars = {
        'charge_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([1.0, 10.0]), axis=0)),
        'mass_i': (['time', 'ion'], np.atleast_2d([2.0, 20.0])),
        'grad_density_e_norm': (['time', 'radius'], np.atleast_2d([1.0])),
        'grad_temperature_e_norm': (['time', 'radius'], np.atleast_2d([1.5])),
        'density_i_norm': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([0.9, 0.01]), axis=0)),
        'temperature_i_norm': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([1.5, 1.5]), axis=0)),
        'grad_density_i_norm': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([0.95, 1.45]), axis=0)),
        'grad_temperature_i_norm': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([2.0, 2.0]), axis=0)),
        'pressure_total_norm': (['time', 'radius'], np.atleast_2d([0.004300469])),
        'grad_pressure_total_norm': (['time', 'radius'], np.atleast_2d([0.047527])),
        'x': (['time', 'radius'], np.atleast_2d([0.55])),
        'epsilon_lcfs': (['time'], np.array([0.25])),
        'aspect_ratio_lcfs': (['time'], np.array([4.0])),
        'gyroradius_ref_norm': (['time', 'radius'], np.atleast_2d([0.006119])),
        'safety_factor_circular': (['time', 'radius'], np.atleast_2d([2.0])),
        'grad_safety_factor_circular': (['time', 'radius'], np.atleast_2d([0.4])),
        'magnetic_shear_circular': (['time', 'radius'], np.atleast_2d([0.055])),
        'effective_charge': (['time', 'radius'], np.atleast_2d([1.9])),
        'grad_effective_charge': (['time', 'radius'], np.atleast_2d([-0.81])),
        'grad_effective_charge_norm': (['time', 'radius'], np.atleast_2d([0.405 / 1.9])),
        'coulomb_logarithm_nrl': (['time', 'radius'], np.atleast_2d([17.04444])),
        'coulomb_logarithm': (['time', 'radius'], np.atleast_2d([18.79925])),
        'collisionality_nrl_norm': (['time', 'radius'], np.atleast_2d([0.04963843])),
        'mach_i': (['time', 'radius', 'ion', 'direction'], np.expand_dims(np.expand_dims(np.atleast_2d([[0.321936, 0.0321936], [0.321936, 0.0321936]]), axis=0), axis=0)),
    }
    return xr.Dataset(coords=coords, data_vars=data_vars)

@pytest.fixture(scope='module')
def physical_3ion_plasma_state():
    coords = {
        'time': np.array([0.0]),
        'radius': np.array([0.5]),
        'ion': np.array(['D', 'Ne', 'Ar']),
        'direction': np.array(['toroidal', 'poloidal']),
    }
    data_vars = {
        'field_axis': (['time'], np.array([3.0])),
        'charge_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([1.0, 10.0, 18.0]), axis=0)),
        'mass_i': (['time', 'ion'], np.atleast_2d([2.0, 20.0, 40.0])),
        'density_e': (['time', 'radius'], np.atleast_2d([1.0e19])),
        'temperature_e': (['time', 'radius'], np.atleast_2d([2.0e3])),
        'grad_density_e': (['time', 'radius'], np.atleast_2d([-2.0e19])),
        'grad_temperature_e': (['time', 'radius'], np.atleast_2d([-6.0e3])),
        'density_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([9.0e18, 9.1e16, 5.0e15]), axis=0)),
        'temperature_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([3.0e3, 3.0e3, 3.0e3]), axis=0)),
        'grad_density_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([-1.836e19, -1.4987e17, -7.85e15]), axis=0)),
        'grad_temperature_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([-1.2e4, -1.2e4, -1.2e4]), axis=0)),
        'r_minor': (['time', 'radius'], np.atleast_2d([0.275])),
        'r_geometric': (['time', 'radius'], np.atleast_2d([2.1])),
        'r_minor_lcfs': (['time'], np.array([0.5])),
        'r_geometric_lcfs': (['time'], np.array([2.0])),
        'field_geometric': (['time', 'radius', 'direction'], np.expand_dims(np.atleast_2d([2.1, 0.1375]), axis=0)),
        'grad_field_geometric': (['time', 'radius', 'direction'], np.expand_dims(np.atleast_2d([7.35, 0.4725]), axis=0)),
    }
    return xr.Dataset(coords=coords, data_vars=data_vars)

@pytest.fixture(scope='module')
def dimensionless_3ion_plasma_state():
    coords = {
        'time': np.array([0.0]),
        'radius': np.array([0.5]),
        'ion': np.array(['D', 'Ne', 'Ar']),
    }
    data_vars = {
        'charge_i': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([1.0, 10.0, 18.0]), axis=0)),
        'mass_i': (['time', 'ion'], np.atleast_2d([2.0, 20.0, 40.0])),
        'grad_density_e_norm': (['time', 'radius'], np.atleast_2d([1.0])),
        'grad_temperature_e_norm': (['time', 'radius'], np.atleast_2d([1.5])),
        'density_i_norm': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([0.9, 0.0091, 0.0005]), axis=0)),
        'temperature_i_norm': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([1.5, 1.5, 1.5]), axis=0)),
        'grad_density_i_norm': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([1.02, 0.823461538461538, 0.785]), axis=0)),
        'grad_temperature_i_norm': (['time', 'radius', 'ion'], np.expand_dims(np.atleast_2d([2.0, 2.0, 2.0]), axis=0)),
        'grad_pressure_total_norm': (['time', 'radius'], np.atleast_2d([0.04813346])),
        'safety_factor_circular': (['time', 'radius'], np.atleast_2d([2.0])),
        'effective_charge': (['time', 'radius'], np.atleast_2d([1.972])),
        'grad_effective_charge': (['time', 'radius'], np.atleast_2d([0.35496])),
        'grad_effective_charge_norm': (['time', 'radius'], np.atleast_2d([-0.09])),
    }
    return xr.Dataset(coords=coords, data_vars=data_vars)