import copy
import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

import datetime
from scipy.integrate import quad  # type: ignore[import-untyped]
from .io import io
from ..utils.plasma_tools import define_ion_species
from ..utils.math_tools import (
    vectorized_numpy_derivative,
    vectorized_numpy_integration,
    vectorized_numpy_interpolation,
    vectorized_numpy_find,
)
from ..utils.eqdsk_tools import (
    read_eqdsk,
)

logger = logging.getLogger('fusio')


class plasma_io(io):


    basevars: Final[Sequence[str]] = [
        'atomic_number_i',
        'type_i',
        'mass_e',
        'mass_i',
        'charge_e',
        'charge_i',
        'field_axis',
        'current',
        'magnetic_flux',
        'r_minor',
        'r_geometric',
        'z_geometric',
        'density_i',
        'temperature_i',
        'density_e',
        'temperature_e',
        'velocity_i',
        'heat_source_e',
        'particle_source_e',
        'heat_source_i',
        'particle_source_i',
        'momentum_source_i',
        'current_source',
        'heat_exchange_ei',
        'contour',
    ]
    units: Final[Mapping[str, str]] = {
        'field_axis': 'T',
        'current': 'A',
        'magnetic_flux': 'Wb/radian',
        'r_minor': 'm',
        'r_geometric': 'm',
        'z_geometric': 'm',
        'density_i': '1/m^3',
        'temperature_i': 'eV',
        'density_e': '1/m^3',
        'temperature_e': 'eV',
        'velocity_i': 'm/s',
        'heat_source_e': 'W/m^3',
        'particle_source_e': '1/m^3/s',
        'heat_source_i': 'W/m^3',
        'particle_source_i': '1/m^3/s',
        'momentum_source_i': 'N/m^2',
        'current_source': 'A/m^2',
        'heat_exchange_ei': 'W/m^3',
        'contour': 'm',
    }
    constants: Final[Mapping[str, float]] = {
        'e_si': 1.60218e-19,  # C
        'u_si': 1.66054e-27,  # kg
        'eps_si': 8.85419e-12,  # F/m
        'mu_si': 4.0e-7 * np.pi,  # H/m
        'atm_si': 1.01325e5,  # Pa
        'me_si': 5.4858e-4 * 1.66054e-27,  # kg
    }
    directions: Final[Sequence[str]] = [
        'toroidal',
        'poloidal',
    ]
    sources: Final[Sequence[str]] = [
        'ohmic',
        'neutral_beam',
        'ion_cyclotron',
        'electron_cyclotron',
        'synchrotron',
        'bremsstrahlung',
        'line_radiation',
        'ionization',
        'charge_exchange',
        'bootstrap',
        'fusion',
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


    def read(
        self,
        path: str | Path,
        side: str = 'output',
    ) -> None:
        if side == 'input':
            self.input = self._read_plasma_file(path)
        else:
            self.output = self._read_plasma_file(path)


    def write(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False
    ) -> None:
        if side == 'input':
            self._write_plasma_file(path, self.input, overwrite=overwrite)
        else:
            self._write_plasma_file(path, self.output, overwrite=overwrite)


    def _read_plasma_file(
        self,
        path: str | Path
    ) -> xr.Dataset:
        data = xr.Dataset()
        if isinstance(path, (str, Path)):
            load_path = Path(path)
            if load_path.exists():
                data = xr.open_dataset(path, engine='netcdf4')
        return data


    def _write_plasma_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        overwrite: bool = False
    ) -> None:
        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            opath = Path(path)
            if not opath.exists() or overwrite:
                data.to_netcdf(opath, mode='w', format='NETCDF4')
                logger.info(f'Saved {self.format} data into {opath.resolve()}')
            else:
                logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    def add_geometry_from_eqdsk(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False,
    ) -> None:
        data = self.input if side == 'input' else self.output
        if isinstance(path, (str, Path)) and 'polflux' in data:
            eqdsk_data = read_eqdsk(path)
            newvars: MutableMapping[str, Any] = {}
            if newvars:
                if side == 'input':
                    self.update_input_data_vars(newvars)
                else:
                    self.update_output_data_vars(newvars)


    def interpolate(
        self,
        v: float | NDArray,
        xvar: str,  # x-coordinate to be interpolated on
        yvar: str,  # y-coordinate to be interpolated
        cvar: str,  # common coordinate in xarray representation
        extrapolate: bool = False,
        side: str = 'input',
    ) -> NDArray:
        data = self.input if side == 'input' else self.output
        if xvar in data and yvar in data and cvar in data[f'{xvar}'].dims and cvar in data[f'{yvar}'].dims:
            idim_y = data[f'{yvar}'].dims.index(cvar)
            dimlist = tuple([f'{v}' for v in data[f'{yvar}'].dims if f'{v}' != f'{cvar}'] + [f'{cvar}'])
            newxdims = {f'{v}': data[f'{v}'].to_numpy() for v in dimlist if v not in data[f'{xvar}'].dims}
            x = data[f'{xvar}'].expand_dims(newxdims).transpose(dimlist).to_numpy()
            y = data[f'{yvar}'].transpose(dimlist).to_numpy()
            outdims = [i for i in range(len(dimlist) - 1)]
            outdims.insert(idim_y, len(dimlist) - 1)
            return np.transpose(vectorized_numpy_interpolation(v, x, y, extrapolate=extrapolate), axes=outdims)
        else:
            logger.error(f'Invalid variable names given to {self.format} interpolate function! Aborting interpolation...')
            return np.array([np.nan]) * v


    def _compute_derived_coordinates(
        self,
    ) -> None:
        newvars: MutableMapping[str, Any] = {}
        if self.has_input:
            data = self.input
            r_minor_lcfs = data['r_minor'].interp(coords={'radius': 1.0}, kwargs={'fill_value': 'extrapolate'})
            r_geometric_lcfs = data['r_geometric'].interp(coords={'radius': 1.0}, kwargs={'fill_value': 'extrapolate'})
            z_geometric_lcfs = data['z_geometric'].interp(coords={'radius': 1.0}, kwargs={'fill_value': 'extrapolate'})
            magnetic_flux_lcfs = data['magnetic_flux'].interp(coords={'radius': 1.0}, kwargs={'fill_value': 'extrapolate'})
            newvars['r_minor_lcfs'] = (['time'], r_minor_lcfs.to_numpy())
            newvars['r_geometric_lcfs'] = (['time'], r_geometric_lcfs.to_numpy())
            newvars['z_geometric_lcfs'] = (['time'], z_geometric_lcfs.to_numpy())
            newvars['r_minor_norm'] = (['time', 'radius'], (data['r_minor'] / r_minor_lcfs).to_numpy())
            newvars['r_geometric_norm'] = (['time', 'radius'], (data['r_geometric'] / r_minor_lcfs).to_numpy())
            newvars['r_geometric_norm_lcfs'] = (['time'], (r_geometric_lcfs / r_minor_lcfs).to_numpy())
            newvars['z_geometric_norm'] = (['time', 'radius'], (data['z_geometric'] / r_minor_lcfs).to_numpy())
            newvars['z_geometric_norm_lcfs'] = (['time'], (z_geometric_lcfs / r_minor_lcfs).to_numpy())
            newvars['aspect_ratio'] = (['time', 'radius'], (data['r_geometric'] / data['r_minor']).to_numpy())
            newvars['aspect_ratio_lcfs'] = (['time'], (r_geometric_lcfs / r_minor_lcfs).to_numpy())
            newvars['epsilon'] = (['time', 'radius'], (data['r_minor'] / data['r_geometric']).to_numpy())
            newvars['epsilon_lcfs'] = (['time'], (r_minor_lcfs / r_geometric_lcfs).to_numpy())
            newvars['magnetic_flux_norm'] = (['time', 'radius', 'direction'], (data['magnetic_flux'] / magnetic_flux_lcfs).to_numpy())
            newvars['rho'] = (['time', 'radius', 'direction'], ((data['magnetic_flux'] / (0.5 * data['field_axis'])) ** 0.5).to_numpy())
            newvars['rho_norm'] = (['time', 'radius', 'direction'], ((data['magnetic_flux'] / magnetic_flux_lcfs) ** 0.5).to_numpy())
            contour_r = (data['contour'] * np.cos(data['angle_geometric']) + data['r_geometric']).to_numpy()
            contour_z = (data['contour'] * np.sin(data['angle_geometric']) + data['z_geometric']).to_numpy()
            xs_area = np.trapezoid(contour_r, contour_z, axis=-1)
            vol = 2.0 * np.pi * data['r_geometric'].to_numpy() * xs_area
            volp = vectorized_numpy_derivative(data['r_minor'].to_numpy(), vol)
            newvars['cross_sectional_area'] = (['time', 'radius'], xs_area)
            newvars['volume'] = (['time', 'radius'], vol)
            newvars['dvolume_dr'] = (['time', 'radius'], volp)
        self.update_input_data_vars(newvars)


    def _compute_derived_reference_quantities(
        self,
    ) -> None:
        newvars: MutableMapping[str, Any] = {}
        if self.has_input:
            data = self.input
            main_species_mask = (np.isclose(data['atomic_number_i'].to_numpy(), 1.0) & (data['type_i'].isin(['thermal'])).to_numpy()).flatten()
            main_species = [i for i in range(len(main_species_mask)) if main_species_mask[i]]
            n_i_vol = vectorized_numpy_integration(
                np.transpose((data['density_i'].isel(ion=main_species) * data['dvolume_dr']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(main_species), axis=1)
            )
            mass_ave = np.sum((data['mass_i'].isel(ion=main_species).to_numpy() * n_i_vol[..., -1] / np.expand_dims(np.sum(n_i_vol[..., -1], axis=1), axis=1)), axis=-1)
            mass_ref = data.get('mass_ref', xr.zeros_like(data['time']) + 2.0)
            length_ref = data.get('length_ref', xr.zeros_like(data['time']) + data['r_minor_lcfs'])
            field_unit = vectorized_numpy_derivative(0.5 * data['r_minor'].to_numpy() ** 2, data['magnetic_flux'].sel(direction='toroidal').to_numpy() / (2.0 * np.pi))
            safety_factor = vectorized_numpy_derivative(data['magnetic_flux'].sel(direction='toroidal').to_numpy(), data['magnetic_flux'].sel(direction='poloidal').to_numpy())
            safety_factor[..., 0] = 2.0 * safety_factor[..., 1] - safety_factor[..., 2]
            newvars['mass_ref'] = (['time'], mass_ref.to_numpy())
            newvars['mass_main_average'] = (['time'], mass_ave)
            newvars['length_ref'] = (['time'], length_ref.to_numpy())
            newvars['field_unit'] = (['time', 'radius'], field_unit)
            newvars['velocity_sound_ref'] = (['time', 'radius'], ((self.constants['e_si'] * data['temperature_e'] / (self.constants['u_si'] * mass_ref)) ** 0.5).to_numpy())
            newvars['velocity_sound_i'] = (['time', 'radius', 'ion'], ((self.constants['e_si'] * data['temperature_e'] / (self.constants['u_si'] * data['mass_i'])) ** 0.5).to_numpy())
            newvars['gyroradius_ref_unit'] = (['time', 'radius'], ((data['temperature_e'] * self.constants['u_si'] * mass_ref / self.constants['e_si']) ** 0.5).to_numpy() / np.abs(field_unit))
            newvars['velocity_thermal_e'] = (['time', 'radius'], ((2.0 * self.constants['e_si'] * data['temperature_e'] / (self.constants['u_si'] * data['mass_e'])) ** 0.5).to_numpy())
            newvars['velocity_thermal_i'] = (['time', 'radius', 'ion'], ((2.0 * self.constants['e_si'] * data['temperature_i'] / (self.constants['u_si'] * data['mass_i'])) ** 0.5).to_numpy())
            newvars['safety_factor'] = (['time', 'radius'], safety_factor)
            newvars['magnetic_shear'] = (['time', 'radius'], (data['r_minor'].to_numpy() / safety_factor) * vectorized_numpy_derivative(data['r_minor'].to_numpy(), safety_factor))
            #newvars['dqdr'] = (['n', 'rho'], vectorized_numpy_derivative(data['r_minor'].to_numpy(), data['q'].to_numpy()))
        self.update_input_data_vars(newvars)


    def _compute_derived_geometry(
        self,
    ) -> None:
        newcoords: MutableMapping[str, Any] = {}
        newvars: MutableMapping[str, Any] = {}
        if self.has_input:
            data = self.input
            n_coeffs = 7
            contour_r = (data['contour'] * np.cos(data['angle_geometric']) + data['r_geometric']).to_numpy()
            contour_z = (data['contour'] * np.sin(data['angle_geometric']) + data['z_geometric']).to_numpy()
            mxh_r0 = (np.nanmax(contour_r, axis=-1) + np.nanmin(contour_r, axis=-1)) / 2.0
            mxh_dr0 = vectorized_numpy_derivative(data['r_minor_norm'].to_numpy(), mxh_r0)
            mxh_z0 = (np.nanmax(contour_z, axis=-1) + np.nanmin(contour_z, axis=-1)) / 2.0
            mxh_dz0 = vectorized_numpy_derivative(data['r_minor_norm'].to_numpy(), mxh_z0)
            mxh_r = (np.nanmax(contour_r, axis=-1) - np.nanmin(contour_r, axis=-1)) / 2.0
            mxh_kappa = (np.nanmax(contour_z, axis=-1) - np.nanmin(contour_z, axis=-1)) / (2.0 * mxh_r)
            mxh_s_kappa = data['r_minor_norm'].to_numpy() / mxh_kappa * vectorized_numpy_derivative(data['r_minor_norm'].to_numpy(), mxh_kappa)
            r_norm = (contour_r - np.expand_dims(mxh_r0, axis=-1)) / np.expand_dims(mxh_r, axis=-1)
            r_norm = np.where(r_norm > 1.0, 1.0, np.where(r_norm < -1.0, -1.0, r_norm))
            z_norm = (contour_z - np.expand_dims(mxh_z0, axis=-1)) / np.expand_dims(mxh_kappa * mxh_r, axis=-1)
            z_norm = np.where(z_norm > 1.0, 1.0, np.where(z_norm < -1.0, -1.0, z_norm))
            angle_r_norm = np.where(z_norm[..., :-1] < 0.0, 2.0 * np.pi - np.arccos(r_norm[..., :-1]), np.arccos(r_norm[..., :-1]))
            angle_r_norm = np.concatenate([angle_r_norm, np.expand_dims(angle_r_norm[..., 0], axis=-1) + 2.0 * np.pi], axis=-1)
            angle_z_norm = np.where(r_norm[..., :-1] < 0.0, np.pi - np.arcsin(z_norm[..., :-1]), np.arcsin(z_norm[..., :-1]))
            angle_z_norm = np.where(angle_z_norm < 0.0, 2.0 * np.pi + angle_z_norm, angle_z_norm)
            angle_z_norm = np.concatenate([angle_z_norm, np.expand_dims(angle_z_norm[..., 0], axis=-1) + 2.0 * np.pi], axis=-1)
            mxh_sin = np.repeat(np.expand_dims(np.zeros_like(mxh_r), axis=-1), n_coeffs, axis=-1)
            mxh_cos = np.repeat(np.expand_dims(np.zeros_like(mxh_r), axis=-1), n_coeffs, axis=-1)
            for i in range(n_coeffs):
                sint_func = lambda angle_z, angle_r: quad(np.interp, 0.0, 2.0 * np.pi, weight='sin', wvar=i, args=(angle_z, angle_r - angle_z))[0]
                sint_vfunc = np.vectorize(sint_func, signature='(n),(n)->()')
                cint_func = lambda angle_z, angle_r: quad(np.interp, 0.0, 2.0 * np.pi, weight='cos', wvar=i, args=(angle_z, angle_r - angle_z))[0]
                cint_vfunc = np.vectorize(cint_func, signature='(n),(n)->()')
                mxh_sin[..., i] = sint_vfunc(angle_z_norm, angle_r_norm) / np.pi
                mxh_cos[..., i] = cint_vfunc(angle_z_norm, angle_r_norm) / np.pi
            mxh_cos[..., 0] = mxh_cos[..., 0] / 2.0
            mxh_sin[:, 0, :] = 2.0 * mxh_sin[:, 1, :] - mxh_sin[:, 2, :]
            mxh_cos[:, 0, :] = 2.0 * mxh_cos[:, 1, :] - mxh_cos[:, 2, :]
            mxh_s_sin = np.expand_dims(data['r_minor_norm'].to_numpy(), axis=-1) * np.transpose(vectorized_numpy_derivative(
                np.transpose(np.repeat(np.expand_dims(data['r_minor_norm'].to_numpy(), axis=-1), n_coeffs, axis=-1), axes=(0, 2, 1)),
                np.transpose(mxh_sin, axes=(0, 2, 1))
            ), axes=(0, 2, 1))
            mxh_s_cos = np.expand_dims(data['r_minor_norm'].to_numpy(), axis=-1) * np.transpose(vectorized_numpy_derivative(
                np.transpose(np.repeat(np.expand_dims(data['r_minor_norm'].to_numpy(), axis=-1), n_coeffs, axis=-1), axes=(0, 2, 1)),
                np.transpose(mxh_cos, axes=(0, 2, 1))
            ), axes=(0, 2, 1))
            mxh_delta = np.sin(mxh_sin[..., 1])
            mxh_s_delta = data['r_minor_norm'].to_numpy() * vectorized_numpy_derivative(data['r_minor_norm'].to_numpy(), mxh_delta)
            mxh_zeta = -mxh_sin[..., 2]
            mxh_s_zeta = data['r_minor_norm'].to_numpy() * vectorized_numpy_derivative(data['r_minor_norm'].to_numpy(), mxh_zeta)
            newcoords['mxh_coefficient'] = np.arange(n_coeffs)
            newvars['mxh_r0'] = (['time', 'radius'], mxh_r0)
            newvars['mxh_dr0'] = (['time', 'radius'], np.where(np.isclose(mxh_dr0, 0.0), 0.0, mxh_dr0))
            newvars['mxh_z0'] = (['time', 'radius'], mxh_z0)
            newvars['mxh_dz0'] = (['time', 'radius'], np.where(np.isclose(mxh_dz0, 0.0), 0.0, mxh_dz0))
            newvars['mxh_r'] = (['time', 'radius'], mxh_r)
            newvars['mxh_kappa'] = (['time', 'radius'], mxh_kappa)
            newvars['mxh_s_kappa'] = (['time', 'radius'], np.where(np.isclose(mxh_s_kappa, 0.0), 0.0, mxh_s_kappa))
            newvars['mxh_delta'] = (['time', 'radius'], mxh_delta)
            newvars['mxh_s_delta'] = (['time', 'radius'], np.where(np.isclose(mxh_s_delta, 0.0), 0.0, mxh_s_delta))
            newvars['mxh_zeta'] = (['time', 'radius'], mxh_zeta)
            newvars['mxh_s_zeta'] = (['time', 'radius'], np.where(np.isclose(mxh_s_zeta, 0.0), 0.0, mxh_s_zeta))
            newvars['mxh_sin'] = (['time', 'radius', 'mxh_coefficient'], mxh_sin)
            newvars['mxh_s_sin'] = (['time', 'radius', 'mxh_coefficient'], np.where(np.isclose(mxh_s_sin, 0.0), 0.0, mxh_s_sin))
            newvars['mxh_cos'] = (['time', 'radius', 'mxh_coefficient'], mxh_cos)
            newvars['mxh_s_cos'] = (['time', 'radius', 'mxh_coefficient'], np.where(np.isclose(mxh_s_cos, 0.0), 0.0, mxh_s_cos))
            newvars['mxh_contour_r'] = (['time', 'radius', 'angle_geometric'], contour_r)
            newvars['mxh_contour_z'] = (['time', 'radius', 'angle_geometric'], contour_z)

            signb = 1.0
            n_theta = 1001
            weight = np.expand_dims(np.expand_dims(np.expand_dims(np.arange(n_coeffs), axis=0), axis=0), axis=0)
            theta = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(-np.pi, np.pi, n_theta), axis=-1), axis=-1), axis=-1)
            a = np.sum(mxh_sin * np.sin(weight * theta) + mxh_cos * np.cos(weight * theta), axis=-1)
            a_r = np.sum(mxh_s_sin * np.sin(weight * theta) + mxh_s_cos * np.cos(weight * theta), axis=-1)
            a_t = np.sum(mxh_sin * weight * np.cos(weight * theta) - mxh_cos * weight * np.sin(weight * theta), axis=-1)
            #a_tt = np.sum(-mxh_sin * weight ** 2 * np.sin(weight * theta) - mxh_cos * weight ** 2 * np.cos(weight * theta), axis=-1)
            r = np.expand_dims(mxh_r0, axis=0) + np.expand_dims(mxh_r, axis=0) * np.cos(a)
            r_r = np.expand_dims(mxh_dr0, axis=0) + np.cos(a) - np.expand_dims(mxh_r, axis=0) * np.sin(a) * a_r
            r_t = np.expand_dims(-mxh_r, axis=0) * a_t * np.sin(a)
            #r_tt = np.expand_dims(-mxh_r, axis=0) * (a_t ** 2 * np.cos(a) + a_tt * np.sin(a))
            z = np.expand_dims(mxh_z0, axis=0) + np.expand_dims(mxh_kappa * mxh_r, axis=0) * np.sin(np.squeeze(theta, axis=-1))
            z_r = np.expand_dims(mxh_dz0, axis=0) + np.expand_dims(mxh_kappa * (1.0 + mxh_s_kappa), axis=0) * np.sin(np.squeeze(theta, axis=-1))
            z_t = np.expand_dims(mxh_kappa * mxh_r, axis=0) * np.cos(np.squeeze(theta, axis=-1))
            #z_tt = np.expand_dims(-mxh_kappa * mxh_r, axis=0) * np.sin(np.squeeze(theta, axis=-1))
            l_t = np.sqrt(r_t ** 2 + z_t ** 2)
            j_r = r * (r_r * z_t - r_t * z_r)
            inv_j_r = 1.0 / np.where(np.isclose(j_r, 0.0), 0.001, j_r)
            grad_r = np.where(np.isclose(j_r, 0.0), 1.0, r * l_t * inv_j_r)
            #r_c = l_t ** 3 / (r_t * z_tt - z_t * r_tt)
            #z_l = np.where(np.isclose(l_t, 0.0), 0.0, z_t / l_t)
            #r_l = np.where(np.isclose(l_t, 0.0), 0.0, r_t / l_t)
            #l_r = z_l * z_r + r_l * r_r
            #nsin = (r_r * r_t + z_r * z_t) / l_t
            c = 2.0 * np.pi * np.sum(l_t[:-1, ...] / (r[:-1, ...] * grad_r[:-1, ...]), axis=0)
            f = 2.0 * np.pi * data['r_minor'].to_numpy() / (np.where(np.isclose(c, 0.0), 1.0, c) / float(n_theta - 1))
            f[..., 0] = 2.0 * f[..., 1] - f[..., 2]
            bt = np.expand_dims(f, axis=0) / r
            bp = np.expand_dims((data['r_minor'] / data['safety_factor']).to_numpy(), axis=0) * grad_r / r
            bt_inner = np.squeeze(np.take_along_axis(bt, np.expand_dims(np.nanargmin(r, axis=0), axis=0), axis=0), axis=0)
            bp_inner = np.squeeze(np.take_along_axis(bp, np.expand_dims(np.nanargmin(r, axis=0), axis=0), axis=0), axis=0)
            bt_outer = np.squeeze(np.take_along_axis(bt, np.expand_dims(np.nanargmax(r, axis=0), axis=0), axis=0), axis=0)
            bp_outer = np.squeeze(np.take_along_axis(bp, np.expand_dims(np.nanargmax(r, axis=0), axis=0), axis=0), axis=0)
            b = signb * np.sqrt(bt ** 2 + bp ** 2)
            r_v = np.expand_dims((data['r_minor'] * data['r_geometric']).to_numpy(), axis=0)
            g_t = r * b * l_t / (np.where(np.isclose(r_v, 0.0), 1.0, r_v) * grad_r)
            g_t[..., 0] = 2.0 * g_t[..., 1] - g_t[..., 2]
            newvars['mxh_dvolume_dr'] = (['time', 'radius'], 2.0 * np.pi * np.where(np.isfinite(c), c, 0.0) / float(n_theta - 1))
            newvars['mxh_surface_area'] = (['time', 'radius'], 2.0 * np.pi * np.sum(l_t[:-1, ...] * r[:-1, ...], axis=0) * 2.0 * np.pi / float(n_theta - 1))
            #newvars['mxh_bt'] = (['n', 'rho'], bt[i1] + (bt[i2] - bt[i1]) * ztheta)
            denom = np.sum(np.where(np.isfinite(g_t), g_t, 0.0)[:-1, ...] / b[:-1, ...], axis=0)
            denom[..., 0] = 2.0 * denom[..., 1] - denom[..., 2]
            b2 = np.stack([np.sum(bp[:-1, ...] ** 2 * g_t[:-1, ...] / b[:-1, ...], axis=0) / denom, np.sum(bt[:-1, ...] ** 2 * g_t[:-1, ...] / b[:-1, ...], axis=0) / denom], axis=-1)
            newvars['mxh_dr_dpsi'] = (['time', 'radius'], np.sum(grad_r[:-1, ...] * g_t[:-1, ...] / b[:-1, ...], axis=0) / denom)
            newvars['mxh_field_squared'] = (['time', 'radius', 'direction'], b2)
            newvars['mxh_cross_sectional_area'] = (['time', 'radius'], np.trapezoid(r, z, axis=0))
            newvars['mxh_field_ref'] = (['time', 'radius'], np.abs(data['field_unit'].to_numpy() * bt_inner))  # For synchrotron
            b_inner = np.stack([np.where(np.isfinite(bt_inner), bt_inner, 0.0), np.where(np.isfinite(bp_inner), bp_inner, 0.0)], axis=-1)
            b_outer = np.stack([np.where(np.isfinite(bt_outer), bt_outer, 0.0), np.where(np.isfinite(bp_outer), bp_outer, 0.0)], axis=-1)
            newvars['mxh_field_inner'] = (['time', 'radius', 'direction'], b_inner)
            newvars['mxh_field_outer'] = (['time', 'radius', 'direction'], b_outer)

            main_species_mask = (np.isclose(data['atomic_number_i'].to_numpy(), 1.0) & (data['type_i'].isin(['thermal'])).to_numpy()).flatten()
            main_species = [i for i in range(len(main_species_mask)) if main_species_mask[i]]
            n_i_vol = vectorized_numpy_integration(
                np.transpose(data['density_i'].isel(ion=main_species).to_numpy() * np.expand_dims(newvars['mxh_dvolume_dr'][-1], axis=-1), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(main_species), axis=1)
            )
            mxh_mass_ave = np.sum((data['mass_i'].isel(ion=main_species).to_numpy() * n_i_vol[..., -1] / np.expand_dims(np.sum(n_i_vol[..., -1], axis=1), axis=1)), axis=-1)
            newvars['mxh_mass_main_average'] = (['time'], mxh_mass_ave)

            newvars['field_squared'] = (['time', 'radius', 'direction'], b2 * np.expand_dims(data['field_unit'].to_numpy() ** 2, axis=-1))
            newvars['field_inner'] = (['time', 'radius', 'direction'], b_inner * np.expand_dims(data['field_unit'].to_numpy() ** 2, axis=-1))
            newvars['field_outer'] = (['time', 'radius', 'direction'], b_outer * np.expand_dims(data['field_unit'].to_numpy() ** 2, axis=-1))

        self.update_input_data_vars(newvars)


    def _compute_extended_local_inputs(
        self
    ) -> None:
        newcoords: MutableMapping[str, Any] = {}
        newvars: MutableMapping[str, Any] = {}
        if self.has_input:
            data = self.input
            main_species_mask = (np.isclose(data['atomic_number_i'].to_numpy(), 1.0) & (data['type_i'].isin(['thermal'])).to_numpy()).flatten()
            main_species = [i for i in range(len(main_species_mask)) if main_species_mask[i]]
            thermal_species_mask = data['type_i'].isin(['thermal']).to_numpy().flatten()
            thermal_species = [i for i in range(len(thermal_species_mask)) if thermal_species_mask[i]]

            pressure_e = self.constants['e_si'] * data['temperature_e'] * data['density_e']
            pressure_i = self.constants['e_si'] * data['temperature_i'] * data['density_i']
            newvars['density_thermal_total_i'] = (['time', 'radius'], data['density_i'].isel(ion=thermal_species).sum('ion').to_numpy())
            newvars['density_total_i'] = (['time', 'radius'], data['density_i'].sum('ion').to_numpy())
            newvars['pressure_e'] = (['time', 'radius'], pressure_e.to_numpy())
            newvars['pressure_i'] = (['time', 'radius', 'ion'], pressure_i.to_numpy())
            newvars['pressure_thermal_total_i'] = (['time', 'radius'], pressure_i.isel(ion=thermal_species).sum('ion').to_numpy())
            newvars['pressure_total_i'] = (['time', 'radius'], pressure_i.sum('ion').to_numpy())
            newvars['pressure_thermal_total'] = (['time', 'radius'], (pressure_e + pressure_i.isel(ion=thermal_species).sum('ion')).to_numpy())
            newvars['pressure_total'] = (['time', 'radius'], (pressure_e + pressure_i.sum('ion')).to_numpy())
            newvars['pressure_fast_total_i'] = (['time', 'radius'], (pressure_i.sum('ion') - pressure_i.isel(ion=thermal_species).sum('ion')).to_numpy())
            newvars['pressure_fast_total'] = (['time', 'radius'], (pressure_i.sum('ion') - pressure_i.isel(ion=thermal_species).sum('ion')).to_numpy())
            newvars['gyrobohm_particle_flux_e'] = (['time', 'radius'], (data['density_e'] * data['velocity_sound_ref'] * (data['gyroradius_ref_unit'] / data['r_minor_lcfs']) ** 2).to_numpy())
            newvars['gyrobohm_particle_flux_i'] = (['time', 'radius', 'ion'], (data['density_i'] * data['velocity_sound_ref'] * (data['gyroradius_ref_unit'] / data['r_minor_lcfs']) ** 2).to_numpy())
            newvars['gyrobohm_heat_flux_e'] = (['time', 'radius'], (self.constants['e_si'] * data['density_e'] * data['temperature_e'] * data['velocity_sound_ref'] * (data['gyroradius_ref_unit'] / data['r_minor_lcfs']) ** 2).to_numpy())
            newvars['gyrobohm_heat_flux_i'] = (['time', 'radius', 'ion'], (self.constants['e_si'] * data['density_i'] * data['temperature_i'] * data['velocity_sound_ref'] * (data['gyroradius_ref_unit'] / data['r_minor_lcfs']) ** 2).to_numpy())
            newvars['gyrobohm_momentum_flux_e'] = (['time', 'radius'], (self.constants['u_si'] * data['density_e'] * data['r_geometric'] * data['mass_e'] * data['velocity_thermal_e'] * data['velocity_sound_ref'] * (data['gyroradius_ref_unit'] / data['r_minor_lcfs']) ** 2).to_numpy())
            newvars['gyrobohm_momentum_flux_i'] = (['time', 'radius', 'ion'], (self.constants['u_si'] * data['density_i'] * data['r_geometric'] * data['mass_i'] * data['velocity_thermal_i'] * data['velocity_sound_ref'] * (data['gyroradius_ref_unit'] / data['r_minor_lcfs']) ** 2).to_numpy())
            newvars['gyrobohm_exchange_flux'] = (['time', 'radius'], (self.constants['e_si'] * data['density_e'] * data['temperature_e'] * data['velocity_sound_ref'] * (data['gyroradius_ref_unit']) ** 2 / (data['r_minor_lcfs'] ** 3)).to_numpy())
            newvars['gyrobohm_convective_heat_flux_e'] = (['time', 'radius'], (1.5 * self.constants['e_si'] * data['density_e'] * data['temperature_e'] * data['velocity_sound_ref'] * (data['gyroradius_ref_unit'] / data['r_minor_lcfs']) ** 2).to_numpy())
            newvars['gyrobohm_convective_heat_flux_i'] = (['time', 'radius', 'ion'], (1.5 * self.constants['e_si'] * data['density_i'] * data['temperature_i'] * data['velocity_sound_ref'] * (data['gyroradius_ref_unit'] / data['r_minor_lcfs']) ** 2).to_numpy())

            beta_e = pressure_e * 2.0 * self.constants['mu_si'] / data['field_squared']
            beta_i = pressure_i * 2.0 * self.constants['mu_si'] / data['field_squared']
            norm = np.expand_dims(data['length_ref'].to_numpy(), axis=-1)
            grad_density_e_norm = norm * vectorized_numpy_derivative(
                data['r_minor'].to_numpy(),
                -np.log(data['density_e'].to_numpy())
            )
            grad_density_i_norm = np.transpose(np.expand_dims(norm, axis=1) * vectorized_numpy_derivative(
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1),
                -np.log(np.transpose(data['density_i'].to_numpy(), axes=(0, 2, 1)))
            ), axes=(0, 2, 1))
            grad_temperature_e_norm = norm * vectorized_numpy_derivative(
                data['r_minor'].to_numpy(),
                -np.log(data['temperature_e'].to_numpy())
            )
            grad_temperature_i_norm = np.transpose(np.expand_dims(norm, axis=1) * vectorized_numpy_derivative(
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1),
                -np.log(np.transpose(data['temperature_i'].to_numpy(), axes=(0, 2, 1)))
            ), axes=(0, 2, 1))
            newvars['pressure_e_norm'] = (['time', 'radius'], (1.0 / ((1.0 / beta_e).sum('direction'))).to_numpy())
            newvars['pressure_i_norm'] = (['time', 'radius', 'ion'], (1.0 / ((1.0 / beta_i).sum('direction'))).to_numpy())
            newvars['density_i_norm'] = (['time', 'radius', 'ion'], (data['density_i'] / data['density_e']).to_numpy())
            newvars['temperature_i_norm'] = (['time', 'radius', 'ion'], (data['temperature_i'] / data['temperature_e']).to_numpy())
            newvars['grad_density_e_norm'] = (['time', 'radius'], grad_density_e_norm)
            newvars['grad_density_i_norm'] = (['time', 'radius', 'ion'], grad_density_i_norm)
            newvars['grad_temperature_e_norm'] = (['time', 'radius'], grad_temperature_e_norm)
            newvars['grad_temperature_i_norm'] = (['time', 'radius', 'ion'], grad_temperature_i_norm)
            newvars['quasineutrality'] = (['time', 'radius'], (1.0 - (data['density_i'] * data['charge_i'] / data['density_e'])).sum('ion').to_numpy())
            newvars['effective_charge'] = (['time', 'radius'], (data['density_i'] * (data['charge_i'] ** 2) / data['density_e']).sum('ion').to_numpy())

            newvars['density_ratio_main'] = (['time', 'radius'], (data['density_i'].isel(ion=main_species).sum('ion') / data['density_e']).to_numpy())
            newvars['temperature_ratio_main'] = (['time', 'radius'], (data['temperature_i'].isel(ion=main_species).mean('ion') / data['temperature_e']).to_numpy())

            debye_e = ((self.constants['eps_si'] / self.constants['e_si']) ** 0.5) * (data['temperature_e'] / (data['density_e'] * (data['charge_e'] ** 2))) ** 0.5  # m
            debye_i = ((self.constants['eps_si'] / self.constants['e_si']) ** 0.5) * (data['temperature_i'] / (data['density_i'] * (data['charge_i'] ** 2))) ** 0.5  # m
            collisionality_prefactor = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
            inverse_approach_factor_ee = (4.0 * np.pi * self.constants['eps_si'] / self.constants['e_si']) * data['temperature_e'] / abs(data['charge_e'] ** 2)
            collisionality_ee = (data['density_e'] / (inverse_approach_factor_ee ** 2)) * (self.constants['e_si'] * data['temperature_e'] / (self.constants['u_si'] * data['mass_e'])) ** 0.5 * np.log(inverse_approach_factor_ee * debye_e)
            inverse_approach_factor_ei = (4.0 * np.pi * self.constants['eps_si'] / self.constants['e_si']) * data['temperature_e'] / abs(data['charge_e'] * data['charge_i'])
            collisionality_ei = (data['density_i'] / (inverse_approach_factor_ei ** 2)) * (self.constants['e_si'] * data['temperature_e'] / (self.constants['u_si'] * data['mass_e'])) ** 0.5 * np.log(inverse_approach_factor_ei * debye_e)
            inverse_approach_factor_ii = (4.0 * np.pi * self.constants['eps_si'] / self.constants['e_si']) * data['temperature_i'] / abs(data['charge_i'] * data['charge_i'])
            collisionality_ii = (data['density_i'] / (inverse_approach_factor_ii ** 2)) * (self.constants['e_si'] * data['temperature_i'] / (self.constants['u_si'] * data['mass_i'])) ** 0.5 * np.log(inverse_approach_factor_ii * debye_i)
            newvars['debye_e'] = (['time', 'radius'], debye_e.to_numpy())
            newvars['debye_e_norm'] = (['time', 'radius'], (debye_e / data['gyroradius_ref_unit']).to_numpy())
            newvars['debye_i'] = (['time', 'radius', 'ion'], debye_i.to_numpy())
            newvars['debye_i_norm'] = (['time', 'radius', 'ion'], (debye_i / data['gyroradius_ref_unit']).to_numpy())
            newvars['debye'] = (['time', 'radius'], ((debye_e ** (-2) + (debye_i ** (-2)).sum('ion')) ** (-0.5)).to_numpy())
            newvars['debye_norm'] = (['time', 'radius'], (((debye_e ** (-2) + (debye_i ** (-2)).sum('ion')) ** (-0.5)) / data['gyroradius_ref_unit']).to_numpy())
            newvars['collisionality_ee'] = (['time', 'radius'], (collisionality_prefactor * collisionality_ee).to_numpy())
            newvars['collisionality_ei'] = (['time', 'radius'], (collisionality_prefactor * collisionality_ei).sum('ion').to_numpy())
            newvars['collisionality_ii'] = (['time', 'radius', 'ion'], (collisionality_prefactor * collisionality_ii).to_numpy())
            newvars['collisionality_ee_norm'] = (['time', 'radius'], (collisionality_prefactor * collisionality_ee * data['length_ref'] / data['velocity_sound_ref']).to_numpy())
            newvars['collisionality_ei_norm'] = (['time', 'radius'], (collisionality_prefactor * collisionality_ei * data['length_ref'] / data['velocity_sound_ref']).sum('ion').to_numpy())
            newvars['collisionality_ii_norm'] = (['time', 'radius', 'ion'], (collisionality_prefactor * collisionality_ii * data['length_ref'] / data['velocity_sound_ref']).to_numpy())

            field = data['field_squared'].sum('direction') ** 0.5
            vperp_tor = data['velocity_i'].sel(direction='toroidal', drop=True) * (data['field_squared'].sel(direction='poloidal', drop=True) ** 0.5) / field
            vperp_pol = data['velocity_i'].sel(direction='poloidal', drop=True) * (data['field_squared'].sel(direction='toroidal', drop=True) ** 0.5) / field
            vpar_tor = data['velocity_i'].sel(direction='toroidal', drop=True) * (data['field_squared'].sel(direction='toroidal', drop=True) ** 0.5) / field
            vpar_pol = data['velocity_i'].sel(direction='poloidal', drop=True) * (data['field_squared'].sel(direction='poloidal', drop=True) ** 0.5) / field
            vperp = vperp_tor - vperp_pol
            vpar = vpar_tor + vpar_pol
            grad_vperp = np.transpose(np.expand_dims(norm, axis=1) * vectorized_numpy_derivative(
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1),
                -np.transpose(vperp.to_numpy(), axes=(0, 2, 1))
            ), axes=(0, 2, 1))
            grad_vpar = np.transpose(np.expand_dims(norm, axis=1) * vectorized_numpy_derivative(
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1),
                -np.transpose(vpar.to_numpy(), axes=(0, 2, 1))
            ), axes=(0, 2, 1))
            newcoords['field_direction'] = np.array(['parallel', 'perpendicular'])
            field_velocity_i = np.concatenate([np.expand_dims(vperp.to_numpy(), axis=-1), np.expand_dims(vpar.to_numpy(), axis=-1)], axis=-1)
            grad_field_velocity_i = np.concatenate([np.expand_dims(grad_vperp, axis=-1), np.expand_dims(grad_vpar, axis=-1)], axis=-1)
            rotation_frequency_sonic = (vperp * field / (data['r_geometric'] * (data['field_squared'].sel(direction='poloidal', drop=True) ** 0.5))).isel(ion=main_species).max('ion').to_numpy()
            exb_norm = (data['r_minor'] / data['safety_factor']).to_numpy()
            exb_shearing_rate = exb_norm * vectorized_numpy_derivative(data['r_minor'].to_numpy(), -vperp.isel(ion=main_species).mean('ion').to_numpy() / np.where(np.isclose(exb_norm, 0.0), 1.0e-4, exb_norm))
            newvars['field_velocity_i'] = (['time', 'radius', 'ion', 'field_direction'], field_velocity_i)
            newvars['field_velocity_i_norm'] = (['time', 'radius', 'ion', 'field_direction'], field_velocity_i / np.expand_dims(np.expand_dims(data['velocity_sound_ref'].to_numpy(), axis=-1), axis=-1))
            newvars['grad_field_velocity_i'] = (['time', 'radius', 'ion', 'field_direction'], grad_field_velocity_i)
            newvars['grad_field_velocity_i_norm'] = (['time', 'radius', 'ion', 'field_direction'], grad_field_velocity_i * np.expand_dims(np.expand_dims((data['length_ref'] * data['velocity_sound_ref']).to_numpy(), axis=-1), axis=-1))
            newvars['exb_shearing_rate'] = (['time', 'radius'], exb_shearing_rate)
            newvars['exb_shearing_rate_norm'] = (['time', 'radius'], exb_shearing_rate * (data['length_ref'] / data['velocity_sound_ref']).to_numpy())
            newvars['rotation_frequency_sonic'] = (['time', 'radius'], rotation_frequency_sonic)
            newvars['rotation_frequency_sonic_norm'] = (['time', 'radius'], rotation_frequency_sonic * (data['length_ref'] / data['velocity_sound_ref']).to_numpy())
            #newvars['alw0'] = (['n', 'rho'], norm * vectorized_numpy_derivative(data['r_minor'].to_numpy(), -np.log(data['omega0'].to_numpy())))
            newvars['grad_rotation_frequency_sonic'] = (['time', 'radius'], vectorized_numpy_derivative(data['r_minor'].to_numpy(), rotation_frequency_sonic))

            radiation_sources = ['synchrotron', 'bremsstrahlung', 'line_radiation']
            auxiliary_sources = ['ohmic', 'neutral_beam', 'ion_cyclotron', 'electron_cyclotron']
            auxiliary_plus_sources = ['ohmic', 'neutral_beam', 'ion_cyclotron', 'electron_cyclotron', 'ionization']
            newvars['heat_source_total_e'] = (['time', 'radius'], data['heat_source_e'].sum('source').to_numpy())
            newvars['heat_source_total_i'] = (['time', 'radius', 'ion'], data['heat_source_i'].sum('source').to_numpy())
            newvars['particle_source_total_e'] = (['time', 'radius'], data['particle_source_e'].sum('source').to_numpy())
            newvars['particle_source_total_i'] = (['time', 'radius', 'ion'], data['particle_source_i'].sum('source').to_numpy())
            newvars['momentum_source_total_i'] = (['time', 'radius', 'ion', 'direction'], data['momentum_source_i'].sum('source').to_numpy())
            newvars['radiation_heat_source'] = (['time', 'radius'], data['heat_source_e'].sel(source=radiation_sources).sum('source').to_numpy())
            newvars['auxiliary_heat_source_e'] = (['time', 'radius'], data['heat_source_e'].sel(source=auxiliary_sources).sum('source').to_numpy())
            newvars['auxiliary_heat_source_i'] = (['time', 'radius', 'ion'], data['heat_source_i'].sel(source=auxiliary_sources).sum('source').to_numpy())
            newvars['auxiliary_plus_heat_source_e'] = (['time', 'radius'], data['heat_source_e'].sel(source=auxiliary_plus_sources).sum('source').to_numpy())
            newvars['auxiliary_plus_heat_source_i'] = (['time', 'radius', 'ion'], data['heat_source_i'].sel(source=auxiliary_plus_sources).sum('source').to_numpy())
            newvars['fusion_heat_source'] = (['time', 'radius'], (data['heat_source_e'].sel(source='fusion') + data['heat_source_i'].sel(source='fusion').sum('ion')).to_numpy())

        self.update_input_coords(newcoords)
        self.update_input_data_vars(newvars)


    def _compute_integrated_quantities(self):

        newvars: MutableMapping[str, Any] = {}

        if self.has_input:
            data = self.input

            line = vectorized_numpy_integration(np.ones_like(data['r_minor'].to_numpy()), data['r_minor'].to_numpy())
            vol = vectorized_numpy_integration(data['mxh_dvolume_dr'].to_numpy(), data['r_minor'].to_numpy())
            newvars['line_average'] = (['time', 'radius'], line)
            newvars['volume_average'] = (['time', 'radius'], vol)

            density_e_line = vectorized_numpy_integration((data['density_e']).to_numpy(), data['r_minor'].to_numpy())
            density_i_line = vectorized_numpy_integration(
                np.transpose((data['density_i']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1)
            )
            newvars['density_e_line_average'] = (['time'], density_e_line[..., -1] / line[..., -1])
            newvars['density_i_line_average'] = (['time', 'ion'], density_i_line[..., -1] / np.expand_dims(line, axis=1)[..., -1])
            newvars['density_ratio_i_line_average'] = (['time', 'ion'], density_i_line[..., -1] / np.expand_dims(density_e_line, axis=1)[..., -1])
            newvars['concentration_i_line_average'] = (['time', 'ion'], density_i_line[..., -1] / np.expand_dims(np.sum(density_i_line, axis=1), axis=1)[..., -1])

            heat_source_e_vol = np.transpose(vectorized_numpy_integration(
                np.transpose((data['heat_source_e'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['source']), axis=1)
            ), axes=(0, 2, 1))
            heat_source_i_vol = np.transpose(vectorized_numpy_integration(
                np.transpose((data['heat_source_i'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 3, 1)),
                np.repeat(np.expand_dims(np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['source']), axis=1), axis=1), len(data['ion']), axis=1)
            ), axes=(0, 3, 1, 2))
            particle_source_e_vol = np.transpose(vectorized_numpy_integration(
                np.transpose((data['particle_source_e'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['source']), axis=1)
            ), axes=(0, 2, 1))
            particle_source_e_conv_vol = 1.5 * self.constants['e_si'] * np.expand_dims(data['temperature_e'].to_numpy(), axis=-1) * particle_source_e_vol  # MW
            particle_source_i_vol = np.transpose(vectorized_numpy_integration(
                np.transpose((data['particle_source_i'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 3, 1)),
                np.repeat(np.expand_dims(np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['source']), axis=1), axis=1), len(data['ion']), axis=1)
            ), axes=(0, 3, 1, 2))
            momentum_source_i_vol = np.transpose(vectorized_numpy_integration(
                np.transpose((data['momentum_source_i'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 3, 4, 1)),
                np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['source']), axis=1), axis=1), len(data['direction']), axis=1), axis=1), len(data['ion']), axis=1)
            ), axes=(0, 4, 1, 2, 3))
            newvars['heat_source_e_vol'] = (['time', 'radius', 'source'], heat_source_e_vol)
            newvars['heat_source_i_vol'] = (['time', 'radius', 'ion', 'source'], heat_source_i_vol)
            newvars['particle_source_e_vol'] = (['time', 'radius', 'source'], particle_source_e_vol)
            newvars['particle_source_e_conv_vol'] = (['time', 'radius', 'source'], particle_source_e_conv_vol)
            newvars['particle_source_i_vol'] = (['time', 'radius', 'ion', 'source'], particle_source_i_vol)
            newvars['momentum_source_i_vol'] = (['time', 'radius', 'ion', 'direction', 'source'], momentum_source_i_vol)
            newvars['heat_source_vol_total'] = (['time', 'radius'], np.sum(heat_source_e_vol + np.sum(heat_source_i_vol, axis=2), axis=-1))
            newvars['particle_source_vol_total'] = (['time', 'radius'], np.sum(particle_source_e_vol, axis=-1))
            newvars['momentum_source_vol_total'] = (['time', 'radius', 'direction'], np.sum(np.sum(momentum_source_i_vol, axis=2), axis=-1))
            newvars['heat_source_total_in'] = (['time'], np.sum(heat_source_e_vol + np.sum(heat_source_i_vol, axis=2), axis=-1)[:, -1])
            newvars['particle_source_total_in'] = (['time'], np.sum(particle_source_e_vol, axis=-1)[:, -1])
            newvars['momentum_source_total_in'] = (['time'], np.sum(np.sum(momentum_source_i_vol, axis=2), axis=-1)[:, -1, 0])

            surface = np.repeat(np.expand_dims(data['mxh_surface_area'].to_numpy(), axis=-1), len(data['source']), axis=-1)
            inverse_surface = 1.0 / np.where(np.isclose(surface, 0.0), 1.0, surface)
            newvars['heat_source_e_flux'] = (['time', 'radius', 'source'], np.where(np.isclose(surface, 0.0), 0.0, heat_source_e_vol * inverse_surface))
            newvars['heat_source_i_flux'] = (['time', 'radius', 'ion', 'source'], np.where(np.isclose(np.repeat(np.expand_dims(surface, axis=2), len(data['ion']), axis=2), 0.0), 0.0, heat_source_i_vol * np.expand_dims(inverse_surface, axis=2)))
            newvars['particle_source_e_flux'] = (['time', 'radius', 'source'], np.where(np.isclose(surface, 0.0), 0.0, particle_source_e_vol * inverse_surface))
            newvars['particle_source_e_conv_flux'] = (['time', 'radius', 'source'], np.where(np.isclose(surface, 0.0), 0.0, particle_source_e_conv_vol * inverse_surface))
            newvars['momentum_source_i_flux'] = (['time', 'radius', 'ion', 'direction', 'source'], np.where(np.repeat(np.expand_dims(np.repeat(np.expand_dims(surface, axis=2), len(data['ion']), axis=2), axis=3), len(data['direction']), axis=3), 0.0, momentum_source_i_vol * np.expand_dims(np.expand_dims(inverse_surface, axis=2), axis=3)))
            #newvars["qratio_surf"] = qi / np.where(qe == 0.0, 1e-10, qe)  # to avoid division by zero

            ohmic_heat_source_vol = vectorized_numpy_integration(
                ((data['heat_source_e'] + data['heat_source_i'].sum('ion')).sel(source='ohmic', drop=True) * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            wave_heat_source_vol = vectorized_numpy_integration(
                ((data['heat_source_e'] + data['heat_source_i'].sum('ion')).sel(source=['electron_cyclotron', 'ion_cyclotron']).sum('source') * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            beam_heat_source_vol = vectorized_numpy_integration(
                ((data['heat_source_e'] + data['heat_source_i'].sum('ion')).sel(source='neutral_beam', drop=True) * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            ionization_heat_source_vol = vectorized_numpy_integration(
                ((data['heat_source_e'] + data['heat_source_i'].sum('ion')).sel(source='ionization', drop=True) * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            radiation_heat_source_vol = vectorized_numpy_integration(
                (data['radiation_heat_source'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            auxiliary_heat_source_e_vol = vectorized_numpy_integration(
                (data['auxiliary_heat_source_e'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            auxiliary_heat_source_i_vol = np.transpose(vectorized_numpy_integration(
                np.transpose((data['auxiliary_heat_source_i'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1)
            ), axes=(0, 2, 1))
            auxiliary_plus_heat_source_e_vol = vectorized_numpy_integration(
                (data['auxiliary_plus_heat_source_e'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            auxiliary_plus_heat_source_i_vol = np.transpose(vectorized_numpy_integration(
                np.transpose((data['auxiliary_plus_heat_source_i'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1)
            ), axes=(0, 2, 1))
            fusion_heat_source_vol = vectorized_numpy_integration(
                (data['fusion_heat_source'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            heat_exchange_ei_vol = vectorized_numpy_integration(
                (data['heat_exchange_ei'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            newvars['ohmic_heat_source_vol'] = (['time', 'radius'], ohmic_heat_source_vol)
            newvars['wave_heat_source_vol'] = (['time', 'radius'], wave_heat_source_vol)
            newvars['beam_heat_source_vol'] = (['time', 'radius'], beam_heat_source_vol)
            newvars['ionization_heat_source_vol'] = (['time', 'radius'], ionization_heat_source_vol)
            newvars['radiation_heat_source_vol'] = (['time', 'radius'], radiation_heat_source_vol)
            newvars['auxiliary_heat_source_e_vol'] = (['time', 'radius'], auxiliary_heat_source_e_vol)
            newvars['auxiliary_heat_source_i_vol'] = (['time', 'radius', 'ion'], auxiliary_heat_source_i_vol)
            newvars['auxiliary_plus_heat_source_e_vol'] = (['time', 'radius'], auxiliary_plus_heat_source_e_vol)
            newvars['auxiliary_plus_heat_source_i_vol'] = (['time', 'radius', 'ion'], auxiliary_plus_heat_source_i_vol)
            newvars['fusion_heat_source_vol'] = (['time', 'radius'], fusion_heat_source_vol)
            newvars['heat_exchange_ei_vol'] = (['time', 'radius'], heat_exchange_ei_vol)

            heating_source_vol = ohmic_heat_source_vol + wave_heat_source_vol + beam_heat_source_vol + ionization_heat_source_vol + fusion_heat_source_vol
            newvars['power_fusion'] = (['time'], 5.0 * fusion_heat_source_vol[..., -1])
            newvars['power_radiation'] = (['time'], radiation_heat_source_vol[..., -1])
            newvars['power_input'] = (['time'], (np.sum(heat_source_e_vol + np.sum(heat_source_i_vol, axis=2), axis=-1))[..., -1])
            newvars['power_heating'] = (['time'], heating_source_vol[..., -1])
            newvars['power_scrape_off_layer'] = (['time'], (heating_source_vol + radiation_heat_source_vol)[..., -1])
            newvars['fusion_gain'] = (['time'], ((5.0 * fusion_heat_source_vol)[..., -1] / (ohmic_heat_source_vol + wave_heat_source_vol + beam_heat_source_vol + ionization_heat_source_vol)[..., -1]))

            density_e_vol = vectorized_numpy_integration(
                (data['density_e'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            temperature_e_vol = vectorized_numpy_integration(
                (data['temperature_e'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            pressure_e_vol = vectorized_numpy_integration(
                (data['pressure_e'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            energy_e_vol = vectorized_numpy_integration(
                (1.5 * data['pressure_e'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            density_i_vol = vectorized_numpy_integration(
                np.transpose((data['density_i'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1)
            )
            temperature_i_vol = vectorized_numpy_integration(
                np.transpose((data['temperature_i'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1)
            )
            pressure_i_vol = vectorized_numpy_integration(
                np.transpose((data['pressure_i'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1)
            )
            energy_i_vol = vectorized_numpy_integration(
                np.transpose((1.5 * data['pressure_i'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['ion']), axis=1)
            )
            density_thermal_i_vol = vectorized_numpy_integration(
                (data['density_thermal_total_i'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            pressure_thermal_i_vol = vectorized_numpy_integration(
                (data['pressure_thermal_total_i'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            energy_thermal_i_vol = vectorized_numpy_integration(
                (1.5 * data['pressure_thermal_total_i'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            newvars['density_e_volume_average'] = (['time'], density_e_vol[..., -1] / vol[..., -1])
            newvars['density_i_volume_average'] = (['time', 'ion'], density_i_vol[..., -1] / np.expand_dims(vol, axis=1)[..., -1])
            newvars['density_ratio_i_volume_average'] = (['time', 'ion'], density_i_vol[..., -1] / np.expand_dims(density_e_vol, axis=1)[..., -1])
            newvars['concentration_i_volume_average'] = (['time', 'ion'], density_i_vol[..., -1] / np.expand_dims(np.sum(density_i_vol, axis=1), axis=1)[..., -1])
            newvars['density_thermal_i_volume_average'] = (['time'], density_thermal_i_vol[..., -1] / vol[..., -1])
            #newvars['density_ratio_thermal_i_volume_average'] = (['time'], density_thermal_i_vol[..., -1] / np.sum(density_i_vol, axis=1)[..., -1])
            newvars['temperature_e_volume_average'] = (['time'], temperature_e_vol[..., -1] / vol[..., -1])
            newvars['temperature_i_volume_average'] = (['time', 'ion'], temperature_i_vol[..., -1] / np.expand_dims(vol, axis=1)[..., -1])
            #newvars['temperature_ratio_i_volume_average'] = (['time', 'ion'], temperature_i_vol[..., -1] / np.expand_dims(temperature_e_vol, axis=1)[..., -1])
            newvars['pressure_e_volume_average'] = (['time'], pressure_e_vol[..., -1] / vol[..., -1])
            newvars['pressure_i_volume_average'] = (['time', 'ion'], pressure_i_vol[..., -1] / np.expand_dims(vol, axis=1)[..., -1])
            newvars['pressure_ratio_i_volume_average'] = (['time', 'ion'], pressure_i_vol[..., -1] / np.expand_dims(pressure_e_vol, axis=1)[..., -1])
            newvars['pressure_thermal_volume_average'] = (['time'], (pressure_e_vol + pressure_thermal_i_vol)[..., -1] / vol[..., -1])
            #newvars['pressure_ratio_thermal_volume_average'] = (['time'], (pressure_e_vol + pressure_thermal_i_vol)[..., -1] / (pressure_e_vol + np.sum(pressure_i_vol, axis=1))[..., -1])
            newvars['pressure_total_volume_average'] = (['time'], (pressure_e_vol + np.sum(pressure_i_vol, axis=1))[..., -1] / vol[..., -1])
            newvars['energy_e_volume_average'] = (['time'], energy_e_vol[..., -1] / vol[..., -1])
            newvars['energy_i_volume_average'] = (['time', 'ion'], energy_i_vol[..., -1] / np.expand_dims(vol, axis=1)[..., -1])
            newvars['energy_ratio_i_volume_average'] = (['time', 'ion'], energy_i_vol[..., -1] / np.expand_dims(energy_e_vol, axis=1)[..., -1])
            newvars['energy_thermal_volume_average'] = (['time'], (energy_e_vol + energy_thermal_i_vol)[..., -1] / vol[..., -1])
            #newvars['energy_ratio_thermal_volume_average'] = (['time'], (energy_e_vol + energy_thermal_i_vol)[..., -1] / (energy_e_vol + np.sum(energy_i_vol, axis=1))[..., -1])
            newvars['energy_total_volume_average'] = (['time'], (energy_e_vol + np.sum(energy_i_vol, axis=1))[..., -1] / vol[..., -1])

            density_e_00 = vectorized_numpy_interpolation(
                0.0,
                np.repeat(np.expand_dims(data['radius'].to_numpy(), axis=0), len(data['time']), axis=0),
                data['density_e'].to_numpy()
            )
            density_e_02 = vectorized_numpy_interpolation(
                0.2,
                np.repeat(np.expand_dims(data['radius'].to_numpy(), axis=0), len(data['time']), axis=0),
                data['density_e'].to_numpy()
            )
            temperature_e_00 = vectorized_numpy_interpolation(
                0.0,
                np.repeat(np.expand_dims(data['radius'].to_numpy(), axis=0), len(data['time']), axis=0),
                data['temperature_e'].to_numpy()
            )
            temperature_e_02 = vectorized_numpy_interpolation(
                0.2,
                np.repeat(np.expand_dims(data['radius'].to_numpy(), axis=0), len(data['time']), axis=0),
                data['temperature_e'].to_numpy()
            )
            density_i_00 = vectorized_numpy_interpolation(
                0.0,
                np.repeat(np.expand_dims(np.repeat(np.expand_dims(data['radius'].to_numpy(), axis=0), len(data['time']), axis=0), axis=1), len(data['ion']), axis=1),
                np.transpose(data['density_i'].to_numpy(), axes=(0, 2, 1))
            )
            density_i_02 = vectorized_numpy_interpolation(
                0.2,
                np.repeat(np.expand_dims(np.repeat(np.expand_dims(data['radius'].to_numpy(), axis=0), len(data['time']), axis=0), axis=1), len(data['ion']), axis=1),
                np.transpose(data['density_i'].to_numpy(), axes=(0, 2, 1))
            )
            temperature_i_00 = vectorized_numpy_interpolation(
                0.0,
                np.repeat(np.expand_dims(np.repeat(np.expand_dims(data['radius'].to_numpy(), axis=0), len(data['time']), axis=0), axis=1), len(data['ion']), axis=1),
                np.transpose(data['temperature_i'].to_numpy(), axes=(0, 2, 1))
            )
            temperature_i_02 = vectorized_numpy_interpolation(
                0.2,
                np.repeat(np.expand_dims(np.repeat(np.expand_dims(data['radius'].to_numpy(), axis=0), len(data['time']), axis=0), axis=1), len(data['ion']), axis=1),
                np.transpose(data['temperature_i'].to_numpy(), axes=(0, 2, 1))
            )
            newvars['density_peaking_e'] = (['time'], density_e_00 / (density_e_vol[..., -1] / vol[..., -1]))
            newvars['density_peaking_off_axis_e'] = (['time'], density_e_02 / (density_e_vol[..., -1] / vol[..., -1]))
            newvars['temperature_peaking_e'] = (['time'], temperature_e_00 / (temperature_e_vol[..., -1] / vol[..., -1]))
            newvars['temperature_peaking_off_axis_e'] = (['time'], temperature_e_02 / (temperature_e_vol[..., -1] / vol[..., -1]))
            newvars['density_peaking_i'] = (['time', 'ion'], density_i_00 / (density_i_vol[..., -1] / np.expand_dims(vol, axis=1)[..., -1]))
            newvars['density_peaking_off_axis_i'] = (['time', 'ion'], density_i_02 / (density_i_vol[..., -1] / np.expand_dims(vol, axis=1)[..., -1]))
            newvars['temperature_peaking_i'] = (['time', 'ion'], temperature_i_00 / (temperature_i_vol[..., -1] / np.expand_dims(vol, axis=1)[..., -1]))
            newvars['temperature_peaking_off_axis_i'] = (['time', 'ion'], temperature_i_02 / (temperature_i_vol[..., -1] / np.expand_dims(vol, axis=1)[..., -1]))

            density_ratio_vol = vectorized_numpy_integration(
                (data['density_ratio_main'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            temperature_ratio_vol = vectorized_numpy_integration(
                (data['temperature_ratio_main'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            effective_charge_vol = vectorized_numpy_integration(
                (data['effective_charge'] * data['mxh_dvolume_dr']).to_numpy(),
                data['r_minor'].to_numpy()
            )
            newvars['density_ratio_vol'] = (['time'], density_ratio_vol[..., -1] / vol[..., -1])
            newvars['temperature_ratio_vol'] = (['time'], temperature_ratio_vol[..., -1] / vol[..., -1])
            newvars['effective_charge_vol'] = (['time'], effective_charge_vol[..., -1] / vol[..., -1])

            #if 'mach' in data:
            #    newvars['mach_vol'] = (['n'], vectorized_numpy_integration((data['mach'] * data['mxh_dvolume_dr']).to_numpy(), data['r_minor'].to_numpy())[:, -1] / vol[:, -1])
            newvars['pressure_total_vol_norm_axis'] = (['time'], ((pressure_e_vol + np.sum(pressure_i_vol, axis=1))[..., -1] / vol[..., -1]) * (2.0 * self.constants['mu_si'] / (data['field_axis'] ** 2)).to_numpy())
            newvars['beta_n_axis'] = (['time'], ((pressure_e_vol + np.sum(pressure_i_vol, axis=1))[..., -1] / vol[..., -1]) * (2.0 * self.constants['mu_si'] * 100.0 * data['r_minor_lcfs'] / (data['field_axis'] * data['current'])).to_numpy())  # pc

            field_squared_vol = vectorized_numpy_integration(
                np.transpose((data['field_squared'] * data['mxh_dvolume_dr']).to_numpy(), axes=(0, 2, 1)),
                np.repeat(np.expand_dims(data['r_minor'].to_numpy(), axis=1), len(data['field_direction']), axis=1)
            )
            newvars['field_squared_vol'] = (['time', 'field_direction'], field_squared_vol[..., -1] / np.expand_dims(vol, axis=1)[..., -1])
            newvars['pressure_total_vol_norm_field'] = (['time', 'field_direction'], np.expand_dims((pressure_e_vol + np.sum(pressure_i_vol, axis=1))[..., -1], axis=-1) * 2.0 * self.constants['mu_si'] / field_squared_vol[..., -1])
            newvars['pressure_total_vol_norm'] = (['time'], (pressure_e_vol + np.sum(pressure_i_vol, axis=1))[..., -1] * 2.0 * self.constants['mu_si'] / np.sum(field_squared_vol, axis=1)[..., -1])
            newvars['beta_n'] = (['time'], ((pressure_e_vol + np.sum(pressure_i_vol, axis=1))[..., -1] * 2.0 * self.constants['mu_si'] / np.sum(field_squared_vol, axis=1)[..., -1]) * (100.0 * data['r_minor_lcfs'] * data['field_axis'] / data['current']).to_numpy())  # pc

            confinement_time_energy = np.where(np.isclose(heating_source_vol[..., -1], 0.0), np.inf, (energy_e_vol + energy_thermal_i_vol)[..., -1] / heating_source_vol[..., -1])
            confinement_time_particle = np.where(np.isclose(np.sum(particle_source_e_vol, axis=-1)[..., -1], 0.0), np.inf, density_e_vol[..., -1] / np.sum(particle_source_e_vol, axis=-1)[..., -1])
            newvars['confinement_time_energy'] = (['time'], confinement_time_energy)
            newvars['confinement_time_particle'] = (['time'], confinement_time_particle)
            newvars['confinement_time_ratio'] = (['time'], np.where(np.isfinite(confinement_time_energy), confinement_time_particle / confinement_time_energy, 0.0))

        self.update_input_data_vars(newvars)


    def _compute_scalings(
        self
    ) -> None:

        data = self.input
        newvars: MutableMapping[str, Any] = {}
        if 'current' in data and 'bcentr' in data:

            greenwald_density = (1.0e14 * data['current'] / (np.pi * data['r_minor_lcfs'] ** 2))
            newvars['greenwald_density'] = (['time'], greenwald_density.to_numpy())
            newvars['greenwald_fraction'] = (['time'], (data['density_e_vol'] / greenwald_density).to_numpy())
            #newvars['greenwald_density_local'] = (['time', 'radius'], (data['density_e'] / greenwald_density).to_numpy())

            newvars['effective_collisionality_angioni'] = (['time', 'radius'], (data['effective_charge_vol'] * data['r_geometric'] * 0.1 * data['density_e'] * data['temperature_e'] ** (-2)).to_numpy())

            newvars['confinement_time_scaling_h98'] = (['time'], (
                0.0562
                * data['current'] ** (0.93)
                * data['r_geometric_lcfs'] ** (1.97)
                * data['kappa'].isel(radius=-1) ** (0.78)
                * data['epsilon_lcfs'] ** (0.58)
                * data['field_axis'] ** (0.15)
                * (1.0e-19 * data['density_e_line_average']) ** (0.41)
                * data['mass_main_average'] ** (0.19)
                * data['power_input'] ** (-0.69)
            ).to_numpy())
            newvars['confinement_time_scaling_h89'] = (['time'], (
                0.048
                * data['current'] ** (0.85)
                * data['r_geometric_lcfs'] ** (1.50)
                * data['kappa'].isel(radius=-1) ** (0.50)
                * data['epsilon_lcfs'] ** (0.30)
                * data['field_axis'] ** (0.20)
                * (1.0e-20 * data['density_e_line_average']) ** (0.10)
                * data['mass_main_average'] ** (0.50)
                * data['power_input'] ** (-0.50)
            ).to_numpy())
            newvars['confinement_time_scaling_l97'] = (['time'], (
                0.023
                * data['current'] ** (0.96)
                * data['r_geometric_lcfs'] ** (1.83)
                * data['kappa'].isel(radius=-1) ** (0.64)
                * data['epsilon_lcfs'] ** (0.06)
                * data['field_axis'] ** (0.03)
                * (1.0e-19 * data['density_e_line_average']) ** (0.40)
                * data['mass_main_average'] ** (0.20)
                * data['power_input'] ** (-0.73)
            ).to_numpy())

            lh_nmin = (
                1.0e19 * 0.07
                * data['current'] ** (0.34)
                * data['field_axis'] ** (0.62)
                * data['r_minor_lcfs'] ** (-0.95)
                * data['epsilon_lcfs'] ** (0.4)
            ).to_numpy()
            nminfactor = np.where(data['density_e_volume_average'].to_numpy() > lh_nmin, (data['density_e_volume_average'].to_numpy() / lh_nmin) ** 2, 1.0)
            p_lh = (
                2.15
                * (1.0e-19 * data['density_e_volume_average']) ** (0.782)
                * data['field_axis'] ** (0.772)
                * data['r_minor_lcfs'] ** (0.975)
                * data['r_geometric_lcfs'] ** (0.999)
                * (2.0 / data['mass_main_average']) ** (1.11)
            ).to_numpy() * nminfactor
            newvars['power_lh_scaling_martin'] = (['time'], p_lh)
            newvars['power_lh_ratio'] = (['time'], data['power_scrape_off_layer'].to_numpy() / p_lh)

            epsilon_95 = vectorized_numpy_interpolation(
                0.95,
                data['magnetic_flux_norm'].sel(direction='poloidal').to_numpy(),
                data['epsilon'].to_numpy()
            )
            kappa_95 = vectorized_numpy_interpolation(
                0.95,
                data['magnetic_flux_norm'].sel(direction='poloidal').to_numpy(),
                data['mxh_kappa'].to_numpy()
            )
            delta_95 = vectorized_numpy_interpolation(
                0.95,
                data['magnetic_flux_norm'].sel(direction='poloidal').to_numpy(),
                data['mxh_delta'].to_numpy()
            )
            uckan_shaping = 1.0 + kappa_95[..., -1] ** 2 * (1.0 + 2.0 * delta_95[..., -1] ** 2 - 1.2 * delta_95[..., -1] ** 3)
            iter_shaping = uckan_shaping * (1.17 - 0.65 * epsilon_95) / (1 - epsilon_95 ** 2) ** 2

            newvars['qstar_uckan'] = (['time'], (
                2.5
                * data['r_geometric_lcfs']
                * data['epsilon_lcfs'] ** 2
                * data['field_axis']
                / data['current']
                * uckan_shaping
            ).to_numpy())
            newvars['qstar_iter'] = (['time'], (
                2.5
                * data['r_geometric_lcfs']
                * data['epsilon_lcfs'] ** 2
                * data['field_axis']
                / data['current']
                * iter_shaping
            ).to_numpy())

            newvars['width_scrape_off_layer_brunner'] = (['time'], (
                0.91
                * (data['pressure_total_volume_average'] / self.constants['atm_si']) ** (-0.48)
            ).to_numpy())
            newvars['width_scrape_off_layer_eich14'] = (['time'], (
                0.63
                * data['field_outer'].sel(direction='poloidal').isel(radius=-1) ** (-1.19)
            ).to_numpy())
            newvars['width_scrape_off_layer_eich15'] = (['time'], (
                1.35
                * data['power_scrape_off_layer'] ** (-0.02)
                * data['field_outer'].sel(direction='poloidal').isel(radius=-1) ** (-0.92)
                * data['r_geometric_lcfs'] ** (0.04)
                * data['epsilon_lcfs'] ** (0.42)
            ).to_numpy())

            # newvars['density_e_upstream'] = (['time'], 0.6 * data['density_e_vol'].isel(rho=-1).to_numpy())
            # debye_e_upstream = ((self.constants['eps_si'] / self.constants['e_si']) ** 0.5) * (data['temperature_e'] / (0.6 * data['density_e_vol'] * (data['charge_e'] ** 2))) ** 0.5  # m
            # collisionality_prefactor = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
            # inverse_approach_factor_ee = (4.0 * np.pi * self.constants['eps_si'] / self.constants['e_si']) * data['temperature_e'] / abs(data['charge_e'] ** 2)
            # log_coulomb_ee = np.log((inverse_approach_factor_ee * debye_e_upstream).isel(rho=-1))
            # te_guess = 1.0
            # p_elfrac = 0.5
            # k0e = 1.0e6 * (3.2 * 3.44e5 * (self.constants['e_si'] ** 2)) / self.constants['me_si']
            # Aqpar = 4.0 * np.pi * (1.0e-3 * data['eps'] * data['rcentr'] / data['q95']).to_numpy() * newvars['lq_brunner'][-1]
            # lpsol = (p_elfrac * 1.0e6 * data['p_sol'] * np.pi * data['rcentr'] * data['q95']).to_numpy()
            # newvars['temperature_e_upstream'] = (['n'], (3.5 * (lpsol / Aqpar) * lnC / k0e) ** (2.0 / 7.0))

        self.update_input_data_vars(newvars)


    def compute_derived_quantities(
        self,
    ) -> None:
        self._compute_derived_coordinates()
        self._compute_derived_reference_quantities()
        self._compute_derived_geometry()
        self._compute_extended_local_inputs()
        self._compute_integrated_quantities()
        self._compute_scalings()


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
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input if side == 'input' else obj.output
            direct_time_map = {
                'masse': 'mass_e',
                'ze': 'charge_e',
                'bcentr': 'field_axis',
                'current': 'current',
            }
            direct_time_ion_map = {
                'mass': 'mass_i',
            }
            direct_time_rho_map = {
                'rmin': 'r_minor',
                'rmaj': 'r_geometric',
                'zmag': 'z_geometric',
                'ne': 'density_e',
                'te': 'temperature_e',
                'qei': 'heat_exchange_ei',
            }
            direct_time_rho_ion_map = {
                'ni': 'density_i',
                'ti': 'temperature_i',
            }
            coords: MutableMapping[str, Any] = {}
            data_vars: MutableMapping[str, Any] = {}
            attrs: MutableMapping[str, Any] = {}
            if 'n' in data and 'rho' in data:
                coords['time'] = data['time'].to_numpy() if 'time' in data else np.arange(len(data['n']))
                coords['radius'] = data['rho'].to_numpy()
                if 'name' in data:
                    coords['ion'] = data['name'].to_numpy()
                    mass_i = np.zeros((len(coords['ion']), ))
                    atomic_number_i = np.zeros((len(coords['ion']), ))
                    for i, short_name in enumerate(coords['ion']):
                        _, sa, sz = define_ion_species(short_name=short_name)
                        mass_i[i] = sa
                        atomic_number_i[i] = sz
                    data_vars['mass_i'] = (['time', 'ion'], np.repeat(np.expand_dims(mass_i, axis=0), len(coords['time']), axis=0))
                    data_vars['atomic_number_i'] = (['time', 'ion'], np.repeat(np.expand_dims(atomic_number_i, axis=0), len(coords['time']), axis=0))
                    if 'type' in data:
                        data_vars['type_i'] = (['time', 'ion'], np.where(data['type'].to_numpy() == '[therm]', 'thermal', 'fast'))
                coords['direction'] = np.array(newobj.directions)
                coords['source'] = np.array(newobj.sources)
                for key, nkey in direct_time_map.items():
                    if key in data:
                        data_vars[nkey] = (['time'], data[key].to_numpy())
                for key, nkey in direct_time_ion_map.items():
                    if key in data and 'name' in data:
                        data_vars[nkey] = (['time', 'ion'], data[key].to_numpy())
                for key, nkey in direct_time_rho_map.items():
                    if key in data:
                        data_vars[nkey] = (['time', 'radius'], data[key].to_numpy())
                for key, nkey in direct_time_rho_ion_map.items():
                    if key in data and 'name' in data:
                        data_vars[nkey] = (['time', 'radius', 'ion'], data[key].to_numpy())
                if 'name' in data and 'z' in data:
                    data_vars['charge_i'] = (['time', 'radius', 'ion'], np.repeat(np.expand_dims(data['z'].to_numpy(), axis=1), len(coords['radius']), axis=1))
                flux = np.repeat(np.expand_dims(np.zeros((len(coords['time']), len(coords['radius']))), axis=-1), len(coords['direction']), axis=-1)
                if 'torflux' in data:
                    flux[..., 0] = data['torflux'].to_numpy()
                elif 'q' in data and 'polflux' in data:
                    flux[..., 0] = vectorized_numpy_integration(data['q'].to_numpy(), data['polflux'].to_numpy())
                if 'polflux' in data:
                    flux[..., 1] = data['polflux'].to_numpy()
                data_vars['magnetic_flux'] = (['time', 'radius', 'direction'], flux)
                if 'q' in data:
                    data_vars['safety_factor'] = (['time', 'radius'], data['q'].to_numpy())
                if 'name' in data:
                    velocity = np.repeat(np.expand_dims(np.zeros((len(coords['time']), len(coords['radius']), len(coords['ion']))), axis=-1), len(coords['direction']), axis=-1)
                    if 'vtor' in data:
                        velocity[..., 0] = data['vtor'].to_numpy()
                    if 'vpol' in data:
                        velocity[..., 1] = data['vpol'].to_numpy()
                    data_vars['velocity_i'] = (['time', 'radius', 'ion', 'direction'], velocity)
                if 'heat_exchange_ei' not in data_vars:
                    data_vars['heat_exchange_ei'] = (['time', 'radius'], np.zeros((len(coords['time']), len(coords['radius']))))
                heat_source_e = np.repeat(np.expand_dims(np.zeros((len(coords['time']), len(coords['radius']))), axis=-1), len(coords['source']), axis=-1)
                if 'qohme' in data:
                    heat_source_e[..., 0] = 1.0e6 * data['qohme'].to_numpy()
                if 'qbeame' in data:
                    heat_source_e[..., 1] = 1.0e6 * data['qbeame'].to_numpy()
                if 'qrfe' in data:
                    heat_source_e[..., 2] = 1.0e6 * data['qrfe'].to_numpy()
                if 'qsync' in data:
                    heat_source_e[..., 4] = -1.0e6 * data['qsync'].to_numpy()
                if 'qbrem' in data:
                    heat_source_e[..., 5] = -1.0e6 * data['qbrem'].to_numpy()
                if 'qline' in data:
                    heat_source_e[..., 6] = -1.0e6 * data['qline'].to_numpy()
                if 'qione' in data:
                    heat_source_e[..., 7] = -1.0e6 * data['qione'].to_numpy()
                if 'qfuse' in data:
                    heat_source_e[..., 10] = 1.0e6 * data['qfuse'].to_numpy()
                data_vars['heat_source_e'] = (['time', 'radius', 'source'], heat_source_e)
                heat_source_i = np.repeat(np.expand_dims(np.zeros((len(coords['time']), len(coords['radius']), len(coords['ion']))), axis=-1), len(coords['source']), axis=-1)
                # Assumes all ion heat sources apply to the first ion species only
                if 'qbeami' in data and 'name' in data:
                    heat_source_i[..., 0, 1] = 1.0e6 * data['qbeami'].to_numpy()
                if 'qrfi' in data and 'name' in data:
                    heat_source_i[..., 0, 2] = 1.0e6 * data['qrfi'].to_numpy()
                if 'qioni' in data and 'name' in data:
                    heat_source_i[..., 0, 7] = -1.0e6 * data['qioni'].to_numpy()
                if 'qcxi' in data and 'name' in data:
                    heat_source_i[..., 0, 8] = -1.0e6 * data['qcxi'].to_numpy()
                if 'qfusi' in data and 'name' in data:
                    heat_source_i[..., 0, 10] = 1.0e6 * data['qfusi'].to_numpy()
                data_vars['heat_source_i'] = (['time', 'radius', 'ion', 'source'], heat_source_i)
                particle_source_e = np.repeat(np.expand_dims(np.zeros((len(coords['time']), len(coords['radius']))), axis=-1), len(coords['source']), axis=-1)
                if 'qpar_beam' in data:
                    particle_source_e[..., 1] = data['qpar_beam'].to_numpy()
                data_vars['particle_source_e'] = (['time', 'radius', 'source'], particle_source_e)
                particle_source_i = np.repeat(np.expand_dims(np.zeros((len(coords['time']), len(coords['radius']), len(coords['ion']))), axis=-1), len(coords['source']), axis=-1)
                # Assumes all ion particle sources apply to the first ion species only
                if 'qpar_beam' in data and 'name' in data:
                    particle_source_i[..., 0, 1] = data['qpar_beam'].to_numpy()
                if 'qpar_wall' in data and 'name' in data:
                    particle_source_i[..., 0, 7] = data['qpar_wall'].to_numpy()
                data_vars['particle_source_i'] = (['time', 'radius', 'ion', 'source'], particle_source_i)
                momentum_source_i = np.repeat(np.expand_dims(np.zeros((len(coords['time']), len(coords['radius']), len(coords['ion']), len(coords['direction']))), axis=-1), len(coords['source']), axis=-1)
                # Assumes all ion momentum sources apply to the first ion species only
                if 'qmom' in data and 'name' in data:
                    momentum_source_i[..., 0, 0, 1] = data['qmom'].to_numpy()
                data_vars['momentum_source_i'] = (['time', 'radius', 'ion', 'direction', 'source'], momentum_source_i)
                current_source = np.repeat(np.expand_dims(np.zeros((len(coords['time']), len(coords['radius']))), axis=-1), len(coords['source']), axis=-1)
                if 'johm' in data:
                    current_source[..., 0] = data['johm'].to_numpy()
                if 'jnb' in data:
                    current_source[..., 1] = data['jnb'].to_numpy()
                if 'jrf' in data:
                    current_source[..., 3] = data['jrf'].to_numpy()
                if 'jbs' in data:
                    current_source[..., 9] = data['jbs'].to_numpy()
                data_vars['current_source'] = (['time', 'radius', 'source'], current_source)
                if 'rmaj' in data and 'zmag' in data and 'rmin' in data and 'kappa' in data:
                    r0 = data['rmaj'].to_numpy()
                    z0 = data['zmag'].to_numpy()
                    r = data['rmin'].to_numpy()
                    kappa = data['kappa'].to_numpy()
                    sin_coeffs = [np.zeros_like(kappa)]
                    cos_coeffs = []
                    if 'delta' in data:
                        sin_coeffs.append(np.sin(data['delta'].to_numpy()))
                    elif 'shape_sin1' in data:
                        sin_coeffs.append(data['shape_sin1'].to_numpy())
                    else:
                        sin_coeffs.append(np.zeros_like(kappa))
                    if 'zeta' in data:
                        sin_coeffs.append(-data['zeta'].to_numpy())
                    elif 'shape_sin2' in data:
                        sin_coeffs.append(data['shape_sin2'].to_numpy())
                    else:
                        sin_coeffs.append(np.zeros_like(kappa))
                    for i in range(3, 7):
                        if f'shape_sin{i}' in data:
                            sin_coeffs.append(data[f'shape_sin{i}'].to_numpy())
                        else:
                            sin_coeffs.append(np.zeros_like(kappa))
                    for i in range(0, 7):
                        if f'shape_cos{i}' in data:
                            cos_coeffs.append(data[f'shape_cos{i}'].to_numpy())
                        else:
                            cos_coeffs.append(np.zeros_like(kappa))
                    if len(sin_coeffs) < len(cos_coeffs):
                        for _ in range(len(cos_coeffs) - len(sin_coeffs)):
                            sin_coeffs.append(np.zeros_like(kappa))
                    if len(cos_coeffs) < len(sin_coeffs):
                        for _ in range(len(sin_coeffs) - len(cos_coeffs)):
                            cos_coeffs.append(np.zeros_like(kappa))
                    theta = np.linspace(0.0, 2.0 * np.pi, 501)
                    theta_z = np.repeat(np.expand_dims(np.repeat(np.expand_dims(theta, axis=0), len(coords['radius']), axis=0), axis=0), len(coords['time']), axis=0)
                    theta_r = copy.deepcopy(theta_z)
                    for i in range(len(sin_coeffs)):
                        theta_r += np.expand_dims(sin_coeffs[i], axis=-1) * np.sin(i * theta_z) + np.expand_dims(cos_coeffs[i], axis=-1) * np.cos(i * theta_z)
                    r_contour = np.expand_dims(r0, axis=-1) + np.expand_dims(r, axis=-1) * np.cos(theta_r)
                    z_contour = np.expand_dims(z0, axis=-1) + np.expand_dims(r * kappa, axis=-1) * np.sin(theta_z)
                    l_contour = np.sqrt((r_contour - np.expand_dims(r0, axis=-1)) ** 2 + (z_contour - np.expand_dims(z0, axis=-1)) ** 2)
                    coords['angle_geometric'] = theta
                    data_vars['contour'] = (['time', 'radius', 'angle_geometric'], l_contour)
            newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj
