import copy
import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

import datetime
from scipy.integrate import cumulative_simpson  # type: ignore[import-untyped]
from .io import io

logger = logging.getLogger('fusio')


class plasma_io(io):


    basevars: Final[Sequence[str]] = [
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
            if not path.exists() or overwrite:
                opath = Path(path)
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
            if newvars:
                if side == 'input':
                    self.update_input_data_vars(newvars)
                else:
                    self.update_output_data_vars(newvars)


    def _compute_derived_coordinates(
        self,
    ) -> None:
        data = self.input
        newvars: MutableMapping[str, Any] = {}
        if 'rho' in data:
            if 'rmin' in data:
                a = data['rmin'].isel(rho=-1)
                newvars['a'] = (['n'], a.to_numpy())
                newvars['roa'] = (['n', 'rho'], (data['rmin'] / a).to_numpy())
                if 'rmaj' in data:
                    newvars['aspect_local'] = (['n', 'rho'], (data['rmaj'] / data['rmin']).to_numpy())
                    newvars['aspect'] = (['n'], (data['rmaj'] / data['rmin']).isel(rho=-1).to_numpy())
                    newvars['eps_local'] = (['n', 'rho'], (data['rmin'] / data['rmaj']).to_numpy())
                    newvars['eps'] = (['n'], (data['rmin'] / data['rmaj']).isel(rho=-1).to_numpy())
                    newvars['rmajoa'] = (['n', 'rho'], (data['rmaj'] / a).to_numpy())
                if 'zmag' in data:
                    newvars['zmagoa'] = (['n', 'rho'], (data['zmag'] / a).to_numpy())
            if 'torfluxa' in data:
                torflux = 2.0 * np.pi * data['torfluxa'] * data['rho'] ** 2
                newvars['torflux'] = (['n', 'rho'], torflux.to_numpy())
                newvars['psi_tor_norm'] = (['n', 'rho'], (torflux / torflux.isel(rho=-1)).to_numpy())
                newvars['rho_tor'] = (['n', 'rho'], np.sqrt((torflux / torflux.isel(rho=-1)).to_numpy()))
            if 'polflux' in data:
                polfluxn = (data['polflux'] - data['polflux'].isel(rho=0)) / (data['polflux'].isel(rho=-1) - data['polflux'].isel(rho=0))
                newvars['psi_pol_norm'] = (['n', 'rho'], polfluxn.to_numpy())
                newvars['rho_pol'] = (['n', 'rho'], np.sqrt(polfluxn.to_numpy()))
        self.update_input_data_vars(newvars)


    def _compute_derived_reference_quantities(
        self,
    ) -> None:
        def derivative(x, y):
            deriv = np.zeros_like(x)
            if x.shape[-1] > 2:
                x1 = np.concatenate([np.expand_dims(x[..., 0], axis=-1), x[..., :-2], np.expand_dims(x[..., -3], axis=-1)], axis=-1)
                x2 = np.concatenate([np.expand_dims(x[..., 1], axis=-1), x[..., 1:-1], np.expand_dims(x[..., -2], axis=-1)], axis=-1)
                x3 = np.concatenate([np.expand_dims(x[..., 2], axis=-1), x[..., 2:], np.expand_dims(x[..., -1], axis=-1)], axis=-1)
                y1 = np.concatenate([np.expand_dims(y[..., 0], axis=-1), y[..., :-2], np.expand_dims(y[..., -3], axis=-1)], axis=-1)
                y2 = np.concatenate([np.expand_dims(y[..., 1], axis=-1), y[..., 1:-1], np.expand_dims(y[..., -2], axis=-1)], axis=-1)
                y3 = np.concatenate([np.expand_dims(y[..., 2], axis=-1), y[..., 2:], np.expand_dims(y[..., -1], axis=-1)], axis=-1)
                deriv = ((x - x1) + (x - x2)) / (x3 - x1) / (x3 - x2) * y3 + ((x - x1) + (x - x3)) / (x2 - x1) / (x2 - x3) * y2 + ((x - x2) + (x - x3)) / (x1 - x2) / (x1 - x3) * y1
            elif x.shape[-1] > 1:
                deriv[..., 0] = np.diff(y, axis=-1) / np.diff(x, axis=-1)
                deriv[..., 1] = np.diff(y, axis=-1) / np.diff(x, axis=-1)
            return deriv
        e_si = 1.60218e-19  # C
        u_si = 1.66054e-27  # kg
        data = self.input
        newvars = {}
        if 'rho' in data:
            mref = data.get('mref', xr.zeros_like(data['n']) + 2.0).to_numpy()
            if 'mref' not in data:
                newvars['mref'] = (['n'], mref)
            if 'rmin' in data and 'torflux' in data:
                bunit = derivative(0.5 * data['rmin'].to_numpy() ** 2, data['torflux'].to_numpy() / (2.0 * np.pi))
                newvars['b_unit'] = (['n', 'rho'], bunit)
            if 'te' in data:
                newvars['c_s'] = (['n', 'rho'], (1.0e3 * e_si * data['te'].to_numpy() / (u_si * mref)) ** (0.5))
                if 'b_unit' in newvars:
                    newvars['rho_s_unit'] = (['n', 'rho'], (1.0e3 * data['te'].to_numpy() * u_si * mref / e_si) ** (0.5) / np.abs(newvars['b_unit'][-1]))
                if 'masse' in data:
                    newvars['c_the'] = (['n', 'rho'], ((2.0 * 1.0e3 * e_si * data['te'] / (u_si * data['masse'])) ** (0.5)).to_numpy())
            if 'ti' in data and 'masse' in data:
                newvars['c_thi'] = (['n', 'rho', 'name'], ((2.0 * 1.0e3 * e_si * data['ti'] / (u_si * data['mass'])) ** (0.5)).to_numpy())
            if 'rcentr' in data:
                newvars['rgeo'] = (['n'], data['rcentr'].to_numpy())
            if 'bcentr' in data:
                newvars['b_zero'] = (['n'], np.abs(data['bcentr'].to_numpy()))
        self.update_input_data_vars(newvars)


    def _compute_derived_geometry(
        self,
    ) -> None:
        def derivative(x, y):
            deriv = np.zeros_like(x)
            if x.shape[-1] > 2:
                x1 = np.concatenate([np.expand_dims(x[..., 0], axis=-1), x[..., :-2], np.expand_dims(x[..., -3], axis=-1)], axis=-1)
                x2 = np.concatenate([np.expand_dims(x[..., 1], axis=-1), x[..., 1:-1], np.expand_dims(x[..., -2], axis=-1)], axis=-1)
                x3 = np.concatenate([np.expand_dims(x[..., 2], axis=-1), x[..., 2:], np.expand_dims(x[..., -1], axis=-1)], axis=-1)
                y1 = np.concatenate([np.expand_dims(y[..., 0], axis=-1), y[..., :-2], np.expand_dims(y[..., -3], axis=-1)], axis=-1)
                y2 = np.concatenate([np.expand_dims(y[..., 1], axis=-1), y[..., 1:-1], np.expand_dims(y[..., -2], axis=-1)], axis=-1)
                y3 = np.concatenate([np.expand_dims(y[..., 2], axis=-1), y[..., 2:], np.expand_dims(y[..., -1], axis=-1)], axis=-1)
                deriv = ((x - x1) + (x - x2)) / (x3 - x1) / (x3 - x2) * y3 + ((x - x1) + (x - x3)) / (x2 - x1) / (x2 - x3) * y2 + ((x - x2) + (x - x3)) / (x1 - x2) / (x1 - x3) * y1
            elif x.shape[-1] > 1:
                deriv[..., 0] = np.diff(y, axis=-1) / np.diff(x, axis=-1)
                deriv[..., 1] = np.diff(y, axis=-1) / np.diff(x, axis=-1)
            return deriv
        data = self.input
        newvars: MutableMapping[str, Any] = {}
        if 'roa' in data:
            signb = 1.0
            n_rho = len(data['rho'])
            if 'kappa' in data:
                s_v = (data['roa'] / data['kappa']).to_numpy() * derivative(data['roa'].to_numpy(), data['kappa'].to_numpy())
                newvars['s_kappa'] = (['n', 'rho'], np.where(np.isclose(s_v, 0.0), 0.0, s_v))
            if 'delta' in data:
                s_v = data['roa'].to_numpy() * derivative(data['roa'].to_numpy(), data['delta'].to_numpy())
                newvars['s_delta'] = (['n', 'rho'], np.where(np.isclose(s_v, 0.0), 0.0, s_v))
            if 'zeta' in data:
                s_v = data['roa'].to_numpy() * derivative(data['roa'].to_numpy(), data['zeta'].to_numpy())
                newvars['s_zeta'] = (['n', 'rho'], np.where(np.isclose(s_v, 0.0), 0.0, s_v))
            if 'rmaj' in data:
                s_v = derivative(data['roa'].to_numpy(), data['rmaj'].to_numpy())
                newvars['drmajdr'] = (['n', 'rho'], np.where(np.isclose(s_v, 0.0), 0.0, s_v))
            if 'zmag' in data:
                s_v = derivative(data['roa'].to_numpy(), data['zmag'].to_numpy())
                newvars['dzmagdr'] = (['n', 'rho'], np.where(np.isclose(s_v, 0.0), 0.0, s_v))
            n_theta = 1001
            theta = np.expand_dims(np.expand_dims(np.linspace(-np.pi, np.pi, n_theta), axis=-1), axis=-1)
            #! A
            #! dA/dr
            #! dA/dtheta
            #! d^2A/dtheta^2
            a = np.repeat(theta, n_rho, axis=-1)
            a_r = np.zeros_like(a)
            a_t = np.ones_like(a)
            a_tt = np.zeros_like(a)
            for i in range(7):
                if f'shape_sin{i:d}' in data:
                    s_v = data['roa'].to_numpy() * derivative(data['roa'].to_numpy(), data[f'shape_sin{i:d}'].to_numpy())
                    newvars[f's_shape_sin{i:d}'] = (['n', 'rho'], np.where(np.isclose(s_v, 0.0), 0.0, s_v))
                    a += np.expand_dims(data[f'shape_sin{i:d}'].to_numpy(), axis=0) * np.sin(float(i) * theta)
                    a_r += np.expand_dims(newvars[f's_shape_sin{i:d}'][-1], axis=0) * np.sin(float(i) * theta)
                    a_t += np.expand_dims(data[f'shape_sin{i:d}'].to_numpy(), axis=0) * float(i) * np.cos(float(i) * theta)
                    a_tt += np.expand_dims(data[f'shape_sin{i:d}'].to_numpy(), axis=0) * float(-i * i) * np.sin(float(i) * theta)
                elif i == 0:
                    newvars[f'shape_sin{i:d}'] = (['n', 'rho'], np.zeros_like(data['kappa'].to_numpy()))
                elif i == 1 and 'delta' in data:
                    s = np.arcsin(data['delta'].to_numpy())
                    s_v = data['roa'].to_numpy() * derivative(data['roa'].to_numpy(), np.where(np.isclose(s, 0.0), 0.0, s))
                    newvars[f'shape_sin{i:d}'] = (['n', 'rho'], np.where(np.isclose(s, 0.0), 0.0, s))
                    newvars[f's_shape_sin{i:d}'] = (['n', 'rho'], np.where(np.isclose(s_v, 0.0), 0.0, s_v))
                    a += np.expand_dims(newvars[f'shape_sin{i:d}'][-1], axis=0) * np.sin(float(i) * theta)
                    a_r += np.expand_dims(newvars[f's_shape_sin{i:d}'][-1], axis=0) * np.sin(float(i) * theta)
                    a_t += np.expand_dims(newvars[f'shape_sin{i:d}'][-1], axis=0) * float(i) * np.cos(float(i) * theta)
                    a_tt += np.expand_dims(newvars[f'shape_sin{i:d}'][-1], axis=0) * float(-i * i) * np.sin(float(i) * theta)
                elif i == 2 and 'zeta' in data:
                    s = -data['zeta'].to_numpy()
                    s_v = data['roa'].to_numpy() * derivative(data['roa'].to_numpy(), np.where(np.isclose(s, 0.0), 0.0, s))
                    newvars[f'shape_sin{i:d}'] = (['n', 'rho'], np.where(np.isclose(s, 0.0), 0.0, s))
                    newvars[f's_shape_sin{i:d}'] = (['n', 'rho'], np.where(np.isclose(s_v, 0.0), 0.0, s_v))
                    a += np.expand_dims(newvars[f'shape_sin{i:d}'][-1], axis=0) * np.sin(float(i) * theta)
                    a_r += np.expand_dims(newvars[f's_shape_sin{i:d}'][-1], axis=0) * np.sin(float(i) * theta)
                    a_t += np.expand_dims(newvars[f'shape_sin{i:d}'][-1], axis=0) * float(i) * np.cos(float(i) * theta)
                    a_tt += np.expand_dims(newvars[f'shape_sin{i:d}'][-1], axis=0) * float(-i * i) * np.sin(float(i) * theta)
                if f'shape_cos{i:d}' in data:
                    s_v = data['roa'].to_numpy() * derivative(data['roa'].to_numpy(), data[f'shape_cos{i:d}'].to_numpy())
                    newvars[f's_shape_cos{i:d}'] = (['n', 'rho'], np.where(np.isclose(s_v, 0.0), 0.0, s_v))
                    a += np.expand_dims(data[f'shape_cos{i:d}'].to_numpy(), axis=0) * np.cos(float(i) * theta)
                    a_r += np.expand_dims(newvars[f's_shape_cos{i:d}'][-1], axis=0) * np.cos(float(i) * theta)
                    a_t += np.expand_dims(data[f'shape_cos{i:d}'].to_numpy(), axis=0) * float(-i) * np.sin(float(i) * theta)
                    a_tt += np.expand_dims(data[f'shape_cos{i:d}'].to_numpy(), axis=0) * float(-i * i) * np.cos(float(i) * theta)
            #! R(theta)
            #! dR/dr
            #! dR/dtheta
            #! d^2R/dtheta^2
            r = np.expand_dims(data['rmaj'].to_numpy(), axis=0) + np.expand_dims(data['rmin'].to_numpy(), axis=0) * np.cos(a)
            r_r = np.expand_dims(newvars['drmajdr'][-1], axis=0) + np.cos(a) - np.expand_dims(data['rmin'].to_numpy(), axis=0) * np.sin(a) * a_r
            r_t = np.expand_dims(-data['rmin'].to_numpy(), axis=0) * a_t * np.sin(a)
            r_tt = np.expand_dims(-data['rmin'].to_numpy(), axis=0) * (a_t**2 * np.cos(a) + a_tt * np.sin(a))
            #! Z(theta)
            #! dZ/dr
            #! dZ/dtheta
            #! d^2Z/dtheta^2
            z = np.expand_dims(data['zmag'].to_numpy(), axis=0) + np.expand_dims((data['kappa'] * data['rmin']).to_numpy(), axis=0) * np.sin(theta)
            z_r = np.expand_dims(newvars['dzmagdr'][-1], axis=0) + np.expand_dims(data['kappa'].to_numpy() * (1.0 + newvars['s_kappa'][-1]), axis=0) * np.sin(theta)
            z_t = np.expand_dims((data['kappa'] * data['rmin']).to_numpy(), axis=0) * np.cos(theta)
            z_tt = np.expand_dims((-data['kappa'] * data['rmin']).to_numpy(), axis=0) * np.sin(theta)
            g_tt = r_t ** 2 + z_t ** 2
            l_t = np.sqrt(g_tt)
            j_r = r * (r_r * z_t - r_t * z_r)
            inv_j_r = 1.0 / np.where(np.isclose(j_r, 0.0), 0.001, j_r)
            grad_r = np.where(np.isclose(j_r, 0.0), 1.0, r * l_t * inv_j_r)
            #r_c = l_t ** 3 / (r_t * z_tt - z_t * r_tt)
            #z_l = np.where(np.isclose(l_t, 0.0), 0.0, z_t / l_t)
            #r_l = np.where(np.isclose(l_t, 0.0), 0.0, r_t / l_t)
            #l_r = z_l * z_r + r_l * r_r
            #nsin = (r_r * r_t + z_r * z_t) / l_t
            c = 2.0 * np.pi * np.sum(l_t[:-1] / (r[:-1] * grad_r[:-1]), axis=0)
            f = 2.0 * np.pi * data['rmin'].to_numpy() / (np.where(np.isclose(c, 0.0), 1.0, c) / float(n_theta - 1))
            f[..., 0] = 2.0 * f[..., 1] - f[..., 2]
            newvars['volp_miller'] = (['n', 'rho'], 2.0 * np.pi * np.where(np.isfinite(c), c, 0.0) / float(n_theta - 1))
            newvars['surf_miller'] = (['n', 'rho'], 2.0 * np.pi * np.sum(l_t[:-1] * r[:-1], axis=0) * 2.0 * np.pi / float(n_theta - 1))
            bt = np.expand_dims(f, axis=0) / r
            bp = np.expand_dims((data['rmin'] / data['q']).to_numpy(), axis=0) * grad_r / r
            b = signb * np.sqrt(bt ** 2 + bp ** 2)
            r_v = np.expand_dims((data['rmin'] * data['rmaj']).to_numpy(), axis=0)
            g_t = r * b * l_t / (np.where(np.isclose(r_v, 0.0), 1.0, r_v) * grad_r)
            g_t[..., 0] = 2.0 * g_t[..., 1] - g_t[..., 2]
            dtheta = 2.0 * np.pi / float(n_theta - 1)
            theta0 = np.pi
            i1 = int(theta0 / dtheta) + 1
            i2 = i1 + 1
            theta1 = (i1 - 1) * dtheta
            ztheta = (theta0 - theta1) / dtheta
            if i2 == n_theta:
                i2 -= 1
            newvars['geo_bt'] = (['n', 'rho'], bt[i1] + (bt[i2] - bt[i1]) * ztheta)
            denom = np.sum(np.where(np.isfinite(g_t), g_t, 0.0)[:-1] / b[:-1], axis=0)
            denom[..., 0] = 2.0 * denom[..., 1] - denom[..., 2]
            newvars['gradr_miller'] = (['n', 'rho'], np.sum(grad_r[:-1] * g_t[:-1] / b[:-1], axis=0) / denom)
            newvars['bp2_miller'] = (['n', 'rho'], np.sum(bt[:-1] ** 2 * g_t[:-1] / b[:-1], axis=0) / denom)
            newvars['bt2_miller'] = (['n', 'rho'], np.sum(bp[:-1] ** 2 * g_t[:-1] / b[:-1], axis=0) / denom)
            newvars['r_surface'] = (['theta', 'n', 'rho'], r)
            newvars['z_surface'] = (['theta', 'n', 'rho'], z)
            newvars['surfxs'] = (['n', 'rho'], np.trapezoid(r, z, axis=0))
            newvars['r_out'] = (['n', 'rho'], np.nanmax(r, axis=0))
            newvars['r_in'] = (['n', 'rho'], np.nanmin(r, axis=0))
            newvars['b_ref'] = (['n', 'rho'], np.abs(data['b_unit'].to_numpy() * newvars['geo_bt'][-1]))  # For synchrotron
            bt = np.squeeze(np.take_along_axis(bt, np.expand_dims(np.argmax(r, axis=0), axis=0), axis=0), axis=0)
            bp = np.squeeze(np.take_along_axis(bp, np.expand_dims(np.argmax(r, axis=0), axis=0), axis=0), axis=0)
            newvars['bt_out'] = (['n', 'rho'], np.where(np.isfinite(bt), bt, 0.0))
            newvars['bp_out'] = (['n', 'rho'], np.where(np.isfinite(bp), bp, 0.0))
        self.update_input_data_vars(newvars)


    def _compute_average_mass(
        self,
    ) -> None:
        data = self.input
        newvars: MutableMapping[str, Any] = {}
        if 'name' in data and 'mass' in data and 'ni' in data and 'volp_miller' in data:
            main_species_mask = (data['name'].isin(['H', 'D', 'T']).to_numpy() & (data['type'].isin(['[therm]'])).to_numpy()).flatten()
            main_species = [i for i in range(len(main_species_mask)) if main_species_mask[i]]
            n_i_vol = cumulative_simpson(
                np.transpose((data['ni'].isel(name=main_species) * data['volp_miller']).to_numpy(), axes=(0, 2, 1)),
                x=np.repeat(np.expand_dims(data['rmin'].to_numpy(), axis=1), len(main_species), axis=1),
                initial=0.0
            )
            f_i_i_vol = n_i_vol[:, :, -1] / np.expand_dims(np.sum(n_i_vol[:, :, -1], axis=1), axis=1)
            mass_factor = data['mass'].isel(name=main_species).to_numpy()
            newvars['mass_i'] = (['n'], np.sum((f_i_i_vol * mass_factor), axis=1))
        self.update_input_data_vars(newvars)


    def _compute_source_terms(
        self,
    ) -> None:

        data = self.input
        newvars: MutableMapping[str, Any] = {}

        qe_terms = {
            'qohme': 1.0,
            'qbeame': 1.0,
            'qrfe': 1.0,
            'qfuse': 1.0,
            'qei': -1.0,
            'qsync': -1.0,
            'qbrem': -1.0,
            'qline': -1.0,
            'qione': 1.0,
        }
        qe = np.zeros_like(data['te'].to_numpy())
        for var in qe_terms:
            if var in data:
                qe += qe_terms[var] * data[var].to_numpy()
        newvars['qe'] = (['n', 'rho'], qe)

        qi_terms = {
            'qbeami': 1.0,
            'qrfi': 1.0,
            'qfusi': 1.0,
            'qei': 1.0,
            'qioni': 1.0,
        }
        qi = np.zeros_like(data['te'].to_numpy())
        for var in qi_terms:
            if var in data:
                qi += qi_terms[var] * data[var].to_numpy()
        newvars['qi'] = (['n', 'rho'], qi)

        ge_terms = {
            'qpar_beam': 1.0, 
            'qpar_wall': 1.0,
        }
        ge = np.zeros_like(data['te'].to_numpy())
        for var in ge_terms:
            if var in data:
                ge += ge_terms[var] * data[var].to_numpy()
        newvars['ge'] = (['n', 'rho'], ge)

        qrad_terms = {
            'qsync': 1.0,
            'qbrem': 1.0,
            'qline': 1.0,
        }
        qrad = np.zeros_like(data['te'].to_numpy())
        for var in qrad_terms:
            if var in data:
                qrad += qrad_terms[var] * data[var].to_numpy()
        newvars['qrad'] = (['n', 'rho'], qrad)

        qe_aux_terms = {
            'qohme': 1.0,
            'qbeame': 1.0,
            'qrfe': 1.0,
        }
        qe_aux = np.zeros_like(data['te'].to_numpy())
        for var in qe_aux_terms:
            if var in data:
                qe_aux += qe_aux_terms[var] * data[var].to_numpy()
        newvars['qe_aux'] = (['n', 'rho'], qe_aux)
        if 'qione' in data:
            qe_aux += data['qione'].to_numpy()
        newvars['qe_aux_ion'] = (['n', 'rho'], qe_aux)

        qi_aux_terms = {
            'qbeami': 1.0,
            'qrfi': 1.0,
        }
        qi_aux = np.zeros_like(data['te'].to_numpy())
        for var in qi_aux_terms:
            if var in data:
                qi_aux += qi_aux_terms[var] * data[var].to_numpy()
        newvars['qi_aux'] = (['n', 'rho'], qi_aux)
        if 'qioni' in data:
            qi_aux += data['qioni'].to_numpy()
        newvars['qi_aux_ion'] = (['n', 'rho'], qi_aux)

        qrf_terms = {
            'qrfe': 1.0,
            'qrfi': 1.0,
        }
        qrf = np.zeros_like(data['te'].to_numpy())
        for var in qrf_terms:
            if var in data:
                qrf += qrf_terms[var] * data[var].to_numpy()
        newvars['qrf'] = (['n', 'rho'], qrf)

        qbeam_terms = {
            'qbeame': 1.0,
            'qbeami': 1.0,
        }
        qbeam = np.zeros_like(data['te'].to_numpy())
        for var in qbeam_terms:
            if var in data:
                qbeam += qbeam_terms[var] * data[var].to_numpy()
        newvars['qbeam'] = (['n', 'rho'], qbeam)

        qion_terms = {
            'qione': 1.0,
            'qioni': 1.0,
        }
        qion = np.zeros_like(data['te'].to_numpy())
        for var in qion_terms:
            if var in data:
                qion += qion_terms[var] * data[var].to_numpy()
        newvars['qion'] = (['n', 'rho'], qion)

        qalpha_terms = {
            'qfuse': 1.0,
            'qfusi': 1.0,
        }
        qalpha = np.zeros_like(data['te'].to_numpy())
        for var in qalpha_terms:
            if var in data:
                qalpha += qalpha_terms[var] * data[var].to_numpy()
        newvars['qalpha'] = (['n', 'rho'], qalpha)

        self.update_input_data_vars(newvars)


    def _compute_extended_local_inputs(
        self
    ) -> None:

        def derivative(x, y):
            deriv = np.zeros_like(x)
            if x.shape[-1] > 2:
                x1 = np.concatenate([np.expand_dims(x[..., 0], axis=-1), x[..., :-2], np.expand_dims(x[..., -3], axis=-1)], axis=-1)
                x2 = np.concatenate([np.expand_dims(x[..., 1], axis=-1), x[..., 1:-1], np.expand_dims(x[..., -2], axis=-1)], axis=-1)
                x3 = np.concatenate([np.expand_dims(x[..., 2], axis=-1), x[..., 2:], np.expand_dims(x[..., -1], axis=-1)], axis=-1)
                y1 = np.concatenate([np.expand_dims(y[..., 0], axis=-1), y[..., :-2], np.expand_dims(y[..., -3], axis=-1)], axis=-1)
                y2 = np.concatenate([np.expand_dims(y[..., 1], axis=-1), y[..., 1:-1], np.expand_dims(y[..., -2], axis=-1)], axis=-1)
                y3 = np.concatenate([np.expand_dims(y[..., 2], axis=-1), y[..., 2:], np.expand_dims(y[..., -1], axis=-1)], axis=-1)
                deriv = ((x - x1) + (x - x2)) / (x3 - x1) / (x3 - x2) * y3 + ((x - x1) + (x - x3)) / (x2 - x1) / (x2 - x3) * y2 + ((x - x2) + (x - x3)) / (x1 - x2) / (x1 - x3) * y1
            elif x.shape[-1] > 1:
                deriv[..., 0] = np.diff(y, axis=-1) / np.diff(x, axis=-1)
                deriv[..., 1] = np.diff(y, axis=-1) / np.diff(x, axis=-1)
            return deriv

        def interpolate(v, x, y):
            vm = np.array([v]) if isinstance(v, (float, int)) else copy.deepcopy(v)
            xm = x.reshape(-1, x.shape[-1])
            ym = y.reshape(-1, y.shape[-1])
            if vm.shape[0] != xm.shape[0]:
                vm = np.repeat(np.expand_dims(vm, axis=0), xm.shape[0], axis=0)
            interp = np.zeros_like(vm)
            for i in range(xm.shape[0]):
                interp[i] = np.interp(vm[i], xm[i], ym[i])
            interp = interp.reshape(*y.shape[:-1])
            return interp

        def find(v, x, y, last=False):
            xm = x.reshape(-1, x.shape[-1])
            ym = y.reshape(-1, y.shape[-1])
            found = np.full((xm.shape[0], ), np.nan)
            for i in range(xm.shape[0]):
                yidx = np.where(((ym[i] - v)[1:] * (ym[i] - v)[:-1]) < 0.0)[0]
                if len(yidx) > 0:
                    yi = yidx[-1] if last else yidx[0]
                    found[i] = (v - ym[i, yi]) * (xm[i, yi + 1] - xm[i, yi]) / (ym[i, yi + 1] - ym[i, yi])
            found = found.reshape(*y.shape[:-1])
            return found

        data = self.input
        newvars: MutableMapping[str, Any] = {}

        e_si = 1.60218e-19
        u_si = 1.66054e-27
        eps_si = 8.85419e-12
        if 'type' in data:
            thermal_species_mask = data['type'].isin(['[therm]']).to_numpy().flatten()
            thermal_species = [i for i in range(len(thermal_species_mask)) if thermal_species_mask[i]]
            #newvars['ni_th'] = data['ni'].isel(name=thermal_species)
            newvars['ni_th_all'] = (['n', 'rho'], data['ni'].isel(name=thermal_species).sum('name').to_numpy())
            newvars['ni_all'] = (['n', 'rho'], data['ni'].sum('name').to_numpy())
            pressure_e = 1.0e16 * e_si * data['te'] * data['ne']  # MPa
            pressure_i = 1.0e16 * e_si * data['ti'] * data['ni']  # MPa
            pressure_i_th = pressure_i.isel(name=thermal_species)  # MPa
            newvars['pressure_e'] = (['n', 'rho'], pressure_e.to_numpy())
            newvars['pressure_i'] = (['n', 'rho', 'name'], pressure_i.to_numpy())
            newvars['pressure_i_th'] = (['n', 'rho'], pressure_i_th.sum('name').to_numpy())
            newvars['ptot_derived'] = (['n', 'rho'], (pressure_e + pressure_i.sum('name')).to_numpy())
            newvars['pth_derived'] = (['n', 'rho'], (pressure_e + pressure_i_th.sum('name')).to_numpy())
            newvars['pfast_derived'] = (['n', 'rho'], (pressure_i.sum('name') - pressure_i_th.sum('name')).to_numpy())
        if 'qmom' not in data:
            newvars['qmom'] = (['n', 'rho'], np.repeat(np.expand_dims(np.zeros_like(data['rho'].to_numpy()), axis=0), len(data['n']), axis=0))
        if 'kappa' in data:
            newvars['kappa95'] = (['n'], interpolate(0.95, data['psi_pol_norm'].to_numpy(), data['kappa'].to_numpy()))
            newvars['kappa995'] = (['n'], interpolate(0.995, data['psi_pol_norm'].to_numpy(), data['kappa'].to_numpy()))
            #newvars['kappa_a'] = (['n'], (data['surfXS'].isel(-1) / np.pi / data['a'] ** 2).to_numpy())
        if 'shape_sin0' not in data:
            newvars['shape_sin0'] = (['n', 'rho'], np.repeat(np.expand_dims(np.zeros_like(data['rho'].to_numpy()), axis=0), len(data['n']), axis=0))
        if 'delta' in data:
            newvars['shape_sin1'] = (['n', 'rho'], np.arcsin(data['delta'].to_numpy()))
            newvars['delta95'] = (['n'], interpolate(0.95, data['psi_pol_norm'].to_numpy(), data['delta'].to_numpy()))
            newvars['delta995'] = (['n'], interpolate(0.995, data['psi_pol_norm'].to_numpy(), data['delta'].to_numpy()))
        if 'zeta' in data:
            newvars['shape_sin2'] = (['n', 'rho'], -1.0 * data['zeta'].to_numpy())
        if 'q' in data and 'psi_pol_norm' in data:
            newvars['q0'] = (['n'], interpolate(0.0, data['psi_pol_norm'].to_numpy(), data['q'].to_numpy()))
            newvars['q95'] = (['n'], interpolate(0.95, data['psi_pol_norm'].to_numpy(), data['q'].to_numpy()))
            newvars['rho_saw'] = (['n'], find(1.0, data['rho_tor'].to_numpy(), data['q'].to_numpy(), last=True))
        if 'rho_s_unit' in data and 'c_s' in data:
            newvars['gammae_gb'] = (['n', 'rho'], (data['ne'] * data['c_s'] * (data['rho_s_unit'] / data['a']) ** 2).to_numpy())
            newvars['gammai_gb'] = (['n', 'rho', 'name'], (data['ni'] * data['c_s'] * (data['rho_s_unit'] / data['a']) ** 2).to_numpy())
            newvars['qe_gb'] = (['n', 'rho'], (1.0e16 * e_si * data['ne'] * data['te'] * data['c_s'] * (data['rho_s_unit'] / data['a']) ** 2).to_numpy())
            newvars['qi_gb'] = (['n', 'rho', 'name'], (1.0e16 * e_si * data['ni'] * data['ti'] * data['c_s'] * (data['rho_s_unit'] / data['a']) ** 2).to_numpy())
            if 'masse' in data and 'c_the' in data:
                newvars['pie_gb'] = (['n', 'rho'], (1.0e19 * data['ne'] * u_si * data['masse'] * data['rmaj'] * data['c_the'] * data['c_s'] * (data['rho_s_unit'] / data['a']) ** 2).to_numpy())
            if 'mass' in data and 'c_thi' in data:
                newvars['pii_gb'] = (['n', 'rho', 'name'], (1.0e19 * data['ni'] * u_si * data['mass'] * data['rmaj'] * data['c_thi'] * data['c_s'] * (data['rho_s_unit'] / data['a']) ** 2).to_numpy())
            newvars['ex_gb'] = (['n', 'rho'], (1.0e16 * e_si * data['ne'] * data['te'] * data['c_s'] * (data['rho_s_unit']) ** 2 / (data['a'] ** 3)).to_numpy())
            newvars['qce_gb'] = (['n', 'rho'], (1.5 * 1.0e16 * e_si * data['ne'] * data['te'] * data['c_s'] * (data['rho_s_unit'] / data['a']) ** 2).to_numpy())
            newvars['qci_gb'] = (['n', 'rho', 'name'], (1.5 * 1.0e16 * e_si * data['ni'] * data['ti'] * data['c_s'] * (data['rho_s_unit'] / data['a']) ** 2).to_numpy())
        if 'surf_miller' in data and 'gradr_miller' in data:
            surf = (data['surf_miller'] / data['gradr_miller']).to_numpy()
            newvars['surf_gacode'] = (['n', 'rho'], np.where(np.isfinite(surf), surf, 0.0))
        if 'q' in data:
            norm = (data['rmin'] / data['q']).to_numpy()
            newvars['s'] = (['n', 'rho'], norm * derivative(data['rmin'].to_numpy(), data['q'].to_numpy()))
            newvars['dqdr'] = (['n', 'rho'], derivative(data['rmin'].to_numpy(), data['q'].to_numpy()))
        if 'bp2_miller' in data and 'b_unit' in data:
            newvars['bp2'] = (['n', 'rho'], (data['bp2_miller'] * data['b_unit'] ** 2).to_numpy())
        if 'bt2_miller' in data and 'b_unit' in data:
            newvars['bt2'] = (['n', 'rho'], (data['bt2_miller'] * data['b_unit'] ** 2).to_numpy())
        if 'te' in data:
            norm = np.expand_dims(data['a'].to_numpy(), axis=-1)
            newvars['alte'] = (['n', 'rho'], norm * derivative(data['rmin'].to_numpy(), -np.log(data['te'].to_numpy())))
            if 'masse' in data:
                v_e_th = (2.0 * (1.0e3 * e_si * data['te']) / (u_si * data['masse'])) ** 0.5  # m/s
                newvars['v_e_th'] = (['n', 'rho'], v_e_th.to_numpy())
            if 'mass_i' in data:
                v_s = (2.0 * (1.0e3 * e_si * data['te']) / (u_si * data['mass_i'])) ** 0.5  # m/s
                newvars['v_s'] = (['n', 'rho'], v_s.to_numpy())
        if 'ti' in data:
            norm = np.expand_dims(np.expand_dims(data['a'].to_numpy(), axis=-1), axis=-1)
            newvars['alti'] = (['n', 'rho', 'name'], np.transpose(norm * derivative(np.repeat(np.expand_dims(data['rmin'].to_numpy(), axis=1), len(data['name']), axis=1), -np.log(np.transpose(data['ti'].to_numpy(), axes=(0, 2, 1)))), axes=(0, 2, 1)))
            if 'te' in data:
                newvars['tite'] = (['n', 'rho', 'name'], (data['ti'] / data['te']).to_numpy())
            if 'mass_i' in data:
                v_i_th = (2.0 * (1.0e3 * e_si * data['ti']) / (u_si * data['mass_i'])) ** 0.5  # m/s
                newvars['v_i_th'] = (['n', 'rho', 'name'], v_i_th.to_numpy())
        if 'ne' in data:
            norm = np.expand_dims(data['a'].to_numpy(), axis=-1)
            newvars['alne'] = (['n', 'rho'], norm * derivative(data['rmin'].to_numpy(), -np.log(data['ne'].to_numpy())))
        if 'ni' in data:
            norm = np.expand_dims(np.expand_dims(data['a'].to_numpy(), axis=-1), axis=-1)
            newvars['alni'] = (['n', 'rho', 'name'], np.transpose(norm * derivative(np.repeat(np.expand_dims(data['rmin'].to_numpy(), axis=1), len(data['name']), axis=1), -np.log(np.transpose(data['ni'].to_numpy(), axes=(0, 2, 1)))), axes=(0, 2, 1)))
            if 'ne' in data:
                newvars['nine'] = (['n', 'rho', 'name'], (data['ni'] / data['ne']).to_numpy())
                zeff = (data['ni'] * data['z'] ** 2 / data['ne']).sum('name')
                newvars['qn_error'] = (['n'], abs(1.0 - (data['ni'] * data['z'] / data['ne']).sum('name')).sum('rho').to_numpy())
                newvars['zeff_derived'] = (['n', 'rho'], zeff.to_numpy())
        if 'ne' in data and 'te' in data and 'rho_s_unit' in data:
            debye_e = (eps_si / e_si) ** 0.5 * (1.0e-16 * data['te'] / (data['ne'] * data['ze'] ** 2)) ** 0.5  # m
            newvars['debye_e'] = (['n', 'rho'], debye_e.to_numpy())
            newvars['debye_e_norm'] = (['n', 'rho'], (debye_e / data['rho_s_unit']).to_numpy())
        if 'ni' in data and 'ti' in data and 'rho_s_unit' in data:
            debye_i = (eps_si / e_si) ** 0.5 * (1.0e-16 * data['ti'] / (data['ni'] * data['z'] ** 2)) ** 0.5  # m
            newvars['debye_i'] = (['n', 'rho', 'name'], debye_i.to_numpy())
            newvars['debye_i_norm'] = (['n', 'rho', 'name'], (debye_i / data['rho_s_unit']).to_numpy())
        if 'omega0' in data:
            norm = np.expand_dims(data['a'].to_numpy(), axis=-1)
            newvars['alw0'] = (['n', 'rho'], norm * derivative(data['rmin'].to_numpy(), -np.log(data['omega0'].to_numpy())))
            newvars['dw0dr'] = (['n', 'rho'], -1.0 * derivative(data['rmin'].to_numpy(), data['omega0'].to_numpy()))
            if 'r_out' in data:
                newvars['mach'] = (['n', 'rho'], (data['omega0'] * data['r_out']).to_numpy() / newvars['v_s'][-1])
        elif 'w0' in data:
            norm = np.expand_dims(data['a'].to_numpy(), axis=-1)
            newvars['alw0'] = (['n', 'rho'], norm * derivative(data['rmin'].to_numpy(), -np.log(data['w0'].to_numpy())))
            newvars['dw0dr'] = (['n', 'rho'], -1.0 * derivative(data['rmin'].to_numpy(), data['w0'].to_numpy()))
            if 'r_out' in data:
                newvars['mach'] = (['n', 'rho'], (data['w0'] * data['r_out']).to_numpy() / newvars['v_s'][-1])

        self.update_input_data_vars(newvars)


    def _compute_secondary_quantities(
        self
    ) -> None:
        def derivative(x, y):
            deriv = np.zeros_like(x)
            if x.shape[-1] > 2:
                x1 = np.concatenate([np.expand_dims(x[..., 0], axis=-1), x[..., :-2], np.expand_dims(x[..., -3], axis=-1)], axis=-1)
                x2 = np.concatenate([np.expand_dims(x[..., 1], axis=-1), x[..., 1:-1], np.expand_dims(x[..., -2], axis=-1)], axis=-1)
                x3 = np.concatenate([np.expand_dims(x[..., 2], axis=-1), x[..., 2:], np.expand_dims(x[..., -1], axis=-1)], axis=-1)
                y1 = np.concatenate([np.expand_dims(y[..., 0], axis=-1), y[..., :-2], np.expand_dims(y[..., -3], axis=-1)], axis=-1)
                y2 = np.concatenate([np.expand_dims(y[..., 1], axis=-1), y[..., 1:-1], np.expand_dims(y[..., -2], axis=-1)], axis=-1)
                y3 = np.concatenate([np.expand_dims(y[..., 2], axis=-1), y[..., 2:], np.expand_dims(y[..., -1], axis=-1)], axis=-1)
                deriv = ((x - x1) + (x - x2)) / (x3 - x1) / (x3 - x2) * y3 + ((x - x1) + (x - x3)) / (x2 - x1) / (x2 - x3) * y2 + ((x - x2) + (x - x3)) / (x1 - x2) / (x1 - x3) * y1
            elif x.shape[-1] > 1:
                deriv[..., 0] = np.diff(y, axis=-1) / np.diff(x, axis=-1)
                deriv[..., 1] = np.diff(y, axis=-1) / np.diff(x, axis=-1)
            return deriv
        data = self.input
        newvars: MutableMapping[str, Any] = {}
        e_si = 1.60218e-19
        u_si = 1.66054e-27
        eps_si = 8.85419e-12
        if 'bp2' in data and 'bt2' in data:
            b = (data['bp2'] + data['bt2']) ** 0.5
            newvars['b'] = (['n', 'rho'], b.to_numpy())
            main_species_mask = (data['name'].isin(['H', 'D', 'T']).to_numpy() & (data['type'].isin(['[therm]'])).to_numpy()).flatten()
            main_species = [i for i in range(len(main_species_mask)) if main_species_mask[i]]
            vperp_tor = xr.zeros_like(b)
            vperp_pol = xr.zeros_like(b)
            vpar_tor = xr.zeros_like(b)
            vpar_pol = xr.zeros_like(b)
            if 'vtor' in data:
                vperp_tor = data['vtor'].isel(name=main_species_mask).mean('name') * data['bp2'] ** 0.5 / b
                vpar_tor = data['vtor'].isel(name=main_species_mask).mean('name') * data['bt2'] ** 0.5 / b
            if 'vpol' in data:
                vperp_pol = data['vpol'].isel(name=main_species_mask).mean('name') * data['bt2'] ** 0.5 / b
                vpar_pol = data['vpol'].isel(name=main_species_mask).mean('name') * data['bp2'] ** 0.5 / b
            newvars['vperp'] = (['n', 'rho'], (vperp_tor - vperp_pol).to_numpy())
            newvars['vpar'] = (['n', 'rho'], (vpar_tor + vpar_pol).to_numpy())
            if 'c_s' in data:
                newvars['machperp'] = (['n', 'rho'], newvars['vperp'][-1] / data['c_s'].to_numpy())
                newvars['machpar'] = (['n', 'rho'], newvars['vpar'][-1] / data['c_s'].to_numpy())
                if 'a' in data:
                    norm = (data['a'] / data['c_s']).to_numpy()
                    newvars['alvperp'] = (['n', 'rho'], norm * derivative(data['rmin'].to_numpy(), -newvars['vperp'][-1]))
                    newvars['alvpar'] = (['n', 'rho'], norm * derivative(data['rmin'].to_numpy(), -newvars['vpar'][-1]))
            if 'rmin' in data and 'q' in data:
                norm = (data['rmin'] / data['q']).to_numpy()
                newvars['gamma_exb'] = (['n', 'rho'], norm * derivative(data['rmin'].to_numpy(), -newvars['vperp'][-1] / np.where(np.isclose(norm, 0.0), 1.0e-4, norm)))
                if 'c_s' in data and 'a' in data:
                    newvars['gamma_exb_norm'] = (['n', 'rho'], newvars['gamma_exb'][-1] * (data['a'] / data['c_s']).to_numpy())
        if 'pressure_e' in data and 'bp2' in data and 'bt2' in data:
            beta_e_p = 1.0e6 * data['pressure_e'] * 2.0 * 4.0e-7 * np.pi / data['bp2']
            beta_e_t = 1.0e6 * data['pressure_e'] * 2.0 * 4.0e-7 * np.pi / data['bt2']
            newvars['beta_e'] = (['n', 'rho'], 1.0 / (beta_e_p ** (-1) + beta_e_t ** (-1)).to_numpy())
        if 'pressure_i_th' in data and 'bp2' in data and 'bt2' in data:
            beta_i_th_p = 1.0e6 * data['pressure_i_th'] * 2.0 * 4.0e-7 * np.pi / data['bp2']
            beta_i_th_t = 1.0e6 * data['pressure_i_th'] * 2.0 * 4.0e-7 * np.pi / data['bt2']
            newvars['beta_i_th'] = (['n', 'rho'], 1.0 / (beta_i_th_p ** (-1) + beta_i_th_t ** (-1)).to_numpy())
        if 'debye_e' in data and 'debye_i' in data:
            newvars['debye'] = (['n', 'rho'], ((data['debye_e'] ** (-2) + (data['debye_i'] ** (-2)).sum('name')) ** (-0.5)).to_numpy())
            f_nu = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
            inv_b90_ee = (4.0 * np.pi * eps_si / e_si) * 1.0e3 * data['te'] / abs(data['ze'] * data['ze'])
            f_ee = 1.0e19 * data['ne'] * (e_si * 1.0e3 * data['te'] / (u_si * data['masse'])) ** 0.5 / (inv_b90_ee ** 2) * np.log(inv_b90_ee * data['debye_e'])
            newvars['nu_ee'] = (['n', 'rho'], (f_ee * f_nu).to_numpy())
            inv_b90_ei = (4.0 * np.pi * eps_si / e_si) * 1.0e3 * data['te'] / abs(data['ze'] * data['z'])
            f_ei = 1.0e19 * data['ni'] * (e_si * 1.0e3 * data['te'] / (u_si * data['masse'])) ** 0.5 / (inv_b90_ei ** 2) * np.log(inv_b90_ei * data['debye_e'])
            newvars['nu_ei'] = (['n', 'rho'], (f_ei * f_nu).sum('name').to_numpy())
            inv_b90_ii = (4.0 * np.pi * eps_si / e_si) * 1.0e3 * data['ti'] / abs(data['z'] * data['z'])
            f_ii = 1.0e19 * data['ni'] * (e_si * 1.0e3 * data['ti'] / (u_si * data['mass'])) ** 0.5 / (inv_b90_ii ** 2) * np.log(inv_b90_ii * data['debye_i'])
            newvars['nu_ii'] = (['n', 'rho', 'name'], (f_ii * f_nu).to_numpy())
            if 'c_s' in data and 'a' in data:
                newvars['nu_ee_norm'] = (['n', 'rho'], (f_ee * f_nu * data['a'] / data['c_s']).to_numpy())
                newvars['nu_ei_norm'] = (['n', 'rho'], (f_ei * f_nu * data['a'] / data['c_s']).sum('name').to_numpy())
                newvars['nu_ii_norm'] = (['n', 'rho', 'name'], (f_ii * f_nu * data['a'] / data['c_s']).to_numpy())
        self.update_input_data_vars(newvars)


    def _compute_integrated_quantities(self):

        def interpolate(v, x, y):
            vm = np.array([v]) if isinstance(v, (float, int)) else copy.deepcopy(v)
            xm = x.reshape(-1, x.shape[-1])
            ym = y.reshape(-1, y.shape[-1])
            if vm.shape[0] != xm.shape[0]:
                vm = np.repeat(np.expand_dims(vm, axis=0), xm.shape[0], axis=0)
            interp = np.zeros_like(vm)
            for i in range(xm.shape[0]):
                interp[i] = np.interp(vm[i], xm[i], ym[i])
            interp = interp.reshape(*y.shape[:-1])
            return interp

        data = self.input
        newvars: MutableMapping[str, Any] = {}

        if 'rmin' in data:
            line = cumulative_simpson(np.ones_like(data['rmin'].to_numpy()), x=data['rmin'].to_numpy(), initial=0.0)
            n_e_line = cumulative_simpson((data['ne']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            n_i_line = cumulative_simpson(
                np.transpose((data['ni']).to_numpy(), axes=(0, 2, 1)),
                x=np.repeat(np.expand_dims(data['rmin'].to_numpy(), axis=1), len(data['name']), axis=1),
                initial=0.0
            )
            newvars['n_e_line'] = (['n'], n_e_line[:, -1] / line[:, -1])
            newvars['n_i_line'] = (['n', 'name'], n_i_line[:, :, -1] / np.expand_dims(line, axis=1)[:, :, -1])
            newvars['f_i_line'] = (['n', 'name'], n_i_line[:, :, -1] / np.expand_dims(n_e_line, axis=1)[:, :, -1])

        if 'rmin' in data and 'volp_miller' in data:

            pe = cumulative_simpson((data['qe'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            pi = cumulative_simpson((data['qi'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            se = cumulative_simpson((data['ge'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            pce = 1.5 * 1.0e-3 * 16.0218 * data['te'].to_numpy() * se  # MW
            mt = cumulative_simpson((data['qmom'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            newvars['pe'] = (['n', 'rho'], pe)
            newvars['pi'] = (['n', 'rho'], pi)
            newvars['se'] = (['n', 'rho'], se)
            newvars['pce'] = (['n', 'rho'], pce)
            newvars['mt'] = (['n', 'rho'], mt)
            newvars['ptotal'] = (['n', 'rho'], pe + pi)
            newvars['s_in'] = (['n'], se[:, -1])

            if 'surf_miller' in data:
                surf = data['surf_miller'].to_numpy()
                inv_surf = 1.0 / np.where(np.isclose(surf, 0.0), 1.0, surf)
                newvars['qe_surf'] = (['n', 'rho'], np.where(np.isclose(surf, 0.0), 0.0, pe * inv_surf))
                newvars['qi_surf'] = (['n', 'rho'], np.where(np.isclose(surf, 0.0), 0.0, pi * inv_surf))
                newvars['ge_surf'] = (['n', 'rho'], np.where(np.isclose(surf, 0.0), 0.0, se * inv_surf))
                newvars['qce_surf'] = (['n', 'rho'], np.where(np.isclose(surf, 0.0), 0.0, pce * inv_surf))
                newvars['tt_surf'] = (['n', 'rho'], np.where(np.isclose(surf, 0.0), 0.0, mt * inv_surf))
                #newvars["qratio_surf"] = qi / np.where(qe == 0.0, 1e-10, qe)  # to avoid division by zero

            pe_aux = cumulative_simpson((data['qe_aux_ion'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            pi_aux = cumulative_simpson((data['qi_aux_ion'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            palpha = cumulative_simpson((data['qalpha'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            prad = cumulative_simpson((data['qrad'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            newvars['pe_aux'] = (['n', 'rho'], cumulative_simpson((data['qe_aux'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0))
            newvars['pi_aux'] = (['n', 'rho'], cumulative_simpson((data['qi_aux'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0))
            newvars['pe_aux_ion'] = (['n', 'rho'], pe_aux)
            newvars['pi_aux_ion'] = (['n', 'rho'], pi_aux)
            newvars['pohm'] = (['n', 'rho'], cumulative_simpson((data['qohme'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0))
            newvars['prf'] = (['n', 'rho'], cumulative_simpson((data['qrf'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0))
            newvars['pbeam'] = (['n', 'rho'], cumulative_simpson((data['qbeam'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0))
            newvars['pion'] = (['n', 'rho'], cumulative_simpson((data['qion'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0))
            newvars['palpha'] = (['n', 'rho'], palpha)
            newvars['prad'] = (['n', 'rho'], prad)
            newvars['ptr'] = (['n', 'rho'], pe_aux + pi_aux + palpha + prad)

            if 'qei' in data:
                newvars['pei'] = (['n', 'rho'], cumulative_simpson((data['qei'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0))
            newvars['p_rad'] = (['n'], prad[:, -1])
            newvars['p_fus'] = (['n'], 5.0 * palpha[:, -1])
            newvars['p_in'] = (['n'], (pe_aux + pi_aux)[:, -1])
            newvars['q_gain'] = (['n'], newvars['p_fus'][-1] / newvars['p_in'][-1])
            newvars['p_heat'] = (['n'], (pe_aux + pi_aux + palpha)[:, -1])
            newvars['p_sol'] = (['n'], (pe_aux + pi_aux + palpha - prad)[:, -1])

            vol = cumulative_simpson(data['volp_miller'].to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            n_e_vol = cumulative_simpson((data['ne'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            t_e_vol = cumulative_simpson((data['te'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            p_e_vol = cumulative_simpson((data['pressure_e'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            w_e_vol = cumulative_simpson((1.5 * data['pressure_e'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            n_i_vol = cumulative_simpson(
                np.transpose((data['ni'] * data['volp_miller']).to_numpy(), axes=(0, 2, 1)),
                x=np.repeat(np.expand_dims(data['rmin'].to_numpy(), axis=1), len(data['name']), axis=1),
                initial=0.0
            )
            t_i_vol = cumulative_simpson(
                np.transpose((data['ti'] * data['volp_miller']).to_numpy(), axes=(0, 2, 1)),
                x=np.repeat(np.expand_dims(data['rmin'].to_numpy(), axis=1), len(data['name']), axis=1),
                initial=0.0
            )
            p_i_vol = cumulative_simpson(
                np.transpose((data['pressure_i'] * data['volp_miller']).to_numpy(), axes=(0, 2, 1)),
                x=np.repeat(np.expand_dims(data['rmin'].to_numpy(), axis=1), len(data['name']), axis=1),
                initial=0.0
            )
            w_i_vol = cumulative_simpson(
                np.transpose((1.5 * data['pressure_i'] * data['volp_miller']).to_numpy(), axes=(0, 2, 1)),
                x=np.repeat(np.expand_dims(data['rmin'].to_numpy(), axis=1), len(data['name']), axis=1),
                initial=0.0
            )
            #n_i_th_vol = cumulative_simpson((data['ni'].isel(name=thermal_species).sum('name') * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            p_i_th_vol = cumulative_simpson((data['pressure_i_th'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            w_i_th_vol = cumulative_simpson((1.5 * data['pressure_i_th'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)
            newvars['vol'] = (['n'], vol[:, -1])
            #newvars['n_e'] = (['n'], n_e_vol[:, -1])
            #newvars['n_i'] = (['n', 'name'], n_i_vol[:, :, -1])
            #newvars['n_i_th'] = (['n'], n_i_th_vol[:, -1])
            #newvars['n_th'] = (['n'], (n_e_vol + n_i_th_vol)[:, -1])
            newvars['w_e'] = (['n'], w_e_vol[:, -1])
            newvars['w_i'] = (['n', 'name'], w_i_vol[:, :, -1])
            newvars['w_i_th'] = (['n'], w_i_th_vol[:, -1])
            newvars['w_th'] = (['n'], (w_e_vol + w_i_th_vol)[:, -1])

            inv_p = 1.0 / np.where(np.isclose(pe_aux + pi_aux + palpha, 0.0), 1.0, pe_aux + pi_aux + palpha)[:, -1]
            inv_s = 1.0 / np.where(np.isclose(se, 0.0), 1.0, se)[:, -1]
            newvars['taue'] = (['n'], np.where(np.isclose((pe_aux + pi_aux + palpha)[:, -1], 0.0), np.inf, (w_e_vol + w_i_th_vol)[:, -1] * inv_p))
            newvars['taup'] = (['n'], np.where(np.isclose(se[:, -1], 0.0), np.inf, n_e_vol[:, -1] * inv_s))
            newvars['tau'] = newvars['taup'][-1] / newvars['taue'][-1]

            newvars['n_e_vol'] = (['n'], n_e_vol[:, -1] / vol[:, -1])
            newvars['n_i_vol'] = (['n', 'name'], n_i_vol[:, :, -1] / np.expand_dims(vol, axis=1)[:, :, -1])
            newvars['t_e_vol'] = (['n'], t_e_vol[:, -1] / vol[:, -1])
            newvars['t_i_vol'] = (['n', 'name'], t_i_vol[:, :, -1] / np.expand_dims(vol, axis=1)[:, :, -1])
            newvars['f_i_vol'] = (['n', 'name'], n_i_vol[:, :, -1] / np.expand_dims(n_e_vol, axis=1)[:, :, -1])

            newvars['nu_ne'] = (['n'], interpolate(0.0, np.repeat(np.expand_dims(data['rho'].to_numpy(), axis=0), len(data['n']), axis=0), data['ne'].to_numpy()) / newvars['n_e_vol'][-1])
            newvars['nu_ne_0.2'] = (['n'], interpolate(0.2, np.repeat(np.expand_dims(data['rho'].to_numpy(), axis=0), len(data['n']), axis=0), data['ne'].to_numpy())  / newvars['n_e_vol'][-1])
            newvars['nu_te'] = (['n'], interpolate(0.0, np.repeat(np.expand_dims(data['rho'].to_numpy(), axis=0), len(data['n']), axis=0), data['te'].to_numpy()) / newvars['t_e_vol'][-1])
            newvars['nu_te_0.2'] = (['n'], interpolate(0.2, np.repeat(np.expand_dims(data['rho'].to_numpy(), axis=0), len(data['n']), axis=0), data['te'].to_numpy()) / newvars['t_e_vol'][-1])
            newvars['nu_ni'] = (['n', 'name'], interpolate(0.0, np.repeat(np.repeat(np.expand_dims(np.expand_dims(data['rho'].to_numpy(), axis=0), axis=0), len(data['name']), axis=0), len(data['n']), axis=0), np.transpose(data['ni'].to_numpy(), axes=(0, 2, 1))) / newvars['n_i_vol'][-1])
            newvars['nu_ni_0.2'] = (['n', 'name'], interpolate(0.2, np.repeat(np.repeat(np.expand_dims(np.expand_dims(data['rho'].to_numpy(), axis=0), axis=0), len(data['name']), axis=0), len(data['n']), axis=0), np.transpose(data['ni'].to_numpy(), axes=(0, 2, 1))) / newvars['n_i_vol'][-1])
            newvars['nu_ti'] = (['n', 'name'], interpolate(0.0, np.repeat(np.repeat(np.expand_dims(np.expand_dims(data['rho'].to_numpy(), axis=0), axis=0), len(data['name']), axis=0), len(data['n']), axis=0), np.transpose(data['ti'].to_numpy(), axes=(0, 2, 1))) / newvars['t_i_vol'][-1])
            newvars['nu_ti_0.2'] = (['n', 'name'], interpolate(0.2, np.repeat(np.repeat(np.expand_dims(np.expand_dims(data['rho'].to_numpy(), axis=0), axis=0), len(data['name']), axis=0), len(data['n']), axis=0), np.transpose(data['ti'].to_numpy(), axes=(0, 2, 1))) / newvars['t_i_vol'][-1])

            newvars['pressure_e_derived_vol'] = (['n'], p_e_vol[:, -1] / vol[:, -1])
            newvars['pressure_tot_derived_vol'] = (['n'], (p_e_vol + np.sum(p_i_vol, axis=1))[:, -1] / vol[:, -1])
            newvars['pressure_th_derived_vol'] = (['n'], (p_e_vol + np.sum(p_i_th_vol, axis=1))[:, -1] / vol[:, -1])
            newvars['pressure_fast_derived_vol'] = (['n'], np.sum(p_i_vol - p_i_th_vol, axis=1)[:, -1] / vol[:, -1])
            newvars['pressure_fast_fraction'] = (['n'], newvars['pressure_fast_derived_vol'][-1] / newvars['pressure_tot_derived_vol'][-1])

            if 'nine' in data:
                newvars['nine_vol'] = (['n', 'name'], cumulative_simpson(np.transpose((data['nine'] * data['volp_miller']).to_numpy(), axes=(0, 2, 1)), x=np.repeat(np.expand_dims(data['rmin'].to_numpy(), axis=1), len(data['name']), axis=1), initial=0.0)[:, :, -1] / np.expand_dims(vol, axis=1)[:, :, -1])
            if 'tite' in data:
                newvars['tite_vol'] = (['n', 'name'], cumulative_simpson(np.transpose((data['tite'] * data['volp_miller']).to_numpy(), axes=(0, 2, 1)), x=np.repeat(np.expand_dims(data['rmin'].to_numpy(), axis=1), len(data['name']), axis=1), initial=0.0)[:, :, -1] / np.expand_dims(vol, axis=1)[:, :, -1])
            if 'zeff_derived' in data:
                newvars['zeff_derived_vol'] = (['n'], cumulative_simpson((data['zeff_derived'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)[:, -1] / vol[:, -1])
                newvars['nueff'] = (['n'], newvars['zeff_derived_vol'][-1] * (data['rcentr'] * 0.1 * data['ne'] * data['te'] ** (-2)).isel(rho=-1).to_numpy())
            if 'mach' in data:
                newvars['mach_vol'] = (['n'], cumulative_simpson((data['mach'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)[:, -1] / vol[:, -1])

            newvars['beta_zero'] = (['n'], 1.0e6 * newvars['pressure_tot_derived_vol'][-1] * 2.0 * 4.0e-7 * np.pi / (data['b_zero'] ** 2).to_numpy())
            newvars['beta_n_eng'] = (['n'], newvars['beta_zero'][-1] * (100.0 * data['a'] * data['b_zero'] / data['current']).to_numpy())  # pc

            newvars['bp2_vol'] = (['n'], cumulative_simpson((data['bp2'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)[:, -1] / vol[:, -1])
            newvars['bt2_vol'] = (['n'], cumulative_simpson((data['bt2'] * data['volp_miller']).to_numpy(), x=data['rmin'].to_numpy(), initial=0.0)[:, -1] / vol[:, -1])
            newvars['beta_p'] = (['n'], 1.0e6 * newvars['pressure_tot_derived_vol'][-1] * 2.0 * 4.0e-7 * np.pi / newvars['bp2_vol'][-1])
            newvars['beta_t'] = (['n'], 1.0e6 * newvars['pressure_tot_derived_vol'][-1] * 2.0 * 4.0e-7 * np.pi / newvars['bt2_vol'][-1])
            newvars['beta'] = (['n'], 1.0 / (newvars['beta_p'][-1] ** (-1) + newvars['beta_t'][-1] ** (-1)))
            newvars['beta_n'] = (['n'], newvars['beta'][-1] * (100.0 * data['a'] * data['b_zero'] / data['current']).to_numpy())  # pc

        self.update_input_data_vars(newvars)


    def _compute_scalings(
        self
    ) -> None:

        data = self.input
        newvars: MutableMapping[str, Any] = {}
        if 'current' in data and 'bcentr' in data:

            newvars['n_gw'] = (['n'], (data['current'] / (np.pi * data['rmin'] ** 2)).to_numpy()[:, -1])
            newvars['f_gw'] = (['n'], 0.1 * data['n_e_vol'].to_numpy() / newvars['n_gw'][-1])
            newvars['f_gw_local'] = (['n', 'rho'], 0.1 * data['ne'].to_numpy() / np.expand_dims(newvars['n_gw'][-1], axis=-1))

            newvars['tau98'] = (['n'], (
                0.0562
                * data['current'] ** (0.93)
                * data['rcentr'] ** (1.97)
                * data['kappa'].isel(rho=-1) ** (0.78)
                * data['eps'] ** (0.58)
                * data['bcentr'] ** (0.15)
                * data['n_e_line'] ** (0.41)
                * data['mass_i'] ** (0.19)
                * data['ptotal'].isel(rho=-1) ** (-0.69)
            ).to_numpy())
            newvars['tau89'] = (['n'], (
                0.048
                * data['current'] ** (0.85)
                * data['rcentr'] ** (1.50)
                * data['kappa'].isel(rho=-1) ** (0.50)
                * data['eps'] ** (0.30)
                * data['bcentr'] ** (0.20)
                * (data['n_e_line'] * 0.1) ** (0.10)
                * data['mass_i'] ** (0.50)
                * data['ptotal'].isel(rho=-1) ** (-0.50)
            ).to_numpy())
            newvars['tau97l'] = (['n'], (
                0.023
                * data['current'] ** (0.96)
                * data['rcentr'] ** (1.83)
                * data['kappa'].isel(rho=-1) ** (0.64)
                * data['eps'] ** (0.06)
                * data['bcentr'] ** (0.03)
                * data['n_e_line'] ** (0.40)
                * data['mass_i'] ** (0.20)
                * data['ptotal'].isel(rho=-1) ** (-0.73)
            ).to_numpy())

            lh_nmin = (
                0.07
                * data['current'] ** (0.34)
                * data['bcentr'] ** (0.62)
                * data['a'] ** (-0.95)
                * data['eps'] ** (0.4)
            ).to_numpy()
            nminfactor = np.where(data['n_e_vol'].to_numpy() > lh_nmin, (data['n_e_vol'].to_numpy() / lh_nmin) ** 2, 1.0)
            newvars['p_lh_martin'] = (['n'], (
                2.15
                * data['n_e_vol'] ** (0.782)
                * data['bcentr'] ** (0.772)
                * data['a'] ** (0.975)
                * data['rcentr'] ** (0.999)
                * (2.0 / data['mass_i']) ** (1.11)
            ).to_numpy() * nminfactor)
            newvars['lhratio'] = (['n'], data['p_sol'].to_numpy() / newvars['p_lh_martin'][-1])

            uckan_shaping = 1.0 + data['kappa95'] ** 2 * (1.0 + 2.0 * data['delta95'] ** 2 - 1.2 * data['delta95'] ** 3)
            iter_shaping = uckan_shaping * (1.17 - 0.65 * data['eps']) / (1 - data['eps'] ** 2) ** 2

            newvars['qstar_uckan'] = (['n'], (
                2.5
                * data['rmaj'].isel(rho=-1)
                * data['eps'] ** 2
                * data['bcentr']
                / data['current']
                * uckan_shaping
            ).to_numpy())
            newvars['qstar_iter'] = (['n'], (
                2.5
                * data['rmaj'].isel(rho=-1)
                * data['eps'] ** 2
                * data['bcentr']
                / data['current']
                * iter_shaping
            ).to_numpy())

            newvars['lq_brunner'] = (['n'], (
                0.91
                * (data['pressure_tot_derived_vol'] / 0.101325) ** (-0.48)
            ).to_numpy())
            newvars['lq_eich14'] = (['n'], (
                0.63
                * data['bp_out'].isel(rho=-1) ** (-1.19)
            ).to_numpy())
            newvars['lq_eich15'] = (['n'], (
                1.35
                * (data['p_sol'] * 1.0e6) ** (-0.02)
                * data['bp_out'].isel(rho=-1) ** (-0.92)
                * data['rcentr'] ** (0.04)
                * data['eps'] ** (0.42)
            ).to_numpy())

            #bp = data['eps'] * data['bcentr'] / data['q95'] #TODO: VERY ROUGH APPROXIMATION!!!!

            #newvars['ne_upstream'] = (['n'], 0.6 * data['n_e_vol'].to_numpy())
            #te_guess = 1.0
            #p_elfrac = 0.5
            #k0e = 1.0e6 * (3.2 * 3.44e5 * 1.60218e-19**2) / 9.1094e-31
            #Aqpar = 4.0 * np.pi * (1.0e-3 * data['eps'] * data['rcentr'] / data['q95']).to_numpy() * newvars['lq_brunner'][-1]
            #lpsol = (p_elfrac * 1.0e6 * data['p_sol'] * np.pi * data['rcentr'] * data['q95']).to_numpy()
            #lnC = 24 - np.log((newvars['ne_upstream'][-1] * 1.0e13) ** 0.5 / te_guess)  # low temperature approximation
            #te_up = (3.5 * (lpsol / Aqpar) * lnC / k0e) ** (2.0 / 7.0)
            #newvars['te_upstream'] = (['n'], te_up)

        self.update_input_data_vars(newvars)


    def compute_derived_quantities(
        self,
    ) -> None:
        self._compute_derived_coordinates()
        self._compute_derived_reference_quantities()
        self._compute_derived_geometry()
        self._compute_average_mass()
        self._compute_source_terms()
        self._compute_extended_local_inputs()
        self._compute_secondary_quantities()
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
                'z': 'charge_i',
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
            coords = {}
            data_vars = {}
            attrs = {}
            if 'n' in data and 'rho' in data:
                coords['time'] = data['time'].to_numpy() if 'time' in data else np.arange(len(data['n']))
                coords['radius'] = data['rho'].to_numpy()
                if 'name' in data:
                    coords['ion'] = data['name'].to_numpy()
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
                flux = np.repeat(np.expand_dims(np.zeros((len(coords['time']), len(coords['radius']))), axis=-1), len(coords['direction']), axis=-1)
                if 'torflux' in data:
                    flux[..., 0] = data['torflux'].to_numpy()
                elif 'q' in data and 'polflux' in data:
                    flux[..., 0] = cumulative_simpson(data['q'].to_numpy(), x=data['polflux'].to_numpy(), initial=0.0)
                if 'polflux' in data:
                    flux[..., 1] = data['polflux'].to_numpy()
                data_vars['magnetic_flux'] = (['time', 'radius', 'direction'], flux)
                if 'name' in data:
                    velocity = np.repeat(np.expand_dims(np.zeros((len(coords['time']), len(coords['radius']), len(coords['ion']))), axis=-1), len(coords['direction']), axis=-1)
                    if 'vtor' in data:
                        velocity[..., 0] = data['vtor'].to_numpy()
                    if 'vpol' in data:
                        velocity[..., 1] = data['vpol'].to_numpy()
                    data_vars['velocity_i'] = (['time', 'radius', 'ion', 'direction'], velocity)
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
                    coords['angle'] = theta
                    data_vars['contour'] = (['time', 'radius', 'angle_geometric'], l_contour)
            newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj
