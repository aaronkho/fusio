import logging
from pathlib import Path
from .io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import xarray as xr

import datetime
from .io import io
from ..utils.plasma_tools import define_ion_species
from ..utils.eqdsk_tools import (
    define_cocos_converter,
    read_eqdsk,
    trace_flux_surfaces,
    calculate_mxh_coefficients,
)

logger = logging.getLogger('fusio')


class gacode_io(io):

    basevars: Final[Sequence[str]] = [
        'nexp',
        'nion',
        'shot',
        'name',
        'type',
        'masse',
        'mass',
        'ze',
        'z',
        'torfluxa',
        'rcentr',
        'bcentr',
        'current',
        'time',
        'polflux',
        'q',
        'rmin',
        'rmaj',
        'zmag',
        'kappa',
        'delta',
        'zeta',
        'shape_cos',
        'shape_sin',
        'ni',
        'ti',
        'ne',
        'te',
        'z_eff',
        'qohme',
        'qbeame',
        'qbeami',
        'qrfe',
        'qrfi',
        'qsync',
        'qbrem',
        'qline',
        'qfuse',
        'qfusi',
        'qei',
        'qione',
        'qioni',
        'qcxi',
        'johm',
        'jbs',
        'jbstor',
        'jrf',
        'jnb',
        'vtor',
        'vpol',
        'omega0',
        'ptot',
        'qpar_beam',
        'qpar_wall',
        'qmom',
    ]
    titles_singleInt: Final[Sequence[str]] = [
        'nexp',
        'nion',
        'shot',
    ]
    titles_singleStr: Final[Sequence[str]] = [
        'name',
        'type',
    ]
    titles_singleFloat: Final[Sequence[str]] = [
        'masse',
        'mass',
        'ze',
        'z',
        'torfluxa',
        'rcentr',
        'bcentr',
        'current',
        'time',
    ]
    units: Final[Mapping[str, str]] = {
        'torfluxa': 'Wb/radian',
        'rcentr': 'm',
        'bcentr': 'T',
        'current': 'MA',
        'polflux': 'Wb/radian',
        'rmin': 'm',
        'rmaj': 'm',
        'zmag': 'm',
        'ni': '10^19/m^3',
        'ti': 'keV',
        'ne': '10^19/m^3',
        'te': 'keV',
        'qohme': 'MW/m^3',
        'qbeame': 'MW/m^3',
        'qbeami': 'MW/m^3',
        'qrfe': 'MW/m^3',
        'qrfi': 'MW/m^3',
        'qsync': 'MW/m^3',
        'qbrem': 'MW/m^3',
        'qline': 'MW/m^3',
        'qfuse': 'MW/m^3',
        'qfusi': 'MW/m^3',
        'qei': 'MW/m^3',
        'qione': 'MW/m^3',
        'qioni': 'MW/m^3',
        'qcxi': 'MW/m^3',
        'johm': 'MA/m^2',
        'jbs': 'MA/m^2',
        'jbstor': 'MA/m^2',
        'jrf': 'MA/m^2',
        'jnb': 'MA/m^2',
        'vtor': 'm/s',
        'vpol': 'm/s',
        'omega0': 'rad/s',
        'ptot': 'Pa',
        'qpar_beam': '1/m^3/s',
        'qpar_wall': '1/m^3/s',
        'qmom': 'N/m^2',
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


    def make_file_header(
        self,
    ) -> str:
        now = datetime.datetime.now()
        gacode_header = [
            f'#  *original : {now.strftime('%a %b %-d %H:%M:%S %Z %Y')}',
            f'# *statefile : null',
            f'#     *gfile : null',
            f'#   *cerfile : null',
            f'#      *vgen : null',
            f'#     *tgyro : null',
            f'#',
        ]
        return '\n'.join(gacode_header)


    def correct_magnetic_fluxes(
        self,
        exponent: int = -1,
        side: str = 'input',
    ) -> None:
        if side == 'input':
            if 'polflux' in self.input:
                self._tree['input']['polflux'] *= np.power(2.0 * np.pi, exponent)
            if 'torfluxa' in self.input:
                self._tree['input']['torfluxa'] *= np.power(2.0 * np.pi, exponent)
        else:
            if 'polflux' in self.output:
                self._tree['output']['polflux'] *= np.power(2.0 * np.pi, exponent)
            if 'torfluxa' in self.output:
                self._tree['output']['torfluxa'] *= np.power(2.0 * np.pi, exponent)


    def add_geometry_from_eqdsk(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False,
    ) -> None:
        data = self.input.to_dataset() if side == 'input' else self.output.to_dataset()
        if isinstance(path, (str, Path)) and 'polflux' in data:
            eqdsk_data = read_eqdsk(path)
            mxh_data = self._calculate_geometry_from_eqdsk(eqdsk_data, data.isel(n=0)['polflux'].to_numpy().flatten())
            newvars = {}
            if overwrite or np.abs(data.get('rmaj', np.array([0.0]))).sum() == 0.0:
                newvars['rmaj'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['rmaj']), axis=0))
            if overwrite or np.abs(data.get('rmin', np.array([0.0]))).sum() == 0.0:
                newvars['rmin'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['rmin']), axis=0))
            if overwrite or np.abs(data.get('zmag', np.array([0.0]))).sum() == 0.0:
                newvars['zmag'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['zmag']), axis=0))
            if overwrite or np.abs(data.get('kappa', np.array([0.0]))).sum() == 0.0:
                newvars['kappa'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['kappa']), axis=0))
            if overwrite or np.abs(data.get('delta', np.array([0.0]))).sum() == 0.0:
                newvars['delta'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['delta']), axis=0))
            if overwrite or np.abs(data.get('zeta', np.array([0.0]))).sum() == 0.0:
                newvars['zeta'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['zeta']), axis=0))
            if overwrite or np.abs(data.get('shape_sin3', np.array([0.0]))).sum() == 0.0:
                newvars['shape_sin3'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin3']), axis=0))
            if overwrite or np.abs(data.get('shape_sin4', np.array([0.0]))).sum() == 0.0:
                newvars['shape_sin4'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin4']), axis=0))
            if overwrite or np.abs(data.get('shape_sin5', np.array([0.0]))).sum() == 0.0:
                newvars['shape_sin5'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin5']), axis=0))
            if overwrite or np.abs(data.get('shape_sin6', np.array([0.0]))).sum() == 0.0:
                newvars['shape_sin6'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin6']), axis=0))
            if overwrite or np.abs(data.get('shape_cos0', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos0'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos0']), axis=0))
            if overwrite or np.abs(data.get('shape_cos1', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos1'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos1']), axis=0))
            if overwrite or np.abs(data.get('shape_cos2', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos2'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos2']), axis=0))
            if overwrite or np.abs(data.get('shape_cos3', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos3'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos3']), axis=0))
            if overwrite or np.abs(data.get('shape_cos4', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos4'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos4']), axis=0))
            if overwrite or np.abs(data.get('shape_cos5', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos5'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos5']), axis=0))
            if overwrite or np.abs(data.get('shape_cos6', np.array([0.0]))).sum() == 0.0:
                newvars['shape_cos6'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos6']), axis=0))
            if newvars:
                if side == 'input':
                    self.update_input_data_vars(newvars)
                else:
                    self.update_output_data_vars(newvars)


    # This could probably be generalized and moved to eqdsk_tools
    def _calculate_geometry_from_eqdsk(
        self,
        eqdsk_data: MutableMapping[str, Any],
        psivec: ArrayLike,
    ) -> MutableMapping[str, list[int | float]]:
        mxh_data: dict[str, list[int| float]] = {
            'rmaj': [],
            'rmin': [],
            'zmag': [],
            'kappa': [],
            'delta': [],
            'zeta': [],
            'sin3': [],
            'sin4': [],
            'sin5': [],
            'sin6': [],
            'cos0': [],
            'cos1': [],
            'cos2': [],
            'cos3': [],
            'cos4': [],
            'cos5': [],
            'cos6': [],
        }
        if isinstance(eqdsk_data, dict) and isinstance(psivec, np.ndarray):
            faxis = False
            rvec = np.linspace(eqdsk_data['rleft'], eqdsk_data['rleft'] + eqdsk_data['rdim'], eqdsk_data['nr'])
            zvec = np.linspace(eqdsk_data['zmid'] - 0.5 * eqdsk_data['zdim'], eqdsk_data['zmid'] + 0.5 * eqdsk_data['zdim'], eqdsk_data['nz'])
            if np.isclose(eqdsk_data['psi'][0, 0], eqdsk_data['psi'][-1, -1]) and np.isclose(eqdsk_data['psi'][0, -1], eqdsk_data['psi'][-1, 0]):
                if eqdsk_data['simagx'] > eqdsk_data['sibdry'] and psivec[-1] < eqdsk_data['psi'][0, 0]:
                    psivec[-1] = eqdsk_data['psi'][0, 0] + 1.0e-6
                elif eqdsk_data['simagx'] < eqdsk_data['sibdry'] and psivec[-1] > eqdsk_data['psi'][0, 0]:
                    psivec[-1] = eqdsk_data['psi'][0, 0] - 1.0e-6
            if eqdsk_data['simagx'] > eqdsk_data['sibdry'] and psivec[0] >= eqdsk_data['simagx']:
                faxis = True
                psivec[0] = eqdsk_data['simagx'] - 1.0e-6
            elif eqdsk_data['simagx'] < eqdsk_data['sibdry'] and psivec[0] <= eqdsk_data['simagx']:
                faxis = True
                psivec[0] = eqdsk_data['simagx'] + 1.0e-6
            rmesh, zmesh = np.meshgrid(rvec, zvec)
            axis = [eqdsk_data['rmagx'], eqdsk_data['zmagx']]
            fs = trace_flux_surfaces(rmesh, zmesh, eqdsk_data['psi'], psivec, axis=axis)
            mxh = {psi: calculate_mxh_coefficients(c[:, 0], c[:, 1], n=6) for psi, c in fs.items()}
            for psi in psivec:
                mxh_data['rmaj'].append(mxh[psi][2][0] if psi in mxh else np.nan)
                mxh_data['rmin'].append(mxh[psi][2][1] if psi in mxh else np.nan)
                mxh_data['zmag'].append(mxh[psi][2][2] if psi in mxh else np.nan)
                mxh_data['kappa'].append(mxh[psi][2][3] if psi in mxh else np.nan)
                mxh_data['delta'].append(np.sin(mxh[psi][1][1]) if psi in mxh else np.nan)
                mxh_data['zeta'].append(-mxh[psi][1][2] if psi in mxh else np.nan)
                mxh_data['sin3'].append(mxh[psi][1][3] if psi in mxh else np.nan)
                mxh_data['sin4'].append(mxh[psi][1][4] if psi in mxh else np.nan)
                mxh_data['sin5'].append(mxh[psi][1][5] if psi in mxh else np.nan)
                mxh_data['sin6'].append(mxh[psi][1][6] if psi in mxh else np.nan)
                mxh_data['cos0'].append(mxh[psi][0][0] if psi in mxh else np.nan)
                mxh_data['cos1'].append(mxh[psi][0][1] if psi in mxh else np.nan)
                mxh_data['cos2'].append(mxh[psi][0][2] if psi in mxh else np.nan)
                mxh_data['cos3'].append(mxh[psi][0][3] if psi in mxh else np.nan)
                mxh_data['cos4'].append(mxh[psi][0][4] if psi in mxh else np.nan)
                mxh_data['cos5'].append(mxh[psi][0][5] if psi in mxh else np.nan)
                mxh_data['cos6'].append(mxh[psi][0][6] if psi in mxh else np.nan)
            for key in mxh_data:
                arr = np.array(mxh_data[key])
                mask = np.isfinite(arr)
                if not np.all(mask) and np.any(mask):
                    inner_idx = np.argmax(mask)
                    mask |= (np.arange(len(mask)) >= inner_idx)
                    if inner_idx > 0:
                        if key == 'rmaj':
                            arr[~mask] = eqdsk_data['rmagx'] + (psivec[~mask] - psivec[0]) * (arr[inner_idx] - eqdsk_data['rmagx']) / (psivec[inner_idx] - psivec[0])
                        elif key == 'zmag':
                            arr[~mask] = eqdsk_data['zmagx'] + (psivec[~mask] - psivec[0]) * (arr[inner_idx] - eqdsk_data['zmagx']) / (psivec[inner_idx] - psivec[0])
                        else:
                            arr[~mask] = arr[inner_idx] + (psivec[~mask] - psivec[inner_idx]) * (arr[inner_idx + 1] - arr[inner_idx]) / (psivec[inner_idx + 1] - psivec[inner_idx])
                        for i in range(inner_idx):
                            mxh_data[key][i] = arr.item(i)
        return mxh_data


    def _compute_derived_coordinates(
        self,
    ) -> None:
        data = self.input.to_dataset()
        newvars = {}
        if 'rho' in data:
            if 'mref' not in data:
                newvars['mref'] = np.zeros(data['n'].to_numpy()) + 2.0
            if 'rmin' in data:
                a = data['rmin'].isel(rho=1)
                newvars['a'] = (['n'], a.to_numpy().flatten())
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
                torflux = 2.0 * np.pi * data['torflux'] * data['rho'] ** 2
                newvars['torflux'] = (['n', 'rho'], torflux.to_numpy())
                newvars['psi_tor_norm'] = (['n', 'rho'], (torflux / torflux.isel(rho=-1)).to_numpy())
                newvars['rho_tor'] = (['n', 'rho'], np.sqrt((torflux / torflux.isel(rho=-1)).to_numpy()))
            if 'polflux' in data:
                polfluxn = (data['polflux'] - data['polflux'].isel(rho=0)) / (data['polflux'].isel(rho=-1) - data['polflux'].isel(rho=0))
                newvars['psi_pol_norm'] = (['n', 'rho'], polfluxn.to_numpy())
                newvars['rho_pol'] = (['n', 'rho'], np.sqrt(polfluxn.to_numpy()))
        self.update_input_data_vars(newvars)


    def compute_derived_quantities(
        self,
    ) -> None:

        def derivative(x, y):
            deriv = np.zeros_like(x)
            if len(x) > 2:
                x1 = np.concatenate([x[0], x[:-2], x[-3]])
                x2 = np.concatenate([x[1], x[1:-1], x[-2]])
                x3 = np.concatenate([x[2], x[2:], x[-1]])
                y1 = np.concatenate([y[0], y[:-2], y[-3]])
                y2 = np.concatenate([y[1], y[1:-1], y[-2]])
                y3 = np.concatenate([y[2], y[2:], y[-1]])
                deriv = ((x - x1) + (x - x2)) / (x3 - x1) / (x3 - x2) * y3 + ((x - x1) + (x - x3)) / (x2 - x1) / (x2 - x3) * y2 + ((x - x2) + (x - x3)) / (x1 - x2) / (x1 - x3) * y1
            elif len(x) > 1:
                deriv += np.diff(y) / np.diff(x)
            return deriv
        def find_interp_last(v, x, y):
            out = np.nan
            yprod = (y - v)[1:] * (y - v)[:-1]
            yidx = np.where(yprod)[0]
            if len(yidx) > 0:
                yi = yidx[-1]
                out = (v - y[yi]) * (x[yi + 1] - x[yi]) / (y[yi + 1] - y[yi])
            return out
        v_interp = np.vectorize(np.interp, signature='(m),(m,n),(m,n)->(m)')
        v_interp_last = np.vectorize(find_interp_last, signature='(m),(m,n),(m,n)->(m)')

        self._compute_derived_coordinates()
        data = self.input.to_dataset()

        newvars = {}
        if 'rho' in data:
            n_zeros = np.zeros_like(data['n'].to_numpy())

            if 'qmom' not in data:
                newvars['qmom'] = (['n', 'rho'], np.repeat(np.atleast_2d(np.zeros_like(data['rho'].to_numpy().flatten())), len(data['n']), axis=0))

            if 'rmin' in data and 'torflux' in data:
                newvars['B_unit'] = (['n', 'rho'], np.atleast_2d(derivative(0.5 * data['rmin'].to_numpy() ** 2 , data['torflux'].to_numpy() / (2.0 * pi))))

            if 'shape_sin0' not in data:
                newvars['shape_sin0'] = (['n', 'rho'], np.repeat(np.atleast_2d(np.zeros_like(data['rho'].to_numpy().flatten())), len(data['n']), axis=0))
            if 'delta' in data:
                newvars['shape_sin1'] = (['n', 'rho'], np.arcsin(data['delta'].to_numpy()))
            if 'zeta' in data:
                newvars['shape_sin2'] = (['n', 'rho'], -1.0 * data['zeta'].to_numpy())

            if 'q' in data and 'psi_pol_norm' in data:
                newvars['q0'] = v_interp(n_zeros, data['psi_pol_norm'].to_numpy(), data['q'].to_numpy())
                newvars['q95'] = v_interp(n_zeros + 0.95, data['psi_pol_norm'].to_numpy(), data['q'].to_numpy())
                newvars['rho_saw'] = v_interp_last(n_zeros + 1.0, data['rho_tor'].to_numpy(), data['q'].to_numpy())

        newvars['c_s'] = PLASMAtools.c_s(
            data['te(keV)'], newvars['mi_ref']
        )
        newvars['rho_s'] = PLASMAtools.rho_s(
            data['te(keV)'], newvars['mi_ref'], newvars['B_unit']
        )

        newvars['q_gb'], newvars['g_gb'], _, _, _ = PLASMAtools.gyrobohmUnits(
            data['te(keV)'],
            data['ne(10^19/m^3)'] * 1e-1,
            newvars['mi_ref'],
            np.abs(newvars['B_unit']),
            data['rmin(m)'][-1],
        )

        # --------- Geometry (only if it doesn't exist or if I ask to recalculate)

        if rederiveGeometry or ('volp_miller' not in newvars):

            self.produce_shape_lists()

            (
                newvars['volp_miller'],
                newvars['surf_miller'],
                newvars['gradr_miller'],
                newvars['bp2_miller'],
                newvars['bt2_miller'],
                newvars['geo_bt'],
            ) = GEOMETRYtools.calculateGeometricFactors(
                self,
                n_theta=n_theta_geo,
            )

            # Calculate flux surfaces
            cn = np.array(self.shape_cos).T
            sn = copy.deepcopy(self.shape_sin)
            sn[0] = data['rmaj(m)']*0.0
            sn[1] = np.arcsin(data['delta(-)'])
            sn[2] = -data['zeta(-)']
            sn = np.array(sn).T
            flux_surfaces = GEQtools.mitim_flux_surfaces()
            flux_surfaces.reconstruct_from_mxh_moments(
                data['rmaj(m)'],
                data['rmin(m)'],
                data['kappa(-)'],
                data['zmag(m)'],
                cn,
                sn)
            newvars['R_surface'],newvars['Z_surface'] = flux_surfaces.R, flux_surfaces.Z
            # -----------------------------------------------

            #cross-sectional area of each flux surface
            newvars['surfXS'] = GEOMETRYtools.xsec_area_RZ(
                newvars['R_surface'],
                newvars['Z_surface']
                )

            newvars['R_LF'] = newvars['R_surface'].max(
                axis=1
            )  # data['rmaj(m)'][0]+data['rmin(m)']

            # For Synchrotron
            newvars['B_ref'] = np.abs(
                newvars['B_unit'] * newvars['geo_bt']
            )

        # --------------------------------------------------------------------------
        # Reference mass
        # --------------------------------------------------------------------------

        # Forcing mass from this specific deriveQuantities call
        if mi_ref is not None:
            newvars['mi_ref'] = mi_ref
            print(f'\t- Using mi_ref={newvars['mi_ref']} provided in this particular deriveQuantities method, subtituting initialization one',typeMsg='i')

        # ---------------------------------------------------------------------------------------------------------------------
        # --------- Important for scaling laws
        # ---------------------------------------------------------------------------------------------------------------------

        newvars['kappa95'] = np.interp(
            0.95, newvars['psi_pol_n'], data['kappa(-)']
        )

        newvars['kappa995'] = np.interp(
            0.995, newvars['psi_pol_n'], data['kappa(-)']
        )

        newvars['kappa_a'] = newvars['surfXS'][-1] / np.pi / newvars['a'] ** 2

        newvars['delta95'] = np.interp(
            0.95, newvars['psi_pol_n'], data['delta(-)']
        )

        newvars['delta995'] = np.interp(
            0.995, newvars['psi_pol_n'], data['delta(-)']
        )

        newvars['Rgeo'] = float(data['rcentr(m)'][-1])
        newvars['B0'] = np.abs(float(data['bcentr(T)'][-1]))

        # ---------------------------------------------------------------------------------------------------------------------

        """
		surf_miller is truly surface area, but because of the GACODE definitions of flux, 
		Surf 		= V' <|grad r|>	 
		Surf_GACODE = V'
		"""

        newvars['surfGACODE_miller'] = (newvars['surf_miller'] / newvars['gradr_miller'])

        newvars['surfGACODE_miller'][np.isnan(newvars['surfGACODE_miller'])] = 0


        """
		In prgen_map_plasmastate:
			qspow_e = expro_qohme+expro_qbeame+expro_qrfe+expro_qfuse-expro_qei &
				-expro_qsync-expro_qbrem-expro_qline
			qspow_i = expro_qbeami+expro_qrfi+expro_qfusi+expro_qei
		"""

        qe_terms = {
            'qohme(MW/m^3)': 1,
            'qbeame(MW/m^3)': 1,
            'qrfe(MW/m^3)': 1,
            'qfuse(MW/m^3)': 1,
            'qei(MW/m^3)': -1,
            'qsync(MW/m^3)': -1,
            'qbrem(MW/m^3)': -1,
            'qline(MW/m^3)': -1,
            'qione(MW/m^3)': 1,
        }

        newvars['qe'] = np.zeros(len(data['rho(-)']))
        for i in qe_terms:
            if i in data:
                newvars['qe'] += qe_terms[i] * data[i]

        qrad = {
            'qsync(MW/m^3)': 1,
            'qbrem(MW/m^3)': 1,
            'qline(MW/m^3)': 1,
        }

        newvars['qrad'] = np.zeros(len(data['rho(-)']))
        for i in qrad:
            if i in data:
                newvars['qrad'] += qrad[i] * data[i]

        qi_terms = {
            'qbeami(MW/m^3)': 1,
            'qrfi(MW/m^3)': 1,
            'qfusi(MW/m^3)': 1,
            'qei(MW/m^3)': 1,
            'qioni(MW/m^3)': 1,
        }

        newvars['qi'] = np.zeros(len(data['rho(-)']))
        for i in qi_terms:
            if i in data:
                newvars['qi'] += qi_terms[i] * data[i]

        # Depends on GACODE version
        ge_terms = {self.varqpar: 1, self.varqpar2: 1}

        newvars['ge'] = np.zeros(len(data['rho(-)']))
        for i in ge_terms:
            if i in data:
                newvars['ge'] += ge_terms[i] * data[i]

        """
		Careful, that's in MW/m^3. I need to find the volumes. Using here the Miller
		calculation. Should be consistent with TGYRO

		profiles_gen puts any missing power into the CX: qioni, qione
		"""

        r = data['rmin(m)']
        volp = newvars['volp_miller']

        newvars['qe_MWmiller'] = CALCtools.integrateFS(newvars['qe'], r, volp)
        newvars['qi_MWmiller'] = CALCtools.integrateFS(newvars['qi'], r, volp)
        newvars['ge_10E20miller'] = CALCtools.integrateFS(
            newvars['ge'] * 1e-20, r, volp
        )  # Because the units were #/sec/m^3

        newvars['geIn'] = newvars['ge_10E20miller'][-1]  # 1E20 particles/sec

        newvars['qe_MWm2'] = newvars['qe_MWmiller'] / (volp)
        newvars['qi_MWm2'] = newvars['qi_MWmiller'] / (volp)
        newvars['ge_10E20m2'] = newvars['ge_10E20miller'] / (volp)

        newvars['QiQe'] = newvars['qi_MWm2'] / np.where(newvars['qe_MWm2'] == 0, 1e-10, newvars['qe_MWm2']) # to avoid division by zero

        # 'Convective' flux
        newvars['ce_MWmiller'] = PLASMAtools.convective_flux(
            data['te(keV)'], newvars['ge_10E20miller']
        )
        newvars['ce_MWm2'] = PLASMAtools.convective_flux(
            data['te(keV)'], newvars['ge_10E20m2']
        )

        # qmom
        newvars['mt_Jmiller'] = CALCtools.integrateFS(
            data[self.varqmom], r, volp
        )
        newvars['mt_Jm2'] = newvars['mt_Jmiller'] / (volp)

        # Extras for plotting in TGYRO for comparison
        P = np.zeros(len(data['rmin(m)']))
        if 'qsync(MW/m^3)' in data:
            P += data['qsync(MW/m^3)']
        if 'qbrem(MW/m^3)' in data:
            P += data['qbrem(MW/m^3)']
        if 'qline(MW/m^3)' in data:
            P += data['qline(MW/m^3)']
        newvars['qe_rad_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        P = data['qei(MW/m^3)']
        newvars['qe_exc_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        """
		---------------------------------------------------------------------------------------------------------------------
		Note that the real auxiliary power is RF+BEAMS+OHMIC, 
		The QIONE is added by TGYRO, but sometimes it includes radiation and direct RF to electrons
		---------------------------------------------------------------------------------------------------------------------
		"""

        # ** Electrons

        P = np.zeros(len(data['rho(-)']))
        for i in ['qrfe(MW/m^3)', 'qohme(MW/m^3)', 'qbeame(MW/m^3)']:
            if i in data:
                P += data[i]

        newvars['qe_auxONLY'] = copy.deepcopy(P)
        newvars['qe_auxONLY_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        for i in ['qione(MW/m^3)']:
            if i in data:
                P += data[i]

        newvars['qe_aux'] = copy.deepcopy(P)
        newvars['qe_aux_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        # ** Ions

        P = np.zeros(len(data['rho(-)']))
        for i in ['qrfi(MW/m^3)', 'qbeami(MW/m^3)']:
            if i in data:
                P += data[i]

        newvars['qi_auxONLY'] = copy.deepcopy(P)
        newvars['qi_auxONLY_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        for i in ['qioni(MW/m^3)']:
            if i in data:
                P += data[i]

        newvars['qi_aux'] = copy.deepcopy(P)
        newvars['qi_aux_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        # ** General

        P = np.zeros(len(data['rho(-)']))
        for i in ['qohme(MW/m^3)']:
            if i in data:
                P += data[i]
        newvars['qOhm_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        P = np.zeros(len(data['rho(-)']))
        for i in ['qrfe(MW/m^3)', 'qrfi(MW/m^3)']:
            if i in data:
                P += data[i]
        newvars['qRF_MWmiller'] = CALCtools.integrateFS(P, r, volp)
        if 'qrfe(MW/m^3)' in data:
            newvars['qRFe_MWmiller'] = CALCtools.integrateFS(
                data['qrfe(MW/m^3)'], r, volp
            )
        if 'qrfi(MW/m^3)' in data:
            newvars['qRFi_MWmiller'] = CALCtools.integrateFS(
                data['qrfi(MW/m^3)'], r, volp
            )

        P = np.zeros(len(data['rho(-)']))
        for i in ['qbeame(MW/m^3)', 'qbeami(MW/m^3)']:
            if i in data:
                P += data[i]
        newvars['qBEAM_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        newvars['qrad_MWmiller'] = CALCtools.integrateFS(newvars['qrad'], r, volp)
        if 'qsync(MW/m^3)' in data:
            newvars['qrad_sync_MWmiller'] = CALCtools.integrateFS(data['qsync(MW/m^3)'], r, volp)
        else:
            newvars['qrad_sync_MWmiller'] = newvars['qrad_MWmiller']*0.0
        if 'qbrem(MW/m^3)' in data:
            newvars['qrad_brem_MWmiller'] = CALCtools.integrateFS(data['qbrem(MW/m^3)'], r, volp)
        else:
            newvars['qrad_brem_MWmiller'] = newvars['qrad_MWmiller']*0.0
        if 'qline(MW/m^3)' in data:    
            newvars['qrad_line_MWmiller'] = CALCtools.integrateFS(data['qline(MW/m^3)'], r, volp)
        else:
            newvars['qrad_line_MWmiller'] = newvars['qrad_MWmiller']*0.0

        P = np.zeros(len(data['rho(-)']))
        for i in ['qfuse(MW/m^3)', 'qfusi(MW/m^3)']:
            if i in data:
                P += data[i]
        newvars['qFus_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        P = np.zeros(len(data['rho(-)']))
        for i in ['qioni(MW/m^3)', 'qione(MW/m^3)']:
            if i in data:
                P += data[i]
        newvars['qz_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        newvars['q_MWmiller'] = (
            newvars['qe_MWmiller'] + newvars['qi_MWmiller']
        )

        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------

        P = np.zeros(len(data['rho(-)']))
        if 'qfuse(MW/m^3)' in data:
            P = data['qfuse(MW/m^3)']
        newvars['qe_fus_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        P = np.zeros(len(data['rho(-)']))
        if 'qfusi(MW/m^3)' in data:
            P = data['qfusi(MW/m^3)']
        newvars['qi_fus_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        P = np.zeros(len(data['rho(-)']))
        if 'qfusi(MW/m^3)' in data:
            newvars['q_fus'] = (
                data['qfuse(MW/m^3)'] + data['qfusi(MW/m^3)']
            ) * 5
            P = newvars['q_fus']
        newvars['q_fus'] = P
        newvars['q_fus_MWmiller'] = CALCtools.integrateFS(P, r, volp)

        """
		Derivatives
		"""
        newvars['aLTe'] = aLT(data['rmin(m)'], data['te(keV)'])
        newvars['aLTi'] = data['ti(keV)'] * 0.0
        for i in range(data['ti(keV)'].shape[1]):
            newvars['aLTi'][:, i] = aLT(
                data['rmin(m)'], data['ti(keV)'][:, i]
            )
        newvars['aLne'] = aLT(
            data['rmin(m)'], data['ne(10^19/m^3)']
        )
        newvars['aLni'] = []
        for i in range(data['ni(10^19/m^3)'].shape[1]):
            newvars['aLni'].append(
                aLT(data['rmin(m)'], data['ni(10^19/m^3)'][:, i])
            )
        newvars['aLni'] = np.transpose(np.array(newvars['aLni']))

        if 'w0(rad/s)' not in data:
            data['w0(rad/s)'] = data['rho(-)'] * 0.0
        newvars['aLw0'] = aLT(data['rmin(m)'], data['w0(rad/s)'])
        newvars['dw0dr'] = -grad(
            data['rmin(m)'], data['w0(rad/s)']
        )

        newvars['dqdr'] = grad(data['rmin(m)'], data['q(-)'])

        """
		Other, performance
		"""
        qFus = newvars['qe_fus_MWmiller'] + newvars['qi_fus_MWmiller']
        newvars['Pfus'] = qFus[-1] * 5

        # Note that in cases with NPRAD=0 in TRANPS, this includes radiation! no way to deal wit this...
        qIn = newvars['qe_aux_MWmiller'] + newvars['qi_aux_MWmiller']
        newvars['qIn'] = qIn[-1]
        newvars['Q'] = newvars['Pfus'] / newvars['qIn']
        newvars['qHeat'] = qIn[-1] + qFus[-1]

        newvars['qTr'] = (
            newvars['qe_aux_MWmiller']
            + newvars['qi_aux_MWmiller']
            + (newvars['qe_fus_MWmiller'] + newvars['qi_fus_MWmiller'])
            - newvars['qrad_MWmiller']
        )

        newvars['Prad'] = newvars['qrad_MWmiller'][-1]
        newvars['Prad_sync'] = newvars['qrad_sync_MWmiller'][-1]
        newvars['Prad_brem'] = newvars['qrad_brem_MWmiller'][-1]
        newvars['Prad_line'] = newvars['qrad_line_MWmiller'][-1]
        newvars['Psol'] = newvars['qHeat'] - newvars['Prad']

        newvars['ni_thr'] = []
        for sp in range(len(self.Species)):
            if self.Species[sp]['S'] == 'therm':
                newvars['ni_thr'].append(data['ni(10^19/m^3)'][:, sp])
        newvars['ni_thr'] = np.transpose(newvars['ni_thr'])
        newvars['ni_thrAll'] = newvars['ni_thr'].sum(axis=1)

        newvars['ni_All'] = data['ni(10^19/m^3)'].sum(axis=1)


        (
            newvars['ptot_manual'],
            newvars['pe'],
            newvars['pi'],
        ) = PLASMAtools.calculatePressure(
            np.expand_dims(data['te(keV)'], 0),
            np.expand_dims(np.transpose(data['ti(keV)']), 0),
            np.expand_dims(data['ne(10^19/m^3)'] * 0.1, 0),
            np.expand_dims(np.transpose(data['ni(10^19/m^3)'] * 0.1), 0),
        )
        newvars['ptot_manual'], newvars['pe'], newvars['pi'] = (
            newvars['ptot_manual'][0],
            newvars['pe'][0],
            newvars['pi'][0],
        )

        (
            newvars['pthr_manual'],
            _,
            newvars['pi_thr'],
        ) = PLASMAtools.calculatePressure(
            np.expand_dims(data['te(keV)'], 0),
            np.expand_dims(np.transpose(data['ti(keV)']), 0),
            np.expand_dims(data['ne(10^19/m^3)'] * 0.1, 0),
            np.expand_dims(np.transpose(newvars['ni_thr'] * 0.1), 0),
        )
        newvars['pthr_manual'], newvars['pi_thr'] = (
            newvars['pthr_manual'][0],
            newvars['pi_thr'][0],
        )

        # -------
        # Content
        # -------

        (
            newvars['We'],
            newvars['Wi_thr'],
            newvars['Ne'],
            newvars['Ni_thr'],
        ) = PLASMAtools.calculateContent(
            np.expand_dims(r, 0),
            np.expand_dims(data['te(keV)'], 0),
            np.expand_dims(np.transpose(data['ti(keV)']), 0),
            np.expand_dims(data['ne(10^19/m^3)'] * 0.1, 0),
            np.expand_dims(np.transpose(newvars['ni_thr'] * 0.1), 0),
            np.expand_dims(volp, 0),
        )

        (
            newvars['We'],
            newvars['Wi_thr'],
            newvars['Ne'],
            newvars['Ni_thr'],
        ) = (
            newvars['We'][0],
            newvars['Wi_thr'][0],
            newvars['Ne'][0],
            newvars['Ni_thr'][0],
        )

        newvars['Nthr'] = newvars['Ne'] + newvars['Ni_thr']
        newvars['Wthr'] = newvars['We'] + newvars['Wi_thr']  # Thermal

        newvars['tauE'] = newvars['Wthr'] / newvars['qHeat']  # Seconds

        newvars['tauP'] = np.where(newvars['geIn'] != 0, newvars['Ne'] / newvars['geIn'], np.inf)   # Seconds
        

        newvars['tauPotauE'] = newvars['tauP'] / newvars['tauE']

        # Dilutions
        newvars['fi'] = data['ni(10^19/m^3)'] / np.atleast_2d(
            data['ne(10^19/m^3)']
        ).transpose().repeat(data['ni(10^19/m^3)'].shape[1], axis=1)

        # Vol-avg density
        newvars['volume'] = CALCtools.integrateFS(np.ones(r.shape[0]), r, volp)[
            -1
        ]  # m^3
        newvars['ne_vol20'] = (
            CALCtools.integrateFS(data['ne(10^19/m^3)'] * 0.1, r, volp)[-1]
            / newvars['volume']
        )  # 1E20/m^3

        newvars['ni_vol20'] = np.zeros(data['ni(10^19/m^3)'].shape[1])
        newvars['fi_vol'] = np.zeros(data['ni(10^19/m^3)'].shape[1])
        for i in range(data['ni(10^19/m^3)'].shape[1]):
            newvars['ni_vol20'][i] = (
                CALCtools.integrateFS(
                    data['ni(10^19/m^3)'][:, i] * 0.1, r, volp
                )[-1]
                / newvars['volume']
            )  # 1E20/m^3
            newvars['fi_vol'][i] = (
                newvars['ni_vol20'][i] / newvars['ne_vol20']
            )

        newvars['fi_onlyions_vol'] = newvars['ni_vol20'] / np.sum(
            newvars['ni_vol20']
        )

        newvars['ne_peaking'] = (
            data['ne(10^19/m^3)'][0] * 0.1 / newvars['ne_vol20']
        )

        xcoord = newvars[
            'rho_pol'
        ]  # to find the peaking at rho_pol (with square root) as in Angioni PRL 2003
        newvars['ne_peaking0.2'] = (
            data['ne(10^19/m^3)'][np.argmin(np.abs(xcoord - 0.2))]
            * 0.1
            / newvars['ne_vol20']
        )

        newvars['Te_vol'] = (
            CALCtools.integrateFS(data['te(keV)'], r, volp)[-1]
            / newvars['volume']
        )  # keV
        newvars['Te_peaking'] = (
            data['te(keV)'][0] / newvars['Te_vol']
        )
        newvars['Ti_vol'] = (
            CALCtools.integrateFS(data['ti(keV)'][:, 0], r, volp)[-1]
            / newvars['volume']
        )  # keV
        newvars['Ti_peaking'] = (
            data['ti(keV)'][0, 0] / newvars['Ti_vol']
        )

        newvars['ptot_manual_vol'] = (
            CALCtools.integrateFS(newvars['ptot_manual'], r, volp)[-1]
            / newvars['volume']
        )  # MPa
        newvars['pthr_manual_vol'] = (
            CALCtools.integrateFS(newvars['pthr_manual'], r, volp)[-1]
            / newvars['volume']
        )  # MPa

        newvars['pfast_manual'] = newvars['ptot_manual'] - newvars['pthr_manual']
        newvars['pfast_manual_vol'] = (
            CALCtools.integrateFS(newvars['pfast_manual'], r, volp)[-1]
            / newvars['volume']
        )  # MPa

        newvars['pfast_fraction'] = newvars['pfast_manual_vol'] / newvars['ptot_manual_vol']

        #approximate pedestal top density
        newvars['ptop(Pa)'] = np.interp(0.90, data['rho(-)'], data['ptot(Pa)'])

        # Quasineutrality
        newvars['QN_Error'] = np.abs(
            1 - np.sum(newvars['fi_vol'] * data['z'])
        )
        newvars['Zeff'] = (
            np.sum(data['ni(10^19/m^3)'] * data['z'] ** 2, axis=1)
            / data['ne(10^19/m^3)']
        )
        newvars['Zeff_vol'] = (
            CALCtools.integrateFS(newvars['Zeff'], r, volp)[-1]
            / newvars['volume']
        )

        newvars['nu_eff'] = PLASMAtools.coll_Angioni07(
            newvars['ne_vol20'] * 1e1,
            newvars['Te_vol'],
            newvars['Rgeo'],
            Zeff=newvars['Zeff_vol'],
        )

        newvars['nu_eff2'] = PLASMAtools.coll_Angioni07(
            newvars['ne_vol20'] * 1e1,
            newvars['Te_vol'],
            newvars['Rgeo'],
            Zeff=2.0,
        )

        # Avg mass
        self.calculateMass()

        params_set_scaling = (
            np.abs(float(data['current(MA)'][-1])),
            newvars['Rgeo'],
            newvars['kappa_a'],
            newvars['ne_vol20'],
            newvars['a'] / newvars['Rgeo'],
            newvars['B0'],
            newvars['mbg_main'],
            newvars['qHeat'],
        )

        newvars['tau98y2'], newvars['H98'] = PLASMAtools.tau98y2(
            *params_set_scaling, tauE=newvars['tauE']
        )
        newvars['tau89p'], newvars['H89'] = PLASMAtools.tau89p(
            *params_set_scaling, tauE=newvars['tauE']
        )
        newvars['tau97L'], newvars['H97L'] = PLASMAtools.tau97L(
            *params_set_scaling, tauE=newvars['tauE']
        )

        """
		Mach number
		"""

        Vtor_LF_Mach1 = PLASMAtools.constructVtorFromMach(
            1.0, data['ti(keV)'][:, 0], newvars['mbg']
        )  # m/s
        w0_Mach1 = Vtor_LF_Mach1 / (newvars['R_LF'])  # rad/s
        newvars['MachNum'] = data['w0(rad/s)'] / w0_Mach1
        newvars['MachNum_vol'] = (
            CALCtools.integrateFS(newvars['MachNum'], r, volp)[-1]
            / newvars['volume']
        )

        # Retain the old beta definition for comparison with 0D modeling
        Beta_old = (newvars['ptot_manual_vol']* 1e6 / (newvars['B0'] ** 2 / (2 * 4 * np.pi * 1e-7)))
        newvars['BetaN_engineering'] = (Beta_old / 
                                        (np.abs(float(data['current(MA)'][-1])) / 
                                         (newvars['a'] * newvars['B0'])
                                         )* 100.0
                                         ) # expressed in percent

        """
        ---------------------------------------------------------------------------------------------------
        Using B_unit, derive <B_p^2> and <Bt^2> for betap and betat calculations.
        Equivalent to GACODE expro_bp2, expro_bt2
        ---------------------------------------------------------------------------------------------------
        """

        newvars['bp2_exp'] = newvars['bp2_miller'] * newvars['B_unit'] ** 2
        newvars['bt2_exp'] = newvars['bt2_miller'] * newvars['B_unit'] ** 2

        # Calculate the volume averages of bt2 and bp2

        P = newvars['bp2_exp']
        newvars['bp2_vol_avg'] = CALCtools.integrateFS(P, r, volp)[-1] / newvars['volume']
        P = newvars['bt2_exp']
        newvars['bt2_vol_avg'] = CALCtools.integrateFS(P, r, volp)[-1] / newvars['volume']

        # calculate beta_poloidal and beta_toroidal using volume averaged values
        # mu0 = 4pi x 10^-7, also need to convert MPa to Pa

        newvars['Beta_p'] = (2 * 4 * np.pi * 1e-7)*newvars['ptot_manual_vol']* 1e6/newvars['bp2_vol_avg']
        newvars['Beta_t'] = (2 * 4 * np.pi * 1e-7)*newvars['ptot_manual_vol']* 1e6/newvars['bt2_vol_avg']

        newvars['Beta'] = 1/(1/newvars['Beta_p']+1/newvars['Beta_t'])

        TroyonFactor = np.abs(float(data['current(MA)'][-1])) / (newvars['a'] * newvars['B0'])

        newvars['BetaN'] = newvars['Beta'] / TroyonFactor * 100.0

        # ---

        nG = PLASMAtools.Greenwald_density(
            np.abs(float(data['current(MA)'][-1])),
            float(data['rmin(m)'][-1]),
        )
        newvars['fG'] = newvars['ne_vol20'] / nG
        newvars['fG_x'] = data['ne(10^19/m^3)']* 0.1 / nG

        newvars['tite'] = data['ti(keV)'][:, 0] / data['te(keV)']
        newvars['tite_vol'] = newvars['Ti_vol'] / newvars['Te_vol']

        newvars['LH_nmin'] = PLASMAtools.LHthreshold_nmin(
            np.abs(float(data['current(MA)'][-1])),
            newvars['B0'],
            newvars['a'],
            newvars['Rgeo'],
        )

        newvars['LH_Martin2'] = (
            PLASMAtools.LHthreshold_Martin2(
                newvars['ne_vol20'],
                newvars['B0'],
                newvars['a'],
                newvars['Rgeo'],
                nmin=newvars['LH_nmin'],
            )
            * (2 / newvars['mbg_main']) ** 1.11
        )

        newvars['LHratio'] = newvars['Psol'] / newvars['LH_Martin2']

        self.readSpecies()

        # -------------------------------------------------------
        # q-star
        # -------------------------------------------------------

        newvars['qstar'] = PLASMAtools.evaluate_qstar(
            data['current(MA)'][0],
            data['rcentr(m)'],
            newvars['kappa95'],
            data['bcentr(T)'],
            newvars['eps'],
            newvars['delta95'],
            ITERcorrection=False,
            includeShaping=True,
        )[0]
        newvars['qstar_ITER'] = PLASMAtools.evaluate_qstar(
            data['current(MA)'][0],
            data['rcentr(m)'],
            newvars['kappa95'],
            data['bcentr(T)'],
            newvars['eps'],
            newvars['delta95'],
            ITERcorrection=True,
            includeShaping=True,
        )[0]

        # -------------------------------------------------------
        # Separatrix estimations
        # -------------------------------------------------------

        # ~~~~ Estimate lambda_q
        pressure_atm = newvars['ptot_manual_vol'] * 1e6 / 101325.0
        Lambda_q = PLASMAtools.calculateHeatFluxWidth_Brunner(pressure_atm)

        # ~~~~ Estimate upstream temperature
        Bt = data['bcentr(T)'][0]
        Bp = newvars['eps'] * Bt / newvars['q95'] #TODO: VERY ROUGH APPROXIMATION!!!!

        newvars['Te_lcfs_estimate'] = PLASMAtools.calculateUpstreamTemperature(
                Lambda_q, 
                newvars['q95'], 
                newvars['ne_vol20'], 
                newvars['Psol'], 
                data['rcentr(m)'][0], 
                Bp, 
                Bt
                )[0]
                
        # ~~~~ Estimate upstream density
        newvars['ne_lcfs_estimate'] = newvars['ne_vol20'] * 0.6


    def read(
        self,
        path: str | Path,
        side: str = 'output',
    ) -> None:
        if side == 'input':
            self.input = self._read_gacode_file(path)
        else:
            self.output = self._read_gacode_file(path)


    def write(
        self,
        path: str | Path,
        side: str = 'input',
        overwrite: bool = False
    ) -> None:
        if side == 'input':
            self._write_gacode_file(path, self.input.to_dataset(), overwrite=overwrite)
        else:
            self._write_gacode_file(path, self.output.to_dataset(), overwrite=overwrite)


    def _read_gacode_file(
        self,
        path: str | Path
    ) -> xr.Dataset:

        coords = {}
        data_vars = {}
        attrs: MutableMapping[str, Any] = {}

        if isinstance(path, (str, Path)):
            ipath = Path(path)
            lines = []
            titles_single: list[str] = []
            if ipath.is_file():
                titles_single.extend(self.titles_singleInt)
                titles_single.extend(self.titles_singleStr)
                titles_single.extend(self.titles_singleFloat)
                with open(ipath, 'r') as f:
                    lines = f.readlines()

            istartProfs = None
            for i in range(len(lines)):
                if '# nexp' in lines[i]:
                    istartProfs = i
                    break
            header = lines[:istartProfs]
            while len(header) > 0 and not header[-1].strip():
                header = header[:-1]
            attrs['header'] = ''.join(header).strip()

            singleLine = False
            title = ''
            var: list[list[int | float]] = []
            found = False
            singles: dict[str, NDArray] = {}
            profiles: dict[str, NDArray] = {}
            for i in range(len(lines)):

                if lines[i].startswith('#') and not lines[i + 1].startswith('#'):
                    # previous
                    if found and not singleLine:
                        profiles[title] = np.array(var)
                        if profiles[title].shape[1] == 1:
                            profiles[title] = profiles[title][:, 0]
                    linebr = lines[i].split('#')[1].split('\n')[0].split()
                    title = linebr[0]
                    #title_orig = linebr[0]
                    #aif len(linebr) > 1:
                    #    unit = lines[i].split('#')[1].split('\n')[0].split()[2]
                    #    title = title_orig
                    #else:
                    #    title = title_orig
                    found, var = True, []
                    if title in titles_single:
                        singleLine = True
                    else:
                        singleLine = False

                elif found:
                    var0 = lines[i].split()
                    if singleLine:
                        if title in self.titles_singleFloat:
                            singles[title] = np.array(var0, dtype=float)
                        elif title in self.titles_singleInt:
                            singles[title] = np.array(var0, dtype=int)
                        else:
                            singles[title] = np.array(var0, dtype=str)
                    else:
                        varT = [float(j) if (j[-4].upper() == 'E' or '.' in j) else 0.0 for j in var0[1:]]
                        var.append(varT)

            # last
            if not singleLine:
                while len(var[-1]) < 1:
                    var = var[:-1]  # Sometimes there's an extra space, remove
                profiles[title] = np.array(var)
                if profiles[title].shape[1] == 1:
                    profiles[title] = profiles[title][:, 0]

            ncoord = 'n'
            rcoord = 'rho' if 'rho' in profiles else 'polflux'
            scoord = 'name' if 'name' in singles else 'z'
            coords[ncoord] = np.atleast_1d([0])
            if rcoord in profiles:
                coords[rcoord] = profiles.pop(rcoord)
            if scoord in singles:
                coords[scoord] = singles.pop(scoord)
            for key, val in profiles.items():
                if key in ['rho', 'polflux', 'rmin']:
                    data_vars[key] = ([ncoord, rcoord], np.expand_dims(val, axis=0))
                elif key in ['ni', 'ti', 'vtor', 'vpol']:
                    data_vars[key] = ([ncoord, rcoord, scoord], np.expand_dims(val, axis=0))
                elif key in ['w0']:
                    data_vars['omega0'] = ([ncoord, rcoord], np.expand_dims(val, axis=0))
                else:
                    data_vars[key] = ([ncoord, rcoord], np.expand_dims(val, axis=0))
            for key, val in singles.items():
                if key in ['name', 'z', 'mass', 'type']:
                    data_vars[key] = ([ncoord, scoord], np.expand_dims(val, axis=0))
                #elif key in ['header']:
                #    attrs[key] = val
                else:
                    data_vars[key] = ([ncoord], val)

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


    def _write_gacode_file(
        self,
        path: str | Path,
        data: xr.Dataset | xr.DataArray,
        item: int = 0,
        overwrite: bool = False
    ) -> None:

        if isinstance(path, (str, Path)) and isinstance(data, xr.Dataset):
            wdata = data.sel(n=item, drop=True)
            opath = Path(path)
            processed_titles = []
            header = wdata.attrs.get('header', '').split('\n')
            lines = [f'{line:<70}\n' for line in header]
            lines += ['#\n']
            processed_titles.append('header')
            for title in self.titles_singleInt:
                newlines = []
                if title in wdata:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(f'{wdata[title]:d}\n')
                    processed_titles.append(title)
                lines += newlines
            for title in self.titles_singleStr:
                newlines = []
                if title in wdata:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(' '.join([f'{val}' for val in np.atleast_1d(wdata[title].values)]) + '\n')
                    processed_titles.append(title)
                lines += newlines
            for title in self.titles_singleFloat:
                newlines = []
                if title in wdata:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    newlines.append(f'# {newtitle}\n')
                    newlines.append(' '.join([f'{val:14.7E}' for val in np.atleast_1d(wdata[title].values)]) + '\n')
                    processed_titles.append(title)
                lines += newlines
            for title in list(wdata.coords) + list(wdata.data_vars):
                newlines = []
                if title not in processed_titles:
                    newtitle = title
                    if title in self.units:
                        newtitle += f' | {self.units[title]}'
                    else:
                        newtitle += f' | -'
                    newlines.append(f'# {newtitle}\n')
                    rcoord = [f'{dim}' for dim in wdata[title].dims if dim in ['rho', 'polflux', 'rmin']]
                    if len(rcoord) > 0:
                        for ii in range(len(wdata[rcoord[0]])):
                            nstr = [f'{ii + 1:3d}']
                            nstr.extend([f'{val:14.7E}' for val in np.atleast_1d(wdata[title].isel({f'{rcoord[0]}': ii}).values)])
                            newlines.append(' '.join(nstr) + '\n')
                    processed_titles.append(title)
                lines += newlines

            with open(opath, 'w') as f:
                f.writelines(lines)
            logger.info(f'Saved {self.format} data into {opath.resolve()}')
            #else:
            #    logger.warning(f'Requested write path, {opath.resolve()}, already exists! Aborting write...')
        else:
            logger.error(f'Invalid path argument given to {self.format} write function! Aborting write...')


    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        input: str | Path | None = None,
        output: str | Path | None = None,
    ) -> Self:
        return cls(path=path, input=input, output=output)  # Places data into output side unless specified


    # Assumed that the self creation method transfers output to input
    @classmethod
    def from_gacode(
        cls,
        obj: io,
        side: str = 'output',
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            newobj.input = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
        return newobj


    @classmethod
    def from_torax(
        cls,
        obj: io,
        side: str = 'output',
        window: Sequence[int | float] | None = None,
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            if 'rho_norm' in data.coords:
                data = data.isel(time=-1)
                zeros = np.zeros_like(data.coords['rho_norm'].to_numpy().flatten())
                coords = {}
                data_vars = {}
                attrs: MutableMapping[str, Any] = {}
                name: list[str] = []
                coords['n'] = np.array([0], dtype=int)
                if 'rho_norm' in data.coords:
                    coords['rho'] = data.coords['rho_norm'].to_numpy().flatten()
                    data_vars['nexp'] = (['n'], np.array([len(coords['rho'])], dtype=int))
                if 'psi' in data:
                    data_vars['polflux'] = (['n', 'rho'], np.expand_dims(data['psi'].to_numpy().flatten(), axis=0))
                if 'r_mid' in data:
                    data_vars['rmin'] = (['n', 'rho'], np.expand_dims(data['r_mid'].to_numpy().flatten(), axis=0))
                data_vars['shot'] = (['n'], np.atleast_1d([0]))
                data_vars['masse'] = (['n'], np.atleast_1d([5.4488748e-04]))
                data_vars['ze'] = (['n'], np.atleast_1d([-1.0]))
                if 'Phi_b' in data:
                    data_vars['torfluxa'] = (['n'], data['Phi_b'].to_numpy().flatten())
                #if 'R_major' in data:
                #    data_vars['rcentr'] = (['n'], data['R_major'].to_numpy().flatten())
                if 'R_out' in data:
                    data_vars['rcentr'] = (['n'], data['R_out'].isel(rho_norm=0).to_numpy().flatten())
                if 'F' in data and 'R_out' in data:
                    data_vars['bcentr'] = (['n'], (data['F'] / data['R_out']).isel(rho_norm=0).to_numpy().flatten())
                if 'Ip' in data:
                    data_vars['current'] = (['n'], 1.0e-6 * data['Ip'].to_numpy().flatten())
                if 'q' in data and 'rho_norm' in data and 'rho_face_norm' in data:
                    q = np.interp(data['rho_norm'].to_numpy().flatten(), data['rho_face_norm'].to_numpy().flatten(), data['q'].to_numpy().flatten())
                    data_vars['q'] = (['n', 'rho'], np.expand_dims(q, axis=0))
                if 'R_in' in data and 'R_out' in data:
                    rmaj = (data['R_in'] + data['R_out']).to_numpy().flatten() / 2.0
                    data_vars['rmaj'] = (['n', 'rho'], np.expand_dims(rmaj, axis=0))
                    data_vars['zmag'] = (['n', 'rho'], np.expand_dims(np.zeros_like(zeros), axis=0))
                if 'elongation' in data:
                    data_vars['kappa'] = (['n', 'rho'], np.expand_dims(data['elongation'].to_numpy().flatten(), axis=0))
                if 'delta' in data:
                    delta = data['delta'].to_numpy().flatten()
                    delta = np.concatenate([np.array([delta[0]]), delta[:-1] + 0.5 * np.diff(delta), np.array([delta[-1]])], axis=0)
                    data_vars['delta'] = (['n', 'rho'], np.expand_dims(delta, axis=0))
                data['zeta'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos0'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos1'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos2'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos3'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos4'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos5'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_cos6'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_sin3'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_sin4'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_sin5'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                data['shape_sin6'] = (['n', 'rho'], np.expand_dims(zeros, axis=0))
                if 'n_i' in data and 'n_e' in data:
                    split_dt = True
                    ne = np.expand_dims(1.0e-19 * data['n_e'].to_numpy().flatten(), axis=-1)
                    ni = np.expand_dims(1.0e-19 * data['n_i'].to_numpy().flatten(), axis=-1)
                    zeff = ni / ne
                    zimps = []
                    if 'n_impurity' in data:
                        nimp = np.expand_dims(1.0e-19 * data['n_impurity'].to_numpy().flatten(), axis=-1)
                        if 'Z_impurity' in data and 'n_e' in data:
                            zimp = np.expand_dims(data['Z_impurity'].to_numpy().flatten(), axis=-1)
                            zimps = [zimp[0, 0]]
                        if split_dt:
                            ni = np.concatenate([0.5 * ni, 0.5 * ni], axis=-1)
                        if 'config' in data.attrs:
                            impdict = data.attrs['config'].get('plasma_composition', {}).get('impurity', {})
                            multi_nimp = []
                            multi_zimp = []
                            for key in impdict:
                                fraction = impdict[key].get('value', ['float', [0.0]])[1][-1]
                                impname, impa, impz = define_ion_species(short_name=key)
                                multi_zimp.append(impz)
                                multi_nimp.append(fraction * nimp)
                            if len(multi_nimp) > 0:
                                nimp = np.concatenate(multi_nimp, axis=-1)
                                zimps = multi_zimp
                        ni = np.concatenate([ni, nimp], axis=-1)
                    names = ['D']
                    types = ['[therm]']
                    masses = [2.0]
                    zs = [1.0]
                    if split_dt:
                        names.append('T')
                        types.append('[therm]')
                        masses.append(3.0)
                        zs.append(1.0)
                    ii = len(names)
                    for zz in range(len(zimps)):
                        impname, impa, impz = define_ion_species(z=zimps[zz])
                        names.append(impname)
                        types.append('[therm]')
                        masses.append(impa)
                        zs.append(impz)
                        zeff += np.expand_dims(ni[:, zz+ii], axis=-1) * (impz ** 2.0) / ne
                    coords['name'] = np.array(names)
                    data_vars['ni'] = (['n', 'rho', 'name'], np.expand_dims(ni, axis=0))
                    data_vars['nion'] = (['n'], np.array([len(names)], dtype=int))
                    data_vars['type'] = (['n', 'name'], np.expand_dims(types, axis=0))
                    data_vars['mass'] = (['n', 'name'], np.expand_dims(masses, axis=0))
                    data_vars['z'] = (['n', 'name'], np.expand_dims(zs, axis=0))
                    data_vars['z_eff'] = (['n', 'rho'], np.expand_dims(zeff.flatten(), axis=0))
                if 'T_i' in data:
                    ti = np.expand_dims(data['T_i'].to_numpy().flatten(), axis=-1)
                    if 'name' in coords and len(coords['name']) > 1:
                        ti = np.repeat(ti, len(coords['name']), axis=-1)
                    data_vars['ti'] = (['n', 'rho', 'name'], np.expand_dims(ti, axis=0))
                if 'n_e' in data:
                    data_vars['ne'] = (['n', 'rho'], np.expand_dims(1.0e-19 * data['n_e'].to_numpy().flatten(), axis=0))
                if 'T_e' in data:
                    data_vars['te'] = (['n', 'rho'], np.expand_dims(data['T_e'].to_numpy().flatten(), axis=0))
                if 'p_ohmic_e' in data:
                    dvec = data['p_ohmic_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qohme'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_generic_heat_e' in data:
                    dvec = data['p_generic_heat_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfe'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                    #data_vars['qbeame'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_generic_heat_i' in data:
                    dvec = data['p_generic_heat_i'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfi'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                    #data_vars['qbeami'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_icrh_e' in data:
                    dvec = data['p_icrh_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfe'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_icrh_i' in data:
                    dvec = data['p_icrh_i'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfi'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_ecrh_e' in data:
                    dvec = data['p_ecrh_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfe'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_ecrh_i' in data:
                    dvec = data['p_ecrh_i'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qrfi'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_cyclotron_radiation_e' in data:
                    dvec = data['p_cyclotron_radiation_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qsync'] = (['n', 'rho'], np.expand_dims(-1.0e-6 * dvec, axis=0))
                if 'p_bremsstrahlung_e' in data:
                    dvec = data['p_bremsstrahlung_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qbrem'] = (['n', 'rho'], np.expand_dims(-1.0e-6 * dvec, axis=0))
                if 'p_impurity_radiation_e' in data:
                    dvec = data['p_impurity_radiation_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qline'] = (['n', 'rho'], np.expand_dims(-1.0e-6 * dvec, axis=0))
                if 'p_alpha_e' in data:
                    dvec = data['p_alpha_e'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qfuse'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'p_alpha_i' in data:
                    dvec = data['p_alpha_i'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qfusi'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'ei_exchange' in data:
                    dvec = data['ei_exchange'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qei'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'j_ohmic' in data:
                    dvec = data['j_ohmic'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['johm'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'j_bootstrap' in data:
                    #dvec = np.concatenate([np.array([np.nan]), data['j_bootstrap'].to_numpy().flatten(), np.array([np.nan])], axis=0)
                    data_vars['jbs'] = (['n', 'rho'], np.expand_dims(1.0e-6 * data['j_bootstrap'].to_numpy().flatten(), axis=0))
                    #data_vars['jbstor'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'j_ecrh' in data:
                    dvec = data['j_ecrh'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['jrf'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'j_external' in data:
                    dvec = data['j_external'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['jrf'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                    #data_vars['jnb'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'j_generic_current' in data:
                    dvec = data['j_generic_current'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['jrf'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                #    data_vars['jnb'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                if 'pressure_thermal_total' in data and 'rho_norm' in data and 'rho_face_norm' in data:
                    data_vars['ptot'] = (['n', 'rho'], np.expand_dims(data['pressure_thermal_total'].to_numpy().flatten(), axis=0))
                if 's_gas_puff' in data:
                    dvec = data['s_gas_puff'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qpar_wall'] = (['n', 'rho'], np.expand_dims(dvec, axis=0))
                if 's_pellet' in data:
                    dvec = data['s_pellet'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qpar_wall'] = (['n', 'rho'], np.expand_dims(dvec, axis=0))
                if 's_generic_particle' in data:
                    dvec = data['s_generic_particle'].to_numpy().flatten()
                    dvec = np.concatenate([np.array([dvec[0]]), dvec, np.array([dvec[-1]])], axis=0)
                    data_vars['qpar_beam'] = (['n', 'rho'], np.expand_dims(dvec, axis=0))
                #'qione'
                #'qioni'
                #'qcxi'
                #'vtor'
                #'vpol'
                #'omega0'
                #'qmom'
                attrs['header'] = newobj.make_file_header()
                newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj


    @classmethod
    def from_imas(
        cls,
        obj: io,
        side: str = 'output',
        window: Sequence[int | float] | None = None,
        transpose_equilibrium: bool = False,
        jetto_style: bool = False,
        **kwargs: Any,
    ) -> Self:

        newobj = cls()
        if isinstance(obj, io):

            data: xr.Dataset = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            obj_cocos = obj.input_cocos if side == 'input' else obj.output_cocos  # type: ignore[attr-defined]
            time_cp = 'core_profiles.time'
            rho_cp_i = 'core_profiles.profiles_1d.grid.rho_tor_norm:i'
            rho_cp = 'core_profiles.profiles_1d.grid.rho_tor_norm'
            ion_cp_i = 'core_profiles.profiles_1d.ion:i'
            ion_cp = 'core_profiles.profiles_1d.ion.label'
            time_eq = 'equilibrium.time'
            psi_eq_i = 'equilibrium.time_slice.profiles_1d.psi:i'
            psi_eq = 'equilibrium.time_slice.profiles_1d.psi'
            rho_eq = 'equilibrium.time_slice.profiles_1d.rho_tor_norm'
            time_cs = 'core_sources.time'
            src_cs_i = 'core_sources.source:i'
            src_cs = 'core_sources.source.identifier.name'
            rho_cs_i = 'core_sources.source.profiles_1d.grid.rho_tor_norm:i'
            rho_cs = 'core_sources.source.profiles_1d.grid.rho_tor_norm'
            ion_cs_i = 'core_sources.source.profiles_1d.ion:i'
            ion_cs = 'core_sources.source.profiles_1d.ion.label'
            ikwargs = {'fill_value': 'extrapolate'}

            cocos_out = 2   # Assumed GACODE has COCOS=2
            if jetto_style and obj_cocos == 11:
                cocos_out = 8
            cocos = define_cocos_converter(obj_cocos, cocos_out)

            dsvec = []

            if time_cp in data.coords:

                time_indices = [-1]
                time = np.array([data.get(time_cp, xr.DataArray()).to_numpy().flatten()[time_indices]]).flatten()  #TODO: Use window argument
                for i, time_index in enumerate(time_indices):

                    coords = {}
                    data_vars = {}
                    attrs: MutableMapping[str, Any] = {}

                    if rho_cp_i in data.dims and rho_cp in data:
                        data = data.isel({time_cp: time_index}).swap_dims({rho_cp_i: rho_cp}).drop_duplicates(rho_cp)
                        if ion_cp_i in data.dims and ion_cp in data:
                            data = data.swap_dims({ion_cp_i: ion_cp})
                        coords['n'] = np.array([i], dtype=int)
                        coords['rho'] = data[rho_cp].to_numpy().flatten()
                        data_vars['nexp'] = (['n'], np.array([len(coords['rho'])], dtype=int))
                        data_vars['shot'] = (['n'], np.atleast_1d([0]))
                        data_vars['masse'] = (['n'], np.atleast_1d([5.4488748e-04]))
                        data_vars['ze'] = (['n'], np.atleast_1d([-1.0]))
                        if ion_cp in data.coords:
                            coords['name'] = data[ion_cp].to_numpy().flatten()
                            data_vars['nion'] = (['n'], np.array([len(coords['name'])], dtype=int))
                            ni = None
                            zi = None
                            tag = 'core_profiles.profiles_1d.ion.density_thermal'
                            if tag in data:
                                types = []
                                for name in coords['name']:
                                    types.extend(['[therm]' if data[tag].sel({ion_cp: name}).sum() > 0.0 else '[fast]'])
                                ni = data[tag]
                                data_vars['ni'] = (['n', 'rho', 'name'], 1.0e-19 * np.expand_dims(ni.to_numpy().T, axis=0))
                                data_vars['type'] = (['n', 'name'], np.expand_dims(types, axis=0))
                            tag = 'core_profiles.profiles_1d.ion.temperature'
                            if tag in data:
                                ti = data[tag]
                                data_vars['ti'] = (['n', 'rho', 'name'], 1.0e-3 * np.expand_dims(ti.to_numpy().T, axis=0))
                            eltag = 'core_profiles.profiles_1d.ion.element:i'
                            tag = 'core_profiles.profiles_1d.ion.element.a'
                            if tag in data:
                                data_vars['mass'] = (['n', 'name'], np.expand_dims(data[tag].isel({eltag: 0}).to_numpy(), axis=0))
                            tag = 'core_profiles.profiles_1d.ion.element.z_n'
                            if tag in data:
                                zi = data[tag].isel({eltag: 0})
                                data_vars['z'] = (['n', 'name'], np.expand_dims(zi.to_numpy(), axis=0))
                            tag = 'core_profiles.profiles_1d.ion.z_ion_1d'  # Potential source of mismatch
                            if tag in data:
                                zi = data[tag]
                            tag = 'core_profiles.profiles_1d.electrons.density_thermal'
                            if tag in data and ni is not None and zi is not None:
                                zeff = (ni * zi * zi).sum(ion_cp) / data[tag]
                                data_vars['z_eff'] = (['n', 'rho'], np.expand_dims(zeff.to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.electrons.density_thermal'
                        if tag in data:
                            ne = data[tag]
                            data_vars['ne'] = (['n', 'rho'], 1.0e-19 * np.expand_dims(ne.to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.electrons.temperature'
                        if tag in data:
                            te = data[tag]
                            data_vars['te'] = (['n', 'rho'], 1.0e-3 * np.expand_dims(te.to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.pressure_thermal'
                        if tag in data:
                            data_vars['ptot'] = (['n', 'rho'], np.expand_dims(data[tag].to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.q'
                        if tag in data:
                            data_vars['q'] = (['n', 'rho'], cocos['spol'] * np.expand_dims(data[tag].to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.j_ohmic'
                        if tag in data:
                            data_vars['johm'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * np.expand_dims(data[tag].to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.j_bootstrap'
                        if tag in data:
                            data_vars['jbs'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * np.expand_dims(data[tag].to_numpy(), axis=0))
                        #tag = 'core_profiles.profiles_1d.momentum_tor'
                        tag = 'core_profiles.profiles_1d.ion.velocity.toroidal'
                        if tag in data:
                            data_vars['vtor'] = (['n', 'rho', 'name'], cocos['scyl'] * np.expand_dims(data[tag].to_numpy().T, axis=0))
                        tag = 'core_profiles.profiles_1d.ion.velocity.poloidal'
                        if tag in data:
                            data_vars['vpol'] = (['n', 'rho', 'name'], cocos['spol'] * np.expand_dims(data[tag].to_numpy().T, axis=0))
                        tag = 'core_profiles.profiles_1d.rotation_frequency_tor_sonic'
                        if tag in data:
                            data_vars['omega0'] = (['n', 'rho'], cocos['scyl'] * np.expand_dims(data[tag].to_numpy(), axis=0))
                        tag = 'core_profiles.profiles_1d.grid.rho_tor'
                        if tag in data and 'core_profiles.vacuum_toroidal_field.b0' in data:
                            torflux = data[tag].interp({rho_cp: np.array([1.0])}, kwargs=ikwargs) ** 2.0 / (np.pi * data['core_profiles.vacuum_toroidal_field.b0'])
                            data_vars['torfluxa'] = (['n'], cocos['scyl'] * torflux.to_numpy().flatten())

                    if time_eq in data.coords and psi_eq_i in data.dims and rho_eq in data and 'rho' in coords:
                        data = data.interp({time_eq: time.item(i)}, kwargs=ikwargs) if data[time_eq].size > 1 else data.isel({time_eq: 0})
                        data = data.swap_dims({psi_eq_i: rho_eq}).drop_duplicates(rho_eq)
                        eqdsk_data = obj.to_eqdsk(time_index=time_index, side=side, transpose=transpose_equilibrium) if hasattr(obj, 'to_eqdsk') else {}
                        rhovec = data.get(rho_eq, xr.DataArray()).to_numpy().flatten()
                        psivec = None
                        tag = 'equilibrium.time_slice.profiles_1d.psi'
                        if tag in data:
                            #ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'psi': (['rho_int'], data[tag].to_numpy().flatten())})
                            #data_vars['polflux'] = (['n', 'rho'], np.expand_dims(ndata['psi'].interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            psivec = data[tag].interp({rho_eq: coords['rho']}, kwargs=ikwargs).to_numpy()
                            data_vars['polflux'] = (['n', 'rho'], np.power(2.0 * np.pi, cocos['eBp']) * cocos['sBp'] * np.expand_dims(psivec, axis=0))
                        tag = 'equilibrium.vacuum_toroidal_field.r0'
                        if tag in data:
                            data_vars['rcentr'] = (['n'], np.atleast_1d(data[tag].to_numpy()))
                        tag = 'equilibrium.vacuum_toroidal_field.b0'
                        if tag in data:
                            data_vars['bcentr'] = (['n'], cocos['scyl'] * np.atleast_1d(data[tag].to_numpy()))
                        tag = 'equilibrium.time_slice.profiles_1d.pressure'
                        if tag in data and 'ptot' not in data_vars:
                            #ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'pressure': (['rho_int'], data[tag].to_numpy().flatten())})
                            #data_vars['ptot'] = (['n', 'rho'], np.expand_dims(ndata['pressure'].interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['ptot'] = (['n', 'rho'], np.expand_dims(data[tag].interp({rho_eq: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        tag = 'equilibrium.time_slice.profiles_1d.q'
                        if tag in data and 'q' not in data_vars:
                            #ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'q': (['rho_int'], data[tag].to_numpy().flatten())})
                            #data_vars['q'] = (['n', 'rho'], np.expand_dims(ndata['q'].interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['q'] = (['n', 'rho'], cocos['spol'] * np.expand_dims(data[tag].interp({rho_eq: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        if eqdsk_data:
                            if psivec is None:
                                psivec = np.linspace(eqdsk_data['simagx'], eqdsk_data['sibdry'], len(coords['rho']))
                            mxh_data = newobj._calculate_geometry_from_eqdsk(eqdsk_data, psivec)
                            data_vars['rmaj'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['rmaj']), axis=0))
                            data_vars['rmin'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['rmin']), axis=0))
                            data_vars['zmag'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['zmag']), axis=0))
                            data_vars['kappa'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['kappa']), axis=0))
                            data_vars['delta'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['delta']), axis=0))
                            data_vars['zeta'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['zeta']), axis=0))
                            data_vars['shape_sin3'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin3']), axis=0))
                            data_vars['shape_sin4'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin4']), axis=0))
                            data_vars['shape_sin5'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin5']), axis=0))
                            data_vars['shape_sin6'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['sin6']), axis=0))
                            data_vars['shape_cos0'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos0']), axis=0))
                            data_vars['shape_cos1'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos1']), axis=0))
                            data_vars['shape_cos2'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos2']), axis=0))
                            data_vars['shape_cos3'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos3']), axis=0))
                            data_vars['shape_cos4'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos4']), axis=0))
                            data_vars['shape_cos5'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos5']), axis=0))
                            data_vars['shape_cos6'] = (['n', 'rho'], np.expand_dims(np.atleast_1d(mxh_data['cos6']), axis=0))
                        tag = 'equilibrium.time_slice.global_quantities.ip'
                        if tag in data:
                            data_vars['current'] = (['n'], 1.0e-6 * cocos['scyl'] * np.atleast_1d(data[tag].to_numpy()))
                        itag = 'equilibrium.time_slice.profiles_1d.r_inboard'
                        otag = 'equilibrium.time_slice.profiles_1d.r_outboard'
                        if itag in data and otag in data and ('rmaj' not in data_vars or 'rmin' not in data_vars):
                            #ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={
                            #    'r_inboard': (['rho_int'], data[itag].to_numpy().flatten()),
                            #    'r_outboard': (['rho_int'], data[otag].to_numpy().flatten())
                            #})
                            #data_vars['rmin'] = (['n', 'rho'], np.expand_dims((0.5 * (ndata['r_outboard'] - ndata['r_inboard'])).interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            #data_vars['rmaj'] = (['n', 'rho'], np.expand_dims((0.5 * (ndata['r_outboard'] + ndata['r_inboard'])).interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['rmin'] = (['n', 'rho'], np.expand_dims((0.5 * (data[otag] - data[itag])).interp({rho_cp: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['rmaj'] = (['n', 'rho'], np.expand_dims((0.5 * (data[otag] + data[itag])).interp({rho_cp: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        #tag = 'equilibrium.time_slice.global_quantities.magnetic_axis.z'
                        #if tag in data and 'zmag' not in data_vars:
                        #    data_vars['zmag'] = (['n', 'rho'], np.expand_dims(np.repeat(data[tag].to_numpy().flatten(), len(coords['rho']), axis=0), axis=0))
                        tag = 'equilibrium.time_slice.profiles_1d.elongation'
                        if tag in data and 'kappa' not in data_vars:
                            #ndata = xr.Dataset(coords={'rho_int': rhovec}, data_vars={'elongation': (['rho_int'], data[tag].to_numpy().flatten())})
                            #data_vars['kappa'] = (['n', 'rho'], np.expand_dims(ndata['elongation'].interp({'rho_int': coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            data_vars['kappa'] = (['n', 'rho'], np.expand_dims(data[tag].interp({rho_cp: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        #if 'equilibrium.time_slice.profiles_1d.triangularity_upper' in data or 'equilibrium.time_slice.profiles_1d.triangularity_lower' in data and 'delta' not in data_vars:
                            #tri = np.zeros(data['rho(-)'].shape)
                            #itri = 0
                            #if hasattr(time_struct.profiles_1d, 'triangularity_upper'):
                            #    tri += time_struct.profiles_1d.triangularity_upper.flatten()
                            #    itri += 1
                            #if hasattr(time_struct.profiles_1d, 'triangularity_lower') and len(time_struct.profiles_1d.triangularity_lower) == data['nexp']:
                            #    tri += time_struct.profiles_1d.triangularity_lower.flatten()
                            #    itri += 1
                            #data['delta(-)'] = tri / float(itri) if itri > 0 else tri

                    if time_cs in data.coords and src_cs_i in data.dims and src_cs in data and rho_cs_i in data.dims and rho_cs in data and 'rho' in coords:
                        data = data.interp({time_cs: time.item(i)}, kwargs=ikwargs) if data[time_cs].size > 1 else data.isel({time_cs: 0})
                        data = data.swap_dims({src_cs_i: src_cs})
                        #if ion_cs_i in data.dims and ion_cs in data:
                        #    data = data.swap_dims({ion_cs_i: ion_cs})
                        srclist = data[src_cs].to_numpy().tolist()
                        qrfe = np.zeros((len(coords['rho']), ))
                        qrfi = np.zeros((len(coords['rho']), ))
                        jrf = np.zeros((len(coords['rho']), ))
                        tag = 'core_sources.source.profiles_1d.electrons.energy'
                        if tag in data:
                            srctag = 'ohmic'
                            if srctag in srclist:
                                data_vars['qohme'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'ec'
                            if srctag in srclist:
                                qrfe += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'ic'
                            if srctag in srclist:
                                qrfe += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'lh'
                            if srctag in srclist:
                                qrfe += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'nbi'
                            if srctag in srclist:
                                data_vars['qbeame'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'synchrotron_radiation'
                            if srctag in srclist:
                                data_vars['qsync'] = (['n', 'rho'], -1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'radiation'
                            if srctag in srclist:
                                data_vars['qline'] = (['n', 'rho'], -1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'bremsstrahlung'
                            if srctag in srclist:
                                data_vars['qbrem'] = (['n', 'rho'], -1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'fusion'
                            if srctag in srclist:
                                data_vars['qfuse'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'collisional_equipartition'
                            if srctag in srclist:
                                data_vars['qei'] = (['n', 'rho'], -1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        if 'qbrem' not in data_vars:  # Why single this one out randomly?
                            data_vars['qbrem'] = (['n', 'rho'], np.expand_dims(np.zeros_like(coords['rho']), axis=0))
                        tag = 'core_sources.source.profiles_1d.total_ion_energy'
                        if tag in data:
                            srctag = 'ic'
                            if srctag in srclist:
                                qrfi += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'lh'
                            if srctag in srclist:
                                qrfi += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'nbi'
                            if srctag in srclist:
                                data_vars['qbeami'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'charge_exchange'
                            if srctag in srclist:
                                data_vars['qcxi'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'fusion'
                            if srctag in srclist:
                                data_vars['qfusi'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        tag = 'core_sources.source.profiles_1d.j_parallel'
                        if tag in data:
                            srctag = 'ohmic'
                            if srctag in srclist and 'johm' not in data_vars:
                                data_vars['johm'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                            srctag = 'j_bootstrap'
                            if srctag in srclist and 'jbs' not in data_vars:
                                data_vars['jbs'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                                #data_vars['jbstor'] = (['n', 'rho'], np.expand_dims(1.0e-6 * dvec, axis=0))
                            srctag = 'ec'
                            if srctag in srclist:
                                jrf += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'ic'
                            if srctag in srclist:
                                jrf += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'lh'
                            if srctag in srclist:
                                jrf += data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy().flatten()
                            srctag = 'nbi'
                            if srctag in srclist:
                                data_vars['jnb'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        tag = 'core_sources.source.profiles_1d.ion.particles'
                        if tag in data and ion_cs_i in data.coords:
                            srctag = 'cold_neutrals'
                            if srctag in srclist:
                                data_vars['qpar_wall'] = (['n', 'rho'], np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).sum(ion_cs_i).to_numpy(), axis=0))
                            srctag = 'nbi'
                            if srctag in srclist:
                                data_vars['qpar_beam'] = (['n', 'rho'], np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).sum(ion_cs_i).to_numpy(), axis=0))
                        tag = 'core_sources.source.profiles_1d.momentum_tor'
                        if tag in data:
                            srctag = 'nbi'
                            if srctag in srclist:
                                data_vars['qmom'] = (['n', 'rho'], cocos['scyl'] * np.expand_dims(data[tag].sel({src_cs: srctag}).swap_dims({rho_cs_i: rho_cs}).drop_duplicates(rho_cs).interp({rho_cs: coords['rho']}, kwargs=ikwargs).to_numpy(), axis=0))
                        if np.abs(qrfe).sum() > 0.0:
                            data_vars['qrfe'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(qrfe, axis=0))
                        if np.abs(qrfi).sum() > 0.0:
                            data_vars['qrfi'] = (['n', 'rho'], 1.0e-6 * np.expand_dims(qrfi, axis=0))
                        if np.abs(jrf).sum() > 0.0:
                            data_vars['jrf'] = (['n', 'rho'], cocos['scyl'] * 1.0e-6 * np.expand_dims(jrf, axis=0))

                    dsvec.append(xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs))

            if len(dsvec) > 0:
                newobj.input = xr.concat(dsvec, dim='n').assign_attrs({'header': newobj.make_file_header()})

        return newobj


    @classmethod
    def from_astra(
        cls,
        obj: io,
        side: str = 'output',
        window: Sequence[int | float] | None = None,
        **kwargs: Any,
    ) -> Self:
        newobj = cls()
        if isinstance(obj, io):
            data = obj.input.to_dataset() if side == 'input' else obj.output.to_dataset()
            coords = {}
            data_vars = {}
            attrs: MutableMapping[str, Any] = {}
            if 'xrho' in data.coords:
                data = data.isel(time=-1)
                zeros = np.zeros_like(data.coords['xrho'].to_numpy().flatten())
                #name = []
                coords['n'] = np.array([0], dtype=int)
                coords['rho'] = data.coords['xrho'].to_numpy().flatten()
                data_vars['nexp'] = (['n'], np.array([len(coords['rho'])], dtype=int))
                if 'te' in data:
                    data_vars['te'] = (['n', 'rho'], np.expand_dims(data['te'].to_numpy().flatten(), axis=0))
                if 'ti' in data:
                    data_vars['ti'] = (['n', 'rho'], np.expand_dims(data['ti'].to_numpy().flatten(), axis=0))
            attrs['header'] = newobj.make_file_header()
            newobj.input = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return newobj
