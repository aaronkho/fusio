"""Roundtrip test: EQDSK -> TORAX vs EQDSK -> GACODE -> IMAS -> TORAX.

Compares StandardGeometryIntermediates fields from two paths:
  1. Direct:    EQDSK -> TORAX (_construct_intermediates_from_eqdsk)
  2. Roundtrip: EQDSK -> fusio gacode_io -> fusio imas_io.from_gacode
                      -> netCDF -> TORAX (geometry_from_IMAS)

The roundtrip path involves a Miller parameterisation approximation in
the GACODE representation, so exact agreement is not expected.

How to run this test
--------------------
This test requires ``torax`` to be installed. If ``torax`` is not found,
the test is automatically skipped.

.. code-block:: bash

   # In a virtualenv with fusio (editable) + torax installed:
   pip install -e /path/to/fusio[test]
   pip install torax

   # Set EQDSK_PATH to the iterhybrid EQDSK file:
   export EQDSK_PATH=/path/to/iterhybrid_cocos11.eqdsk

   # Run:
   python -m pytest /path/to/fusio/tests/test_roundtrip_eqdsk_gacode_imas.py -v
"""

import dataclasses
import os
import shutil
import tempfile

import numpy as np
import pytest
from scipy import interpolate
import xarray as xr

from fusio.classes.gacode import gacode_io
from fusio.classes.imas import imas_io
from fusio.utils.eqdsk_tools import read_eqdsk

try:
    from torax._src.geometry import eqdsk as torax_eqdsk
    from torax._src.geometry import geometry
    from torax._src.geometry import standard_geometry
    from torax._src.imas_tools.input import equilibrium as imas_geometry
    from torax._src.geometry import imas as torax_imas

    _HAS_TORAX = True
except ImportError:
    _HAS_TORAX = False

pytestmark = pytest.mark.skipif(
    not _HAS_TORAX, reason='torax is not installed'
)

_SKIP_FIELDS = frozenset({
    'geometry_type',
    'Ip_from_parameters',
    'face_centers',
    'hires_factor',
    'diverted',
    'connection_length_target',
    'connection_length_divertor',
    'angle_of_incidence_target',
    'R_OMP',
    'R_target',
    'B_pol_OMP',
    'z_magnetic_axis',
    'delta_upper_face',
    'delta_lower_face',
    'flux_surf_avg_grad_psi2',
    'flux_surf_avg_grad_psi2_over_R2',
})

_ABS_FIELDS = frozenset({
    'psi',
    'Ip_profile',
})

_TRIM_FIELDS = frozenset({
    'psi',
    'vpr',
    'int_dl_over_Bp',
    'Ip_profile',
    'flux_surf_avg_grad_psi',
})

_TRIM_BOUNDARY_FIELDS = frozenset({
    'psi',
})

_TIGHT_TOL = 0.05
_LOOSE_TOL = 0.40

_LOOSE_FIELDS = frozenset({
    'psi',
    'vpr',
    'int_dl_over_Bp',
    'flux_surf_avg_grad_psi',
    'Ip_profile',
    'flux_surf_avg_1_over_B2',
    'flux_surf_avg_B2',
    'flux_surf_avg_1_over_R',
    'flux_surf_avg_1_over_R2',
})

_N_TRIM = 3


def _get_eqdsk_path(filename: str = 'iterhybrid_cocos11.eqdsk') -> str:
    if 'EQDSK_PATH' in os.environ:
        return os.environ['EQDSK_PATH']
    import torax
    geo_dir = os.path.join(os.path.dirname(torax.__file__), 'data', 'geo')
    return os.path.join(geo_dir, filename)


def _rhon_from_intermediates(intermediates):
    Phi = np.asarray(intermediates.Phi, dtype=float)
    return np.sqrt(Phi / Phi[-1])


def _build_intermediates_from_eqdsk(eqdsk_path: str, cocos: int = 11):
    config = torax_eqdsk.EQDSKConfig(
        geometry_file=os.path.basename(eqdsk_path),
        geometry_directory=os.path.dirname(eqdsk_path),
        cocos=cocos,
        Ip_from_parameters=False,
    )
    intermediates = torax_eqdsk._construct_intermediates_from_eqdsk(
        geometry_directory=config.geometry_directory,
        geometry_file=config.geometry_file,
        Ip_from_parameters=config.Ip_from_parameters,
        face_centers=config.get_face_centers(),
        hires_factor=config.hires_factor,
        cocos=config.cocos,
        n_surfaces=config.n_surfaces,
        last_surface_factor=config.last_surface_factor,
    )
    return intermediates


def _gacode_from_eqdsk(eqdsk_path: str) -> gacode_io:
    eqdsk_data = read_eqdsk(eqdsk_path)
    n_rho = eqdsk_data['nr']
    psi_axis = eqdsk_data['simagx']
    psi_bdry = eqdsk_data['sibdry']
    polflux = np.linspace(psi_axis, psi_bdry, n_rho)
    q = eqdsk_data['qpsi']
    rcentr = eqdsk_data['rcentr']
    bcentr = eqdsk_data['bcentr']
    current_MA = eqdsk_data['cpasma'] / 1.0e6
    psi_eqdsk = np.linspace(psi_axis, psi_bdry, n_rho)
    torfluxa_val = np.trapezoid(q, psi_eqdsk) / (2.0 * np.pi)
    rho = np.linspace(0.0, 1.0, n_rho)
    ds = xr.Dataset(
        data_vars={
            'polflux': (['n', 'rho'], np.expand_dims(polflux, axis=0)),
            'q': (['n', 'rho'], np.expand_dims(q, axis=0)),
            'rcentr': (['n'], np.array([rcentr])),
            'bcentr': (['n'], np.array([bcentr])),
            'current': (['n'], np.array([current_MA])),
            'torfluxa': (['n'], np.array([np.abs(torfluxa_val)])),
        },
        coords={
            'n': np.arange(1),
            'rho': rho,
        },
    )
    gc = gacode_io()
    gc.input = ds
    gc.add_geometry_from_eqdsk(eqdsk_path, side='input', overwrite=True)
    gc._compute_derived_coordinates()
    gc._compute_derived_reference_quantities()
    gc._compute_derived_geometry()
    return gc


def _build_intermediates_from_roundtrip(eqdsk_path: str, cocos: int = 11):
    gc = _gacode_from_eqdsk(eqdsk_path)
    imas_obj = imas_io.from_gacode(gc, side='input', time=0.0)
    tmp_dir = tempfile.mkdtemp()
    try:
        imas_obj.write(tmp_dir, side='input', overwrite=True)
        nc_path = os.path.join(tmp_dir, 'equilibrium.nc')
        if not os.path.exists(nc_path):
            raise FileNotFoundError(
                f'Expected {nc_path} to be written by imas_io.write, but it '
                f'does not exist. Contents of {tmp_dir}: '
                f'{os.listdir(tmp_dir)}'
            )
        config = torax_imas.IMASConfig(
            imas_filepath='equilibrium.nc',
            geometry_directory=tmp_dir,
            Ip_from_parameters=False,
            explicit_convert=False,
        )
        inputs = imas_geometry.geometry_from_IMAS(
            geometry_directory=config.geometry_directory,
            imas_filepath=config.imas_filepath,
            Ip_from_parameters=config.Ip_from_parameters,
            face_centers=config.get_face_centers(),
            hires_factor=config.hires_factor,
            explicit_convert=config.explicit_convert,
        )
        intermediates = standard_geometry.StandardGeometryIntermediates(
            geometry_type=geometry.GeometryType.IMAS, **inputs
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return intermediates


def _compare(direct, roundtrip) -> dict[str, tuple[float, float]]:
    """Return {field_name: (max_rel_error, mean_rel_error)}.

    Sign-convention-sensitive fields (``psi``, ``Ip_profile``) are
    compared by absolute value.  Fields that are zero on-axis are trimmed
    to exclude the first ``_N_TRIM`` radial points where the relative
    error is dominated by the near-zero denominator.
    """
    rhon_direct = _rhon_from_intermediates(direct)
    rhon_roundtrip = _rhon_from_intermediates(roundtrip)
    results = {}
    for field in dataclasses.fields(direct):
        name = field.name
        if name in _SKIP_FIELDS:
            continue
        val_direct = getattr(direct, name)
        val_roundtrip = getattr(roundtrip, name)
        if val_direct is None or val_roundtrip is None:
            continue
        val_direct = np.asarray(val_direct, dtype=float)
        val_roundtrip = np.asarray(val_roundtrip, dtype=float)
        if val_direct.size == 0:
            continue

        if name in _ABS_FIELDS:
            val_direct = np.abs(val_direct)
            val_roundtrip = np.abs(val_roundtrip)

        if val_direct.ndim == 1 and val_roundtrip.ndim == 1:
            if val_direct.shape != val_roundtrip.shape:
                interp_fn = interpolate.interp1d(
                    rhon_roundtrip,
                    val_roundtrip,
                    kind='cubic',
                    fill_value='extrapolate',
                )
                val_roundtrip = interp_fn(rhon_direct)

        if name in _TRIM_FIELDS and val_direct.ndim == 1:
            val_direct = val_direct[_N_TRIM:]
            val_roundtrip = val_roundtrip[_N_TRIM:]

        if name in _TRIM_BOUNDARY_FIELDS and val_direct.ndim == 1:
            val_direct = val_direct[:-_N_TRIM]
            val_roundtrip = val_roundtrip[:-_N_TRIM]

        denom = np.where(np.abs(val_direct) > 1e-30, val_direct, 1.0)
        rel_err_arr = np.abs((val_roundtrip - val_direct) / denom)
        results[name] = (float(np.max(rel_err_arr)), float(np.mean(rel_err_arr)))
    return results


class TestRoundtripEQDSKGacodeIMAS:

    @pytest.fixture(autouse=True, scope='class')
    def intermediates(self, request):
        eqdsk_path = _get_eqdsk_path()
        if not os.path.exists(eqdsk_path):
            pytest.skip(f'EQDSK file not found: {eqdsk_path}')
        request.cls.direct = _build_intermediates_from_eqdsk(eqdsk_path)
        request.cls.roundtrip = _build_intermediates_from_roundtrip(eqdsk_path)
        request.cls.comparison = _compare(
            request.cls.direct, request.cls.roundtrip
        )

    def test_all_fields_present(self):
        for field in dataclasses.fields(self.direct):
            if field.name in _SKIP_FIELDS:
                continue
            val_d = getattr(self.direct, field.name)
            val_r = getattr(self.roundtrip, field.name)
            if val_d is not None:
                assert val_r is not None, (
                    f'{field.name} is None in roundtrip but not in direct'
                )

    def test_scalars_match(self):
        for name in ('R_major', 'a_minor', 'B_0'):
            if name in self.comparison:
                max_err, _ = self.comparison[name]
                assert max_err < _TIGHT_TOL, (
                    f'{name}: max_rel_err={max_err:.4f} > {_TIGHT_TOL}'
                )

    def test_tight_fields(self):
        tight_fields = {
            k for k in self.comparison if k not in _LOOSE_FIELDS
        }
        for name in sorted(tight_fields):
            max_err, _ = self.comparison[name]
            assert max_err < _TIGHT_TOL, (
                f'{name}: max_rel_err={max_err:.4f} > {_TIGHT_TOL}'
            )

    def test_loose_fields(self):
        for name in sorted(_LOOSE_FIELDS):
            if name not in self.comparison:
                continue
            max_err, _ = self.comparison[name]
            assert max_err < _LOOSE_TOL, (
                f'{name}: max_rel_err={max_err:.4f} > {_LOOSE_TOL}'
            )

    def test_print_summary(self):
        for name, (max_err, mean_err) in sorted(self.comparison.items()):
            tol = _LOOSE_TOL if name in _LOOSE_FIELDS else _TIGHT_TOL
            status = '✓' if max_err < tol else '✗'
            print(
                f'  {status} {name:45s}  max={max_err:.6f}'
                f'  mean={mean_err:.6f}'
            )
        n_pass = sum(
            1
            for name, (max_err, _) in self.comparison.items()
            if max_err
            < (_LOOSE_TOL if name in _LOOSE_FIELDS else _TIGHT_TOL)
        )
        n_total = len(self.comparison)
        print(f'\n{n_pass}/{n_total} fields within tolerance')
