import pytest
import numpy as np
import xarray as xr
from fusio.classes.gacode import gacode_io
from fusio.classes.plasma import plasma_io


@pytest.fixture(scope='module')
def plasma_state(gacode_file_path):
    g = gacode_io(input=gacode_file_path)
    return plasma_io.from_gacode(g, side='input')


@pytest.fixture(scope='module')
def plasma_as_gacode(plasma_state):
    return gacode_io.from_plasma(plasma_state, side='input')


class TestPlasmaToGacodeConversion:

    def test_result_is_gacode_io(self, plasma_as_gacode):
        assert isinstance(plasma_as_gacode, gacode_io)

    def test_has_input_data(self, plasma_as_gacode):
        assert plasma_as_gacode.has_input

    def test_coordinates_present(self, plasma_as_gacode):
        data = plasma_as_gacode.input
        for coord in ('n', 'rho', 'name'):
            assert coord in data.coords, f"coord '{coord}' missing"

    def test_rho_coordinate_length(self, plasma_as_gacode):
        # plasma_state was built from test_input.gacode which has nexp=70
        assert len(plasma_as_gacode.input.coords['rho']) == 70

    def test_ion_count(self, plasma_as_gacode):
        # test_input.gacode has nion=4
        assert len(plasma_as_gacode.input.coords['name']) == 4

    def test_ion_names(self, plasma_as_gacode):
        assert list(plasma_as_gacode.input.coords['name'].values) == ['T', 'D', 'He', 'Ne']

    def test_nexp_value(self, plasma_as_gacode):
        assert int(plasma_as_gacode.input['nexp'].values[0]) == 70

    def test_nion_value(self, plasma_as_gacode):
        assert int(plasma_as_gacode.input['nion'].values[0]) == 4

    def test_electron_fields_present(self, plasma_as_gacode):
        data = plasma_as_gacode.input
        for var in ('ne', 'te', 'masse', 'ze'):
            assert var in data, f"var '{var}' missing"

    def test_ion_fields_present(self, plasma_as_gacode):
        data = plasma_as_gacode.input
        for var in ('ni', 'ti', 'mass', 'z', 'type'):
            assert var in data, f"var '{var}' missing"

    def test_geometry_fields_present(self, plasma_as_gacode):
        data = plasma_as_gacode.input
        for var in ('rmin', 'rmaj', 'zmag', 'rcentr', 'bcentr', 'current', 'torfluxa', 'polflux', 'q'):
            assert var in data, f"var '{var}' missing"

    def test_derived_fields_present(self, plasma_as_gacode):
        data = plasma_as_gacode.input
        for var in ('z_eff', 'ptot'):
            assert var in data, f"var '{var}' missing"

    def test_ne_dimensions(self, plasma_as_gacode):
        assert plasma_as_gacode.input['ne'].dims == ('n', 'rho')

    def test_te_dimensions(self, plasma_as_gacode):
        assert plasma_as_gacode.input['te'].dims == ('n', 'rho')

    def test_ni_dimensions(self, plasma_as_gacode):
        assert plasma_as_gacode.input['ni'].dims == ('n', 'rho', 'name')

    def test_ti_dimensions(self, plasma_as_gacode):
        assert plasma_as_gacode.input['ti'].dims == ('n', 'rho', 'name')

    def test_mass_dimensions(self, plasma_as_gacode):
        assert plasma_as_gacode.input['mass'].dims == ('n', 'name')

    def test_z_dimensions(self, plasma_as_gacode):
        assert plasma_as_gacode.input['z'].dims == ('n', 'name')

    def test_bcentr_dimensions(self, plasma_as_gacode):
        assert plasma_as_gacode.input['bcentr'].dims == ('n',)

    def test_polflux_dimensions(self, plasma_as_gacode):
        assert plasma_as_gacode.input['polflux'].dims == ('n', 'rho')

    def test_q_dimensions(self, plasma_as_gacode):
        assert plasma_as_gacode.input['q'].dims == ('n', 'rho')

    # q and polflux values are not compared against the original gacode source: both are transformed by add_safety_factor_profile during the gacode→plasma step and do not reproduce the originals exactly.

    def test_density_e_unit_conversion(self, plasma_state, plasma_as_gacode):
        ne_plasma = plasma_state.input['density_e'].to_numpy()   # m^-3
        ne_gacode = plasma_as_gacode.input['ne'].to_numpy()      # 10^19 m^-3
        np.testing.assert_allclose(ne_gacode, 1.0e-19 * ne_plasma, rtol=1e-10)

    def test_temperature_e_unit_conversion(self, plasma_state, plasma_as_gacode):
        te_plasma = plasma_state.input['temperature_e'].to_numpy()  # eV
        te_gacode = plasma_as_gacode.input['te'].to_numpy()         # keV
        np.testing.assert_allclose(te_gacode, 1.0e-3 * te_plasma, rtol=1e-10)

    def test_density_i_unit_conversion(self, plasma_state, plasma_as_gacode):
        ni_plasma = plasma_state.input['density_i'].to_numpy()   # m^-3
        ni_gacode = plasma_as_gacode.input['ni'].to_numpy()      # 10^19 m^-3
        np.testing.assert_allclose(ni_gacode, 1.0e-19 * ni_plasma, rtol=1e-10)

    def test_temperature_i_unit_conversion(self, plasma_state, plasma_as_gacode):
        ti_plasma = plasma_state.input['temperature_i'].to_numpy()  # eV
        ti_gacode = plasma_as_gacode.input['ti'].to_numpy()         # keV
        np.testing.assert_allclose(ti_gacode, 1.0e-3 * ti_plasma, rtol=1e-10)

    def test_field_axis_preserved(self, plasma_state, plasma_as_gacode):
        np.testing.assert_allclose(
            plasma_as_gacode.input['bcentr'].to_numpy(),
            plasma_state.input['field_axis'].to_numpy(),
            rtol=1e-10,
        )

    def test_mass_e_preserved(self, plasma_state, plasma_as_gacode):
        np.testing.assert_allclose(
            plasma_as_gacode.input['masse'].to_numpy(),
            plasma_state.input['mass_e'].to_numpy(),
            rtol=1e-10,
        )

    def test_charge_e_preserved(self, plasma_state, plasma_as_gacode):
        np.testing.assert_allclose(
            plasma_as_gacode.input['ze'].to_numpy(),
            plasma_state.input['charge_e'].to_numpy(),
            rtol=1e-10,
        )

    def test_current_unit_conversion(self, plasma_state, plasma_as_gacode):
        # plasma_io holds the current in A, GACODE in MA
        np.testing.assert_allclose(
            plasma_as_gacode.input['current'].to_numpy(),
            1.0e-6 * plasma_state.input['current'].to_numpy(),
            rtol=1e-10,
        )

    def test_r_minor_preserved(self, plasma_state, plasma_as_gacode):
        np.testing.assert_allclose(
            plasma_as_gacode.input['rmin'].to_numpy(),
            plasma_state.input['r_minor'].to_numpy(),
            rtol=1e-10,
        )

    def test_r_geometric_preserved(self, plasma_state, plasma_as_gacode):
        np.testing.assert_allclose(
            plasma_as_gacode.input['rmaj'].to_numpy(),
            plasma_state.input['r_geometric'].to_numpy(),
            rtol=1e-10,
        )

    def test_z_geometric_preserved(self, plasma_state, plasma_as_gacode):
        np.testing.assert_allclose(
            plasma_as_gacode.input['zmag'].to_numpy(),
            plasma_state.input['z_geometric'].to_numpy(),
            rtol=1e-10,
        )

    def test_torfluxa_from_magnetic_flux(self, plasma_state, plasma_as_gacode):
        expected = plasma_state.input['magnetic_flux'].isel(radius=-1).sel(direction='toroidal', drop=True).to_numpy()
        np.testing.assert_allclose(
            plasma_as_gacode.input['torfluxa'].to_numpy(),
            expected,
            rtol=1e-10,
        )

    def test_mass_i_preserved(self, plasma_state, plasma_as_gacode):
        np.testing.assert_allclose(
            plasma_as_gacode.input['mass'].to_numpy(),
            plasma_state.input['mass_i'].to_numpy(),
            rtol=1e-10,
        )

    def test_type_encoding(self, plasma_as_gacode):
        types = plasma_as_gacode.input['type'].to_numpy()
        assert np.all((types == '[therm]') | (types == '[fast]'))

    def test_ne_positive(self, plasma_as_gacode):
        assert np.all(plasma_as_gacode.input['ne'].to_numpy() > 0)

    def test_te_positive(self, plasma_as_gacode):
        assert np.all(plasma_as_gacode.input['te'].to_numpy() > 0)

    def test_ni_positive(self, plasma_as_gacode):
        assert np.all(plasma_as_gacode.input['ni'].to_numpy() > 0)

    def test_ti_positive(self, plasma_as_gacode):
        assert np.all(plasma_as_gacode.input['ti'].to_numpy() > 0)

    def test_ptot_positive(self, plasma_as_gacode):
        assert np.all(plasma_as_gacode.input['ptot'].to_numpy() > 0)

    def test_z_eff_not_less_than_one(self, plasma_as_gacode):
        assert np.all(plasma_as_gacode.input['z_eff'].to_numpy() >= 1.0)


class TestDerivedGeometry:

    def test_mxh_dvolume_dr_matches_contour_volume(self, gacode_file_path):
        # dV/dr from the MXH metric (uses mxh_dr0, mxh_dz0, mxh_s_*) must match the radial derivative
        # of the volume enclosed by the input contours
        from fusio.utils.math_tools import vectorized_numpy_derivative
        p = plasma_io.from_gacode(gacode_io(input=gacode_file_path), side='input')
        p.compute_derived_quantities(side='input')
        d = p.input
        r = d['contour'].sel(grid='r').to_numpy()
        z = d['contour'].sel(grid='z').to_numpy()
        vol = np.pi * np.abs(np.sum(0.5 * (r[..., 1:] ** 2 + r[..., :-1] ** 2) * np.diff(z, axis=-1), axis=-1))
        dvdr = vectorized_numpy_derivative(d['r_minor'].to_numpy(), vol)
        x = d['r_minor_norm'].to_numpy()
        mask = (x > 0.1) & (x < 0.95)
        err = np.abs(d['mxh_dvolume_dr'].to_numpy() / dvdr - 1.0)[mask]
        assert np.median(err) < 1.0e-3
        assert np.max(err) < 1.0e-2



def _plasma_with_flipped_signs(gacode_file_path, keys):
    g = gacode_io(input=gacode_file_path)
    d = g.input
    for k in keys:
        if k in d:
            d[k] = -d[k]
    g.input = d
    p = plasma_io.from_gacode(g, side='input')
    p.compute_derived_quantities(side='input')
    return p


@pytest.fixture(scope='module')
def derived_reference(gacode_file_path):
    return _plasma_with_flipped_signs(gacode_file_path, []).input


SIGN_FLIPS = {
    'all': ['polflux', 'q', 'current', 'bcentr', 'torfluxa'],
    'current': ['current', 'polflux'],
    'field': ['bcentr', 'torfluxa'],
    'safety_factor': ['q'],
    'poloidal_flux': ['polflux'],
}


class TestSignAgnostic:
    # plasma_io retains the input field signs, while every derived magnitude stays independent of them

    @pytest.mark.parametrize('keys', SIGN_FLIPS.values(), ids=SIGN_FLIPS.keys())
    def test_input_signs_retained(self, gacode_file_path, keys):
        g = gacode_io(input=gacode_file_path).input
        d = _plasma_with_flipped_signs(gacode_file_path, keys).input
        flipped = lambda k: -1.0 if k in keys else 1.0
        assert np.sign(d['magnetic_flux'].sel(direction='poloidal').isel(radius=-1)).item() == flipped('polflux') * np.sign(g['polflux'].isel(rho=-1)).item()
        assert np.sign(d['magnetic_flux'].sel(direction='toroidal').isel(radius=-1)).item() == flipped('torfluxa') * np.sign(g['torfluxa']).item()
        assert np.sign(d['safety_factor'].isel(radius=-1)).item() == flipped('q') * np.sign(g['q'].isel(rho=-1)).item()
        assert np.sign(d['field_axis']).item() == flipped('bcentr') * np.sign(g['bcentr']).item()
        assert np.sign(d['current']).item() == flipped('current') * np.sign(g['current']).item()

    @pytest.mark.parametrize('keys', SIGN_FLIPS.values(), ids=SIGN_FLIPS.keys())
    def test_derived_magnitudes_unchanged(self, gacode_file_path, derived_reference, keys):
        d = _plasma_with_flipped_signs(gacode_file_path, keys).input
        for var in derived_reference.data_vars:
            if derived_reference[var].dtype.kind != 'f':
                continue
            ref = np.abs(derived_reference[var].to_numpy())
            new = np.abs(d[var].to_numpy())
            np.testing.assert_array_equal(np.isfinite(new), np.isfinite(ref), err_msg=var)
            mask = np.isfinite(ref)
            np.testing.assert_allclose(new[mask], ref[mask], rtol=1.0e-9, atol=1.0e-12 * np.max(ref[mask], initial=0.0), err_msg=var)

    def test_normalized_flux_ignores_axis_offset(self, derived_reference):
        flux_norm = derived_reference['magnetic_flux_norm'].to_numpy()
        assert np.allclose(flux_norm[:, 0, :], 0.0)
        assert np.allclose(flux_norm[:, -1, :], 1.0)
        assert np.all(np.isfinite(derived_reference['rho_norm'].to_numpy()))


class TestCocos:
    # Convention implied by the signs of Ip, B0, the outward change in psi and q, assuming right-handed (R, phi, Z):
    # sigma_Bp = sign(dpsi) * sign(Ip) and sigma_rhothetaphi = sign(q) * sign(Ip) * sign(B0)
    COCOS_FROM_SIGMAS = {(1, 1): 1, (-1, -1): 3, (1, -1): 5, (-1, 1): 7}

    @pytest.mark.parametrize('keys', SIGN_FLIPS.values(), ids=SIGN_FLIPS.keys())
    def test_cocos_recorded_from_gacode_signs(self, gacode_file_path, keys):
        g = gacode_io(input=gacode_file_path).input
        flip = lambda k: -1 if k in keys else 1
        s_ip = flip('current') * np.sign(g['current'].isel(n=0)).item()
        s_bt = flip('bcentr') * np.sign(g['bcentr'].isel(n=0)).item()
        s_psi = flip('polflux') * np.sign((g['polflux'].isel(n=0, rho=-1) - g['polflux'].isel(n=0, rho=0))).item()
        s_q = flip('q') * np.sign(g['q'].isel(n=0, rho=-1)).item()
        expected = self.COCOS_FROM_SIGMAS[(int(s_psi * s_ip), int(s_q * s_ip * s_bt))]
        assert _plasma_with_flipped_signs(gacode_file_path, keys).input_cocos == expected

    @pytest.mark.parametrize('keys', SIGN_FLIPS.values(), ids=SIGN_FLIPS.keys())
    def test_toroidal_flux_follows_field_without_reference(self, gacode_file_path, keys):
        # Without torfluxa, Phi is rebuilt from q and psi with the COCOS sign, which must follow the sign of B0
        g = gacode_io(input=gacode_file_path)
        d = g.input
        for k in keys:
            d[k] = -d[k]
        g.input = d.drop_vars('torfluxa')
        p = plasma_io.from_gacode(g, side='input')
        phi = p.input['magnetic_flux'].sel(direction='toroidal').isel(time=0, radius=-1)
        assert np.sign(phi).item() == np.sign(p.input['field_axis'].isel(time=0)).item()

    def test_cocos_inferred_when_not_recorded(self, gacode_file_path, derived_reference):
        expected = _plasma_with_flipped_signs(gacode_file_path, []).input_cocos
        ds = derived_reference.copy()
        ds.attrs = {k: v for k, v in derived_reference.attrs.items() if k != 'cocos'}
        p = plasma_io()
        p.input = ds
        assert 'cocos' not in p.input.attrs
        assert p.input_cocos == expected

    @pytest.mark.parametrize('name, per_radian', [('sample_cocos02_input.geqdsk', True), ('sample_cocos11_input.geqdsk', False)])
    def test_psi_per_radian_detected_from_ampere_law(self, name, per_radian):
        from pathlib import Path
        from fusio.utils.eqdsk_tools import read_eqdsk, detect_psi_per_radian
        assert detect_psi_per_radian(read_eqdsk(Path(__file__).parent / 'data' / name)) is per_radian

    def test_current_in_amps_from_gacode(self, gacode_file_path):
        g = gacode_io(input=gacode_file_path)
        p = plasma_io.from_gacode(g, side='input')
        np.testing.assert_allclose(p.input['current'].to_numpy(), 1.0e6 * g.input['current'].to_numpy(), rtol=1e-12)
        back = p.to('gacode', side='input')
        np.testing.assert_allclose(back.input['current'].to_numpy(), g.input['current'].to_numpy(), rtol=1e-12)

    def test_eqdsk_insertion_independent_of_source_cocos(self, gacode_file_path):
        # The same equilibrium given per radian (COCOS 2) and in Wb (COCOS 11) must produce the same plasma state
        states = []
        for name in ['sample_cocos02_input.geqdsk', 'sample_cocos11_input.geqdsk']:
            p = plasma_io.from_gacode(gacode_io(input=gacode_file_path), side='input')
            p.add_geometry_from_eqdsk(gacode_file_path.parent / name, side='input')
            states.append(p.input)
        for var in ['magnetic_flux', 'safety_factor', 'contour', 'r_minor']:
            np.testing.assert_allclose(states[1][var].to_numpy(), states[0][var].to_numpy(), rtol=1.0e-8, atol=1.0e-8 * float(np.max(np.abs(states[0][var]))), err_msg=var)
