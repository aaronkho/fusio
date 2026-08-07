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

    def test_current_preserved(self, plasma_state, plasma_as_gacode):
        np.testing.assert_allclose(
            plasma_as_gacode.input['current'].to_numpy(),
            plasma_state.input['current'].to_numpy(),
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
