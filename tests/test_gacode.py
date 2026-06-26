import pytest
import numpy as np
import xarray as xr
from fusio.classes.io import io
from fusio.classes.gacode import gacode_io
from fusio.classes.plasma import plasma_io


@pytest.mark.usefixtures('gacode_file_path')
class TestInitialization():

    def test_empty_class_creation(self):
        assert isinstance(gacode_io(), gacode_io)

    def test_initialized_input_class_creation(self, gacode_file_path):
        g = gacode_io(input=gacode_file_path)
        assert isinstance(g, gacode_io)
        assert g.has_input

    def test_initialized_output_class_creation(self, gacode_file_path):
        g = gacode_io(output=gacode_file_path)
        assert isinstance(g, gacode_io)
        assert g.has_output

    def test_read_native_input(self, gacode_file_path):
        g = gacode_io()
        g.read(gacode_file_path, side='input')
        assert g.has_input

    def test_read_native_output(self, gacode_file_path):
        g = gacode_io()
        g.read(gacode_file_path, side='output')
        assert g.has_output


@pytest.fixture(scope='module')
def gacode_as_plasma(gacode_file_path):
    g = gacode_io(input=gacode_file_path)
    return plasma_io.from_gacode(g, side='input')

@pytest.fixture(scope='module')
def source_gacode(gacode_file_path):
    return gacode_io(input=gacode_file_path)


class TestGacodeToPlasmaConversion:

    def test_result_is_plasma_io(self, gacode_as_plasma):
        assert isinstance(gacode_as_plasma, plasma_io)

    def test_has_input_data(self, gacode_as_plasma):
        assert gacode_as_plasma.has_input

    def test_coordinates_present(self, gacode_as_plasma):
        data = gacode_as_plasma.input
        for coord in ('time', 'radius', 'ion', 'direction', 'source'):
            assert coord in data.coords, f"coord '{coord}' missing"

    def test_radius_coordinate_length(self, gacode_as_plasma):
        # test_input.gacode has nexp=70
        assert len(gacode_as_plasma.input.coords['radius']) == 70

    def test_ion_count(self, gacode_as_plasma):
        # test_input.gacode has nion=4
        assert len(gacode_as_plasma.input.coords['ion']) == 4

    def test_ion_names(self, gacode_as_plasma):
        assert list(gacode_as_plasma.input.coords['ion'].values) == ['T', 'D', 'He', 'Ne']

    def test_electron_fields_present(self, gacode_as_plasma):
        data = gacode_as_plasma.input
        for var in ('density_e', 'temperature_e', 'mass_e', 'charge_e'):
            assert var in data, f"var '{var}' missing"

    def test_ion_fields_present(self, gacode_as_plasma):
        data = gacode_as_plasma.input
        for var in ('density_i', 'temperature_i', 'charge_i', 'mass_i', 'atomic_number_i', 'velocity_i'):
            assert var in data, f"var '{var}' missing"

    def test_geometry_fields_present(self, gacode_as_plasma):
        data = gacode_as_plasma.input
        for var in ('r_minor', 'r_geometric', 'z_geometric', 'magnetic_flux', 'safety_factor', 'field_axis', 'contour'):
            assert var in data, f"var '{var}' missing"

    # safety_factor and magnetic_flux[poloidal] values are not verified against source q/polflux: add_safety_factor_profile interpolates q(polflux) at stored flux coordinates, which does not exactly reproduce the originals.

    def test_density_e_dimensions(self, gacode_as_plasma):
        assert gacode_as_plasma.input['density_e'].dims == ('time', 'radius')

    def test_density_i_dimensions(self, gacode_as_plasma):
        assert gacode_as_plasma.input['density_i'].dims == ('time', 'radius', 'ion')

    def test_temperature_i_dimensions(self, gacode_as_plasma):
        assert gacode_as_plasma.input['temperature_i'].dims == ('time', 'radius', 'ion')

    def test_charge_i_dimensions(self, gacode_as_plasma):
        assert gacode_as_plasma.input['charge_i'].dims == ('time', 'radius', 'ion')

    def test_velocity_i_dimensions(self, gacode_as_plasma):
        assert gacode_as_plasma.input['velocity_i'].dims == ('time', 'radius', 'ion', 'direction')

    def test_magnetic_flux_dimensions(self, gacode_as_plasma):
        assert gacode_as_plasma.input['magnetic_flux'].dims == ('time', 'radius', 'direction')

    def test_radius_coordinate_is_rho_tor_norm(self, gacode_as_plasma):
        assert gacode_as_plasma.input.attrs.get('radius') == 'rho_tor_norm'

    def test_density_e_unit_conversion(self, source_gacode, gacode_as_plasma):
        ne_gacode = source_gacode.input['ne'].to_numpy()   # 10^19 m^-3
        ne_plasma = gacode_as_plasma.input['density_e'].to_numpy()  # m^-3
        np.testing.assert_allclose(ne_plasma, 1.0e19 * ne_gacode, rtol=1e-10)

    def test_temperature_e_unit_conversion(self, source_gacode, gacode_as_plasma):
        te_gacode = source_gacode.input['te'].to_numpy()   # keV
        te_plasma = gacode_as_plasma.input['temperature_e'].to_numpy()  # eV
        np.testing.assert_allclose(te_plasma, 1.0e3 * te_gacode, rtol=1e-10)

    def test_density_i_unit_conversion(self, source_gacode, gacode_as_plasma):
        ni_gacode = source_gacode.input['ni'].to_numpy()   # 10^19 m^-3
        ni_plasma = gacode_as_plasma.input['density_i'].to_numpy()  # m^-3
        np.testing.assert_allclose(ni_plasma, 1.0e19 * ni_gacode, rtol=1e-10)

    def test_temperature_i_unit_conversion(self, source_gacode, gacode_as_plasma):
        ti_gacode = source_gacode.input['ti'].to_numpy()   # keV
        ti_plasma = gacode_as_plasma.input['temperature_i'].to_numpy()  # eV
        np.testing.assert_allclose(ti_plasma, 1.0e3 * ti_gacode, rtol=1e-10)

    def test_field_axis_preserved(self, source_gacode, gacode_as_plasma):
        np.testing.assert_allclose(
            gacode_as_plasma.input['field_axis'].to_numpy(),
            source_gacode.input['bcentr'].to_numpy(),
            rtol=1e-10,
        )

    def test_r_minor_preserved(self, source_gacode, gacode_as_plasma):
        np.testing.assert_allclose(
            gacode_as_plasma.input['r_minor'].to_numpy(),
            source_gacode.input['rmin'].to_numpy(),
            rtol=1e-10,
        )

    def test_r_geometric_preserved(self, source_gacode, gacode_as_plasma):
        np.testing.assert_allclose(
            gacode_as_plasma.input['r_geometric'].to_numpy(),
            source_gacode.input['rmaj'].to_numpy(),
            rtol=1e-10,
        )

    def test_density_e_positive(self, gacode_as_plasma):
        assert np.all(gacode_as_plasma.input['density_e'].to_numpy() > 0)

    def test_temperature_e_positive(self, gacode_as_plasma):
        assert np.all(gacode_as_plasma.input['temperature_e'].to_numpy() > 0)

    def test_density_i_positive(self, gacode_as_plasma):
        assert np.all(gacode_as_plasma.input['density_i'].to_numpy() > 0)

    def test_temperature_i_positive(self, gacode_as_plasma):
        assert np.all(gacode_as_plasma.input['temperature_i'].to_numpy() > 0)
