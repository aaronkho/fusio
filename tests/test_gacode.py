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


def _contour_volume(r, z, axis):
    """Enclosed volume of closed (R, Z) contours, V = pi * |closed integral of R^2 dZ|."""
    r, z = np.moveaxis(r, axis, -1), np.moveaxis(z, axis, -1)
    return np.pi * np.abs(np.sum(0.5 * (r[..., 1:] ** 2 + r[..., :-1] ** 2) * np.diff(z, axis=-1), axis=-1))


class TestDerivedGeometry:

    def test_volp_miller_matches_contour_volume(self, gacode_file_path):
        # dV/dr from the Miller metric (uses the radial derivatives drmajdr, dzmagdr, s_*) must match
        # the radial derivative of the volume enclosed by the reconstructed contours
        from fusio.utils.math_tools import vectorized_numpy_derivative
        g = gacode_io(input=gacode_file_path)
        # compute_derived_quantities fails later on this file (nu_ni), so run the geometry steps only
        g._compute_derived_coordinates(side='input')
        g._compute_derived_reference_quantities(side='input')
        g._compute_derived_geometry(side='input')
        d = g.input
        vol = _contour_volume(d['r_surface'].to_numpy(), d['z_surface'].to_numpy(), axis=0)
        dvdr = vectorized_numpy_derivative(d['rmin'].to_numpy(), vol)
        roa = d['roa'].to_numpy()
        mask = (roa > 0.1) & (roa < 0.95)
        err = np.abs(d['volp_miller'].to_numpy() / dvdr - 1.0)[mask]
        assert np.median(err) < 1.0e-3
        assert np.max(err) < 1.0e-2

    def test_aspect_local_is_extrapolated_only_on_the_axis(self, gacode_file_path):
        g = gacode_io(input=gacode_file_path)
        g._compute_derived_coordinates(side='input')
        on_axis = g.input['aspect_local'].to_numpy()
        assert np.all(np.isfinite(on_axis))
        # A grid which starts off the axis keeps its own innermost value
        h = gacode_io(input=gacode_file_path)
        h.input = h.input.isel(rho=slice(1, None))
        h._compute_derived_coordinates(side='input')
        off_axis = h.input['aspect_local'].to_numpy()
        expected = (h.input['rmaj'] / h.input['rmin']).to_numpy()
        np.testing.assert_allclose(off_axis[..., 0], expected[..., 0])



class TestWrite:

    @pytest.mark.parametrize('key, expected', [
        ('ne', True),
        ('omega0', True),
        ('shape_cos0', True),
        ('shape_cos6', True),
        ('shape_sin3', True),
        ('shape_sin6', True),
        ('shape_sin0', False),
        ('shape_sin1', False),
        ('shape_sin2', False),
        ('s_shape_sin1', False),
        ('shape_cos', False),
        ('volp_miller', False),
    ])
    def test_base_var_recognition(self, key, expected):
        assert gacode_io().is_base_var(key) == expected

    def test_round_trip_preserves_file_fields(self, tmp_path, gacode_file_path):
        g = gacode_io(input=gacode_file_path)
        g.write(tmp_path / 'input.gacode', side='input', overwrite=True)
        h = gacode_io(input=tmp_path / 'input.gacode')
        assert set(h.input.data_vars) == set(g.input.data_vars)
        assert set(h.input.coords) == set(g.input.coords)
        for key in g.input.data_vars:
            if g.input[key].dtype.kind == 'f':
                np.testing.assert_allclose(h.input[key].to_numpy(), g.input[key].to_numpy(), rtol=1.0e-6)

    def test_derived_quantities_are_kept_but_not_written(self, tmp_path, gacode_file_path):
        g = gacode_io(input=gacode_file_path)
        g.compute_derived_quantities(side='input')
        g.write(tmp_path / 'input.gacode', side='input', overwrite=True)
        h = gacode_io(input=tmp_path / 'input.gacode')
        written = [f'{key}' for key in h.input.data_vars] + [f'{key}' for key in h.input.coords if key != 'n']
        assert all(g.is_base_var(key) for key in written)
        assert set(written) == set([f'{key}' for key in gacode_io(input=gacode_file_path).input.variables if key != 'n'])
        # The internal fields, including the sine moments carried by delta and zeta, remain available
        for key in ['shape_sin0', 'shape_sin1', 'shape_sin2', 's_shape_sin1', 'volp_miller']:
            assert key in g.input



class TestLumpedMainIons:

    def test_mixture_name_and_mass_survive_plasma_round_trip(self, gacode_file_path):
        g = gacode_io(input=gacode_file_path)
        data = g.input
        names = [f'{name}' for name in data['name'].to_numpy()]
        masses = data['mass'].to_numpy().copy()
        masses[:, names.index('D')] = 2.4
        names[names.index('D')] = 'DT'
        g.input = data.assign_coords(name=names).assign(mass=(('n', 'name'), masses))
        p = plasma_io.from_gacode(g, side='input')
        np.testing.assert_allclose(p.input['mass_i'].sel(ion='DT').to_numpy(), 2.4)
        np.testing.assert_allclose(p.input['atomic_number_i'].sel(ion='DT').to_numpy(), 1.0)
        h = gacode_io.from_plasma(p, side='input').input
        assert 'DT' in [f'{name}' for name in h['name'].to_numpy()]
        np.testing.assert_allclose(h['mass'].sel(name='DT').to_numpy(), 2.4)


class TestFixtureSelfConsistency:

    # rho is the normalized toroidal flux coordinate, so Phi(rho) = torfluxa * rho^2 and
    # q = dPhi/dpolflux. The fixture's torfluxa was once 277x too small, which left
    # gacode_io's b_unit, and everything normalized by it, wrong by the same factor,
    # while plasma_io was unaffected because from_gacode rebuilds the toroidal flux
    # from q and polflux. Only this identity catches that.
    def test_safety_factor_follows_from_torfluxa_and_polflux(self, gacode_file_path):
        from fusio.utils.math_tools import vectorized_numpy_derivative
        d = gacode_io(input=gacode_file_path).input
        rho = d['rho'].to_numpy()
        phi = np.ravel(d['torfluxa'].to_numpy())[-1] * rho ** 2
        q_implied = vectorized_numpy_derivative(d['polflux'].to_numpy(), phi[None, :])[0]
        q = d['q'].to_numpy()[0]
        # Skipping the innermost points, where the finite difference is least accurate
        np.testing.assert_allclose(q_implied[5:], q[5:], rtol=2.0e-3)

    def test_toroidal_flux_matches_the_field_through_the_cross_section(self, gacode_file_path):
        d = gacode_io(input=gacode_file_path).input
        a = d['rmin'].to_numpy()[0, -1]
        area = np.pi * a ** 2 * d['kappa'].to_numpy()[0, -1]
        flux_estimate = abs(float(d['bcentr'].to_numpy()[0])) * area
        # A flux-surface-averaged 1/R estimate, so only good to order ten percent
        assert 0.5 < abs(float(d['torfluxa'].to_numpy()[0])) / flux_estimate < 2.0
