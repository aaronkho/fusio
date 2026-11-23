import pytest
import numpy as np
import xarray as xr
from fusio.classes.gacode import gacode_io
from fusio.classes.omas import omas_io
from fusio.classes.plasma import plasma_io
from fusio.classes.torax import torax_io, add_legacy_output_names, TORAX_V2_OUTPUT_RENAMES


def _gacode_with_renamed_ion(gacode_file_path, old, new, mass):
    g = gacode_io(input=gacode_file_path)
    data = g.input
    names = [f'{name}' for name in data['name'].to_numpy()]
    masses = data['mass'].to_numpy().copy()
    masses[:, names.index(old)] = mass
    names[names.index(old)] = new
    g.input = data.assign_coords(name=names).assign(mass=(('n', 'name'), masses))
    return g


def _main_ion_fractions(t):
    return {f'{name}': float(val) for name, val in zip(t['main_ion'].to_numpy(), t['plasma_composition.main_ion'].to_numpy()[:, 0])}


def _omas_state(labels, masses, charges, fractions):
    ti = 'core_profiles.profiles_1d:i'
    ri = 'core_profiles.profiles_1d.grid.rho_tor_norm:i'
    ii = 'core_profiles.profiles_1d.ion:i'
    ei = 'core_profiles.profiles_1d.ion.element:i'
    rho = np.linspace(0.0, 1.0, 5)
    ne = np.full((1, len(rho)), 1.0e20)
    data = xr.Dataset(
        coords={ti: np.arange(1), ri: np.arange(len(rho)), ii: np.arange(len(labels)), ei: np.arange(1)},
        data_vars={
            'core_profiles.time': ([ti], np.array([1.0])),
            'core_profiles.profiles_1d.grid.rho_tor_norm': ([ti, ri], np.atleast_2d(rho)),
            'core_profiles.profiles_1d.ion.label': ([ti, ii], np.atleast_2d(labels)),
            'core_profiles.profiles_1d.ion.density': ([ti, ii, ri], np.stack([f * ne for f in fractions], axis=1)),
            'core_profiles.profiles_1d.ion.element.a': ([ti, ii, ei], np.array(masses, dtype=float).reshape(1, -1, 1)),
            'core_profiles.profiles_1d.ion.element.z_n': ([ti, ii, ei], np.array(charges, dtype=float).reshape(1, -1, 1)),
            'core_profiles.profiles_1d.electrons.density': ([ti, ri], ne),
            'core_profiles.profiles_1d.electrons.temperature': ([ti, ri], np.full((1, len(rho)), 5.0e3)),
        },
    )
    o = omas_io()
    o.input = data
    return o


class TestLumpedMainIons:

    def test_gacode_mixture_is_expanded_into_isotopes(self, gacode_file_path):
        reference = _main_ion_fractions(torax_io.from_gacode(gacode_io(input=gacode_file_path), side='input').input)
        g = _gacode_with_renamed_ion(gacode_file_path, 'D', 'DT', 2.4)
        t = torax_io.from_gacode(g, side='input').input
        fractions = _main_ion_fractions(t)
        assert set(fractions) == {'D', 'T'}
        # The former D share is now split 60 / 40 between D and T by the mixture mass
        np.testing.assert_allclose(fractions['D'], 0.6 * reference['D'])
        np.testing.assert_allclose(fractions['T'], reference['T'] + 0.4 * reference['D'])
        assert 'DT' not in [f'{name}' for name in t['impurity'].to_numpy()]

    def test_gacode_hydrogen_deuterium_mixture(self, gacode_file_path):
        g = _gacode_with_renamed_ion(gacode_file_path, 'T', 'HD', 1.5)
        fractions = _main_ion_fractions(torax_io.from_gacode(g, side='input').input)
        assert set(fractions) == {'H', 'D'}
        np.testing.assert_allclose(sum(fractions.values()), 1.0)

    def test_omas_mixture_is_expanded_into_isotopes(self):
        o = _omas_state(['DT', 'T', 'Ne'], [2.4, 3.0, 20.0], [1.0, 1.0, 10.0], [0.5, 0.4, 0.01])
        t = torax_io.from_omas(o, side='input').input
        fractions = _main_ion_fractions(t)
        np.testing.assert_allclose(fractions['D'], 0.6 * 0.5 / 0.9)
        np.testing.assert_allclose(fractions['T'], (0.4 * 0.5 + 0.4) / 0.9)
        assert [f'{name}' for name in t['impurity'].to_numpy()] == ['Ne']


def _torax_v1_output(ntime=2, ncell=10):
    """Small synthetic TORAX v1 output, with C and W impurities bundled into n_impurity."""
    face = np.linspace(0.0, 1.0, ncell + 1)
    cell = 0.5 * (face[1:] + face[:-1])
    rho = np.concatenate([[0.0], cell, [1.0]])
    time = np.linspace(1.0, 2.0, ntime)
    one = np.ones((ntime, 1))

    def prof(core, edge, grid):
        return one * (edge + (core - edge) * (1.0 - grid ** 2))

    symbols = ['C', 'W']
    zs = np.stack([np.full((ntime, ncell), 6.0), prof(60.0, 40.0, cell)])
    ns = np.stack([prof(2.0e18, 1.0e18, cell), prof(1.0e15, 5.0e14, cell)])
    s1 = (ns * zs).sum(axis=0)
    s2 = (ns * zs ** 2).sum(axis=0)
    zbundled = np.array([np.interp(rho, cell, s2[i] / s1[i]) for i in range(ntime)])
    nbundled = np.array([np.interp(rho, cell, s1[i] ** 2 / s2[i]) for i in range(ntime)])
    a_minor = np.full(ntime, 1.0)
    ne = prof(1.0e20, 3.0e19, rho)
    data = xr.Dataset(
        coords={'time': time, 'rho_norm': rho, 'rho_face_norm': face, 'rho_cell_norm': cell, 'impurity_symbol': symbols},
        data_vars={
            'R_major': (['time'], np.full(ntime, 3.0)),
            'a_minor': (['time'], a_minor),
            'B_0': (['time'], np.full(ntime, 5.0)),
            'z_magnetic_axis': (['time'], np.zeros(ntime)),
            'R_out': (['time', 'rho_norm'], 3.0 + one * rho),
            'R_in': (['time', 'rho_norm'], 3.0 - one * rho),
            'elongation': (['time', 'rho_norm'], prof(1.2, 1.6, rho)),
            'delta': (['time', 'rho_face_norm'], prof(0.0, 0.3, face)),
            'psi': (['time', 'rho_norm'], one * rho ** 2),
            'psi_norm': (['time', 'rho_face_norm'], one * face ** 2),
            'q': (['time', 'rho_face_norm'], prof(1.0, 4.0, face)),
            'magnetic_shear': (['time', 'rho_face_norm'], one * 2.0 * face ** 2),
            'n_e': (['time', 'rho_norm'], ne),
            'T_e': (['time', 'rho_norm'], prof(10.0, 0.5, rho)),
            'T_i': (['time', 'rho_norm'], prof(9.0, 0.5, rho)),
            'n_i': (['time', 'rho_norm'], 0.8 * ne),
            'A_i': (['time'], np.full(ntime, 2.5)),
            'Z_i': (['time', 'rho_norm'], one * np.ones_like(rho)),
            'A_impurity': (['time', 'rho_norm'], one * np.full_like(rho, 20.0)),
            'Z_eff': (['time', 'rho_norm'], one * np.full_like(rho, 1.5)),
            'n_impurity': (['time', 'rho_norm'], nbundled),
            'Z_impurity': (['time', 'rho_norm'], zbundled),
            'n_impurity_species': (['impurity_symbol', 'time', 'rho_cell_norm'], ns),
            'Z_impurity_species': (['impurity_symbol', 'time', 'rho_cell_norm'], zs),
            'Ip': (['time'], np.full(ntime, 1.0e7)),
            'j_bootstrap': (['time', 'rho_norm'], prof(0.0, 1.0e5, rho)),
        },
    )
    return data


def _torax_v2_output(v1):
    """Rename a v1 output into the TORAX v2 layout, adding the plain impurity sum and average."""
    ns = v1['n_impurity_species']
    zs = v1['Z_impurity_species']
    total = ns.sum('impurity_symbol').interp(rho_cell_norm=v1['rho_norm'].to_numpy(), kwargs={'fill_value': 'extrapolate'}).rename({'rho_cell_norm': 'rho_norm'})
    average = ((ns * zs).sum('impurity_symbol') / ns.sum('impurity_symbol')).interp(rho_cell_norm=v1['rho_norm'].to_numpy(), kwargs={'fill_value': 'extrapolate'}).rename({'rho_cell_norm': 'rho_norm'})
    v2 = v1.rename({'n_impurity': 'n_impurity_bundled', 'Z_impurity': 'Z_impurity_bundled', 'Ip': 'I_p', 'j_bootstrap': 'j_toroidal_bootstrap'})
    return v2.assign(n_impurity_total=total.drop_vars('rho_norm'), Z_impurity_average=average.drop_vars('rho_norm'))


def _torax_with_output(data):
    t = torax_io()
    t.output = data
    return t


class TestLegacyOutputNames:

    def test_v2_names_are_aliased_onto_v1_names(self):
        v2 = _torax_v2_output(_torax_v1_output())
        aliased = add_legacy_output_names(v2)
        xr.testing.assert_equal(aliased['n_impurity'], v2['n_impurity_bundled'].rename('n_impurity'))
        xr.testing.assert_equal(aliased['Z_impurity'], v2['Z_impurity_bundled'].rename('Z_impurity'))
        xr.testing.assert_equal(aliased['Ip'], v2['I_p'].rename('Ip'))
        assert 'n_impurity' not in v2

    def test_v1_output_is_left_as_is_apart_from_the_total_impurity(self):
        v1 = _torax_v1_output()
        aliased = add_legacy_output_names(v1)
        assert set(aliased.data_vars) - set(v1.data_vars) == {'n_impurity_total'}
        xr.testing.assert_equal(aliased['n_impurity_total'], v1['n_impurity'].rename('n_impurity_total'))

    def test_v2_total_impurity_is_kept(self):
        v2 = _torax_v2_output(_torax_v1_output())
        xr.testing.assert_equal(add_legacy_output_names(v2)['n_impurity_total'], v2['n_impurity_total'])

    def test_rename_targets_are_unique(self):
        assert len(set(TORAX_V2_OUTPUT_RENAMES.values())) == len(TORAX_V2_OUTPUT_RENAMES)


class TestParametersAcrossVersions:

    @pytest.mark.parametrize('code', ['tglf', 'cgyro', 'qualikiz'])
    @pytest.mark.parametrize('impurity_kwargs', [{}, {'lump_impurities': True}, {'full_impurities': True}])
    def test_v1_and_v2_give_the_same_parameters(self, code, impurity_kwargs):
        v1 = _torax_v1_output()
        v2 = _torax_v2_output(v1)
        # Pressure takes the total impurity density, which v1 approximated by the bundled one
        v2 = v2.assign(n_impurity_total=v2['n_impurity_bundled'])
        p1 = getattr(_torax_with_output(v1), f'to_{code}_parameters')(**impurity_kwargs)
        p2 = getattr(_torax_with_output(v2), f'to_{code}_parameters')(**impurity_kwargs)
        assert len(p1.data_vars) > 0
        xr.testing.assert_allclose(p1, p2)

    def test_v2_lumped_impurity_is_the_bundled_one(self):
        v2 = _torax_v2_output(_torax_v1_output())
        params = _torax_with_output(v2).to_tglf_parameters(rho=[0.5])
        expected = v2['Z_impurity_bundled'].interp(rho_norm=params['rho'].to_numpy()).to_numpy()
        np.testing.assert_allclose(params['ZS_3'].to_numpy(), expected)

    def test_v2_pressure_uses_the_total_impurity_density(self):
        v2 = _torax_v2_output(_torax_v1_output())
        bundled = v2.assign(n_impurity_total=v2['n_impurity_bundled'])
        alpha_total = _torax_with_output(v2).to_qualikiz_parameters()['alpha']
        alpha_bundled = _torax_with_output(bundled).to_qualikiz_parameters()['alpha']
        assert not np.allclose(alpha_total.to_numpy()[:, 1:], alpha_bundled.to_numpy()[:, 1:])


class TestFastIonsFromOmas:

    def test_fast_mixture_is_not_a_main_ion(self):
        o = _omas_state(['D', 'T', 'DT_fast', 'Ne'], [2.0, 3.0, 2.5, 20.0], [1.0, 1.0, 1.0, 10.0], [0.6, 0.3, 0.05, 0.01])
        t = torax_io.from_omas(o, side='input').input
        fractions = _main_ion_fractions(t)
        np.testing.assert_allclose(fractions['D'], 0.6 / 0.9)
        np.testing.assert_allclose(fractions['T'], 0.3 / 0.9)
        assert [f'{name}' for name in t['impurity'].to_numpy()] == ['Ne']

    def test_fast_isotope_is_not_an_impurity(self):
        o = _omas_state(['D', 'Ne', 'D_fast'], [2.0, 20.0, 2.0], [1.0, 10.0, 1.0], [0.9, 0.01, 0.05])
        t = torax_io.from_omas(o, side='input').input
        assert set(_main_ion_fractions(t)) == {'D'}
        assert [f'{name}' for name in t['impurity'].to_numpy()] == ['Ne']

    def test_species_without_thermal_density_is_fast(self):
        o = _omas_state(['D', 'T', 'Ne'], [2.0, 3.0, 20.0], [1.0, 1.0, 10.0], [0.6, 0.3, 0.01])
        thermal = o.input['core_profiles.profiles_1d.ion.density'].copy()
        thermal[:, 1, :] = 0.0
        o.input = o.input.assign({'core_profiles.profiles_1d.ion.density_thermal': thermal})
        fractions = _main_ion_fractions(torax_io.from_omas(o, side='input').input)
        assert set(fractions) == {'D'}

    def test_each_impurity_keeps_its_own_name(self):
        o = _omas_state(['D', 'Ne', 'W'], [2.0, 20.0, 184.0], [1.0, 10.0, 74.0], [0.9, 0.01, 1.0e-5])
        t = torax_io.from_omas(o, side='input').input
        assert [f'{name}' for name in t['impurity'].to_numpy()] == ['Ne', 'W']


class TestPlasmaFromToraxVersions:

    @pytest.mark.parametrize('version', ['v1', 'v2'])
    def test_impurities_take_their_own_charge(self, version):
        v1 = _torax_v1_output()
        data = v1 if version == 'v1' else _torax_v2_output(v1)
        p = plasma_io.from_torax(_torax_with_output(data), side='output').input
        charge = p['charge_i'].isel(time=-1)
        np.testing.assert_allclose(charge.sel(ion='C').to_numpy(), 6.0)
        expected = v1['Z_impurity_species'].sel(impurity_symbol='W').isel(time=-1).interp(rho_cell_norm=p['radius'].to_numpy(), kwargs={'fill_value': 'extrapolate'})
        np.testing.assert_allclose(charge.sel(ion='W').to_numpy(), expected.to_numpy())

    def test_v2_plasma_current_is_read(self):
        v2 = _torax_v2_output(_torax_v1_output())
        p = plasma_io.from_torax(_torax_with_output(v2), side='output').input
        np.testing.assert_allclose(p['current'].to_numpy(), v2['I_p'].to_numpy())

    def test_gacode_from_torax_falls_back_to_zero_zmag(self):
        v2 = _torax_v2_output(_torax_v1_output()).drop_vars('z_magnetic_axis')
        v2 = v2.assign(toroidal_angular_velocity=(['time', 'rho_norm'], np.full_like(v2['T_e'].to_numpy(), 1.0e4)))
        v2.attrs['config'] = {'plasma_composition': {'main_ion': {'D': {'value': ['float', [1.0]]}}}}
        gc = gacode_io.from_torax(_torax_with_output(v2), side='output').input
        assert 'zmag' in gc
        assert 'zeta' in gc
        assert 'shape_cos0' in gc
        assert 'omega0' in gc
        np.testing.assert_allclose(gc['zmag'].to_numpy(), 0.0)
        np.testing.assert_allclose(gc['zeta'].to_numpy(), 0.0)
        np.testing.assert_allclose(gc['omega0'].to_numpy(), 1.0e4)


class TestImpuritiesFromGacode:

    def test_impurities_collapsed_onto_one_species_are_summed(self, gacode_file_path):
        reference = torax_io.from_gacode(gacode_io(input=gacode_file_path), side='input').input
        g = _gacode_with_renamed_ion(gacode_file_path, 'He', 'Na', 23.0)
        t = torax_io.from_gacode(g, side='input').input
        assert [f'{name}' for name in t['impurity'].to_numpy()] == ['Ne']
        # Na is radiated as Ne, with its density rescaled to keep the charge it carries
        data = g.input
        nna = (data['ni'].sel(name='Na') / data['ne']).to_numpy().flatten() * 11.0 / 10.0
        nne = reference['plasma_composition.impurity.species'].sel(impurity='Ne').to_numpy().flatten()
        np.testing.assert_allclose(t['plasma_composition.impurity.species'].sel(impurity='Ne').to_numpy().flatten(), nne + nna)


class TestParameterSide:

    @pytest.mark.parametrize('code', ['tglf', 'cgyro', 'qualikiz'])
    def test_input_side_is_refused(self, code):
        t = _torax_with_output(_torax_v1_output())
        with pytest.raises(NotImplementedError, match='only from simulation output'):
            getattr(t, f'to_{code}_parameters')(side='input')

    def test_generator_does_not_overwrite_the_output(self):
        from fusio.scripts import standalone
        v1 = _torax_v1_output()
        t = _torax_with_output(v1)
        with pytest.raises(NotImplementedError):
            standalone.call_parameter_generator(t, 'tglf', 'input')
        xr.testing.assert_identical(t.output, v1)


try:
    import torax as _torax_pkg
    _HAS_TORAX = True
except ImportError:
    _torax_pkg = None
    _HAS_TORAX = False


def _base_circular_torax(t_initial=0.0, t_final=0.05):
    t = torax_io()
    t.time_coordinate([t_initial, t_final])
    t.radial_coordinate(np.linspace(0.0, 1.0, 11))
    t.set_cosine_initial_and_constant_kinetic_boundary_condition(
        tebc=200.0,
        tibc=200.0,
        nebc=2.0e19,
        tecore=5.0e3,
        ticore=5.0e3,
        necore=8.0e19,
    )
    t.update_input_data_vars({
        'profile_conditions.I_p': (['time'], np.array([1.0e7, 1.0e7])),
    })
    t.set_constant_flat_main_ion_composition({'D': 0.5, 'T': 0.5})
    t.set_constant_flat_impurity_composition_by_electron_density_fraction({'Ne': 0.01})
    t.add_geometry('circular')
    t.set_numerics(t_initial, t_final)
    t.add_fixed_linear_solver(dt_fixed=0.05)
    return t


class TestBohmGyroBohmTransport:

    def test_default_and_custom_multipliers(self):
        t = _base_circular_torax()
        t.add_pedestal_by_temperature(nped=5.0e19, tped=2.0e3, wrho=0.9)
        t.add_bohm_gyrobohm_transport(
            bohm_multiplier=1.5,
            gyrobohm_multiplier=0.75,
            d_c1=0.8,
            d_c2=0.25,
            v_coeff=-0.2,
            rho_min=0.1,
            merge_mode='overwrite',
        )
        cfg = t.to_config()
        model = cfg['transport']['core_transport_models']['bohm_gyrobohm_0']
        assert model['model_name'] == 'bohm_gyrobohm'
        assert model['chi_e_bohm_multiplier'] == 1.5
        assert model['chi_i_bohm_multiplier'] == 1.5
        assert model['chi_e_gyrobohm_multiplier'] == 0.75
        assert model['chi_i_gyrobohm_multiplier'] == 0.75
        assert model['D_face_c1'] == 0.8
        assert model['D_face_c2'] == 0.25
        assert model['V_face_coeff'] == -0.2
        assert model['rho_min'] == 0.1
        assert np.isclose(model['rho_max'], 0.9)
        assert model['merge_mode'] == 'overwrite'

    @pytest.mark.parametrize('kwargs', [
        {'bohm_multiplier': 0.0},
        {'bohm_multiplier': -1.0},
        {'gyrobohm_multiplier': 0.0},
        {'d_c1': 0.0},
        {'d_c2': -0.1},
        {'merge_mode': 'invalid'},
    ])
    def test_invalid_arguments_raise(self, kwargs):
        t = _base_circular_torax()
        with pytest.raises(ValueError):
            t.add_bohm_gyrobohm_transport(**kwargs)

    @pytest.mark.skipif(not _HAS_TORAX, reason='torax is not installed')
    def test_torax_v2_validation_and_simulation(self, tmp_path):
        t = _base_circular_torax(0.0, 0.05)
        t.add_bohm_gyrobohm_transport(bohm_multiplier=1.2, gyrobohm_multiplier=0.8)
        t.set_default_exchange_source()
        t.set_default_ohmic_source()
        t.set_gaussian_generic_heat_source(0.0, 0.2, 1.0e7)
        cfg = t.to_config()
        torax_cfg = _torax_pkg.ToraxConfig.from_dict(cfg)
        bgb_cfg = torax_cfg.transport.core_transport_models['bohm_gyrobohm_0']
        assert bgb_cfg.model_name == 'bohm_gyrobohm'
        np.testing.assert_allclose(bgb_cfg.chi_e_bohm_multiplier.get_value(0.0), 1.2)
        np.testing.assert_allclose(bgb_cfg.chi_i_bohm_multiplier.get_value(0.0), 1.2)
        np.testing.assert_allclose(bgb_cfg.chi_e_gyrobohm_multiplier.get_value(0.0), 0.8)
        np.testing.assert_allclose(bgb_cfg.chi_i_gyrobohm_multiplier.get_value(0.0), 0.8)
        state_history = _torax_pkg.run_simulation(torax_cfg)
        assert state_history.sim_error == _torax_pkg.SimError.NO_ERROR
        out_path = tmp_path / 'bgb_out.nc'
        state_history.simulation_output_to_xr().to_netcdf(out_path)
        loaded = torax_io(output=out_path).output
        assert 'T_e' in loaded
        assert 'turbulent_transport.core.bohm_gyrobohm_0.chi_bohm_e' in loaded
        assert 'turbulent_transport.core.bohm_gyrobohm_0.chi_gyrobohm_i' in loaded


class TestToraxV2Config:

    def test_prescribed_sources_and_json_roundtrip(self):
        t = _base_circular_torax()
        rho = np.linspace(0.0, 1.0, 11)
        t.add_constant_transport(chi_i=1.0, chi_e=0.0, D_e=0.5, V_e=0.0)
        t.set_prescribed_generic_heat_source(rho, np.full_like(rho, 1.0e5), np.full_like(rho, 2.0e5))
        t.set_prescribed_generic_particle_source(rho, np.full_like(rho, 1.0e19))
        t.set_prescribed_generic_current_source(rho, np.full_like(rho, 1.0e5))
        t.set_prescribed_fusion_source(rho, np.full_like(rho, 5.0e4), np.full_like(rho, 2.0e4))
        t.set_prescribed_ohmic_source(rho, np.full_like(rho, 1.0e4))
        t.set_prescribed_exchange_source(rho, np.full_like(rho, 1.0e4))
        cfg = t.to_config(json_compatible=True)
        assert cfg['sources']['generic_heat']['mode'] == 'prescribed'
        assert set(cfg['sources']['generic_heat']['prescribed_values']) == {'power_density_i', 'power_density_e'}
        assert set(cfg['sources']['generic_particle']['prescribed_values']) == {'particle_source_e'}
        assert set(cfg['sources']['generic_current']['prescribed_values']) == {'j_parallel'}
        assert set(cfg['sources']['fusion']['prescribed_values']) == {'power_density_i', 'power_density_e'}
        assert set(cfg['sources']['ohmic']['prescribed_values']) == {'power_density_e'}
        assert cfg['sources']['ei_exchange']['mode'] == 'zero'
        assert 'prescribed_values' not in cfg['sources']['ei_exchange']
        rt = torax_io()
        rt.input = rt.from_dict(cfg)
        rt_cfg = rt.to_config(json_compatible=True)
        assert set(rt_cfg['sources']['generic_heat']['prescribed_values']) == {'power_density_i', 'power_density_e'}
        assert rt_cfg['transport']['core_transport_models']['prescribed_0']['disable_chi_e'] is True
        if _HAS_TORAX:
            _torax_pkg.ToraxConfig.from_dict(rt_cfg)

    def test_solver_switching_and_fixed_vs_adaptive_keys(self):
        t = _base_circular_torax()
        t.add_constant_transport(chi_i=1.0, chi_e=1.0, D_e=0.5, V_e=0.0)
        t.add_adaptive_newton_raphson_solver(dt_mult=25.0, max_dt=0.2, max_steps=500)
        cfg_adapt = t.to_config()
        assert cfg_adapt['solver']['use_fixed_point'] is True
        assert cfg_adapt['solver']['n_fixed_point_iterations'] == 10
        assert cfg_adapt['time_step_calculator']['calculator_type'] == 'chi'
        assert cfg_adapt['time_step_calculator']['max_dt'] == 0.2
        assert cfg_adapt['time_step_calculator']['max_steps'] == 500
        assert 'dt' not in cfg_adapt['time_step_calculator']
        if _HAS_TORAX:
            _torax_pkg.ToraxConfig.from_dict(cfg_adapt)

        t.add_fixed_linear_solver(dt_fixed=0.05)
        cfg_fixed = t.to_config()
        assert cfg_fixed['solver']['solver_type'] == 'linear'
        assert 'log_iterations' not in cfg_fixed['solver']
        assert cfg_fixed['time_step_calculator']['calculator_type'] == 'fixed'
        assert cfg_fixed['time_step_calculator']['dt'] == 0.05
        assert 'max_dt' not in cfg_fixed['time_step_calculator']
        assert 'max_steps' not in cfg_fixed['time_step_calculator']
        if _HAS_TORAX:
            _torax_pkg.ToraxConfig.from_dict(cfg_fixed)

    def test_multitime_eqdsk_dict_roundtrip(self):
        t = _base_circular_torax()
        t.add_geometry('eqdsk', geofiles={0.0: 'a.eqdsk', 1.0: 'b.eqdsk'}, geodir='/tmp', cocos=11)
        cfg = t.to_config(json_compatible=True)
        assert cfg['geometry']['model_name'] == 'eqdsk'
        assert cfg['geometry']['geometry_file'] == {0.0: 'a.eqdsk', 1.0: 'b.eqdsk'}
        assert 'Ip_from_parameters' not in cfg['geometry']
        assert cfg['profile_conditions']['psi_source'] == 'geometry'
        assert cfg['profile_conditions']['I_p_source'] == 'geometry'
        rt = torax_io()
        rt.input = rt.from_dict(cfg)
        rt_cfg = rt.to_config(json_compatible=True)
        assert rt_cfg['geometry']['geometry_file'] == {0.0: 'a.eqdsk', 1.0: 'b.eqdsk'}
        t.add_geometry('eqdsk', geofiles={0.0: 'a.eqdsk', 1.0: 'b.eqdsk'}, geodir='/tmp', ip_from_parameters=True, cocos=11)
        assert t.to_config()['profile_conditions']['I_p_source'] == 'profile_conditions'

    @pytest.mark.skipif(not _HAS_TORAX, reason='torax is not installed')
    def test_pedestals_ibc_patches_and_transport_models_validate(self):
        # Pressure pedestal + QLKNN + inner/outer patches
        t1 = _base_circular_torax()
        t1.add_pedestal_by_pressure(pped=5.0e4, nped=5.0e19, tpedratio=1.0, wrho=0.9)
        t1.add_qlknn_transport(rho_min=0.1, rho_max=0.85)
        t1.add_transport_inner_patch(0.5, 0.0, 1.0, 1.0, rho=0.1)
        t1.add_transport_outer_patch(0.2, 0.0, 0.5, 0.5, rho=0.85)
        t1.add_pedestal_critical_gradient_transport()
        t1.set_gaussian_generic_current_source(0.3, 0.1, 1.0e5)
        cfg1 = t1.to_config()
        assert cfg1['transport']['core_transport_models']['prescribed_1']['merge_mode'] == 'overwrite'
        assert cfg1['transport']['core_transport_models']['prescribed_2']['merge_mode'] == 'overwrite'
        _torax_pkg.ToraxConfig.from_dict(cfg1)

        # Temperature pedestal via IBC + CGM
        t2 = _base_circular_torax()
        t2.add_internal_boundary_with_pedestal(rho=0.85)
        t2.add_critical_gradient_transport()
        _torax_pkg.ToraxConfig.from_dict(t2.to_config())

        # Prescribed IBC + TGLFNN
        t3 = _base_circular_torax()
        t3.add_internal_boundary('T_e', rho=0.85)
        t3.add_tglfnn_transport()
        cfg3 = t3.to_config(json_compatible=True)
        assert cfg3['transport']['core_transport_models']['tglfnn_0']['model_name'] == 'tglfnn_ukaea'
        _torax_pkg.ToraxConfig.from_dict(cfg3)

    @pytest.mark.skipif(not _HAS_TORAX, reason='torax is not installed')
    def test_from_gacode_validates_in_torax_v2(self, gacode_file_path):
        g = gacode_io(input=gacode_file_path)
        t = torax_io.from_gacode(g, side='input')
        t.add_geometry('circular')
        t.add_bohm_gyrobohm_transport()
        t.set_numerics(0.0, 0.1)
        t.add_fixed_linear_solver(dt_fixed=0.05)
        _torax_pkg.ToraxConfig.from_dict(t.to_config())

    def test_model_state_groups_are_namespaced_on_read(self, tmp_path):
        nc_path = tmp_path / 'synthetic_v2.nc'
        t_coord = np.array([0.0, 0.1])
        dt = xr.DataTree.from_dict({
            '/': xr.Dataset(coords={'time': t_coord}),
            '/scalars': xr.Dataset({'rho_norm_top': ('time', np.array([0.9, 0.9]))}, coords={'time': t_coord}),
            '/model_state/pedestal': xr.Dataset({'rho_norm_top': ('time', np.array([0.85, 0.85]))}, coords={'time': t_coord}),
        })
        dt.to_netcdf(nc_path)
        ds = torax_io(output=nc_path).output
        np.testing.assert_allclose(ds['rho_norm_top'].to_numpy(), [0.9, 0.9])
        np.testing.assert_allclose(ds['model_state.pedestal.rho_norm_top'].to_numpy(), [0.85, 0.85])


class TestPrescribedTransport:

    def test_scalars_1d_and_2d_arrays_on_distinct_grids(self):
        t = _base_circular_torax(0.0, 0.1)
        rho1 = np.linspace(0.0, 1.0, 6)
        chi_1d = np.linspace(0.5, 2.0, 6)
        t.add_prescribed_transport(chi_i=chi_1d, chi_e=1.5, D_e=0.0, V_e=-0.1, rho=rho1)

        rho2 = np.linspace(0.0, 0.5, 4)
        times2 = np.array([0.0, 0.05, 0.1])
        chi_2d = np.outer(np.array([1.0, 1.2, 1.4]), np.linspace(0.2, 0.8, 4))
        t.add_prescribed_transport(chi_i=chi_2d, chi_e=chi_2d, D_e=0.3, V_e=0.0, rho=rho2, time=times2, rho_max=0.5, merge_mode='overwrite')

        cfg = t.to_config(json_compatible=True)
        m0 = cfg['transport']['core_transport_models']['prescribed_0']
        m1 = cfg['transport']['core_transport_models']['prescribed_1']
        assert m0['model_name'] == 'prescribed'
        assert m0['disable_D_e'] is True
        assert 'D_e' not in m0
        np.testing.assert_allclose(m0['chi_i'][1], rho1)
        np.testing.assert_allclose(m0['chi_i'][2][0], chi_1d)
        assert m1['merge_mode'] == 'overwrite'
        np.testing.assert_allclose(m1['chi_i'][0], times2)
        np.testing.assert_allclose(m1['chi_i'][1], rho2)
        np.testing.assert_allclose(m1['chi_i'][2], chi_2d)

        rt = torax_io()
        rt.input = rt.from_dict(cfg)
        rt_cfg = rt.to_config(json_compatible=True)
        np.testing.assert_allclose(rt_cfg['transport']['core_transport_models']['prescribed_1']['chi_i'][2], chi_2d)
        if _HAS_TORAX:
            _torax_pkg.ToraxConfig.from_dict(cfg)
            _torax_pkg.ToraxConfig.from_dict(rt_cfg)

    def test_1d_array_uses_dataset_rho_when_omitted(self):
        t = _base_circular_torax()
        chi = np.linspace(0.5, 1.5, 11)
        t.add_prescribed_transport(chi_i=chi, chi_e=chi, D_e=0.4)
        cfg = t.to_config()
        da = cfg['transport']['core_transport_models']['prescribed_0']['chi_i']
        np.testing.assert_allclose(da['rho_norm'].to_numpy(), np.linspace(0.0, 1.0, 11))
        np.testing.assert_allclose(da.isel(time=0).to_numpy(), chi)
        if _HAS_TORAX:
            _torax_pkg.ToraxConfig.from_dict(cfg)

    def test_invalid_shapes_and_nonpositive_values_raise(self):
        t = _base_circular_torax()
        with pytest.raises(ValueError, match='rho'):
            t.add_prescribed_transport(chi_i=np.ones(5))
        with pytest.raises(ValueError, match='length'):
            t.add_prescribed_transport(chi_i=np.ones(5), rho=np.linspace(0.0, 1.0, 4))
        with pytest.raises(ValueError, match='shape'):
            t.add_prescribed_transport(chi_i=np.ones((3, 4)), rho=np.linspace(0.0, 1.0, 4))
        with pytest.raises(ValueError, match='strictly positive'):
            t.add_prescribed_transport(chi_i=np.array([0.0, 1.0]), rho=np.array([0.0, 1.0]))

    def test_add_constant_transport_alias_matches_prescribed(self):
        t1 = _base_circular_torax()
        t2 = _base_circular_torax()
        t1.add_constant_transport(chi_i=1.0, chi_e=2.0, D_e=0.5, V_e=-0.1, rho_min=0.1, rho_max=0.9, merge_mode='overwrite')
        t2.add_prescribed_transport(chi_i=1.0, chi_e=2.0, D_e=0.5, V_e=-0.1, rho_min=0.1, rho_max=0.9, merge_mode='overwrite')
        assert t1.to_config(json_compatible=True) == t2.to_config(json_compatible=True)
