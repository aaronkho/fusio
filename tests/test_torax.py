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
