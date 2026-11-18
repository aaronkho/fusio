import pytest
import numpy as np
import xarray as xr
from fusio.classes.gacode import gacode_io
from fusio.classes.omas import omas_io
from fusio.classes.torax import torax_io


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
