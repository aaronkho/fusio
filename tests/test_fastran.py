import pytest
import numpy as np
import xarray as xr
from fusio.classes.fastran import fastran_io


def _main_ion_dataset(amain, zmain=1.0, ntime=2, nrho=5):
    coords = {
        'time': np.arange(ntime, dtype=float),
        'rho': np.linspace(0.0, 1.0, nrho),
    }
    ni = np.repeat(np.expand_dims(np.linspace(10.0, 2.0, nrho), axis=0), ntime, axis=0)
    data_vars = {
        'amain': (['time'], np.full(ntime, amain)),
        'zmain': (['time'], np.full(ntime, zmain)),
        'ni': (['time', 'rho'], ni),
        'ti': (['time', 'rho'], np.ones_like(ni)),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


class TestMainIonSplit:

    @pytest.mark.parametrize('amain, expected', [
        (2.0, None),
        (2.0 + 1.0e-12, None),
        (2.014, None),
        (2.016, None),
        (2.019, None),
        (2.021, 0.021),
        (2.979, 0.979),
        (3.016, None),
        (1.008, None),
        (2.5, 0.5),
        (2.25, 0.25),
        (2.9, 0.9),
        (1.5, None),
    ])
    def test_tritium_fraction(self, amain, expected):
        fraction = fastran_io().resolve_tritium_fraction(np.array([amain]), np.array([1.0]))
        if expected is None:
            assert fraction is None
        else:
            assert fraction is not None
            np.testing.assert_allclose(fraction, [expected])

    def test_non_hydrogenic_main_ion_is_not_split(self):
        assert fastran_io().resolve_tritium_fraction(np.array([2.5]), np.array([2.0])) is None

    def test_lumped_main_ion_is_split_into_d_and_t(self):
        data = _main_ion_dataset(2.5)
        names, types, masses, charges, densities, temperatures = fastran_io().resolve_ion_species(data)
        assert names == ['D', 'T']
        assert types == ['[therm]', '[therm]']
        np.testing.assert_allclose(np.stack(masses, axis=-1)[-1], [2.0, 3.0])
        np.testing.assert_allclose(np.stack(charges, axis=-1)[-1], [1.0, 1.0])
        np.testing.assert_allclose(densities[0], densities[1])

    def test_split_conserves_density_charge_and_mass(self):
        data = _main_ion_dataset(2.25)
        _, _, masses, charges, densities, _ = fastran_io().resolve_ion_species(data)
        ni = data['ni'].to_numpy()
        n = np.stack(densities, axis=-1)
        a = np.expand_dims(np.stack(masses, axis=-1), axis=1)
        z = np.expand_dims(np.stack(charges, axis=-1), axis=1)
        np.testing.assert_allclose(n.sum(axis=-1), ni)
        np.testing.assert_allclose((n * z).sum(axis=-1), ni)
        np.testing.assert_allclose((n * a).sum(axis=-1) / n.sum(axis=-1), 2.25)

    def test_split_can_be_disabled(self):
        data = _main_ion_dataset(2.5)
        names, _, masses, _, densities, _ = fastran_io().resolve_ion_species(data, split_main_ion=False)
        assert names == ['D']
        np.testing.assert_allclose(masses[0], 2.5)
        np.testing.assert_allclose(densities[0], data['ni'].to_numpy())

    def test_single_isotope_keeps_its_file_mass(self):
        data = _main_ion_dataset(2.014)
        names, _, masses, _, _, _ = fastran_io().resolve_ion_species(data)
        assert names == ['D']
        np.testing.assert_allclose(masses[0], 2.014)

    def test_lumped_mixture_takes_the_given_name(self):
        data = _main_ion_dataset(2.5)
        names, _, masses, _, _, _ = fastran_io().resolve_ion_species(data, split_main_ion=False, lumped_main_ion_name='MAIN')
        assert names == ['MAIN']
        np.testing.assert_allclose(masses[0], 2.5)

    def test_single_isotope_ignores_the_lumped_name(self):
        data = _main_ion_dataset(2.016)
        names, _, _, _, _, _ = fastran_io().resolve_ion_species(data, split_main_ion=False, lumped_main_ion_name='MAIN')
        assert names == ['D']


class TestGacodeConversion:

    @pytest.mark.parametrize('split, names, masses', [
        (True, ['D', 'T'], [2.0, 3.0]),
        (False, ['MAIN'], [2.5]),
    ])
    def test_main_ion_names(self, split, names, masses):
        from fusio.classes.gacode import gacode_io
        f = fastran_io()
        f.output = _main_ion_dataset(2.5)
        g = gacode_io.from_fastran(f, split_main_ion=split)
        assert [f'{name}' for name in g.input['name'].to_numpy()] == names
        np.testing.assert_allclose(g.input['mass'].to_numpy()[-1], masses)
