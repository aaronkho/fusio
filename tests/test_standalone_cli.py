import copy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fusio.scripts import convert, standalone
from fusio.scripts import standalone_cgyro, standalone_qualikiz, standalone_tglf


@pytest.fixture(scope='module')
def conversion_map():
    return convert.build_conversion_map()


@pytest.fixture(scope='module')
def producer_map():
    return standalone.build_producer_map()


@pytest.fixture(scope='session')
def eqdsk_file_path():
    return Path(__file__).parent / 'data' / 'sample_cocos11_input.geqdsk'


class TestProducerMap:

    def test_every_code_is_mapped(self, producer_map):
        for code in standalone.SUPPORTED_CODES:
            assert code in producer_map

    def test_torax_produces_all_three(self, producer_map):
        for code in ['tglf', 'qualikiz', 'cgyro']:
            assert 'torax' in producer_map[code]

    def test_imas_produces_cgyro_only(self, producer_map):
        assert 'imas' in producer_map['cgyro']
        assert 'imas' not in producer_map['tglf']
        assert 'imas' not in producer_map['qualikiz']


class TestProducerRoutes:

    def test_native_source_needs_no_route(self, producer_map, conversion_map):
        route = standalone.resolve_producer_route('torax', producer_map['tglf'], conversion_map, None)
        assert route == ['torax']

    def test_native_only_producer_is_unreachable_by_conversion(self, producer_map, conversion_map):
        for source in ['gacode', 'plasma', 'imas', 'omas']:
            route = standalone.resolve_producer_route(source, producer_map['tglf'], conversion_map, None)
            assert route is None, f'{source} should not be able to synthesize TORAX output'

    def test_explicit_route_into_native_only_producer_raises(self, producer_map, conversion_map):
        with pytest.raises(ValueError, match='read directly'):
            standalone.resolve_producer_route('gacode', producer_map['tglf'], conversion_map, ['torax'])

    def test_cgyro_route_prefers_the_hub(self, producer_map, conversion_map):
        route = standalone.resolve_producer_route('gacode', producer_map['cgyro'], conversion_map, None)
        assert route == ['gacode', 'plasma', 'imas'], (
            'the hub carries a more complete mapping than the direct gacode to imas edge'
        )

    def test_imas_source_produces_cgyro_natively(self, producer_map, conversion_map):
        route = standalone.resolve_producer_route('imas', producer_map['cgyro'], conversion_map, None)
        assert route == ['imas']

    def test_explicit_via_is_validated(self, producer_map, conversion_map):
        with pytest.raises(ValueError, match='not implemented'):
            standalone.resolve_producer_route('gacode', producer_map['cgyro'], conversion_map, ['omas', 'imas'])

    def test_via_must_end_at_a_producer(self, producer_map, conversion_map):
        with pytest.raises(ValueError, match='cannot compute these parameters'):
            standalone.resolve_producer_route('gacode', producer_map['cgyro'], conversion_map, ['plasma'])


class TestEqdskHop:

    def test_hub_is_preferred_over_an_earlier_capable_format(self):
        assert convert.select_eqdsk_hop(['gacode', 'plasma', 'imas']) == 'plasma'

    def test_falls_back_to_the_only_capable_format(self):
        assert convert.select_eqdsk_hop(['gacode', 'torax']) == 'gacode'

    def test_none_when_no_format_accepts_one(self):
        assert convert.select_eqdsk_hop(['torax', 'omas']) is None


def _reference_pointwise_plan(params, code, template, species_template, species_count):
    """The plan building loop as it was written inline in the original scripts."""
    plan = []
    for j, _ in enumerate(params['time']):
        for i, _ in enumerate(params['rho']):
            runpars = {k: v for k, v in template.items()}
            for s in range(int(params[species_count].isel(time=j, drop=True).isel(rho=i, drop=True).to_numpy())):
                runpars.update({k + f'_{s + 1:d}': v for k, v in species_template.items()})
            for key in params.data_vars:
                if key in [species_count]:
                    runpars[key] = int(params[key].isel(time=j, drop=True).isel(rho=i, drop=True).to_numpy())
                elif key in runpars:
                    runpars[key] = float(params[key].isel(time=j, drop=True).isel(rho=i, drop=True).to_numpy())
            for key in list(runpars.keys()):
                if runpars[key] is None:
                    runpars.pop(key)
            plan.append({
                'inputs': copy.deepcopy(runpars),
                'location': f'{code}_run_t{j:05d}_r{i:04d}',
            })
    return plan


@pytest.fixture(scope='module')
def tglf_parameters():
    rng = np.random.default_rng(0)
    nt, nr = 2, 3
    data_vars = {'NS': (['time', 'rho'], np.full((nt, nr), 3.0))}
    for key in ['RLNS', 'RLTS', 'TAUS', 'AS', 'ZS', 'MASS']:
        for s in range(1, 4):
            data_vars[f'{key}_{s}'] = (['time', 'rho'], rng.random((nt, nr)))
    for key in ['Q_LOC', 'KAPPA_LOC', 'BETAE', 'XNUE', 'ZEFF', '#RHO', 'SHAT_SA', '#BUNIT_BY_BREF', 'UNMAPPED']:
        data_vars[key] = (['time', 'rho'], rng.random((nt, nr)))
    return xr.Dataset(data_vars, coords={'time': np.arange(nt) * 1.0, 'rho': np.linspace(0.2, 0.8, nr)})


@pytest.fixture(scope='module')
def qualikiz_parameters():
    rng = np.random.default_rng(1)
    nt, nr = 2, 4
    data_vars = {'nion': (['time'], np.full(nt, 2.0))}
    for key in ['Ti', 'ni', 'Ati', 'Ani', 'Zi', 'Ai']:
        for s in range(2):
            data_vars[f'{key}{s}'] = (['time', 'rho'], rng.random((nt, nr)))
    for key in ['Ro', 'Rmin', 'Bo', 'x', 'q', 'smag', 'alpha', 'Te', 'ne', 'Ate', 'Ane', '#Bunit_by_Bo', 'UNMAPPED']:
        data_vars[key] = (['time', 'rho'], rng.random((nt, nr)))
    return xr.Dataset(data_vars, coords={'time': np.arange(nt) * 1.0, 'rho': np.linspace(0.2, 0.8, nr)})


class TestPointwisePlan:

    def test_matches_the_original_tglf_loop(self, tglf_parameters):
        reference = _reference_pointwise_plan(
            tglf_parameters, 'tglf', standalone_tglf.TGLF_INPUT_TEMPLATE,
            standalone_tglf.TGLF_SPECIES_TEMPLATE, 'NS',
        )
        built = standalone.build_pointwise_plan(
            tglf_parameters, 'tglf',
            template=standalone_tglf.TGLF_INPUT_TEMPLATE,
            species_template=standalone_tglf.TGLF_SPECIES_TEMPLATE,
            species_count='NS', species_offset=1,
        )
        assert built == reference

    def test_one_run_per_time_and_radius(self, tglf_parameters):
        built = standalone.build_pointwise_plan(
            tglf_parameters, 'tglf', template=standalone_tglf.TGLF_INPUT_TEMPLATE,
            species_template=standalone_tglf.TGLF_SPECIES_TEMPLATE, species_count='NS',
        )
        assert len(built) == tglf_parameters.sizes['time'] * tglf_parameters.sizes['rho']
        assert built[0]['location'] == 'tglf_run_t00000_r0000'

    def test_species_count_stays_an_integer(self, tglf_parameters):
        built = standalone.build_pointwise_plan(
            tglf_parameters, 'tglf', template=standalone_tglf.TGLF_INPUT_TEMPLATE,
            species_template=standalone_tglf.TGLF_SPECIES_TEMPLATE, species_count='NS',
        )
        assert isinstance(built[0]['inputs']['NS'], int)

    def test_keys_outside_the_template_are_dropped(self, tglf_parameters):
        built = standalone.build_pointwise_plan(
            tglf_parameters, 'tglf', template=standalone_tglf.TGLF_INPUT_TEMPLATE,
            species_template=standalone_tglf.TGLF_SPECIES_TEMPLATE, species_count='NS',
        )
        assert 'UNMAPPED' not in built[0]['inputs']

    def test_overlay_all_keeps_every_key(self, tglf_parameters):
        built = standalone.build_pointwise_plan(tglf_parameters, 'cgyro', template={}, overlay_all=True)
        assert 'UNMAPPED' in built[0]['inputs']


class TestProfilePlan:

    def test_matches_the_original_qualikiz_loop(self, qualikiz_parameters):
        params = qualikiz_parameters
        reference = []
        for j, _ in enumerate(params['time']):
            runpars = {k: v for k, v in standalone_qualikiz.QUALIKIZ_INPUT_TEMPLATE.items()}
            for s in range(int(params['nion'].isel(time=j, drop=True).to_numpy())):
                runpars.update({k + f'{s:d}': v for k, v in standalone_qualikiz.QUALIKIZ_ION_TEMPLATE.items()})
            for key in params.data_vars:
                if key in ['nion']:
                    runpars[key] = int(params[key].isel(time=j, drop=True).to_numpy())
                elif key in runpars:
                    runpars[key] = params[key].isel(time=j, drop=True).to_numpy().flatten()
            runpars['rho'] = params['rho'].to_numpy().flatten()
            for key in list(runpars.keys()):
                if runpars[key] is None:
                    runpars.pop(key)
            num_ions = runpars.pop('nion', 1)
            conversion = runpars.pop('#Bunit_by_Bo', np.ones_like(runpars['rho']))
            reference.append({
                'inputs': copy.deepcopy(runpars),
                'location': f'qualikiz_run_t{j:05d}',
                'nion': int(num_ions),
                'field_conversion': copy.deepcopy(conversion),
            })

        built = standalone.build_profile_plan(
            params, 'qualikiz',
            template=standalone_qualikiz.QUALIKIZ_INPUT_TEMPLATE,
            species_template=standalone_qualikiz.QUALIKIZ_ION_TEMPLATE,
            species_count='nion', species_offset=0, species_separator='',
        )
        assert len(built) == len(reference)
        for got, want in zip(built, reference):
            assert got['location'] == want['location']
            assert got['nion'] == want['nion']
            assert np.allclose(got['field_conversion'], want['field_conversion'])
            assert sorted(got['inputs']) == sorted(want['inputs'])
            for key in want['inputs']:
                assert np.allclose(np.asarray(got['inputs'][key]), np.asarray(want['inputs'][key]))

    def test_one_run_per_time(self, qualikiz_parameters):
        built = standalone.build_profile_plan(
            qualikiz_parameters, 'qualikiz',
            template=standalone_qualikiz.QUALIKIZ_INPUT_TEMPLATE,
            species_template=standalone_qualikiz.QUALIKIZ_ION_TEMPLATE,
            species_count='nion', species_offset=0, species_separator='',
        )
        assert len(built) == qualikiz_parameters.sizes['time']

    def test_ion_keys_use_a_zero_based_bare_suffix(self, qualikiz_parameters):
        built = standalone.build_profile_plan(
            qualikiz_parameters, 'qualikiz',
            template=standalone_qualikiz.QUALIKIZ_INPUT_TEMPLATE,
            species_template=standalone_qualikiz.QUALIKIZ_ION_TEMPLATE,
            species_count='nion', species_offset=0, species_separator='',
        )
        assert 'Ai0' in built[0]['inputs']
        assert 'Ai1' in built[0]['inputs']


class TestInputFileWriting:

    def test_writes_one_directory_per_run(self, tmp_path):
        plan = [
            {'inputs': {'A': 1, 'B': 2.5}, 'location': 'run_a'},
            {'inputs': {'A': 3}, 'location': 'run_b'},
        ]
        n = standalone.write_input_files(plan, tmp_path / 'runs', 'input.test')
        assert n == 2
        assert (tmp_path / 'runs' / 'run_a' / 'input.test').read_text() == 'A=1\nB=2.5'
        assert (tmp_path / 'runs' / 'run_b' / 'input.test').read_text() == 'A=3'

    def test_existing_input_is_replaced(self, tmp_path):
        plan = [{'inputs': {'A': 1}, 'location': 'run_a'}]
        standalone.write_input_files(plan, tmp_path / 'runs', 'input.test')
        standalone.write_input_files([{'inputs': {'A': 9}, 'location': 'run_a'}], tmp_path / 'runs', 'input.test')
        assert (tmp_path / 'runs' / 'run_a' / 'input.test').read_text() == 'A=9'


class TestCgyroEndToEnd:

    @pytest.fixture(scope='class')
    def cgyro_plan(self, gacode_file_path, eqdsk_file_path):
        return standalone_cgyro.convert_to_cgyro_input(
            gacode_file_path,
            rho=[0.3, 0.5, 0.7],
            eqdsk=eqdsk_file_path,
        )

    def test_one_run_per_requested_radius(self, cgyro_plan):
        assert len(cgyro_plan) == 3

    def test_species_parameters_are_present(self, cgyro_plan):
        inputs = cgyro_plan[1]['inputs']
        n_species = int(inputs['N_SPECIES'])
        assert n_species >= 2
        for s in range(1, n_species + 1):
            for key in ['MASS', 'Z', 'DENS', 'TEMP', 'DLNNDR', 'DLNTDR']:
                assert f'{key}_{s:d}' in inputs

    def test_geometry_and_physics_parameters_are_present(self, cgyro_plan):
        inputs = cgyro_plan[0]['inputs']
        for key in ['Q', 'S', 'RMIN', 'RMAJ', 'KAPPA', 'S_KAPPA', 'DELTA', 'BETAE_UNIT', 'NU_EE']:
            assert key in inputs, f'{key} missing from the generated CGYRO input'

    def test_mxh_shape_series_is_carried_through(self, cgyro_plan):
        inputs = cgyro_plan[0]['inputs']
        assert any(key.startswith('SHAPE_SIN') for key in inputs)
        assert any(key.startswith('SHAPE_S_SIN') for key in inputs)

    def test_template_defaults_are_included(self, cgyro_plan):
        inputs = cgyro_plan[0]['inputs']
        assert inputs['EQUILIBRIUM_MODEL'] == 2
        assert inputs['N_SPECIES'] == int(inputs['N_SPECIES'])

    def test_run_directories_are_written(self, tmp_path, cgyro_plan):
        standalone_cgyro.generate_cgyro_run_directories(cgyro_plan, tmp_path / 'cgyro_runs')
        written = sorted([p.name for p in (tmp_path / 'cgyro_runs').iterdir()])
        assert written == ['cgyro_run_t00000_r0000', 'cgyro_run_t00000_r0001', 'cgyro_run_t00000_r0002']
        text = (tmp_path / 'cgyro_runs' / 'cgyro_run_t00000_r0000' / 'input.cgyro').read_text()
        assert all('=' in line for line in text.splitlines())


class TestFailureModes:

    def test_tglf_from_a_non_torax_source_reports_why(self, gacode_file_path):
        with pytest.raises(ValueError, match='cannot reach any of the tglf parameter producers'):
            standalone_tglf.convert_to_tglf_input(gacode_file_path)

    def test_unknown_format_raises(self, tmp_path):
        ipath = tmp_path / 'mystery.dat'
        ipath.write_text('nothing useful here')
        with pytest.raises(ValueError, match='Could not determine the format'):
            standalone_cgyro.convert_to_cgyro_input(ipath)

    def test_cgyro_without_an_equilibrium_map_reports_why(self, gacode_file_path):
        # The gacode to plasma to imas route carries no 2-D flux map without an EQDSK,
        # so the contour tracing the CGYRO parameters rely on cannot run
        with pytest.raises((ValueError, KeyError)):
            standalone_cgyro.convert_to_cgyro_input(gacode_file_path, rho=[0.5])
