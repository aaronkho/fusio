import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fusio.scripts import convert


@pytest.fixture(scope='module')
def conversion_map():
    return convert.build_conversion_map()


class TestConversionMap:

    def test_known_edges_are_present(self, conversion_map):
        expected = {
            'plasma': ['gacode', 'imas', 'torax', 'fastran'],
            'gacode': ['plasma', 'imas', 'torax', 'fastran'],
            'imas': ['plasma', 'gacode', 'omas'],
            'torax': ['gacode', 'omas'],
        }
        for target, sources in expected.items():
            for source in sources:
                assert source in conversion_map[target], (
                    f'{target} <- {source} missing from the conversion map'
                )

    def test_stub_conversions_are_excluded(self, conversion_map):
        for target, source in convert.STUB_CONVERSIONS:
            assert source not in conversion_map.get(target, [])

    def test_every_writable_format_is_mapped(self, conversion_map):
        for fmt in convert.WRITABLE_FORMATS:
            assert fmt in conversion_map

    def test_rendered_map_names_every_format(self, conversion_map):
        rendered = convert.format_conversion_map(conversion_map)
        for fmt in conversion_map:
            assert fmt in rendered


class TestRouteResolution:

    def test_direct_route(self, conversion_map):
        assert convert.resolve_route('gacode', 'plasma', None, conversion_map) == ['gacode', 'plasma']

    def test_missing_direct_route_raises(self, conversion_map):
        with pytest.raises(ValueError, match='not implemented'):
            convert.resolve_route('imas', 'torax', None, conversion_map)

    def test_explicit_via_route(self, conversion_map):
        route = convert.resolve_route('imas', 'torax', ['plasma', 'gacode'], conversion_map)
        assert route == ['imas', 'plasma', 'gacode', 'torax']

    def test_invalid_via_route_raises(self, conversion_map):
        with pytest.raises(ValueError, match='not implemented'):
            convert.resolve_route('gacode', 'plasma', ['omas'], conversion_map)

    def test_auto_route_is_shortest(self, conversion_map):
        route = convert.resolve_route('imas', 'torax', ['auto'], conversion_map)
        assert route[0] == 'imas'
        assert route[-1] == 'torax'
        for src, tgt in zip(route[:-1], route[1:]):
            assert src in conversion_map[tgt]

    def test_auto_route_without_connection_raises(self):
        with pytest.raises(ValueError, match='No conversion route'):
            convert.resolve_route('transp', 'gacode', ['auto'], {'gacode': [], 'transp': []})

    def test_identity_without_conversion_is_passthrough(self, conversion_map):
        assert convert.resolve_route('plasma', 'plasma', None, conversion_map) == ['plasma']

    def test_identity_with_conversion_is_a_hop(self, conversion_map):
        assert convert.resolve_route('gacode', 'gacode', None, conversion_map) == ['gacode', 'gacode']


class TestFormatDetection:

    def test_gacode_file(self, gacode_file_path):
        assert convert.detect_format(gacode_file_path) == 'gacode'

    def test_gacode_by_canonical_name(self, tmp_path, gacode_file_path):
        ipath = tmp_path / 'input.gacode'
        ipath.write_text(Path(gacode_file_path).read_text())
        assert convert.detect_format(ipath) == 'gacode'

    def test_torax_config_json(self, tmp_path):
        jpath = tmp_path / 'config.json'
        jpath.write_text(json.dumps({
            'profile_conditions': {},
            'numerics': {},
            'geometry': {},
            'sources': {},
        }))
        assert convert.detect_format(jpath) == 'torax'

    def test_omas_json(self, tmp_path):
        jpath = tmp_path / 'ods.json'
        jpath.write_text(json.dumps({
            'core_profiles': {},
            'equilibrium': {},
            'dataset_description': {},
        }))
        assert convert.detect_format(jpath) == 'omas'

    def test_plasma_netcdf(self, tmp_path):
        npath = tmp_path / 'state.nc'
        xr.Dataset(
            data_vars={
                'density_e': (['time', 'radius'], np.ones((1, 3))),
                'temperature_e': (['time', 'radius'], np.ones((1, 3))),
                'r_minor': (['time', 'radius'], np.ones((1, 3))),
            },
            coords={'time': [0.0], 'radius': [0.0, 0.5, 1.0]},
        ).to_netcdf(npath)
        assert convert.detect_format(npath) == 'plasma'

    def test_imas_directory(self, tmp_path):
        dpath = tmp_path / 'imasdir'
        dpath.mkdir()
        xr.Dataset().to_netcdf(dpath / 'equilibrium.nc')
        assert convert.detect_format(dpath) == 'imas'

    def test_astra_run_directory(self, tmp_path):
        dpath = tmp_path / 'astra_run'
        (dpath / 'ncdf_out').mkdir(parents=True)
        assert convert.detect_format(dpath) == 'astra'

    def test_unrecognized_file(self, tmp_path):
        ipath = tmp_path / 'mystery.dat'
        ipath.write_text('nothing useful here')
        assert convert.detect_format(ipath) is None

    def test_missing_path(self, tmp_path):
        assert convert.detect_format(tmp_path / 'absent.nc') is None


class TestDumpDetection:

    def test_dumped_file_reports_its_class(self, tmp_path, gacode_file_path):
        from fusio.classes.gacode import gacode_io
        dpath = tmp_path / 'dumped.nc'
        gacode_io.from_file(output=gacode_file_path).dump(dpath, overwrite=True)
        assert convert.read_dumped_format(dpath) == 'gacode'
        assert convert.detect_format(dpath) == 'gacode'

    def test_plain_netcdf_is_not_a_dump(self, tmp_path):
        npath = tmp_path / 'plain.nc'
        xr.Dataset({'a': ('x', np.ones(2))}).to_netcdf(npath)
        assert convert.read_dumped_format(npath) is None


class TestOutputPaths:

    def test_single_input_uses_output_verbatim(self, tmp_path):
        opath = convert.resolve_output_path(tmp_path / 'out.nc', Path('in.gacode'), 'plasma', 1)
        assert opath == tmp_path / 'out.nc'

    def test_multiple_inputs_treat_output_as_directory(self, tmp_path):
        opath = convert.resolve_output_path(tmp_path, Path('/data/in.gacode'), 'plasma', 3)
        assert opath == tmp_path / 'in_plasma.nc'

    def test_default_sits_beside_the_input(self):
        opath = convert.resolve_output_path(None, Path('/data/in.gacode'), 'torax', 1)
        assert opath == Path('/data/in_torax.json')

    def test_directory_target_has_no_suffix(self):
        opath = convert.resolve_output_path(None, Path('/data/in.gacode'), 'imas', 1)
        assert opath == Path('/data/in_imas')


class TestEndToEnd:

    def test_gacode_to_plasma(self, tmp_path, gacode_file_path, conversion_map):
        opath = convert.convert_file(
            Path(gacode_file_path),
            'plasma',
            output=tmp_path / 'state.nc',
            overwrite=True,
            cmap=conversion_map,
        )
        assert opath is not None and opath.is_file()
        data = xr.open_dataset(opath)
        assert 'density_e' in data
        assert 'temperature_e' in data

    def test_derived_quantities_are_computed_by_default(self, tmp_path, gacode_file_path, conversion_map):
        derived = convert.convert_file(
            Path(gacode_file_path), 'plasma', output=tmp_path / 'derived.nc', overwrite=True, cmap=conversion_map,
        )
        plain = convert.convert_file(
            Path(gacode_file_path), 'plasma', output=tmp_path / 'plain.nc', overwrite=True, derive=False, cmap=conversion_map,
        )
        assert derived is not None and plain is not None
        n_derived = len(xr.open_dataset(derived).data_vars)
        n_plain = len(xr.open_dataset(plain).data_vars)
        assert n_derived > n_plain

    def test_unknown_format_raises(self, tmp_path, conversion_map):
        ipath = tmp_path / 'mystery.dat'
        ipath.write_text('nothing useful here')
        with pytest.raises(ValueError, match='Could not determine the format'):
            convert.convert_file(ipath, 'plasma', output=tmp_path / 'out.nc', cmap=conversion_map)

    def test_existing_output_without_overwrite_raises(self, tmp_path, gacode_file_path, conversion_map):
        opath = tmp_path / 'state.nc'
        convert.convert_file(
            Path(gacode_file_path), 'plasma', output=opath, overwrite=True, cmap=conversion_map,
        )
        with pytest.raises(OSError, match='already exists'):
            convert.convert_file(
                Path(gacode_file_path), 'plasma', output=opath, overwrite=False, cmap=conversion_map,
            )
