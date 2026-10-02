import numpy as np
import pytest
from fusio.classes.gacode import gacode_io
from fusio.classes.plasma import plasma_io
from fusio.classes.imas import imas_io
from fusio.utils.eqdsk_tools import define_cocos_converter, determine_cocos_from_signs


SIGN_FLIPS = {
    'as_is': [],
    'all_flipped': ['polflux', 'q', 'current', 'bcentr', 'torfluxa'],
    'current_flipped': ['current', 'polflux'],
    'field_flipped': ['bcentr', 'torfluxa'],
}


def _plasma_with_flipped_signs(gacode_file_path, keys):
    g = gacode_io(input=gacode_file_path)
    d = g.input
    for k in keys:
        d[k] = -d[k]
    g.input = d
    p = plasma_io.from_gacode(g, side='input')
    p.output = p.input
    return p


@pytest.mark.parametrize('keys', SIGN_FLIPS.values(), ids=SIGN_FLIPS.keys())
def test_plasma_to_imas_is_cocos_17(gacode_file_path, tmp_path, keys):
    p = _plasma_with_flipped_signs(gacode_file_path, keys)
    p.to('imas', side='output').write(tmp_path / 'imas', side='input')
    d = imas_io(input=tmp_path / 'imas').input
    psi = d['equilibrium.time_slice.profiles_1d.psi'].to_numpy().flatten()
    q = d['equilibrium.time_slice.profiles_1d.q'].to_numpy().flatten()
    ip = d['equilibrium.time_slice.global_quantities.ip'].to_numpy().flatten()[0]
    b0 = d['equilibrium.vacuum_toroidal_field.b0'].to_numpy().flatten()[0]
    assert determine_cocos_from_signs(ip, b0, psi[-1] - psi[0], q[-1], per_radian=False) == 17


@pytest.mark.parametrize('keys', SIGN_FLIPS.values(), ids=SIGN_FLIPS.keys())
def test_plasma_imas_roundtrip(gacode_file_path, tmp_path, keys):
    p = _plasma_with_flipped_signs(gacode_file_path, keys)
    p.to('imas', side='output').write(tmp_path / 'imas', side='input')
    back = imas_io(input=tmp_path / 'imas').to('plasma', side='input')
    assert back.input_cocos == 7  # Per radian member of the COCOS 17 family
    cocos = define_cocos_converter(p.output_cocos, back.input_cocos)
    src = p.output.isel(time=-1)
    dst = back.input.isel(time=-1)
    factors = {
        ('magnetic_flux', 'poloidal'): cocos['sBp'] * cocos['scyl'],
        ('magnetic_flux', 'toroidal'): cocos['scyl'],
        ('safety_factor', None): cocos['spol'],
        ('field_axis', None): cocos['scyl'],
        ('current', None): cocos['scyl'],
        ('density_i', None): 1,
        ('temperature_i', None): 1,
    }
    for (var, direction), factor in factors.items():
        a = src[var].sel(direction=direction) if direction else src[var]
        b = dst[var].sel(direction=direction) if direction else dst[var]
        np.testing.assert_allclose(b.transpose(*a.dims).to_numpy(), factor * a.to_numpy(), rtol=1.0e-12, atol=1.0e-12 * float(np.max(np.abs(a))), err_msg=var)


def test_plasma_to_imas_rectangular_psi(gacode_file_path, tmp_path):
    import imas
    from pathlib import Path
    eqdsk = Path(gacode_file_path).parent / 'sample_cocos02_input.geqdsk'
    p = plasma_io.from_gacode(gacode_io(input=gacode_file_path), side='input')
    p.add_geometry_from_eqdsk(eqdsk, side='input')
    p.output = p.input
    p.to('imas', side='output').write(tmp_path / 'imas', side='input')
    with imas.DBEntry(str(tmp_path / 'imas' / 'equilibrium.nc'), 'r') as db:
        ts = db.get('equilibrium').time_slice[0]
    # Rectangular map first, then the flux-surface contours, each on its own (unpadded) grid
    assert [p2d.grid_type.name for p2d in ts.profiles_2d] == ['rectangular', 'inverse_rhotornorm_polar']
    rect, polar = ts.profiles_2d
    assert rect.psi.shape == (p.input['r_map'].size, p.input['z_map'].size)
    np.testing.assert_allclose(rect.grid.dim1, p.input['r_map'].to_numpy())
    np.testing.assert_allclose(rect.grid.dim2, p.input['z_map'].to_numpy())
    assert polar.psi.shape == (p.input['radius'].size, p.input['poloidal_index'].size)
    assert np.all(np.isfinite(polar.r))
    # Map and 1D flux are converted to COCOS 17 with the same factor
    cocos = define_cocos_converter(p.output_cocos, 17)
    scale = 2.0 * np.pi * cocos['sBp'] * cocos['scyl']
    np.testing.assert_allclose(rect.psi, scale * p.input['poloidal_flux_map'].isel(time=0).to_numpy())
    np.testing.assert_allclose(ts.profiles_1d.psi, scale * p.input['magnetic_flux'].sel(direction='poloidal').isel(time=0).to_numpy())


def test_plasma_to_imas_eqdsk_profiles_and_reference(gacode_file_path, tmp_path):
    import imas
    from pathlib import Path
    from fusio.utils.eqdsk_tools import read_eqdsk
    eqdsk = Path(gacode_file_path).parent / 'sample_cocos02_input.geqdsk'
    p = plasma_io.from_gacode(gacode_io(input=gacode_file_path), side='input')
    p.add_geometry_from_eqdsk(eqdsk, side='input')
    p.output = p.input
    p.to('imas', side='output').write(tmp_path / 'imas', side='input')
    with imas.DBEntry(str(tmp_path / 'imas' / 'equilibrium.nc'), 'r') as db:
        ids = db.get('equilibrium')
    eq = read_eqdsk(eqdsk)
    assert np.isclose(float(ids.vacuum_toroidal_field.r0), eq['rcentr'])
    assert np.isclose(abs(float(ids.time_slice[0].global_quantities.ip)), abs(eq['cpasma']))
    cocos = define_cocos_converter(p.output_cocos, 17)
    psi_scale = 2.0 * np.pi * cocos['sBp'] * cocos['scyl']
    p1d = ids.time_slice[0].profiles_1d
    src = p.output.isel(time=0)
    np.testing.assert_allclose(p1d.f, cocos['scyl'] * src['diamagnetic_function'].to_numpy())
    np.testing.assert_allclose(p1d.pressure, src['pressure_equilibrium'].to_numpy())
    np.testing.assert_allclose(p1d.f_df_dpsi, src['f_df_dpsi'].to_numpy() / psi_scale)
    np.testing.assert_allclose(p1d.dpressure_dpsi, src['dpressure_dpsi'].to_numpy() / psi_scale)
    # GS consistency of the exported convention: FF' = F dF/dpsi
    np.testing.assert_allclose(np.gradient(0.5 * p1d.f ** 2, p1d.psi)[5:-5], p1d.f_df_dpsi[5:-5], rtol=5e-2, atol=5e-2 * np.max(np.abs(p1d.f_df_dpsi)))


def test_plasma_to_imas_eqdsk_roundtrip(gacode_file_path, tmp_path):
    from pathlib import Path
    from fusio.utils.eqdsk_tools import read_eqdsk
    eqdsk = Path(gacode_file_path).parent / 'sample_cocos02_input.geqdsk'
    p = plasma_io.from_gacode(gacode_io(input=gacode_file_path), side='input')
    p.add_geometry_from_eqdsk(eqdsk, side='input')
    p.output = p.input
    p.to('imas', side='output').write(tmp_path / 'imas', side='input')
    # The rectangular map is padded to the contour grid inside imas_io, which must not leak into the EQDSK.
    # Written back in the convention inferred on insertion, as signs cannot tell COCOS 2 from 1
    a = read_eqdsk(eqdsk)
    inferred = determine_cocos_from_signs(a['cpasma'], a['bcentr'], a['sibdry'] - a['simagx'], a['qpsi'][-1], per_radian=True)
    imas_io(input=tmp_path / 'imas').generate_eqdsk_file(tmp_path / 'regen.geqdsk', side='input', cocos=inferred)
    b = read_eqdsk(tmp_path / 'regen.geqdsk')
    assert (b['nr'], b['nz']) == (a['nr'], a['nz'])
    for key in ['rcentr', 'bcentr', 'cpasma', 'simagx', 'sibdry']:
        assert np.isclose(b[key], a[key], rtol=1e-6), key
    np.testing.assert_allclose(b['psi'], a['psi'], rtol=1e-6, atol=1e-9 * np.max(np.abs(a['psi'])))
    # Resampled through the radial grid of the plasma state, which is coarse near the edge for this GACODE file,
    # so the steep derivative profiles get a looser tolerance
    for key, tol in [('fpol', 1e-2), ('pres', 1e-2), ('ffprime', 5e-2), ('pprime', 5e-2)]:
        np.testing.assert_allclose(b[key], a[key], rtol=0.0, atol=tol * np.max(np.abs(a[key])), err_msg=key)
