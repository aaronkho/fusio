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
