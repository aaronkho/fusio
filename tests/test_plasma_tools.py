import pytest
import numpy as np
import xarray as xr
from fusio.utils import plasma_tools as pt


@pytest.mark.usefixtures(
    'physical_2ion_plasma_state',
    'dimensionless_2ion_plasma_state',
    'physical_3ion_plasma_state',
    'dimensionless_3ion_plasma_state',
)
class TestPlasmaTools():

    def test_constants(self):
        c = pt.constants_si()
        assert c.get('e') == 1.60218e-19
        assert c.get('u') == 1.66054e-27
        assert c.get('eps') == 8.85419e-12
        assert c.get('mu') == 4.0e-7 * np.pi
        assert c.get('me') == 5.4858e-4 * 1.66054e-27
        assert c.get('mp') == (1.0 + 7.2764e-3) * 1.66054e-27

    def test_unnormalize(self):
        assert pt.unnormalize(0.5, 5.0) == 2.5

    def test_normalize(self):
        assert pt.normalize(2.5, 5.0) == 0.5

    def test_calc_a_from_epsilon(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        epsilon = dimensionless_2ion_plasma_state['epsilon_lcfs']
        rmaj = physical_2ion_plasma_state['r_geometric_lcfs']
        a = pt.calc_a_from_epsilon(epsilon, rmaj)
        xr.testing.assert_allclose(a, physical_2ion_plasma_state['r_minor_lcfs'])

    def test_calc_epsilon_from_a(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        a = physical_2ion_plasma_state['r_minor_lcfs']
        rmaj = physical_2ion_plasma_state['r_geometric_lcfs']
        epsilon = pt.calc_epsilon_from_a(a, rmaj)
        xr.testing.assert_allclose(epsilon, dimensionless_2ion_plasma_state['epsilon_lcfs'])

    def test_calc_r_from_x(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        x = dimensionless_2ion_plasma_state['x']
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        rmin = pt.calc_r_from_x(x, lref)
        xr.testing.assert_allclose(rmin, physical_2ion_plasma_state['r_minor'])

    def test_calc_x_from_r(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        rmin = physical_2ion_plasma_state['r_minor']
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        x = pt.calc_x_from_r(rmin, lref)
        xr.testing.assert_allclose(x, dimensionless_2ion_plasma_state['x'])

    def test_calc_ni_from_ninorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ninorm = dimensionless_2ion_plasma_state['density_i_norm']
        nref = physical_2ion_plasma_state['density_e']
        ni = pt.calc_ni_from_ninorm(ninorm, nref, nscale=1.0)
        xr.testing.assert_allclose(ni, physical_2ion_plasma_state['density_i'])

    def test_calc_ti_from_tinorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        tinorm = dimensionless_2ion_plasma_state['temperature_i_norm']
        tref = physical_2ion_plasma_state['temperature_e']
        ti = pt.calc_ti_from_tinorm(tinorm, tref, tscale=1.0)
        xr.testing.assert_allclose(ti, physical_2ion_plasma_state['temperature_i'])

    def test_calc_ninorm_from_ni(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ni = physical_2ion_plasma_state['density_i']
        nref = physical_2ion_plasma_state['density_e']
        ninorm = pt.calc_ninorm_from_ni(ni, nref)
        xr.testing.assert_allclose(ninorm, dimensionless_2ion_plasma_state['density_i_norm'])

    def test_calc_tinorm_from_ti(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ti = physical_2ion_plasma_state['temperature_i']
        tref = physical_2ion_plasma_state['temperature_e']
        tinorm = pt.calc_tinorm_from_ti(ti, tref)
        xr.testing.assert_allclose(tinorm, dimensionless_2ion_plasma_state['temperature_i_norm'])

    def test_calc_ak_from_grad_k(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        k = physical_2ion_plasma_state['temperature_e']
        grad_k = physical_2ion_plasma_state['grad_temperature_e']
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ak = pt.calc_ak_from_grad_k(grad_k, k, lref)
        xr.testing.assert_allclose(ak, dimensionless_2ion_plasma_state['grad_temperature_e_norm'])

    def test_calc_grad_k_from_ak(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        k = physical_2ion_plasma_state['temperature_e']
        ak = dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        grad_k = pt.calc_grad_k_from_ak(ak, k, lref)
        xr.testing.assert_allclose(grad_k, physical_2ion_plasma_state['grad_temperature_e'])

    def test_calc_q_circular_from_bp(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        bp = physical_2ion_plasma_state['field_geometric'].sel(direction='poloidal', drop=True)
        bt = physical_2ion_plasma_state['field_geometric'].sel(direction='toroidal', drop=True)
        rmin = physical_2ion_plasma_state['r_minor']
        rmaj = physical_2ion_plasma_state['r_geometric']
        q = pt.calc_q_circular_from_bp(bp, rmin, bt, rmaj)
        xr.testing.assert_allclose(q, dimensionless_2ion_plasma_state['safety_factor_circular'])

    def test_calc_bp_from_q_circular(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        bt = physical_2ion_plasma_state['field_geometric'].sel(direction='toroidal', drop=True)
        rmin = physical_2ion_plasma_state['r_minor']
        rmaj = physical_2ion_plasma_state['r_geometric']
        bp = pt.calc_bp_from_q_circular(q, rmin, bt, rmaj)
        xr.testing.assert_allclose(bp, physical_2ion_plasma_state['field_geometric'].sel(direction='poloidal', drop=True))

    def test_calc_grad_q_circular_from_s(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        s = dimensionless_2ion_plasma_state['magnetic_shear_circular']
        bp = physical_2ion_plasma_state['field_geometric'].sel(direction='poloidal', drop=True)
        bt = physical_2ion_plasma_state['field_geometric'].sel(direction='toroidal', drop=True)
        rmaj = physical_2ion_plasma_state['r_geometric']
        grad_q = pt.calc_grad_q_circular_from_s(s, bp, bt, rmaj)
        xr.testing.assert_allclose(grad_q, dimensionless_2ion_plasma_state['grad_safety_factor_circular'])

    def test_calc_grad_q_from_s_and_q(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        s = dimensionless_2ion_plasma_state['magnetic_shear_circular']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        rmin = physical_2ion_plasma_state['r_minor']
        grad_q = pt.calc_grad_q_from_s_and_q(s, q, rmin)
        xr.testing.assert_allclose(grad_q, dimensionless_2ion_plasma_state['grad_safety_factor_circular'])

    def test_calc_s_circular_from_grad_bp(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        grad_bp = physical_2ion_plasma_state['grad_field_geometric'].sel(direction='poloidal', drop=True)
        bp = physical_2ion_plasma_state['field_geometric'].sel(direction='poloidal', drop=True)
        rmin = physical_2ion_plasma_state['r_minor']
        s = pt.calc_s_circular_from_grad_bp(grad_bp, bp, rmin)
        xr.testing.assert_allclose(s, dimensionless_2ion_plasma_state['magnetic_shear_circular'])

    def test_calc_grad_bp_from_grad_q_circular(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        grad_q = dimensionless_2ion_plasma_state['grad_safety_factor_circular']
        bp = physical_2ion_plasma_state['field_geometric'].sel(direction='poloidal', drop=True)
        bt = physical_2ion_plasma_state['field_geometric'].sel(direction='toroidal', drop=True)
        rmin = physical_2ion_plasma_state['r_minor']
        rmaj = physical_2ion_plasma_state['r_geometric']
        grad_bp = pt.calc_grad_bp_from_grad_q_circular(grad_q, bp, rmin, bt, rmaj)
        xr.testing.assert_allclose(grad_bp, physical_2ion_plasma_state['grad_field_geometric'].sel(direction='poloidal', drop=True))

    def test_calc_grad_bp_from_s_circular(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        s = dimensionless_2ion_plasma_state['magnetic_shear_circular']
        bp = physical_2ion_plasma_state['field_geometric'].sel(direction='poloidal', drop=True)
        rmin = physical_2ion_plasma_state['r_minor']
        grad_bp = pt.calc_grad_bp_from_s_circular(s, bp, rmin)
        xr.testing.assert_allclose(grad_bp, physical_2ion_plasma_state['grad_field_geometric'].sel(direction='poloidal', drop=True))

    def test_calc_ninorm_from_zeff_and_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        ninormc = pt.calc_ninorm_from_zeff_and_quasineutrality(zeff, zia, zib, zic, ninorma)
        xr.testing.assert_allclose(ninormc, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True))

    def test_calc_ninorm_from_quasineutrality(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = pt.calc_ninorm_from_quasineutrality(zia, zib, ninorma)
        xr.testing.assert_allclose(ninormb, dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True))

    def test_calc_2ion_ninorm_from_ni_and_quasineutrality(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        ni_test1 = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        ni_test2 = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        ninorma_test1, ninormb_test1 = pt.calc_2ion_ninorm_from_ninorm_and_quasineutrality(ni_test1, zia, zib, ne=ne)
        ninorma_test2, ninormb_test2 = pt.calc_2ion_ninorm_from_ninorm_and_quasineutrality(ni_test2, zib, zia, ne=ne)
        xr.testing.assert_allclose(ninorma_test1, dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(ninormb_test1, dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(ninorma_test2, dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(ninormb_test2, dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True))

    def test_calc_2ion_ninorm_from_ninorm_and_quasineutrality(self, dimensionless_2ion_plasma_state):
        ninorm_test1 = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninorm_test2 = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        ninorma_test1, ninormb_test1 = pt.calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorm_test1, zia, zib, ne=None)
        ninorma_test2, ninormb_test2 = pt.calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorm_test2, zib, zia, ne=None)
        xr.testing.assert_allclose(ninorma_test1, ninorm_test1)
        xr.testing.assert_allclose(ninormb_test1, ninorm_test2)
        xr.testing.assert_allclose(ninorma_test2, ninorm_test2)
        xr.testing.assert_allclose(ninormb_test2, ninorm_test1)

    def test_calc_3ion_ninorm_from_ni_zeff_and_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        ni_test1 = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        ni_test2 = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        ninorma_test1, ninormb_test1, ninormc_test1 = pt.calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ni_test1, zeff, zia, zib, zic, ne=ne)
        ninorma_test2, ninormb_test2, ninormc_test2 = pt.calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ni_test2, zeff, zib, zic, zia, ne=ne)
        xr.testing.assert_allclose(ninorma_test1, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(ninormb_test1, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(ninormc_test1, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(ninorma_test2, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(ninormb_test2, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(ninormc_test2, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True))

    def test_calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(self, dimensionless_3ion_plasma_state):
        ninorm_test1 = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninorm_test2 = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        ninorma_test1, ninormb_test1, ninormc_test1 = pt.calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ninorm_test1, zeff, zia, zib, zic, ne=None)
        ninorma_test2, ninormb_test2, ninormc_test2 = pt.calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ninorm_test2, zeff, zib, zic, zia, ne=None)
        xr.testing.assert_allclose(ninorma_test1, ninorm_test1)
        xr.testing.assert_allclose(ninormb_test1, ninorm_test2)
        xr.testing.assert_allclose(ninormc_test1, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(ninorma_test2, ninorm_test2)
        xr.testing.assert_allclose(ninormb_test2, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(ninormc_test2, ninorm_test1)

    def test_calc_3ion_ninorm_from_ni_and_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        ninorma_test1, ninormb_test1, ninormc_test1 = pt.calc_3ion_ninorm_from_ninorm_and_quasineutrality(nia, nib, zia, zib, zic, ne=ne)
        ninorma_test2, ninormb_test2, ninormc_test2 = pt.calc_3ion_ninorm_from_ninorm_and_quasineutrality(nib, nic, zib, zic, zia, ne=ne)
        xr.testing.assert_allclose(ninorma_test1, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(ninormb_test1, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(ninormc_test1, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(ninorma_test2, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(ninormb_test2, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(ninormc_test2, dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True))

    def test_calc_3ion_ninorm_from_ninorm_and_quasineutrality(self, dimensionless_3ion_plasma_state):
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        ninorma_test1, ninormb_test1, ninormc_test1 = pt.calc_3ion_ninorm_from_ninorm_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=None)
        ninorma_test2, ninormb_test2, ninormc_test2 = pt.calc_3ion_ninorm_from_ninorm_and_quasineutrality(ninormb, ninormc, zib, zic, zia, ne=None)
        xr.testing.assert_allclose(ninorma_test1, ninorma)
        xr.testing.assert_allclose(ninormb_test1, ninormb)
        xr.testing.assert_allclose(ninormc_test1, ninormc)
        xr.testing.assert_allclose(ninorma_test2, ninormb)
        xr.testing.assert_allclose(ninormb_test2, ninormc)
        xr.testing.assert_allclose(ninormc_test2, ninorma)

    def test_calc_ni_from_zeff_and_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        nic = pt.calc_ni_from_zeff_and_quasineutrality(zeff, zia, zib, zic, nia, ne)
        xr.testing.assert_allclose(nic, physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True))

    def test_calc_ni_from_quasineutrality(self, physical_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        nib = pt.calc_ni_from_quasineutrality(zia, zib, nia, ne)
        xr.testing.assert_allclose(nib, physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True))

    def test_calc_2ion_ni_from_ni_and_quasineutrality(self, physical_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        ni_test1 = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        ni_test2 = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        nia_test1, nib_test1 = pt.calc_2ion_ni_from_ni_and_quasineutrality(ni_test1, zia, zib, ne, norm_inputs=False)
        nia_test2, nib_test2 = pt.calc_2ion_ni_from_ni_and_quasineutrality(ni_test2, zib, zia, ne, norm_inputs=False)
        xr.testing.assert_allclose(nia_test1, ni_test1)
        xr.testing.assert_allclose(nib_test1, ni_test2)
        xr.testing.assert_allclose(nia_test2, ni_test2)
        xr.testing.assert_allclose(nib_test2, ni_test1)

    def test_calc_2ion_ni_from_ninorm_and_quasineutrality(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        ninorm_test1 = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninorm_test2 = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        nia_test1, nib_test1 = pt.calc_2ion_ni_from_ni_and_quasineutrality(ninorm_test1, zia, zib, ne, norm_inputs=True)
        nia_test2, nib_test2 = pt.calc_2ion_ni_from_ni_and_quasineutrality(ninorm_test2, zib, zia, ne, norm_inputs=True)
        xr.testing.assert_allclose(nia_test1, physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(nib_test1, physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(nia_test2, physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(nib_test2, physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True))

    def test_calc_3ion_ni_from_ni_zeff_and_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        ni_test1 = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        ni_test2 = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        nia_test1, nib_test1, nic_test1 = pt.calc_3ion_ni_from_ni_zeff_and_quasineutrality(ni_test1, zeff, zia, zib, zic, ne, norm_inputs=False)
        nia_test2, nib_test2, nic_test2 = pt.calc_3ion_ni_from_ni_zeff_and_quasineutrality(ni_test2, zeff, zib, zic, zia, ne, norm_inputs=False)
        xr.testing.assert_allclose(nia_test1, ni_test1)
        xr.testing.assert_allclose(nib_test1, ni_test2)
        xr.testing.assert_allclose(nic_test1, physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(nia_test2, ni_test2)
        xr.testing.assert_allclose(nib_test2, physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(nic_test2, ni_test1)

    def test_calc_3ion_ni_from_ninorm_zeff_and_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        ninorm_test1 = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninorm_test2 = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        nia_test1, nib_test1, nic_test1 = pt.calc_3ion_ni_from_ni_zeff_and_quasineutrality(ninorm_test1, zeff, zia, zib, zic, ne, norm_inputs=True)
        nia_test2, nib_test2, nic_test2 = pt.calc_3ion_ni_from_ni_zeff_and_quasineutrality(ninorm_test2, zeff, zib, zic, zia, ne, norm_inputs=True)
        xr.testing.assert_allclose(nia_test1, physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(nib_test1, physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(nic_test1, physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(nia_test2, physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(nib_test2, physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(nic_test2, physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True))

    def test_calc_3ion_ni_from_ni_and_quasineutrality(self, physical_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        nia_test1, nib_test1, nic_test1 = pt.calc_3ion_ni_from_ni_and_quasineutrality(nia, nib, zia, zib, zic, ne, norm_inputs=False)
        nia_test2, nib_test2, nic_test2 = pt.calc_3ion_ni_from_ni_and_quasineutrality(nib, nic, zib, zic, zia, ne, norm_inputs=False)
        xr.testing.assert_allclose(nia_test1, nia)
        xr.testing.assert_allclose(nib_test1, nib)
        xr.testing.assert_allclose(nic_test1, nic)
        xr.testing.assert_allclose(nia_test2, nib)
        xr.testing.assert_allclose(nib_test2, nic)
        xr.testing.assert_allclose(nic_test2, nia)

    def test_calc_3ion_ni_from_ninorm_and_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        nia_test1, nib_test1, nic_test1 = pt.calc_3ion_ni_from_ni_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne, norm_inputs=True)
        nia_test2, nib_test2, nic_test2 = pt.calc_3ion_ni_from_ni_and_quasineutrality(ninormb, ninormc, zib, zic, zia, ne, norm_inputs=True)
        xr.testing.assert_allclose(nia_test1, physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(nib_test1, physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(nic_test1, physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(nia_test2, physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(nib_test2, physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(nic_test2, physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True))

    def test_calc_ani_from_azeff_and_gradient_quasineutrality(self, dimensionless_3ion_plasma_state):
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        azeff = dimensionless_3ion_plasma_state['grad_effective_charge_norm']
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        anic = pt.calc_ani_from_azeff_and_gradient_quasineutrality(azeff, zeff, zia, zib, zic, ninorma, ninormc, ane, ania)
        xr.testing.assert_allclose(anic, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True))

    def test_calc_ani_from_gradient_quasineutrality(self, dimensionless_2ion_plasma_state):
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        anib = pt.calc_ani_from_gradient_quasineutrality(zia, zib, ninorma, ninormb, ane, ania)
        xr.testing.assert_allclose(anib, dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True))

    def test_calc_2ion_ani_from_ani_and_gradient_quasineutrality(self, dimensionless_2ion_plasma_state):
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ninorm_test1 = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninorm_test2 = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ani_test1 = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        ani_test2 = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        ania_test1, anib_test1 = pt.calc_2ion_ani_from_ani_and_gradient_quasineutrality(ani_test1, zia, zib, ane, ninorm_test1, ne=None, lref=None)
        ania_test2, anib_test2 = pt.calc_2ion_ani_from_ani_and_gradient_quasineutrality(ani_test2, zib, zia, ane, ninorm_test2, ne=None, lref=None)
        xr.testing.assert_allclose(ania_test1, ani_test1)
        xr.testing.assert_allclose(anib_test1, ani_test2)
        xr.testing.assert_allclose(ania_test2, ani_test2)
        xr.testing.assert_allclose(anib_test2, ani_test1)

    def test_calc_2ion_ani_from_grad_ni_and_gradient_quasineutrality(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        ni_test1 = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        ni_test2 = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        grad_ni_test1 = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_ni_test2 = physical_2ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        ania_test1, anib_test1 = pt.calc_2ion_ani_from_ani_and_gradient_quasineutrality(grad_ni_test1, zia, zib, grad_ne, ni_test1, ne=ne, lref=lref)
        ania_test2, anib_test2 = pt.calc_2ion_ani_from_ani_and_gradient_quasineutrality(grad_ni_test2, zib, zia, grad_ne, ni_test2, ne=ne, lref=lref)
        xr.testing.assert_allclose(ania_test1, dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(anib_test1, dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(ania_test2, dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(anib_test2, dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True))

    def test_calc_3ion_ani_from_ani_azeff_and_gradient_quasineutrality(self, dimensionless_3ion_plasma_state):
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ninorm_test1 = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninorm_test2 = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ani_test1 = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        ani_test2 = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        azeff = dimensionless_3ion_plasma_state['grad_effective_charge_norm']
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        ania_test1, anib_test1, anic_test1 = pt.calc_3ion_ani_from_ani_azeff_and_gradient_quasineutrality(ani_test1, azeff, zeff, zia, zib, zic, ane, ninorm_test1, ne=None, lref=None)
        ania_test2, anib_test2, anic_test2 = pt.calc_3ion_ani_from_ani_azeff_and_gradient_quasineutrality(ani_test2, azeff, zeff, zib, zic, zia, ane, ninorm_test2, ne=None, lref=None)
        xr.testing.assert_allclose(ania_test1, ani_test1)
        xr.testing.assert_allclose(anib_test1, ani_test2)
        xr.testing.assert_allclose(anic_test1, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(ania_test2, ani_test2)
        xr.testing.assert_allclose(anib_test2, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(anic_test2, ani_test1)

    def test_calc_3ion_ani_from_grad_ni_grad_zeff_and_gradient_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        ni_test1 = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        ni_test2 = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        grad_ni_test1 = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_ni_test2 = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        grad_zeff = dimensionless_3ion_plasma_state['grad_effective_charge']
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        ania_test1, anib_test1, anic_test1 = pt.calc_3ion_ani_from_ani_azeff_and_gradient_quasineutrality(grad_ni_test1, grad_zeff, zeff, zia, zib, zic, grad_ne, ni_test1, ne=ne, lref=lref)
        ania_test2, anib_test2, anic_test2 = pt.calc_3ion_ani_from_ani_azeff_and_gradient_quasineutrality(grad_ni_test2, grad_zeff, zeff, zib, zic, zia, grad_ne, ni_test2, ne=ne, lref=lref)
        xr.testing.assert_allclose(ania_test1, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(anib_test1, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(anic_test1, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(ania_test2, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(anib_test2, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(anic_test2, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True))

    def test_calc_3ion_ani_from_ani_and_gradient_quasineutrality(self, dimensionless_3ion_plasma_state):
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        anic = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        ania_test1, anib_test1, anic_test1 = pt.calc_3ion_ani_from_ani_and_gradient_quasineutrality(ania, anib, zia, zib, zic, ane, ninorma, ninormb, ne=None, lref=None)
        ania_test2, anib_test2, anic_test2 = pt.calc_3ion_ani_from_ani_and_gradient_quasineutrality(anib, anic, zib, zic, zia, ane, ninormb, ninormc, ne=None, lref=None)
        xr.testing.assert_allclose(ania_test1, ania)
        xr.testing.assert_allclose(anib_test1, anib)
        xr.testing.assert_allclose(anic_test1, anic)
        xr.testing.assert_allclose(ania_test2, anib)
        xr.testing.assert_allclose(anib_test2, anic)
        xr.testing.assert_allclose(anic_test2, ania)

    def test_calc_3ion_ani_from_grad_ni_and_gradient_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_nic = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        ania_test1, anib_test1, anic_test1 = pt.calc_3ion_ani_from_ani_and_gradient_quasineutrality(grad_nia, grad_nib, zia, zib, zic, grad_ne, nia, nib, ne=ne, lref=lref)
        ania_test2, anib_test2, anic_test2 = pt.calc_3ion_ani_from_ani_and_gradient_quasineutrality(grad_nib, grad_nic, zib, zic, zia, grad_ne, nib, nic, ne=ne, lref=lref)
        xr.testing.assert_allclose(ania_test1, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(anib_test1, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(anic_test1, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(ania_test2, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(anib_test2, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(anic_test2, dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True))

    def test_calc_grad_ni_from_grad_zeff_and_gradient_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        grad_zeff = dimensionless_3ion_plasma_state['grad_effective_charge']
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        grad_nic = pt.calc_grad_ni_from_grad_zeff_and_gradient_quasineutrality(grad_zeff, zeff, zia, zib, zic, nia, nic, grad_nia, ne, grad_ne, lref)
        xr.testing.assert_allclose(grad_nic, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True))

    def test_calc_grad_ni_from_gradient_quasineutrality(self, physical_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        grad_nib = pt.calc_grad_ni_from_gradient_quasineutrality(zia, zib, nia, nib, grad_nia, ne, grad_ne, lref)
        xr.testing.assert_allclose(grad_nib, physical_2ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True))

    def test_calc_2ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(self, physical_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        ni_test1 = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        ni_test2 = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        grad_ni_test1 = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_ni_test2 = physical_2ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        grad_nia_test1, grad_nib_test1 = pt.calc_2ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(grad_ni_test1, zia, zib, ni_test1, grad_ne, ne, lref, norm_inputs=False)
        grad_nia_test2, grad_nib_test2 = pt.calc_2ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(grad_ni_test2, zib, zia, ni_test2, grad_ne, ne, lref, norm_inputs=False)
        xr.testing.assert_allclose(grad_nia_test1, grad_ni_test1)
        xr.testing.assert_allclose(grad_nib_test1, grad_ni_test2)
        xr.testing.assert_allclose(grad_nia_test2, grad_ni_test2)
        xr.testing.assert_allclose(grad_nib_test2, grad_ni_test1)

    def test_calc_2ion_grad_ni_from_ani_and_gradient_quasineutrality(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ninorm_test1 = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninorm_test2 = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ani_test1 = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        ani_test2 = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        grad_nia_test1, grad_nib_test1 = pt.calc_2ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(ani_test1, zia, zib, ninorm_test1, ane, ne, lref, norm_inputs=True)
        grad_nia_test2, grad_nib_test2 = pt.calc_2ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(ani_test2, zib, zia, ninorm_test2, ane, ne, lref, norm_inputs=True)
        xr.testing.assert_allclose(grad_nia_test1, physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(grad_nib_test1, physical_2ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(grad_nia_test2, physical_2ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(grad_nib_test2, physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True))

    def test_calc_3ion_grad_ni_from_grad_ni_grad_zeff_and_gradient_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        ni_test1 = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        ni_test2 = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        grad_ni_test1 = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_ni_test2 = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        grad_zeff = dimensionless_3ion_plasma_state['grad_effective_charge']
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        grad_nia_test1, grad_nib_test1, grad_nic_test1 = pt.calc_3ion_grad_ni_from_grad_ni_grad_zeff_and_gradient_quasineutrality(grad_ni_test1, grad_zeff, zeff, zia, zib, zic, ni_test1, grad_ne, ne, lref, norm_inputs=False)
        grad_nia_test2, grad_nib_test2, grad_nic_test2 = pt.calc_3ion_grad_ni_from_grad_ni_grad_zeff_and_gradient_quasineutrality(grad_ni_test2, grad_zeff, zeff, zib, zic, zia, ni_test2, grad_ne, ne, lref, norm_inputs=False)
        xr.testing.assert_allclose(grad_nia_test1, grad_ni_test1)
        xr.testing.assert_allclose(grad_nib_test1, grad_ni_test2)
        xr.testing.assert_allclose(grad_nic_test1, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(grad_nia_test2, grad_ni_test2)
        xr.testing.assert_allclose(grad_nib_test2, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(grad_nic_test2, grad_ni_test1)

    def test_calc_3ion_grad_ni_from_ani_azeff_and_gradient_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ninorm_test1 = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninorm_test2 = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ani_test1 = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        ani_test2 = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        azeff = dimensionless_3ion_plasma_state['grad_effective_charge_norm']
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        grad_nia_test1, grad_nib_test1, grad_nic_test1 = pt.calc_3ion_grad_ni_from_grad_ni_grad_zeff_and_gradient_quasineutrality(ani_test1, azeff, zeff, zia, zib, zic, ninorm_test1, ane, ne, lref, norm_inputs=True)
        grad_nia_test2, grad_nib_test2, grad_nic_test2 = pt.calc_3ion_grad_ni_from_grad_ni_grad_zeff_and_gradient_quasineutrality(ani_test2, azeff, zeff, zib, zic, zia, ninorm_test2, ane, ne, lref, norm_inputs=True)
        xr.testing.assert_allclose(grad_nia_test1, physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(grad_nib_test1, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(grad_nic_test1, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(grad_nia_test2, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(grad_nib_test2, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(grad_nic_test2, physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True))

    def test_calc_3ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(self, physical_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_nic = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        grad_nia_test1, grad_nib_test1, grad_nic_test1 = pt.calc_3ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(grad_nia, grad_nib, zia, zib, zic, nia, nib, grad_ne, ne, lref, norm_inputs=False)
        grad_nia_test2, grad_nib_test2, grad_nic_test2 = pt.calc_3ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(grad_nib, grad_nic, zib, zic, zia, nib, nic, grad_ne, ne, lref, norm_inputs=False)
        xr.testing.assert_allclose(grad_nia_test1, grad_nia)
        xr.testing.assert_allclose(grad_nib_test1, grad_nib)
        xr.testing.assert_allclose(grad_nic_test1, grad_nic)
        xr.testing.assert_allclose(grad_nia_test2, grad_nib)
        xr.testing.assert_allclose(grad_nib_test2, grad_nic)
        xr.testing.assert_allclose(grad_nic_test2, grad_nia)

    def test_calc_3ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        anic = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        grad_nia_test1, grad_nib_test1, grad_nic_test1 = pt.calc_3ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(ania, anib, zia, zib, zic, ninorma, ninormb, ane, ne, lref, norm_inputs=True)
        grad_nia_test2, grad_nib_test2, grad_nic_test2 = pt.calc_3ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(anib, anic, zib, zic, zia, ninormb, ninormc, ane, ne, lref, norm_inputs=True)
        xr.testing.assert_allclose(grad_nia_test1, physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True))
        xr.testing.assert_allclose(grad_nib_test1, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(grad_nic_test1, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(grad_nia_test2, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True))
        xr.testing.assert_allclose(grad_nib_test2, physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True))
        xr.testing.assert_allclose(grad_nic_test2, physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True))

    def test_calc_p_from_pnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        pnorm = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sel(ion='D', drop=True)
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        p = pt.calc_p_from_pnorm(pnorm, ne, te)
        xr.testing.assert_allclose(p, pt.e_si * (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['temperature_i']).sel(ion='D', drop=True))

    def test_calc_pnorm_from_p(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        p = pt.e_si * (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['temperature_i']).sel(ion='D', drop=True)
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        pnorm = pt.calc_pnorm_from_p(p, ne, te)
        xr.testing.assert_allclose(pnorm, (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sel(ion='D', drop=True))

    def test_calc_2ion_pnorm_with_2ions_norm(self, dimensionless_2ion_plasma_state):
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        pnorm = pt.calc_2ion_pnorm_with_2ions(ninorma, ninormb, tinorma, tinormb, ne=None, te=None)
        xr.testing.assert_allclose(pnorm, 1.0 + (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion'))

    def test_calc_2ion_pnorm_with_2ions_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        pnorm = pt.calc_2ion_pnorm_with_2ions(nia, nib, tia, tib, ne=ne, te=te)
        xr.testing.assert_allclose(pnorm, 1.0 + (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion'))

    def test_calc_2ion_pnorm_with_1ion_and_quasineutrality_norm(self, dimensionless_2ion_plasma_state):
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        pnorm = pt.calc_2ion_pnorm_with_1ion_and_quasineutrality(ninorma, tinorma, tinormb, zia, zib, ne=None, te=None)
        xr.testing.assert_allclose(pnorm, 1.0 + (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion'))

    def test_calc_2ion_pnorm_with_1ion_and_quasineutrality_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        pnorm = pt.calc_2ion_pnorm_with_1ion_and_quasineutrality(nia, tia, tib, zia, zib, ne=ne, te=te)
        xr.testing.assert_allclose(pnorm, 1.0 + (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion'))

    def test_calc_3ion_pnorm_with_3ions_norm(self, dimensionless_3ion_plasma_state):
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        pnorm = pt.calc_3ion_pnorm_with_3ions(ninorma, ninormb, ninormc, tinorma, tinormb, tinormc)
        xr.testing.assert_allclose(pnorm, 1.0 + (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion'))

    def test_calc_3ion_pnorm_with_3ions_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        pnorm = pt.calc_3ion_pnorm_with_3ions(nia, nib, nic, tia, tib, tic, ne=ne, te=te)
        xr.testing.assert_allclose(pnorm, 1.0 + (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion'))

    def test_calc_3ion_pnorm_with_2ions_and_quasineutrality_norm(self, dimensionless_3ion_plasma_state):
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        pnorm = pt.calc_3ion_pnorm_with_2ions_and_quasineutrality(ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic)
        xr.testing.assert_allclose(pnorm, 1.0 + (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion'))

    def test_calc_3ion_pnorm_with_2ions_and_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        pnorm = pt.calc_3ion_pnorm_with_2ions_and_quasineutrality(nia, nib, tia, tib, tic, zia, zib, zic, ne=ne, te=te)
        xr.testing.assert_allclose(pnorm, 1.0 + (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion'))

    def test_calc_3ion_pnorm_with_1ion_zeff_and_quasineutrality_norm(self, dimensionless_3ion_plasma_state):
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        pnorm = pt.calc_3ion_pnorm_with_1ion_zeff_and_quasineutrality(ninorma, tinorma, tinormb, tinormc, zeff, zia, zib, zic)
        xr.testing.assert_allclose(pnorm, 1.0 + (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion'))

    def test_calc_3ion_pnorm_with_1ion_zeff_and_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        pnorm = pt.calc_3ion_pnorm_with_1ion_zeff_and_quasineutrality(nia, tia, tib, tic, zeff, zia, zib, zic, ne=ne, te=te)
        xr.testing.assert_allclose(pnorm, 1.0 + (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion'))

    def test_calc_2ion_p_with_2ions_unnorm(self, physical_2ion_plasma_state):
        c = pt.constants_si()
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        p = pt.calc_2ion_p_with_2ions(nia, nib, tia, tib, ne, te, norm_inputs=False)
        xr.testing.assert_allclose(p, c['e'] * (physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['temperature_e'] + (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')))

    def test_calc_2ion_p_with_2ions_norm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        c = pt.constants_si()
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        p = pt.calc_2ion_p_with_2ions(ninorma, ninormb, tinorma, tinormb, ne, te, norm_inputs=True)
        xr.testing.assert_allclose(p, c['e'] * (physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['temperature_e'] + (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')))

    def test_calc_2ion_p_with_1ion_and_quasineutrality_unnorm(self, physical_2ion_plasma_state):
        c = pt.constants_si()
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        p = pt.calc_2ion_p_with_1ion_and_quasineutrality(nia, tia, tib, zia, zib, ne, te, norm_inputs=False)
        xr.testing.assert_allclose(p, c['e'] * (physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['temperature_e'] + (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')))

    def test_calc_2ion_p_with_1ion_and_quasineutrality_norm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        c = pt.constants_si()
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        p = pt.calc_2ion_p_with_1ion_and_quasineutrality(ninorma, tinorma, tinormb, zia, zib, ne, te, norm_inputs=True)
        xr.testing.assert_allclose(p, c['e'] * (physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['temperature_e'] + (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')))

    def test_calc_3ion_p_with_3ions_unnorm(self, physical_3ion_plasma_state):
        c = pt.constants_si()
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        p = pt.calc_3ion_p_with_3ions(nia, nib, nic, tia, tib, tic, ne, te, norm_inputs=False)
        xr.testing.assert_allclose(p, c['e'] * (physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['temperature_e'] + (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')))

    def test_calc_3ion_p_with_3ions_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        c = pt.constants_si()
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        p = pt.calc_3ion_p_with_3ions(ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, ne, te, norm_inputs=True)
        xr.testing.assert_allclose(p, c['e'] * (physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['temperature_e'] + (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')))

    def test_calc_3ion_p_with_2ions_and_quasineutrality_unnorm(self, physical_3ion_plasma_state):
        c = pt.constants_si()
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        p = pt.calc_3ion_p_with_2ions_and_quasineutrality(nia, nib, tia, tib, tic, zia, zib, zic, ne, te, norm_inputs=False)
        xr.testing.assert_allclose(p, c['e'] * (physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['temperature_e'] + (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')))

    def test_calc_3ion_p_with_2ions_and_quasineutrality_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        c = pt.constants_si()
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        p = pt.calc_3ion_p_with_2ions_and_quasineutrality(ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, ne, te, norm_inputs=True)
        xr.testing.assert_allclose(p, c['e'] * (physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['temperature_e'] + (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')))

    def test_calc_3ion_p_with_1ion_zeff_and_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        c = pt.constants_si()
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        p = pt.calc_3ion_p_with_1ion_zeff_and_quasineutrality(nia, tia, tib, tic, zeff, zia, zib, zic, ne, te, norm_inputs=False)
        xr.testing.assert_allclose(p, c['e'] * (physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['temperature_e'] + (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')))

    def test_calc_3ion_p_with_1ion_zeff_and_quasineutrality_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        c = pt.constants_si()
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        p = pt.calc_3ion_p_with_1ion_zeff_and_quasineutrality(ninorma, tinorma, tinormb, tinormc, zeff, zia, zib, zic, ne, te, norm_inputs=True)
        xr.testing.assert_allclose(p, c['e'] * (physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['temperature_e'] + (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')))

    def test_calc_grad_p_from_ap(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        c = pt.constants_si()
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        grad_npe = physical_2ion_plasma_state['grad_density_e'] * physical_2ion_plasma_state['temperature_e']
        grad_tpe = physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_2ion_plasma_state['grad_density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['grad_temperature_i']).sum('ion')
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = ape + anpi + atpi
        grad_p = pt.calc_grad_p_from_ap(ap, ne, te, lref)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_ap_from_grad_p(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        c = pt.constants_si()
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        grad_npe = physical_2ion_plasma_state['grad_density_e'] * physical_2ion_plasma_state['temperature_e']
        grad_tpe = physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_2ion_plasma_state['grad_density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['grad_temperature_i']).sum('ion')
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        grad_p = c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi)
        ap = pt.calc_ap_from_grad_p(grad_p, ne, te, lref)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_2ion_ap_with_2ions_norm(self, dimensionless_2ion_plasma_state):
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        atia = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = pt.calc_2ion_ap_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, ane, ate, ne=None, te=None, lref=None)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_2ion_ap_with_2ions_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        grad_te = physical_2ion_plasma_state['grad_temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_2ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_tia = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = pt.calc_2ion_ap_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, grad_ne, grad_te, ne=ne, te=te, lref=lref)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_2ion_ap_with_1ion_and_gradient_quasineutrality_norm(self, dimensionless_2ion_plasma_state):
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        atia = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = pt.calc_2ion_ap_with_1ion_and_gradient_quasineutrality(ania, atia, atib, ninorma, tinorma, tinormb, zia, zib, ane, ate, ne=None, te=None, lref=None)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_2ion_ap_with_1ion_and_gradient_quasineutrality_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        grad_te = physical_2ion_plasma_state['grad_temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_tia = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = pt.calc_2ion_ap_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, grad_ne, grad_te, ne=ne, te=te, lref=lref)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_3ion_ap_with_3ions_norm(self, dimensionless_3ion_plasma_state):
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        anic = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        ape = dimensionless_3ion_plasma_state['grad_density_e_norm'] + dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_3ion_plasma_state['grad_density_i_norm'] * dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['grad_temperature_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = pt.calc_3ion_ap_with_3ions(ania, anib, anic, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, ane, ate, ne=None, te=None, lref=None)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_3ion_ap_with_3ions_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_nic = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        ape = dimensionless_3ion_plasma_state['grad_density_e_norm'] + dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_3ion_plasma_state['grad_density_i_norm'] * dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['grad_temperature_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = pt.calc_3ion_ap_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, grad_ne, grad_te, ne=ne, te=te, lref=lref)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_3ion_ap_with_2ions_and_gradient_quasineutrality_norm(self, dimensionless_3ion_plasma_state):
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        ape = dimensionless_3ion_plasma_state['grad_density_e_norm'] + dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_3ion_plasma_state['grad_density_i_norm'] * dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['grad_temperature_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = pt.calc_3ion_ap_with_2ions_and_gradient_quasineutrality(ania, anib, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, ane, ate, ne=None, te=None, lref=None)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_3ion_ap_with_2ions_and_gradient_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        ape = dimensionless_3ion_plasma_state['grad_density_e_norm'] + dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_3ion_plasma_state['grad_density_i_norm'] * dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['grad_temperature_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = pt.calc_3ion_ap_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, grad_ne, grad_te, ne=ne, te=te, lref=lref)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality_norm(self, dimensionless_3ion_plasma_state):
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        azeff = dimensionless_3ion_plasma_state['grad_effective_charge_norm']
        ape = dimensionless_3ion_plasma_state['grad_density_e_norm'] + dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_3ion_plasma_state['grad_density_i_norm'] * dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['grad_temperature_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = pt.calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(ania, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic, ane, ate, ne=None, te=None, lref=None)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        grad_zeff = dimensionless_3ion_plasma_state['grad_effective_charge']
        ape = dimensionless_3ion_plasma_state['grad_density_e_norm'] + dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_3ion_plasma_state['grad_density_i_norm'] * dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_3ion_plasma_state['density_i_norm'] * dimensionless_3ion_plasma_state['grad_temperature_i_norm'] * dimensionless_3ion_plasma_state['temperature_i_norm']).sum('ion')
        ap = pt.calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, grad_ne, grad_te, ne=ne, te=te, lref=lref)
        xr.testing.assert_allclose(ap, ape + anpi + atpi)

    def test_calc_2ion_grad_p_with_2ions_unnorm(self, physical_2ion_plasma_state):
        c = pt.constants_si()
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        grad_te = physical_2ion_plasma_state['grad_temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_2ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_tia = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_npe = physical_2ion_plasma_state['grad_density_e'] * physical_2ion_plasma_state['temperature_e']
        grad_tpe = physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_2ion_plasma_state['grad_density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['grad_temperature_i']).sum('ion')
        grad_p = pt.calc_2ion_grad_p_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, grad_ne, grad_te, ne, te, lref, norm_inputs=False)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_2ion_grad_p_with_2ions_norm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        c = pt.constants_si()
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        atia = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        grad_npe = physical_2ion_plasma_state['grad_density_e'] * physical_2ion_plasma_state['temperature_e']
        grad_tpe = physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_2ion_plasma_state['grad_density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['grad_temperature_i']).sum('ion')
        grad_p = pt.calc_2ion_grad_p_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, ane, ate, ne, te, lref, norm_inputs=True)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_2ion_grad_p_with_1ion_and_gradient_quasineutrality_unnorm(self, physical_2ion_plasma_state):
        c = pt.constants_si()
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        grad_te = physical_2ion_plasma_state['grad_temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_tia = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        grad_npe = physical_2ion_plasma_state['grad_density_e'] * physical_2ion_plasma_state['temperature_e']
        grad_tpe = physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_2ion_plasma_state['grad_density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['grad_temperature_i']).sum('ion')
        grad_p = pt.calc_2ion_grad_p_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, grad_ne, grad_te, ne, te, lref, norm_inputs=False)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_2ion_grad_p_with_1ion_and_gradient_quasineutrality_norm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        c = pt.constants_si()
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        atia = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        grad_npe = physical_2ion_plasma_state['grad_density_e'] * physical_2ion_plasma_state['temperature_e']
        grad_tpe = physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_2ion_plasma_state['grad_density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['grad_temperature_i']).sum('ion')
        grad_p = pt.calc_2ion_grad_p_with_1ion_and_gradient_quasineutrality(ania, atia, atib, ninorma, tinorma, tinormb, zia, zib, ane, ate, ne, te, lref, norm_inputs=True)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_3ion_grad_p_with_3ions_unnorm(self, physical_3ion_plasma_state):
        c = pt.constants_si()
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_nic = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        grad_npe = physical_3ion_plasma_state['grad_density_e'] * physical_3ion_plasma_state['temperature_e']
        grad_tpe = physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_3ion_plasma_state['grad_density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['grad_temperature_i']).sum('ion')
        grad_p = pt.calc_3ion_grad_p_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, grad_ne, grad_te, ne, te, lref, norm_inputs=False)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_3ion_grad_p_with_3ions_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        c = pt.constants_si()
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        anic = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        grad_npe = physical_3ion_plasma_state['grad_density_e'] * physical_3ion_plasma_state['temperature_e']
        grad_tpe = physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_3ion_plasma_state['grad_density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['grad_temperature_i']).sum('ion')
        grad_p = pt.calc_3ion_grad_p_with_3ions(ania, anib, anic, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, ane, ate, ne, te, lref, norm_inputs=True)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_3ion_grad_p_with_2ions_and_gradient_quasineutrality_unnorm(self, physical_3ion_plasma_state):
        c = pt.constants_si()
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        grad_npe = physical_3ion_plasma_state['grad_density_e'] * physical_3ion_plasma_state['temperature_e']
        grad_tpe = physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_3ion_plasma_state['grad_density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['grad_temperature_i']).sum('ion')
        grad_p = pt.calc_3ion_grad_p_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, grad_ne, grad_te, ne, te, lref, norm_inputs=False)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_3ion_grad_p_with_2ions_and_gradient_quasineutrality_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        c = pt.constants_si()
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        grad_npe = physical_3ion_plasma_state['grad_density_e'] * physical_3ion_plasma_state['temperature_e']
        grad_tpe = physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_3ion_plasma_state['grad_density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['grad_temperature_i']).sum('ion')
        grad_p = pt.calc_3ion_grad_p_with_2ions_and_gradient_quasineutrality(ania, anib, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, ane, ate, ne, te, lref, norm_inputs=True)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        c = pt.constants_si()
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        grad_zeff = dimensionless_3ion_plasma_state['grad_effective_charge']
        grad_npe = physical_3ion_plasma_state['grad_density_e'] * physical_3ion_plasma_state['temperature_e']
        grad_tpe = physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_3ion_plasma_state['grad_density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['grad_temperature_i']).sum('ion')
        grad_p = pt.calc_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, grad_ne, grad_te, ne, te, lref, norm_inputs=False)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        c = pt.constants_si()
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        azeff = dimensionless_3ion_plasma_state['grad_effective_charge_norm']
        grad_npe = physical_3ion_plasma_state['grad_density_e'] * physical_3ion_plasma_state['temperature_e']
        grad_tpe = physical_3ion_plasma_state['density_e'] * physical_3ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_3ion_plasma_state['grad_density_i'] * physical_3ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_3ion_plasma_state['density_i'] * physical_3ion_plasma_state['grad_temperature_i']).sum('ion')
        grad_p = pt.calc_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(ania, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic, ane, ate, ne, te, lref, norm_inputs=True)
        xr.testing.assert_allclose(grad_p, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi))

    def test_calc_zeff_from_2ion_ninorm_with_2ions_norm(self, dimensionless_2ion_plasma_state):
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zeff = pt.calc_zeff_from_2ion_ninorm_with_2ions(ninorma, ninormb, zia, zib, ne=None)
        xr.testing.assert_allclose(zeff, dimensionless_2ion_plasma_state['effective_charge'])

    def test_calc_zeff_from_2ion_ninorm_with_2ions_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zeff = pt.calc_zeff_from_2ion_ninorm_with_2ions(nia, nib, zia, zib, ne=ne)
        xr.testing.assert_allclose(zeff, dimensionless_2ion_plasma_state['effective_charge'])

    def test_calc_zeff_from_2ion_ninorm_with_1ion_and_quasineutrality_norm(self, dimensionless_2ion_plasma_state):
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zeff = pt.calc_zeff_from_2ion_ninorm_with_1ion_and_quasineutrality(ninorma, zia, zib, ne=None)
        xr.testing.assert_allclose(zeff, dimensionless_2ion_plasma_state['effective_charge'])

    def test_calc_zeff_from_2ion_ninorm_with_1ion_and_quasineutrality_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zeff = pt.calc_zeff_from_2ion_ninorm_with_1ion_and_quasineutrality(nia, zia, zib, ne=ne)
        xr.testing.assert_allclose(zeff, dimensionless_2ion_plasma_state['effective_charge'])

    def test_calc_zeff_from_3ion_ninorm_with_3ions_norm(self, dimensionless_3ion_plasma_state):
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = pt.calc_zeff_from_3ion_ninorm_with_3ions(ninorma, ninormb, ninormc, zia, zib, zic, ne=None)
        xr.testing.assert_allclose(zeff, dimensionless_3ion_plasma_state['effective_charge'])

    def test_calc_zeff_from_3ion_ninorm_with_3ions_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = pt.calc_zeff_from_3ion_ninorm_with_3ions(nia, nib, nic, zia, zib, zic, ne=ne)
        xr.testing.assert_allclose(zeff, dimensionless_3ion_plasma_state['effective_charge'])

    def test_calc_zeff_from_3ion_ninorm_with_2ions_and_quasineutrality_norm(self, dimensionless_3ion_plasma_state):
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = pt.calc_zeff_from_3ion_ninorm_with_2ions_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=None)
        xr.testing.assert_allclose(zeff, dimensionless_3ion_plasma_state['effective_charge'])

    def test_calc_zeff_from_3ion_ninorm_with_2ions_and_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = pt.calc_zeff_from_3ion_ninorm_with_2ions_and_quasineutrality(nia, nib, zia, zib, zic, ne=ne)
        xr.testing.assert_allclose(zeff, dimensionless_3ion_plasma_state['effective_charge'])

    def test_calc_azeff_from_2ion_ani_with_2ions_norm(self, dimensionless_2ion_plasma_state):
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        azeff = pt.calc_azeff_from_2ion_ani_with_2ions(ania, anib, ninorma, ninormb, zia, zib, ane, ne=None, lref=None)
        xr.testing.assert_allclose(azeff, dimensionless_2ion_plasma_state['grad_effective_charge_norm'])

    def test_calc_azeff_from_2ion_ani_with_2ions_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_2ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        azeff = pt.calc_azeff_from_2ion_ani_with_2ions(grad_nia, grad_nib, nia, nib, zia, zib, grad_ne, ne=ne, lref=lref)
        xr.testing.assert_allclose(azeff, dimensionless_2ion_plasma_state['grad_effective_charge_norm'])

    def test_calc_azeff_from_2ion_ani_with_1ion_and_gradient_quasineutrality_norm(self, dimensionless_2ion_plasma_state):
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        azeff = pt.calc_azeff_from_2ion_ani_with_1ion_and_gradient_quasineutrality(ania, ninorma, zia, zib, ane, ne=None, lref=None)
        xr.testing.assert_allclose(azeff, dimensionless_2ion_plasma_state['grad_effective_charge_norm'])

    def test_calc_azeff_from_2ion_ani_with_1ion_and_gradient_quasineutrality_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        azeff = pt.calc_azeff_from_2ion_ani_with_1ion_and_gradient_quasineutrality(grad_nia, nia, zia, zib, grad_ne, ne=ne, lref=lref)
        xr.testing.assert_allclose(azeff, dimensionless_2ion_plasma_state['grad_effective_charge_norm'])

    def test_calc_azeff_from_3ion_ani_with_3ions_norm(self, dimensionless_3ion_plasma_state):
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        anic = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        azeff = pt.calc_azeff_from_3ion_ani_with_3ions(ania, anib, anic, ninorma, ninormb, ninormc, zia, zib, zic, ane, ne=None, lref=None)
        xr.testing.assert_allclose(azeff, dimensionless_3ion_plasma_state['grad_effective_charge_norm'])

    def test_calc_azeff_from_3ion_ani_with_3ions_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_nic = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        azeff = pt.calc_azeff_from_3ion_ani_with_3ions(grad_nia, grad_nib, grad_nic, nia, nib, nic, zia, zib, zic, grad_ne, ne=ne, lref=lref)
        xr.testing.assert_allclose(azeff, dimensionless_3ion_plasma_state['grad_effective_charge_norm'])

    def test_calc_azeff_from_3ion_ani_with_2ions_and_gradient_quasineutrality_norm(self, dimensionless_3ion_plasma_state):
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        azeff = pt.calc_azeff_from_3ion_ani_with_2ions_and_gradient_quasineutrality(ania, anib, ninorma, ninormb, zia, zib, zic, ane, ne=None, lref=None)
        xr.testing.assert_allclose(azeff, dimensionless_3ion_plasma_state['grad_effective_charge_norm'])

    def test_calc_azeff_from_3ion_ani_with_2ions_and_gradient_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        ne = physical_3ion_plasma_state['density_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        azeff = pt.calc_azeff_from_3ion_ani_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, nia, nib, zia, zib, zic, grad_ne, ne=ne, lref=lref)
        xr.testing.assert_allclose(azeff, dimensionless_3ion_plasma_state['grad_effective_charge_norm'])

    def test_calc_ne_from_beta_and_pnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        te = physical_2ion_plasma_state['temperature_e']
        beta = dimensionless_2ion_plasma_state['pressure_total_norm']
        pnorm = 1.0 + (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        ne = pt.calc_ne_from_beta_and_pnorm(beta, pnorm, btot, te)
        xr.testing.assert_allclose(ne, physical_2ion_plasma_state['density_e'])

    def test_calc_te_from_beta_and_pnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        beta = dimensionless_2ion_plasma_state['pressure_total_norm']
        pnorm = 1.0 + (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        te = pt.calc_te_from_beta_and_pnorm(beta, pnorm, btot, ne)
        xr.testing.assert_allclose(te, physical_2ion_plasma_state['temperature_e'])

    def test_calc_btot_from_beta_and_pnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        beta = dimensionless_2ion_plasma_state['pressure_total_norm']
        pnorm = 1.0 + (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        btot = pt.calc_btot_from_beta_and_pnorm(beta, pnorm, ne, te)
        #xr.testing.assert_allclose(bref, physical_2ion_plasma_state['field_axis'].expand_dims({'radius': physical_2ion_plasma_state['radius'].to_numpy().flatten()}, axis=-1))
        xr.testing.assert_allclose(btot, (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5)

    def test_calc_btot_from_beta_and_p(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        c = pt.constants_si()
        beta = dimensionless_2ion_plasma_state['pressure_total_norm']
        p = c['e'] * (physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['temperature_e'] + (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion'))
        btot = pt.calc_btot_from_beta_and_p(beta, p)
        xr.testing.assert_allclose(btot, (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5)

    def test_calc_beta_from_p(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        c = pt.constants_si()
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        p = c['e'] * (physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['temperature_e'] + (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion'))
        beta = pt.calc_beta_from_p(p, btot)
        xr.testing.assert_allclose(beta, dimensionless_2ion_plasma_state['pressure_total_norm'])

    def test_calc_beta_from_pnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        pnorm = 1.0 + (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        beta = pt.calc_beta_from_pnorm(pnorm, btot, ne, te)
        xr.testing.assert_allclose(beta, dimensionless_2ion_plasma_state['pressure_total_norm'])

    def test_calc_ne_from_alpha_and_ap(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        te = physical_2ion_plasma_state['temperature_e']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = dimensionless_2ion_plasma_state['grad_pressure_total_norm']
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        ne = pt.calc_ne_from_alpha_and_ap(alpha, ape + anpi + atpi, q, btot, te)
        xr.testing.assert_allclose(ne, physical_2ion_plasma_state['density_e'])

    def test_calc_te_from_alpha_and_ap(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = dimensionless_2ion_plasma_state['grad_pressure_total_norm']
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        te = pt.calc_te_from_alpha_and_ap(alpha, ape + anpi + atpi, q, btot, ne)
        xr.testing.assert_allclose(te, physical_2ion_plasma_state['temperature_e'])

    def test_calc_btot_from_alpha_and_ap(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = dimensionless_2ion_plasma_state['grad_pressure_total_norm']
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        btot = pt.calc_btot_from_alpha_and_ap(alpha, ape + anpi + atpi, q, ne, te)
        xr.testing.assert_allclose(btot, (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5)

    def test_calc_btot_from_alpha_and_grad_p(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        c = pt.constants_si()
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = dimensionless_2ion_plasma_state['grad_pressure_total_norm']
        grad_npe = physical_2ion_plasma_state['grad_density_e'] * physical_2ion_plasma_state['temperature_e']
        grad_tpe = physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_2ion_plasma_state['grad_density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['grad_temperature_i']).sum('ion')
        btot = pt.calc_btot_from_alpha_and_grad_p(alpha, c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi), q, lref)
        xr.testing.assert_allclose(btot, (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5)

    def test_calc_alpha_from_grad_p(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        c = pt.constants_si()
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        grad_npe = physical_2ion_plasma_state['grad_density_e'] * physical_2ion_plasma_state['temperature_e']
        grad_tpe = physical_2ion_plasma_state['density_e'] * physical_2ion_plasma_state['grad_temperature_e']
        grad_npi = (physical_2ion_plasma_state['grad_density_i'] * physical_2ion_plasma_state['temperature_i']).sum('ion')
        grad_tpi = (physical_2ion_plasma_state['density_i'] * physical_2ion_plasma_state['grad_temperature_i']).sum('ion')
        alpha = pt.calc_alpha_from_grad_p(c['e'] * (grad_npe + grad_tpe + grad_npi + grad_tpi), q, btot, lref)
        xr.testing.assert_allclose(alpha, dimensionless_2ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_ap(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        alpha = pt.calc_alpha_from_ap(ape + anpi + atpi, q, btot, ne, te)
        xr.testing.assert_allclose(alpha, dimensionless_2ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_2ion_ap_with_2ions_norm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        atia = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_2ion_ap_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, q, ane, ate, btot, ne, te, lref=None)
        xr.testing.assert_allclose(alpha, dimensionless_2ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_2ion_ap_with_2ions_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        grad_te = physical_2ion_plasma_state['grad_temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_2ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_tia = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_2ion_ap_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, q, grad_ne, grad_te, btot, ne, te, lref=lref)
        xr.testing.assert_allclose(alpha, dimensionless_2ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_2ion_ap_with_1ion_and_gradient_quasineutrality_norm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        atia = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_2ion_ap_with_1ion_and_gradient_quasineutrality(ania, atia, atib, ninorma, tinorma, tinormb, zia, zib, q, ane, ate, btot, ne, te, lref=None)
        xr.testing.assert_allclose(alpha, dimensionless_2ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_2ion_ap_with_1ion_and_gradient_quasineutrality_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        grad_te = physical_2ion_plasma_state['grad_temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_tia = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_2ion_ap_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, q, grad_ne, grad_te, btot, ne, te, lref=lref)
        xr.testing.assert_allclose(alpha, dimensionless_2ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_2ion_grad_p_with_2ions_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        grad_te = physical_2ion_plasma_state['grad_temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_2ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_2ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_tia = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_2ion_grad_p_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False)
        xr.testing.assert_allclose(alpha, dimensionless_2ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_2ion_grad_p_with_2ions_norm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        atia = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_2ion_grad_p_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, q, ane, ate, btot, ne, te, lref, norm_inputs=True)
        xr.testing.assert_allclose(alpha, dimensionless_2ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_2ion_grad_p_with_1ion_and_gradient_quasineutrality_unnorm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        grad_ne = physical_2ion_plasma_state['grad_density_e']
        grad_te = physical_2ion_plasma_state['grad_temperature_e']
        nia = physical_2ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_2ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_2ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        grad_nia = physical_2ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_tia = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_2ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        zia = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_2ion_grad_p_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False)
        xr.testing.assert_allclose(alpha, dimensionless_2ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_2ion_grad_p_with_1ion_and_gradient_quasineutrality_norm(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        ane = dimensionless_2ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_2ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_2ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        ania = dimensionless_2ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        atia = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_2ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        zia = dimensionless_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_2ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_2ion_grad_p_with_1ion_and_gradient_quasineutrality(ania, atia, atib, ninorma, tinorma, tinormb, zia, zib, q, ane, ate, btot, ne, te, lref, norm_inputs=True)
        xr.testing.assert_allclose(alpha, dimensionless_2ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_ap_with_3ions_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        anic = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_ap_with_3ions(ania, anib, anic, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, q, ane, ate, btot, ne, te, lref=None)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_ap_with_3ions_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_nic = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_ap_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, q, grad_ne, grad_te, btot, ne, te, lref=lref)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_ap_with_2ions_and_gradient_quasineutrality_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        ne = physical_3ion_plasma_state['density_e']
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        te = physical_3ion_plasma_state['temperature_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_ap_with_2ions_and_gradient_quasineutrality(ania, anib, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, q, ane, ate, btot, ne, te, lref=None)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_ap_with_2ions_and_gradient_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_ap_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, q, grad_ne, grad_te, btot, ne, te, lref=lref)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        azeff = dimensionless_3ion_plasma_state['grad_effective_charge_norm']
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(ania, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic, q, ane, ate, btot, ne, te, lref=None)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        grad_zeff = dimensionless_3ion_plasma_state['grad_effective_charge']
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, q, grad_ne, grad_te, btot, ne, te, lref=lref)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_grad_p_with_3ions_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        nic = physical_3ion_plasma_state['density_i'].sel(ion='Ar', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_nic = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ar', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_grad_p_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_grad_p_with_3ions_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        ninormc = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ar', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        anic = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ar', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_grad_p_with_3ions(ania, anib, anic, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, q, ane, ate, btot, ne, te, lref, norm_inputs=True)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_grad_p_with_2ions_and_gradient_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        nib = physical_3ion_plasma_state['density_i'].sel(ion='Ne', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_nib = physical_3ion_plasma_state['grad_density_i'].sel(ion='Ne', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_grad_p_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_grad_p_with_2ions_and_gradient_quasineutrality_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        ninormb = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='Ne', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        anib = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='Ne', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_grad_p_with_2ions_and_gradient_quasineutrality(ania, anib, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, q, ane, ate, btot, ne, te, lref, norm_inputs=True)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality_unnorm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        grad_ne = physical_3ion_plasma_state['grad_density_e']
        grad_te = physical_3ion_plasma_state['grad_temperature_e']
        nia = physical_3ion_plasma_state['density_i'].sel(ion='D', drop=True)
        tia = physical_3ion_plasma_state['temperature_i'].sel(ion='D', drop=True)
        tib = physical_3ion_plasma_state['temperature_i'].sel(ion='Ne', drop=True)
        tic = physical_3ion_plasma_state['temperature_i'].sel(ion='Ar', drop=True)
        grad_nia = physical_3ion_plasma_state['grad_density_i'].sel(ion='D', drop=True)
        grad_tia = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='D', drop=True)
        grad_tib = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ne', drop=True)
        grad_tic = physical_3ion_plasma_state['grad_temperature_i'].sel(ion='Ar', drop=True)
        zia = physical_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = physical_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = physical_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        grad_zeff = dimensionless_3ion_plasma_state['grad_effective_charge']
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_alpha_from_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality_norm(self, physical_3ion_plasma_state, dimensionless_3ion_plasma_state):
        lref = physical_3ion_plasma_state['r_minor_lcfs']
        btot = (physical_3ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ne = physical_3ion_plasma_state['density_e']
        te = physical_3ion_plasma_state['temperature_e']
        ane = dimensionless_3ion_plasma_state['grad_density_e_norm']
        ate = dimensionless_3ion_plasma_state['grad_temperature_e_norm']
        ninorma = dimensionless_3ion_plasma_state['density_i_norm'].sel(ion='D', drop=True)
        tinorma = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='D', drop=True)
        tinormb = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ne', drop=True)
        tinormc = dimensionless_3ion_plasma_state['temperature_i_norm'].sel(ion='Ar', drop=True)
        ania = dimensionless_3ion_plasma_state['grad_density_i_norm'].sel(ion='D', drop=True)
        atia = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='D', drop=True)
        atib = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ne', drop=True)
        atic = dimensionless_3ion_plasma_state['grad_temperature_i_norm'].sel(ion='Ar', drop=True)
        zia = dimensionless_3ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        zib = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ne', drop=True)
        zic = dimensionless_3ion_plasma_state['charge_i'].sel(ion='Ar', drop=True)
        zeff = dimensionless_3ion_plasma_state['effective_charge']
        azeff = dimensionless_3ion_plasma_state['grad_effective_charge_norm']
        q = dimensionless_3ion_plasma_state['safety_factor_circular']
        alpha = pt.calc_alpha_from_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(ania, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic, q, ane, ate, btot, ne, te, lref, norm_inputs=True)
        xr.testing.assert_allclose(alpha, dimensionless_3ion_plasma_state['grad_pressure_total_norm'])

    def test_calc_coulomb_logarithm_nrl_from_ne_and_te(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        cl = pt.calc_coulomb_logarithm_nrl_from_ne_and_te(ne, te)
        xr.testing.assert_allclose(cl, dimensionless_2ion_plasma_state['coulomb_logarithm_nrl'])

    def test_calc_nustar_nrl_from_ne_and_te(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        rmin = physical_2ion_plasma_state['r_minor']
        rmaj = physical_2ion_plasma_state['r_geometric']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        zeff = dimensionless_2ion_plasma_state['effective_charge']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        nustar = pt.calc_nustar_nrl_from_ne_and_te(zeff, q, rmin, rmaj, ne, te)
        xr.testing.assert_allclose(nustar, dimensionless_2ion_plasma_state['collisionality_nrl_norm'])

    def test_calc_lognustar_from_nustar(self, dimensionless_2ion_plasma_state):
        nustar = dimensionless_2ion_plasma_state['collisionality_nrl_norm']
        lognustar = pt.calc_lognustar_from_nustar(nustar)
        xr.testing.assert_allclose(lognustar, np.log10(nustar))

    def test_calc_nustar_from_lognustar(self, dimensionless_2ion_plasma_state):
        lognustar = np.log10(dimensionless_2ion_plasma_state['collisionality_nrl_norm'])
        nustar = pt.calc_nustar_from_lognustar(lognustar)
        xr.testing.assert_allclose(nustar, dimensionless_2ion_plasma_state['collisionality_nrl_norm'])

    def test_calc_ne_from_nustar_nrl(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        rmin = physical_2ion_plasma_state['r_minor']
        rmaj = physical_2ion_plasma_state['r_geometric']
        te = physical_2ion_plasma_state['temperature_e']
        nustar = dimensionless_2ion_plasma_state['collisionality_nrl_norm']
        zeff = dimensionless_2ion_plasma_state['effective_charge']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        ne = pt.calc_ne_from_nustar_nrl(nustar, zeff, q, rmin, rmaj, te)
        xr.testing.assert_allclose(ne, physical_2ion_plasma_state['density_e'])

    def test_calc_te_from_nustar_nrl(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        rmin = physical_2ion_plasma_state['r_minor']
        rmaj = physical_2ion_plasma_state['r_geometric']
        ne = physical_2ion_plasma_state['density_e']
        nustar = dimensionless_2ion_plasma_state['collisionality_nrl_norm']
        zeff = dimensionless_2ion_plasma_state['effective_charge']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        te = pt.calc_te_from_nustar_nrl(nustar, zeff, q, rmin, rmaj, ne)
        xr.testing.assert_allclose(te, physical_2ion_plasma_state['temperature_e'])

    def test_calc_zeff_from_nustar_nrl(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        rmin = physical_2ion_plasma_state['r_minor']
        rmaj = physical_2ion_plasma_state['r_geometric']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        nustar = dimensionless_2ion_plasma_state['collisionality_nrl_norm']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        zeff = pt.calc_zeff_from_nustar_nrl(nustar, q, rmin, rmaj, ne, te)
        xr.testing.assert_allclose(zeff, dimensionless_2ion_plasma_state['effective_charge'])

    def test_calc_rhos_from_ts_and_btot(self, physical_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ti = physical_2ion_plasma_state['temperature_i']
        ai = physical_2ion_plasma_state['mass_i']
        zi = physical_2ion_plasma_state['charge_i']
        rhoi = pt.calc_rhos_from_ts_and_btot(ti, btot, _as=ai, _zs=zi)
        xr.testing.assert_allclose(rhoi, physical_2ion_plasma_state['gyroradius_i'])

    def test_calc_btot_from_rhos(self, physical_2ion_plasma_state):
        ti = physical_2ion_plasma_state['temperature_i']
        ai = physical_2ion_plasma_state['mass_i']
        zi = physical_2ion_plasma_state['charge_i']
        rhoi = physical_2ion_plasma_state['gyroradius_i']
        btot = pt.calc_btot_from_rhos(rhoi, ti, _as=ai, _zs=zi)
        xr.testing.assert_allclose(btot, ((physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5).expand_dims({'ion': physical_2ion_plasma_state['ion'].to_numpy().flatten()}, axis=-1))

    def test_calc_ts_from_rhos(self, physical_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ai = physical_2ion_plasma_state['mass_i']
        zi = physical_2ion_plasma_state['charge_i']
        rhoi = physical_2ion_plasma_state['gyroradius_i']
        ti = pt.calc_ts_from_rhos(rhoi, btot, _as=ai, _zs=zi)
        xr.testing.assert_allclose(ti, physical_2ion_plasma_state['temperature_i'])

    def test_calc_rhoref_from_te_and_btot(self, physical_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        te = physical_2ion_plasma_state['temperature_e']
        ai = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        rhoref = pt.calc_rhoref_from_te_and_btot(te, btot, ai)
        xr.testing.assert_allclose(rhoref, physical_2ion_plasma_state['gyroradius_ref'])

    def test_calc_btot_from_rhoref(self, physical_2ion_plasma_state):
        te = physical_2ion_plasma_state['temperature_e']
        ai = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        rhos = physical_2ion_plasma_state['gyroradius_ref']
        btot = pt.calc_btot_from_rhoref(rhos, te, ai)
        xr.testing.assert_allclose(btot, (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5)

    def test_calc_te_from_rhos(self, physical_2ion_plasma_state):
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ai = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        rhos = physical_2ion_plasma_state['gyroradius_ref']
        te = pt.calc_te_from_rhoref(rhos, btot, ai)
        xr.testing.assert_allclose(te, physical_2ion_plasma_state['temperature_e'])

    def test_calc_rhostar_from_te_and_btot(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        te = physical_2ion_plasma_state['temperature_e']
        ai = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        rhostar = pt.calc_rhostar_from_te_and_btot(te, btot, ai, lref)
        xr.testing.assert_allclose(rhostar, dimensionless_2ion_plasma_state['gyroradius_ref_norm'])

    def test_calc_btot_from_rhostar(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        te = physical_2ion_plasma_state['temperature_e']
        ai = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        rhostar = dimensionless_2ion_plasma_state['gyroradius_ref_norm']
        btot = pt.calc_btot_from_rhostar(rhostar, te, ai, lref)
        xr.testing.assert_allclose(btot, (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5)

    def test_calc_te_from_rhostar(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        ai = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        rhostar = dimensionless_2ion_plasma_state['gyroradius_ref_norm']
        te = pt.calc_te_from_rhostar(rhostar, btot, ai, lref)
        xr.testing.assert_allclose(te, physical_2ion_plasma_state['temperature_e'])

    def test_calc_ne_from_alpha_and_rhostar(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ai = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        zi = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        rhostar = dimensionless_2ion_plasma_state['gyroradius_ref_norm']
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        alpha = dimensionless_2ion_plasma_state['grad_pressure_total_norm']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        ne = pt.calc_ne_from_alpha_and_rhostar(alpha, rhostar, ape + anpi + atpi, ai, zi, q, lref)
        xr.testing.assert_allclose(ne, physical_2ion_plasma_state['density_e'])

    def test_calc_btot_from_alpha_and_rhostar(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        ai = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        zi = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        rhostar = dimensionless_2ion_plasma_state['gyroradius_ref_norm']
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        alpha = dimensionless_2ion_plasma_state['grad_pressure_total_norm']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        btot = pt.calc_btot_from_alpha_and_rhostar(alpha, rhostar, ape + anpi + atpi, ai, zi, q, ne, te, lref)
        xr.testing.assert_allclose(btot, (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5)

    def test_calc_ne_from_beta_and_rhostar(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ai = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        zi = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        rhostar = dimensionless_2ion_plasma_state['gyroradius_ref_norm']
        pnorm = 1.0 + (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        beta = dimensionless_2ion_plasma_state['pressure_total_norm']
        ne = pt.calc_ne_from_beta_and_rhostar(beta, rhostar, pnorm, ai, zi, lref)
        xr.testing.assert_allclose(ne, physical_2ion_plasma_state['density_e'])

    def test_calc_btot_from_beta_and_rhostar(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        lref = physical_2ion_plasma_state['r_minor_lcfs']
        ne = physical_2ion_plasma_state['density_e']
        te = physical_2ion_plasma_state['temperature_e']
        ai = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        zi = physical_2ion_plasma_state['charge_i'].sel(ion='D', drop=True)
        rhostar = dimensionless_2ion_plasma_state['gyroradius_ref_norm']
        pnorm = 1.0 + (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        beta = dimensionless_2ion_plasma_state['pressure_total_norm']
        btot = pt.calc_btot_from_beta_and_rhostar(beta, rhostar, pnorm, ai, zi, ne, te, lref)
        xr.testing.assert_allclose(btot, (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5)

    def test_calc_ne_from_nustar_nrl_alpha_and_ap(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        rmin = physical_2ion_plasma_state['r_minor']
        rmaj = physical_2ion_plasma_state['r_geometric']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        nustar = dimensionless_2ion_plasma_state['collisionality_nrl_norm']
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        alpha = dimensionless_2ion_plasma_state['grad_pressure_total_norm']
        zeff = dimensionless_2ion_plasma_state['effective_charge']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        ne = pt.calc_ne_from_nustar_nrl_alpha_and_ap(nustar, alpha, ape + anpi + atpi, zeff, q, btot, rmin, rmaj)
        xr.testing.assert_allclose(ne, physical_2ion_plasma_state['density_e'])

    def test_calc_te_from_nustar_nrl_alpha_and_ap(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        rmin = physical_2ion_plasma_state['r_minor']
        rmaj = physical_2ion_plasma_state['r_geometric']
        btot = (physical_2ion_plasma_state['field_geometric'] ** 2).sum('direction') ** 0.5
        nustar = dimensionless_2ion_plasma_state['collisionality_nrl_norm']
        ape = dimensionless_2ion_plasma_state['grad_density_e_norm'] + dimensionless_2ion_plasma_state['grad_temperature_e_norm']
        anpi = (dimensionless_2ion_plasma_state['grad_density_i_norm'] * dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        atpi = (dimensionless_2ion_plasma_state['density_i_norm'] * dimensionless_2ion_plasma_state['grad_temperature_i_norm'] * dimensionless_2ion_plasma_state['temperature_i_norm']).sum('ion')
        alpha = dimensionless_2ion_plasma_state['grad_pressure_total_norm']
        zeff = dimensionless_2ion_plasma_state['effective_charge']
        q = dimensionless_2ion_plasma_state['safety_factor_circular']
        te = pt.calc_te_from_nustar_nrl_alpha_and_ap(nustar, alpha, ape + anpi + atpi, zeff, q, btot, rmin, rmaj)
        xr.testing.assert_allclose(te, physical_2ion_plasma_state['temperature_e'])

    def test_calc_vref_from_te_and_mref(self, physical_2ion_plasma_state):
        te = physical_2ion_plasma_state['temperature_e']
        aref = physical_2ion_plasma_state['mass_i'].sel(ion='D', drop=True)
        vref = pt.calc_vref_from_te_and_mref(te, aref)
        xr.testing.assert_allclose(vref, physical_2ion_plasma_state['velocity_ref'])

    def test_calc_vths_from_ts(self, physical_2ion_plasma_state):
        te = physical_2ion_plasma_state['temperature_e']
        ae = physical_2ion_plasma_state['mass_e']
        vthe = pt.calc_vths_from_ts(te, _as=ae)
        xr.testing.assert_allclose(vthe, physical_2ion_plasma_state['velocity_thermal_e'])

    def test_calc_mach_from_u(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        ui = physical_2ion_plasma_state['velocity_i']
        vref = physical_2ion_plasma_state['velocity_ref']
        machi = pt.calc_mach_from_u(ui, vref)
        xr.testing.assert_allclose(machi, dimensionless_2ion_plasma_state['mach_i'])

    def test_calc_u_from_mach(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
        machi = dimensionless_2ion_plasma_state['mach_i']
        vref = physical_2ion_plasma_state['velocity_ref']
        ui = pt.calc_u_from_mach(machi, vref)
        xr.testing.assert_allclose(ui, physical_2ion_plasma_state['velocity_i'])

    # def test_calc_machpar_from_machtor_and_toroidal_assumption(self, physical_2ion_plasma_state, dimensionless_2ion_plasma_state):
    #     machpar = (machtor, q, epsilon, x)
    #     btorbyb = ((q ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    #     machpar = machtor * btorbyb
    #     return machpar

    # def calc_aupar_from_autor_and_puretor(autor, machtor, s, q, epsilon, x):
    #     btorbyb = ((q ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    #     bpolbyb = (((epsilon * x) ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    #     grad_btorbyb = btorbyb * (bpolbyb ** 2) * (s - 1.0) / (epsilon * x)
    #     aupar = autor * btorbyb - machtor * grad_btorbyb
    #     return aupar

    # def calc_gammae_from_aupar_without_grad_dpi(aupar, q, epsilon):
    #     gammae = -(epsilon / q) * aupar
    #     return gammae

    # def calc_grad_dpi_from_gammae_machtor_and_autor(gammae, machtor, autor, q, r, ro):
    #     return None

    # def calc_bunit_from_bo(bo, sfac):
    #     bunit = normalize(bo, sfac)
    #     return bunit

    # def calc_bo_from_bunit(bunit, sfac):
    #     bo = unnormalize(bunit, sfac)
    #     return bo

    # def calc_rhos_from_ts_ms_and_b(ts, ms, b):
    #     c = constants_si()
    #     rhos = (ts * c['u'] * ms / c['e']) ** 0.5 / b
    #     return rhos

    # def calc_vsound_from_te_and_mref(te, mref):
    #     c = constants_si()
    #     vsound = (c['e'] * te / (c['u'] * mref)) ** 0.5
    #     return vsound

    # def calc_vtherms_from_ts_and_ms(ts, ms):
    #     c = constants_si()
    #     vths = (2.0 * c['e'] * ts / (c['u'] * ms)) ** 0.5
    #     return vths

    # def calc_lds_from_ts_and_ns(ts, ns, zs):
    #     c = constants_si()
    #     lds = (c['eps'] * ts / (c['e'] * ns * (zs ** 2))) ** 0.5
    #     return lds

    # def calc_ldsnorm_from_lds(lds, rhos):
    #     ldsnorm = normalize(lds, rhos)
    #     return ldsnorm

    # def calc_ldenorm_from_te_ne_and_rhos(te, ne, rhos, ze=1.0):
    #     lde = calc_lds_from_ts_and_ns(te, ne, ze)
    #     ldenorm = calc_ldsnorm_from_lds(lde, rhos)
    #     return ldenorm

    # def calc_coulomb_logarithm_from_te_and_ne(te, ne, ze=1.0):
    #     c = constants_si()
    #     lda = calc_lds_from_ts_and_ns(te, ne, ze)
    #     inv_b90 = (4.0 * np.pi * c['eps'] / c['e']) * te / (ze * ze)
    #     cl = np.log(inv_b90 * lda)
    #     return cl

    # def calc_nu_from_t_and_n(ta, na, nb, ma, za, zb):
    #     c = constants_si()
    #     factor = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    #     inv_b90 = (4.0 * np.pi * c['eps'] / c['e']) * ta / (za * zb)
    #     cl = calc_coulomb_logarithm_from_te_and_ne(ta, na, za) + np.log(za / zb)
    #     nu = factor * nb * (c['e'] * ta / (c['u'] * ma)) ** 0.5 / (inv_b90 ** 2) * cl
    #     return nu

    # def calc_nuei_from_te_ne_and_zeff(te, ne, zeff, zi, ze=1.0):
    #     c = constants_si()
    #     factor = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    #     inv_b90 = (4.0 * np.pi * c['eps'] / c['e']) * te / (ze * ((zeff * ze) ** 0.5))
    #     cl = calc_coulomb_logarithm_from_te_and_ne(te, ne, ze) + np.log(ze / zi)
    #     nuei = factor * ne * (c['e'] * te / (c['me'])) ** 0.5 / (inv_b90 ** 2) * cl
    #     return nuei

    # def calc_nunorm_from_nu(nu, gref):
    #     nunorm = normalize(nu, gref)
    #     return nunorm

    # def calc_nueenorm_from_te_and_ne(te, ne, gref, ze=1.0):
    #     c = constants_si()
    #     me = c['me'] / c['u']
    #     nuee = calc_nu_from_t_and_n(te, ne, ne, me, ze, ze)
    #     nueenorm = calc_nunorm_from_nu(nuee, gref)
    #     return nueenorm

    # def calc_nueinorm_from_te_ne_and_ni(te, ne, ni, zi, gref, ze=1.0):
    #     c = constants_si()
    #     me = c['me'] / c['u']
    #     nuei = calc_nu_from_t_and_n(te, ne, ni, me, ze, zi)
    #     nueinorm = calc_nunorm_from_nu(nuei, gref)
    #     return nueinorm

    # def calc_nueinorm_from_te_ne_and_zeff(te, ne, zeff, zi, gref, ze=1.0):
    #     nuei = calc_nuei_from_te_ne_and_zeff(te, ne, zeff, zi, ze)
    #     nueinorm = calc_nunorm_from_nu(nuei, gref)
    #     return nueinorm

    # def calc_nuiinorm_from_te_ne_and_ni(tia, nia, nib, mia, zia, zib, gref):
    #     nuii = calc_nu_from_t_and_n(tia, nia, nib, mia, zia, zib)
    #     nuiinorm = calc_nunorm_from_nu(nuii, gref)
    #     return nuiinorm

    # def calc_te_from_betae_and_ldenorm(betae, ldenorm, sfac, mref, ze=1.0):
    #     c = constants_si()
    #     bc = (ze ** 2) * c['u'] * mref / (2.0 * c['eps'] * c['e'] * c['mu'])
    #     te = bc * betae * (ldenorm ** 2) * (sfac ** 2)
    #     return te

    # def calc_ne_from_nueenorm(nueenorm, te, mref, lref, ze=1.0):
    #     c = constants_si()
    #     f_nu = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    #     inv_b90_ee = 4.0 * np.pi * (c['eps'] / c['e']) * te / (ze ** 2)
    #     nc = f_nu * lref * (c['u'] * mref / c['me']) ** 0.5
    #     nl = inv_b90_ee * (c['eps'] / c['e']) ** 0.5 * (te / (ze ** 2)) ** 0.5
    #     data = {'logterm': np.log(nl), 'constant': nueenorm * (inv_b90_ee ** 2) / nc}
    #     rootdata = pd.DataFrame(data)
    #     logger.debug(rootdata)
    #     func_ne20 = lambda row: root_scalar(
    #         lambda ne: (row['logterm'] - 0.5 * np.log(ne * 1.0e20)) * (ne * 1.0e20) - row['constant'],
    #         x0=0.01,
    #         x1=1.0,
    #         maxiter=100,
    #     )
    #     sol_ne20 = rootdata.apply(func_ne20, axis=1)
    #     retry = sol_ne20.apply(lambda sol: not sol.converged)
    #     if np.any(retry):
    #         func_ne20_v2 = lambda row: root_scalar(
    #             lambda ne: (row['logterm'] - 0.5 * np.log(ne * 1.0e20)) * (ne * 1.0e20) - row['constant'],
    #             x0=1.0,
    #             x1=0.1,
    #             maxiter=100,
    #         )
    #         sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
    #     ne = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    #     return ne

    # def calc_bunit_from_ldenorm(ldenorm, ne, mref, ze=1.0):
    #     c = constants_si()
    #     bunit = (ne * (ze ** 2) * c['u'] * mref / c['eps']) ** 0.5 * ldenorm
    #     return bunit

    # def calc_ne_from_nustar(nustar, zeff, q, r, ro, te):
    #     c = constants_si()
    #     eom = c['e'] / c['me']
    #     tb = q * ro * ((r / ro) ** (-1.5)) / ((eom * te) ** 0.5)
    #     kk = (1.0e4 / 1.09) * zeff * ((te * 1.0e-3) ** (-1.5))
    #     nu = nustar / (tb * kk)
    #     data = {'te': te * 1.0e-3, 'knu': nu}
    #     rootdata = pd.DataFrame(data)
    #     logger.debug(rootdata)
    #     func_ne20 = lambda row: root_scalar(
    #         lambda ne: calc_coulomb_logarithm_from_te_and_ne(row['te'] * 1.0e3, ne * 1.0e20) * ne - row['knu'],
    #         x0=0.01,
    #         x1=1.0,
    #         maxiter=100,
    #     )
    #     sol_ne20 = rootdata.apply(func_ne20, axis=1)
    #     retry = sol_ne20.apply(lambda sol: not sol.converged)
    #     if np.any(retry):
    #         func_ne20_v2 = lambda row: root_scalar(
    #             lambda ne: calc_coulomb_logarithm_from_te_and_ne(row['te'] * 1.0e3, ne * 1.0e20) * ne - row['knu'],
    #             x0=1.0,
    #             x1=0.1,
    #             maxiter=100,
    #         )
    #         sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
    #     ne = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    #     logger.debug(f'<{calc_ne_from_nustar.__name__}>: data')
    #     logger.debug(pd.DataFrame(data={'nustar': nustar, 'te': te, 'ne': ne}))
    #     return ne

    # def calc_te_from_nustar(nustar, zeff, q, r, ro, ne, verbose=0):
    #     c = constants_si()
    #     moe = c['me'] / c['e']
    #     kk = (10.0 ** 0.5) * (1.0e2 / 1.09) * zeff * q * ro * ((r / ro) ** (-1.5)) * (moe ** 0.5) * (ne * 1.0e-20)
    #     nu = nustar / kk
    #     data = {'ne': ne * 1.0e-20, 'knu': nu}
    #     rootdata = pd.DataFrame(data)
    #     logger.debug(rootdata)
    #     func_te3 = lambda row: root_scalar(
    #         lambda te: calc_coulomb_logarithm_from_te_and_ne(te * 1.0e3, row['ne'] * 1.0e20) / (te ** 2) - row['knu'],
    #         x0=1.0,
    #         x1=0.1,
    #         maxiter=100,
    #     )
    #     sol_te3 = rootdata.apply(func_te3, axis=1)
    #     retry = sol_te3.apply(lambda sol: not sol.converged)
    #     if np.any(retry):
    #         func_te3_v2 = lambda row: root_scalar(
    #             lambda te: calc_coulomb_logarithm_from_te_and_ne(te * 1.0e3, row['ne'] * 1.0e20) / (te ** 2) - row['knu'],
    #             x0=0.01,
    #             x1=0.1,
    #             maxiter=100,
    #         )
    #         sol_te3.loc[retry] = rootdata.loc[retry].apply(func_te3_v2, axis=1)
    #     te = sol_te3.apply(lambda sol: 1.0e3 * sol.root).to_numpy()
    #     logger.debug(f'<{calc_te_from_nustar.__name__}>: data')
    #     logger.debug(pd.DataFrame(data={'nustar': nustar, 'ne': ne, 'te': te}))
    #     return te

    # def calc_zeff_from_nustar(nustar, q, r, ro, ne, te):
    #     c = constants_si()
    #     cl = calc_coulomb_logarithm_from_te_and_ne(te, ne)
    #     nt = (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    #     kk = (1.0e4 / 1.09) * q * ro * ((r / ro) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    #     zeff = nustar / (cl * nt * kk)
    #     return zeff

    # def calc_nustar(zeff, q, r, ro, ne, te):
    #     c = constants_si()
    #     cl = calc_coulomb_logarithm_from_te_and_ne(te, ne)
    #     nt = (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    #     kk = (1.0e4 / 1.09) * q * ro * ((r / ro) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    #     nustar = cl * zeff * nt * kk
    #     return nustar
