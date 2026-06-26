import copy
import warnings
import logging
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr
from scipy.optimize import root_scalar  # type: ignore[import-untyped]
from scipy.integrate import trapezoid  # type: ignore[import-untyped]

logger = logging.getLogger('fusio')

np_itypes = (np.int8, np.int16, np.int32, np.int64)
np_utypes = (np.uint8, np.uint16, np.uint32, np.uint64)
np_ftypes = (np.float16, np.float32, np.float64)

number_types = (float, int, np_itypes, np_utypes, np_ftypes)
array_types = (list, tuple, np.ndarray)
string_types = (str, np.str_)
pandas_types = (pd.Series, pd.DataFrame)
xarray_types = (xr.DataArray, xr.Dataset)
class_types = (pandas_types, xarray_types)


def ensure_numpy(val):
    """Coerce a scalar, sequence, or pandas/xarray object to a numpy array."""
    if isinstance(val, number_types):
        return np.atleast_1d([val])
    elif isinstance(val, array_types):
        return np.atleast_1d(val)
    elif isinstance(val, class_types):
        return val.to_numpy()
    else:
        return val


def ensure_type_match(val, other):
    """Cast a numpy array to the same scalar or container type as `other`."""
    if isinstance(other, number_types) and val.size == 1:
        return other.__class__(val.item(0))
    elif isinstance(other, xarray_types):
        return other.__class__(val, coords=other.coords)
    else:
        return other.__class__(val)


e_si = 1.60218e-19       # C
u_si = 1.66054e-27       # kg
mu_si = 4.0e-7 * np.pi   # H/m
eps_si = 8.85419e-12     # F/m
me_si = 5.4858e-4 * u_si
mp_si = (1.0 + 7.2764e-3) * u_si
md_si = 2.0141 * u_si


def constants_si():
    """Return a dict of fundamental SI physical constants keyed by short name."""
    return {
        'e': e_si,
        'u': u_si,
        'mu': mu_si,
        'eps': eps_si,
        'me': me_si,
        'mp': mp_si,
        'md': md_si,
    }


def define_ion_species(z=None, a=None, short_name=None, long_name=None, user_mass=False):
    """Look up element symbol, mass number, and charge number for a plasma ion species."""

    specieslist = {
         "e":  (  0.000544617, -1.0), "n":  (  1.000866492,  0.0),
         "H":  (  1.0,  1.0),  "D":  (  2.0,  1.0),  "T":  (  3.0,  1.0), "He":  (  4.0,  2.0),
        "Li":  (  7.0,  3.0), "Be":  (  9.0,  4.0),  "B":  ( 11.0,  5.0),  "C":  ( 12.0,  6.0),
         "N":  ( 14.0,  7.0),  "O":  ( 16.0,  8.0),  "F":  ( 19.0,  9.0), "Ne":  ( 20.0, 10.0),
        "Na":  ( 23.0, 11.0), "Mg":  ( 24.0, 12.0), "Al":  ( 27.0, 13.0), "Si":  ( 28.0, 14.0),
         "P":  ( 31.0, 15.0),  "S":  ( 32.0, 16.0), "Cl":  ( 35.0, 17.0), "Ar":  ( 40.0, 18.0),
         "K":  ( 39.0, 19.0), "Ca":  ( 40.0, 20.0), "Sc":  ( 45.0, 21.0), "Ti":  ( 48.0, 22.0),
         "V":  ( 51.0, 23.0), "Cr":  ( 52.0, 24.0), "Mn":  ( 55.0, 25.0), "Fe":  ( 56.0, 26.0),
        "Co":  ( 59.0, 27.0), "Ni":  ( 58.0, 28.0), "Cu":  ( 63.0, 29.0), "Zn":  ( 64.0, 30.0),
        "Ga":  ( 69.0, 31.0), "Ge":  ( 72.0, 32.0), "As":  ( 75.0, 33.0), "Se":  ( 80.0, 34.0),
        "Br":  ( 79.0, 35.0), "Kr":  ( 84.0, 36.0), "Rb":  ( 85.0, 37.0), "Sr":  ( 88.0, 38.0),
         "Y":  ( 89.0, 39.0), "Zr":  ( 90.0, 40.0), "Nb":  ( 93.0, 41.0), "Mo":  ( 96.0, 42.0),
        "Tc":  ( 99.0, 43.0), "Ru":  (102.0, 44.0), "Rh":  (103.0, 45.0), "Pd":  (106.0, 46.0),
        "Ag":  (107.0, 47.0), "Cd":  (114.0, 48.0), "In":  (115.0, 49.0), "Sn":  (120.0, 50.0),
        "Sb":  (121.0, 51.0), "Te":  (128.0, 52.0),  "I":  (127.0, 53.0), "Xe":  (131.0, 54.0),
        "Cs":  (133.0, 55.0), "Ba":  (138.0, 56.0), "La":  (139.0, 57.0),                      
        "Lu":  (175.0, 71.0), "Hf":  (178.0, 72.0), "Ta":  (181.0, 73.0),  "W":  (184.0, 74.0),
        "Re":  (186.0, 75.0), "Os":  (190.0, 76.0), "Ir":  (193.0, 77.0), "Pt":  (195.0, 78.0),
        "Au":  (197.0, 79.0), "Hg":  (200.0, 80.0), "Tl":  (205.0, 81.0), "Pb":  (208.0, 82.0),
        "Bi":  (209.0, 83.0), "Po":  (209.0, 84.0), "At":  (210.0, 85.0), "Rn":  (222.0, 86.0)
    }

    tz = None
    ta = None
    sn = None
    ln = None
    if isinstance(z, number_types) and int(np.rint(z)) >= -1:
        tz = int(np.ceil(z))
    if isinstance(a, number_types) and int(np.rint(a)) > 0:
        ta = int(np.rint(a))
    if isinstance(short_name, string_types) and short_name in specieslist:
        sn = short_name
    if isinstance(long_name, string_types):
        print("Long name species identifier not yet implemented")

    # Determines atomic charge number based on atomic mass number, if no charge number is given
    if isinstance(ta, int) and tz is None:
        for key, val in specieslist.items():
            if ta == int(np.rint(val[0])):
                ta = int(np.rint(val[0]))
                tz = int(np.rint(val[1]))
    if tz is None:
        ta = None

    # Enforces default return value as deuterium if no arguments or improper arguments are given
    if ta is None and sn is None and ln is None:
        if tz is None:
            tz = 1
        if tz == 1:
            ta = 2

    sz = None
    sa = None
    sname = None
#    lname = None

    periodic_table = [
         "n", # The first 'element' should always be the neutron, for consistency with the numbering
         "H",                                                                                "He",
        "Li","Be",                                                   "B", "C", "N", "O", "F","Ne",
        "Na","Mg",                                                  "Al","Si", "P", "S","Cl","Ar",
         "K","Ca","Sc","Ti", "V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
        "Rb","Sr", "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te", "I","Xe",
        "Cs","Ba","La",     "Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
                       "Hf","Ta", "W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
        "Fr","Ra","Ac",     "Th","Pa", "U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
                       "Rf","Db","Sg","Bh","Hs",
        "e"   # The last 'element' should always be the electron, for consistency with the numbering
    ]

    # Prioritize shorthand element name over atomic charge argument
    if sn is not None and sn in specieslist:
        sname = sn
        (sa, sz) = specieslist[sname]
    elif tz is not None:
        sname = periodic_table[tz] if tz < len(periodic_table) - 1 else periodic_table[-1]
        (sa, sz) = specieslist[sname]

    if sname is not None and sa is not None and sz is not None:
        # Allow user specification of mass according to broad heuristic isotopic limits
        if user_mass and isinstance(a, number_types) and float(a) >= sz and float(a) <= sz*3:
            sa = float(a)
        elif ta is not None and float(ta) != sa:
            sa = float(ta)
        for key, val in specieslist.items():
            if sa == int(np.rint(val[0])) and sz == int(np.rint(val[1])):
                sname = key

    return sname, sa, sz


### ----- Helper conversion functions -----------------------------------------------------------------------

def unnormalize(norm, ref):
    """Recover a dimensional quantity from its normalized form: val = norm * ref."""
    val = norm * ref
    return val

def normalize(val, ref):
    """Normalize a dimensional quantity by a reference value: norm = val / ref."""
    norm = val / ref
    return norm

def calc_a_from_epsilon(epsilon, lref):
    """Compute minor radius from inverse aspect ratio: a = epsilon * lref."""
    a = unnormalize(epsilon, lref)
    return a

def calc_epsilon_from_a(a, lref):
    """Compute inverse aspect ratio from minor radius: epsilon = a / lref."""
    epsilon = normalize(a, lref)
    return epsilon

def calc_r_from_x(x, a):
    """Recover physical minor radius from normalized radial coordinate: r = x * a."""
    r = unnormalize(x, a)
    return r

def calc_x_from_r(r, a):
    """Normalize minor radius by the minor radius at the LCFS: x = r / a."""
    x = normalize(r, a)
    return x

def calc_ni_from_ninorm(ninorm, ne, nscale=1.0e19):
    """Recover ion density from its normalized form: ni = ninorm * nscale * ne."""
    nref = nscale * ne
    ni = unnormalize(ninorm, nref)
    return ni

def calc_ti_from_tinorm(tinorm, te, tscale=1.0e3):
    """Recover ion temperature from its normalized form: ti = tinorm * tscale * te."""
    tref = tscale * te
    ti = unnormalize(tinorm, tref)
    return ti

def calc_ninorm_from_ni(ni, ne):
    """Normalize ion density by electron density: ninorm = ni / ne."""
    ninorm = normalize(ni, ne)
    return ninorm

def calc_tinorm_from_ti(ti, te):
    """Normalize ion temperature by electron temperature: tinorm = ti / te."""
    tinorm = normalize(ti, te)
    return tinorm

def calc_ak_from_grad_k(grad_k, k, lref):
    """Compute the logarithmic gradient scale length a/L_k = -lref * (dk/dr) / k."""
    ak = -lref * grad_k / k
    return ak

def calc_grad_k_from_ak(ak, k, lref):
    """Recover the physical radial gradient of k from its log-gradient scale length a/L_k."""
    grad_k = ak * k / -lref
    return grad_k

def calc_q_circular_from_bp(bp, r, bo, ro):
    """Compute safety factor q from poloidal field under the circular-geometry approximation."""
    q = r * bo / (ro * bp)
    return q

def calc_bp_from_q_circular(q, r, bo, ro):
    """Compute poloidal field Bp from safety factor q under the circular-geometry approximation."""
    bp = r * bo / (ro * q)
    return bp

def calc_grad_q_circular_from_s(s, bp, bo, ro):
    """Compute dq/dr from magnetic shear s and Bp under the circular-geometry approximation."""
    grad_q = s * bo / (ro * bp)
    return grad_q

def calc_grad_q_from_s_and_q(s, q, r):
    """Compute the radial gradient of the safety factor from magnetic shear s and q."""
    grad_q = calc_grad_k_from_ak(s, q, -r)
    return grad_q

def calc_s_circular_from_grad_bp(grad_bp, bp, r):
    """Compute magnetic shear s from the radial gradient of Bp under the circular-geometry approximation."""
    s = 1.0 - r * grad_bp / bp
    return s

def calc_grad_bp_from_grad_q_circular(grad_q, bp, r, bo, ro):
    """Compute d(Bp)/dr from dq/dr under the circular-geometry approximation."""
    grad_bp = (1.0 - ro * bp * grad_q / bo) * bp / r
    return grad_bp

def calc_grad_bp_from_s_circular(s, bp, r):
    """Compute d(Bp)/dr from magnetic shear s under the circular-geometry approximation."""
    grad_bp = (1.0 - s) * bp / r
    return grad_bp

def calc_ninorm_from_zeff_and_quasineutrality(zeff, zia, zib, zi_target, ninorma, ze=1.0):
    """Solve for ni/ne of a target ion species using Zeff and quasineutrality with two known species."""
    zze = (zib - zeff) * ze
    zza = (zib - zia) * zia
    zz_target = (zib - zi_target) * zi_target
    ninorm_target = (zze - ninorma * zza) / zz_target
    return ninorm_target

# def calc_ninorm_from_quasineutrality(zia, zib, zi_target, ninorma, ninormb, ze=1.0):
#    ninorm_target = (ze - ninorma * zia - ninormb * zib) / zi_target
#    return ninorm_target

def calc_ninorm_from_quasineutrality(zi, zi_target, ninorm, ze=1.0):
    """Solve for ni/ne of a target ion species by enforcing quasineutrality given other species."""
    ninorm_target = (ze - ninorm * zi) / zi_target
    return ninorm_target

def calc_2ion_ninorm_from_ninorm_and_quasineutrality(ni, zia, zib, ne=None):
    """Return normalized densities for two ion species, deriving the second via quasineutrality."""
    ninorma = calc_ninorm_from_ni(ni, ne) if ne is not None else copy.deepcopy(ni)
    ninormb = calc_ninorm_from_quasineutrality(zia, zib, ninorma)
    return ninorma, ninormb

def calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ni, zeff, zia, zib, zic, ne=None):
    """Return normalized densities for three ion species using one known density, Zeff, and quasineutrality."""
    ninorma = calc_ninorm_from_ni(ni, ne) if ne is not None else copy.deepcopy(ni)
    ninormb = calc_ninorm_from_zeff_and_quasineutrality(zeff, zia, zic, zib, ninorma)
    ninormc = calc_ninorm_from_quasineutrality(1.0, zic, ninorma * zia + ninormb * zib)
    return ninorma, ninormb, ninormc

def calc_3ion_ninorm_from_ninorm_and_quasineutrality(nia, nib, zia, zib, zic, ne=None):
    """Return normalized densities for three ion species using two known densities and quasineutrality."""
    ninorma = calc_ninorm_from_ni(nia, ne) if ne is not None else copy.deepcopy(nia)
    ninormb = calc_ninorm_from_ni(nib, ne) if ne is not None else copy.deepcopy(nib)
    ninormc = calc_ninorm_from_quasineutrality(1.0, zic, ninorma * zia + ninormb * zib)
    return ninorma, ninormb, ninormc

def calc_ni_from_zeff_and_quasineutrality(zeff, zia, zib, zi_target, nia, ne):
    """Compute physical density of a target ion species from Zeff and quasineutrality."""
    ninorma = calc_ninorm_from_ni(nia, ne)
    ninorm_target = calc_ninorm_from_zeff_and_quasineutrality(zeff, zia, zib, zi_target, ninorma)
    ni_target = calc_ni_from_ninorm(ninorm_target, ne, nscale=1.0)
    return ni_target

# def calc_ni_from_quasineutrality(zia, zib, zi_target, nia, nib, ne):
#     ninorma = calc_ninorm_from_ni(nia, ne)
#     ninormb = calc_ninorm_from_ni(nib, ne)
#     ninorm_target = calc_ninorm_from_quasineutrality(1.0, zi_target, ninorma * zia + ninormb * zib)
#     ni_target = calc_ni_from_ninorm(ninorm_target, ne, nscale=1.0)
#     return ni_target

def calc_ni_from_quasineutrality(zi, zi_target, ni, ne):
    """Compute physical density of a target ion species by enforcing quasineutrality."""
    ninorm = calc_ninorm_from_ni(ni, ne)
    ninorm_target = calc_ninorm_from_quasineutrality(zi, zi_target, ninorm)
    ni_target = calc_ni_from_ninorm(ninorm_target, ne, nscale=1.0)
    return ni_target

def calc_2ion_ni_from_ni_and_quasineutrality(ni, zia, zib, ne, norm_inputs=False):
    """Return physical densities for two ion species, deriving the second via quasineutrality."""
    nia = calc_ni_from_ninorm(ni, ne, nscale=1.0) if norm_inputs else copy.deepcopy(ni)
    nib = calc_ni_from_quasineutrality(zia, zib, nia, ne)
    return nia, nib

def calc_3ion_ni_from_ni_zeff_and_quasineutrality(ni, zeff, zia, zib, zic, ne, norm_inputs=False):
    """Return physical densities for three ion species using one known density, Zeff, and quasineutrality."""
    nia = calc_ni_from_ninorm(ni, ne, nscale=1.0) if norm_inputs else copy.deepcopy(ni)
    nib = calc_ni_from_zeff_and_quasineutrality(zeff, zia, zic, zib, nia, ne)
    nic = calc_ni_from_quasineutrality(1.0, zic, nia * zia + nib * zib, ne)
    return nia, nib, nic

def calc_3ion_ni_from_ni_and_quasineutrality(nia, nib, zia, zib, zic, ne, norm_inputs=False):
    """Return physical densities for three ion species using two known densities and quasineutrality."""
    nia = calc_ni_from_ninorm(nia, ne, nscale=1.0) if norm_inputs else copy.deepcopy(nia)
    nib = calc_ni_from_ninorm(nib, ne, nscale=1.0) if norm_inputs else copy.deepcopy(nib)
    nic = calc_ni_from_quasineutrality(1.0, zic, nia * zia + nib * zib, ne)
    return nia, nib, nic

def calc_ani_from_azeff_and_gradient_quasineutrality(azeff, zeff, zia, zib, zi_target, ninorma, ninorm_target, ane, ania, ze=1.0):
    """Compute the normalized density gradient of a target species from the Zeff gradient and gradient quasineutrality."""
    zze = (zeff - zib) * ze
    zza = (zib - zia) * zia
    zz_target = (zi_target - zib) * zi_target
    ani_target = (azeff * zeff * ze + ane * zze + ania * ninorma * zza) / (ninorm_target * zz_target)
    return ani_target

# def calc_ani_from_gradient_quasineutrality(zia, zib, zi_target, ninorma, ninormb, ninorm_target, ane, ania, anib, ze=1.0):
#     ani_target = (ane * ze - ninorma * ania * zia - ninormb * anib * zib) / (ninorm_target * zi_target)
#     return ani_target

def calc_ani_from_gradient_quasineutrality(zi, zi_target, ninorm, ninorm_target, ane, ani, ze=1.0):
    """Compute the normalized density gradient of a target ion species from gradient quasineutrality."""
    ani_target = (ane * ze - ninorm * ani * zi) / (ninorm_target * zi_target)
    return ani_target

def calc_2ion_ani_from_ani_and_gradient_quasineutrality(ani, zia, zib, ane, ninorm, ne=1.0, lref=None):
    """Return normalized density gradients for two ion species, deriving the second via gradient quasineutrality."""
    nenorm = copy.deepcopy(ne) if lref is not None else 1.0
    ane = calc_ak_from_grad_k(ane, nenorm, lref) if lref is not None else copy.deepcopy(ane)
    ania = calc_ak_from_grad_k(ani, ninorm, lref) if lref is not None else copy.deepcopy(ani)
    ninorma, ninormb = calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorm, zia, zib, ne=nenorm)
    anib = calc_ani_from_gradient_quasineutrality(zia, zib, ninorma, ninormb, ane, ania)
    return ania, anib

def calc_3ion_ani_from_ani_azeff_and_gradient_quasineutrality(ani, azeff, zeff, zia, zib, zic, ane, ninorm, ne=1.0, lref=None):
    """Return normalized density gradients for three ion species using one known gradient, Zeff gradient, and gradient quasineutrality."""
    nenorm = copy.deepcopy(ne) if lref is not None else 1.0
    ane = calc_ak_from_grad_k(ane, nenorm, lref) if lref is not None else copy.deepcopy(ane)
    ania = calc_ak_from_grad_k(ani, ninorm, lref) if lref is not None else copy.deepcopy(ani)
    azeff_temp = calc_ak_from_grad_k(azeff, zeff, lref) if lref is not None else copy.deepcopy(azeff)
    ninorma, ninormb, ninormc = calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ninorm, zeff, zia, zib, zic, ne=nenorm)
    anib = calc_ani_from_azeff_and_gradient_quasineutrality(azeff_temp, zeff, zia, zic, zib, ninorma, ninormb, ane, ania)
    anic = calc_ani_from_gradient_quasineutrality(1.0, zic, 1.0, ninormc, ane, ninorma * ania * zia + ninormb * anib * zib)
    return ania, anib, anic

def calc_3ion_ani_from_ani_and_gradient_quasineutrality(ania, anib, zia, zib, zic, ane, ninorma, ninormb, ne=1.0, lref=None):
    """Return normalized density gradients for three ion species using two known gradients and gradient quasineutrality."""
    nenorm = copy.deepcopy(ne) if lref is not None else 1.0
    ane = calc_ak_from_grad_k(ane, nenorm, lref) if lref is not None else copy.deepcopy(ane)
    ania_temp = calc_ak_from_grad_k(ania, ninorma, lref) if lref is not None else copy.deepcopy(ania)
    anib_temp = calc_ak_from_grad_k(anib, ninormb, lref) if lref is not None else copy.deepcopy(anib)
    ninorma_temp, ninormb_temp, ninormc = calc_3ion_ninorm_from_ninorm_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=nenorm)
    anic = calc_ani_from_gradient_quasineutrality(1.0, zic, 1.0, ninormc, ane, ninorma_temp * ania_temp * zia + ninormb_temp * anib_temp * zib)
    return ania_temp, anib_temp, anic

def calc_grad_ni_from_grad_zeff_and_gradient_quasineutrality(grad_zeff, zeff, zia, zib, zi_target, nia, ni_target, grad_nia, ne, grad_ne, lref):
    """Compute d(ni)/dr for a target species from the Zeff gradient and gradient quasineutrality."""
    azeff = calc_ak_from_grad_k(grad_zeff, zeff, lref)
    ninorma = calc_ninorm_from_ni(nia, ne)
    ninorm_target = calc_ninorm_from_ni(ni_target, ne)
    ane = calc_ak_from_grad_k(grad_ne, ne, lref)
    ania = calc_ak_from_grad_k(grad_nia, nia, lref)
    ani_target = calc_ani_from_azeff_and_gradient_quasineutrality(azeff, zeff, zia, zib, zi_target, ninorma, ninorm_target, ane, ania)
    grad_ni_target = calc_grad_k_from_ak(ani_target, ni_target, lref)
    return grad_ni_target

# def calc_grad_ni_from_gradient_quasineutrality(zia, zib, zi_target, nia, nib, ni_target, grad_ne, grad_nia, grad_nib, ne, lref):
#     ninorma = calc_ninorm_from_ni(nia, ne)
#     ninormb = calc_ninorm_from_ni(nib, ne)
#     ninorm_target = calc_ninorm_from_ni(ni_target, ne)
#     ane = calc_ak_from_grad_k(grad_ne, ne, lref)
#     ania = calc_ak_from_grad_k(grad_nia, nia, lref)
#     anib = calc_ak_from_grad_k(grad_nib, nib, lref)
#     ani_target = calc_ani_from_gradient_quasineutrality(zia, zib, zi_target, ninorma, ninormb, ninorm_target, ane, ania, anib)
#     grad_ni_target = calc_grad_k_from_ak(ani_target, ni_target, lref)
#     return grad_ni_target

def calc_grad_ni_from_gradient_quasineutrality(zi, zi_target, ni, ni_target, grad_ni, ne, grad_ne, lref):
    """Compute d(ni)/dr for a target ion species by differentiating the quasineutrality constraint."""
    ninorm = calc_ninorm_from_ni(ni, ne)
    ninorm_target = calc_ninorm_from_ni(ni_target, ne)
    ane = calc_ak_from_grad_k(grad_ne, ne, lref)
    ani = calc_ak_from_grad_k(grad_ni, ni, lref)
    ani_target = calc_ani_from_gradient_quasineutrality(zi, zi_target, ninorm, ninorm_target, ane, ani)
    grad_ni_target = calc_grad_k_from_ak(ani_target, ni_target, lref)
    return grad_ni_target

def calc_2ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(grad_ni, zia, zib, ni, grad_ne, ne, lref, norm_inputs=False):
    """Return d(ni)/dr for two ion species, deriving the second via gradient quasineutrality."""
    nia, nib = calc_2ion_ni_from_ni_and_quasineutrality(ni, zia, zib, ne, norm_inputs=norm_inputs)
    grad_ne_temp = calc_grad_k_from_ak(grad_ne, ne, lref) if norm_inputs else copy.deepcopy(grad_ne)
    grad_nia = calc_grad_k_from_ak(grad_ni, nia, lref) if norm_inputs else copy.deepcopy(grad_ni)
    grad_nib = calc_grad_ni_from_gradient_quasineutrality(zia, zib, nia, nib, grad_nia, ne, grad_ne_temp, lref)
    return grad_nia, grad_nib

def calc_3ion_grad_ni_from_grad_ni_grad_zeff_and_gradient_quasineutrality(grad_ni, grad_zeff, zeff, zia, zib, zic, ni, grad_ne, ne, lref, norm_inputs=False):
    """Return d(ni)/dr for three ion species using one known gradient, Zeff gradient, and gradient quasineutrality."""
    nia, nib, nic = calc_3ion_ni_from_ni_zeff_and_quasineutrality(ni, zeff, zia, zib, zic, ne, norm_inputs=norm_inputs)
    grad_ne_temp = calc_grad_k_from_ak(grad_ne, ne, lref) if norm_inputs else copy.deepcopy(grad_ne)
    grad_nia = calc_grad_k_from_ak(grad_ni, nia, lref) if norm_inputs else copy.deepcopy(grad_ni)
    grad_zeff_temp = calc_grad_k_from_ak(grad_zeff, zeff, lref) if norm_inputs else copy.deepcopy(grad_zeff)
    grad_nib = calc_grad_ni_from_grad_zeff_and_gradient_quasineutrality(grad_zeff_temp, zeff, zia, zic, zib, nia, nib, grad_nia, ne, grad_ne_temp, lref)
    grad_nic = calc_grad_ni_from_gradient_quasineutrality(1.0, zic, ne, nic, grad_nia * zia + grad_nib * zib, ne, grad_ne_temp, lref)
    return grad_nia, grad_nib, grad_nic

def calc_3ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(grad_nia, grad_nib, zia, zib, zic, nia, nib, grad_ne, ne, lref, norm_inputs=False):
    """Return d(ni)/dr for three ion species using two known gradients and gradient quasineutrality."""
    nia_temp, nib_temp, nic = calc_3ion_ni_from_ni_and_quasineutrality(nia, nib, zia, zib, zic, ne, norm_inputs=norm_inputs)
    grad_ne_temp = calc_grad_k_from_ak(grad_ne, ne, lref) if norm_inputs else copy.deepcopy(grad_ne)
    grad_nia_temp = calc_grad_k_from_ak(grad_nia, nia_temp, lref) if norm_inputs else copy.deepcopy(grad_nia)
    grad_nib_temp = calc_grad_k_from_ak(grad_nib, nib_temp, lref) if norm_inputs else copy.deepcopy(grad_nib)
    grad_nic = calc_grad_ni_from_gradient_quasineutrality(1.0, zic, ne, nic, grad_nia_temp * zia + grad_nib_temp * zib, ne, grad_ne_temp, lref)
    return grad_nia_temp, grad_nib_temp, grad_nic

def calc_p_from_pnorm(pnorm, ne, te):
    """Recover physical pressure from its normalized value: p = e * ne * te * pnorm."""
    c = constants_si()
    p = c['e'] * ne * te * pnorm
    return p

def calc_pnorm_from_p(p, ne, te):
    """Normalize pressure by the electron thermal energy density: pnorm = p / (e * ne * te)."""
    c = constants_si()
    pnorm = p / (c['e'] * ne * te)
    return pnorm

def calc_2ion_pnorm_with_2ions(ninorma, ninormb, tinorma, tinormb, ne=None, te=None):
    """Compute total normalized pressure for a two-ion system given both species' densities and temperatures."""
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    tinorma_temp = calc_tinorm_from_ti(tinorma, te) if te is not None else copy.deepcopy(tinorma)
    tinormb_temp = calc_tinorm_from_ti(tinormb, te) if te is not None else copy.deepcopy(tinormb)
    pnorm = 1.0 + ninorma_temp * tinorma_temp + ninormb_temp * tinormb_temp
    return pnorm

def calc_2ion_pnorm_with_1ion_and_quasineutrality(ninorma, tinorma, tinormb, zia, zib, ne=None, te=None):
    """Compute total normalized pressure for a two-ion system, deriving the second density via quasineutrality."""
    ninorma_temp, ninormb = calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorma, zia, zib, ne=ne)
    pnorm = calc_2ion_pnorm_with_2ions(ninorma_temp, ninormb, tinorma, tinormb, ne=None, te=te)
    return pnorm

def calc_3ion_pnorm_with_3ions(ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, ne=None, te=None):
    """Compute total normalized pressure for a three-ion system given all species' densities and temperatures."""
    ninormc_temp = calc_ninorm_from_ni(ninormc, ne) if ne is not None else copy.deepcopy(ninormc)
    tinormc_temp = calc_tinorm_from_ti(tinormc, te) if te is not None else copy.deepcopy(tinormc)
    pnorm_2ion = calc_2ion_pnorm_with_2ions(ninorma, ninormb, tinorma, tinormb, ne=ne, te=te)
    pnorm = pnorm_2ion + ninormc_temp * tinormc_temp
    return pnorm

def calc_3ion_pnorm_with_2ions_and_quasineutrality(ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, ne=None, te=None):
    """Compute total normalized pressure for a three-ion system, inferring the third density via quasineutrality."""
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    ninormc = calc_ninorm_from_quasineutrality(1.0, zic, ninorma_temp * zia + ninormb_temp * zib)
    pnorm = calc_3ion_pnorm_with_3ions(ninorma_temp, ninormb_temp, ninormc, tinorma, tinormb, tinormc, ne=None, te=te)
    return pnorm

def calc_3ion_pnorm_with_1ion_zeff_and_quasineutrality(ninorma, tinorma, tinormb, tinormc, zeff, zia, zib, zic, ne=None, te=None):
    """Compute total normalized pressure for a three-ion system using one known density, Zeff, and quasineutrality."""
    ninorma_temp, ninormb, ninormc = calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ninorma, zeff, zia, zib, zic, ne=ne)
    pnorm = calc_3ion_pnorm_with_3ions(ninorma_temp, ninormb, ninormc, tinorma, tinormb, tinormc, ne=None, te=te)
    return pnorm

def calc_2ion_p_with_2ions(nia, nib, tia, tib, ne, te, norm_inputs=False):
    """Compute physical total pressure for a two-ion system given both species' densities and temperatures."""
    ninorma = calc_ninorm_from_ni(nia, ne) if not norm_inputs else copy.deepcopy(nia)
    ninormb = calc_ninorm_from_ni(nib, ne) if not norm_inputs else copy.deepcopy(nib)
    tinorma = calc_tinorm_from_ti(tia, te) if not norm_inputs else copy.deepcopy(tia)
    tinormb = calc_tinorm_from_ti(tib, te) if not norm_inputs else copy.deepcopy(tib)
    pnorm = calc_2ion_pnorm_with_2ions(ninorma, ninormb, tinorma, tinormb)
    p = calc_p_from_pnorm(pnorm, ne, te)
    return p

def calc_2ion_p_with_1ion_and_quasineutrality(nia, tia, tib, zia, zib, ne, te, norm_inputs=False):
    """Compute physical total pressure for a two-ion system, inferring the second density via quasineutrality."""
    nia_temp, nib = calc_2ion_ni_from_ni_and_quasineutrality(nia, zia, zib, ne, norm_inputs=norm_inputs)
    tia_temp = calc_ti_from_tinorm(tia, te, tscale=1.0) if norm_inputs else copy.deepcopy(tia)
    tib_temp = calc_ti_from_tinorm(tib, te, tscale=1.0) if norm_inputs else copy.deepcopy(tib)
    p = calc_2ion_p_with_2ions(nia_temp, nib, tia_temp, tib_temp, ne, te, norm_inputs=False)
    return p

def calc_3ion_p_with_3ions(nia, nib, nic, tia, tib, tic, ne, te, norm_inputs=False):
    """Compute physical total pressure for a three-ion system given all species' densities and temperatures."""
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    pnorm = calc_3ion_pnorm_with_3ions(nia, nib, nic, tia, tib, tic, ne=ne_temp, te=te_temp)
    p = calc_p_from_pnorm(pnorm, ne, te)
    return p

def calc_3ion_p_with_2ions_and_quasineutrality(nia, nib, tia, tib, tic, zia, zib, zic, ne, te, norm_inputs=False):
    """Compute physical total pressure for a three-ion system, inferring the third density via quasineutrality."""
    nia_temp, nib_temp, nic = calc_3ion_ni_from_ni_and_quasineutrality(nia, nib, zia, zib, zic, ne, norm_inputs=norm_inputs)
    tia_temp = calc_ti_from_tinorm(tia, te, tscale=1.0) if norm_inputs else copy.deepcopy(tia)
    tib_temp = calc_ti_from_tinorm(tib, te, tscale=1.0) if norm_inputs else copy.deepcopy(tib)
    tic_temp = calc_ti_from_tinorm(tic, te, tscale=1.0) if norm_inputs else copy.deepcopy(tic)
    p = calc_3ion_p_with_3ions(nia_temp, nib_temp, nic, tia_temp, tib_temp, tic_temp, ne, te, norm_inputs=False)
    return p

def calc_3ion_p_with_1ion_zeff_and_quasineutrality(nia, tia, tib, tic, zeff, zia, zib, zic, ne, te, norm_inputs=False):
    """Compute physical total pressure for a three-ion system using one known density, Zeff, and quasineutrality."""
    nia_temp, nib, nic = calc_3ion_ni_from_ni_zeff_and_quasineutrality(nia, zeff, zia, zib, zic, ne, norm_inputs=norm_inputs)
    tia_temp = calc_ti_from_tinorm(tia, te, tscale=1.0) if norm_inputs else copy.deepcopy(tia)
    tib_temp = calc_ti_from_tinorm(tib, te, tscale=1.0) if norm_inputs else copy.deepcopy(tib)
    tic_temp = calc_ti_from_tinorm(tic, te, tscale=1.0) if norm_inputs else copy.deepcopy(tic)
    p = calc_3ion_p_with_3ions(nia_temp, nib, nic, tia_temp, tib_temp, tic_temp, ne, te, norm_inputs=False)
    return p

def calc_grad_p_from_ap(ap, ne, te, lref):
    """Recover the physical pressure gradient from the normalized scale length a/Lp."""
    c = constants_si()
    grad_p = calc_grad_k_from_ak(ap, c['e'] * ne * te, lref)
    return grad_p

def calc_ap_from_grad_p(grad_p, ne, te, lref):
    """Compute the normalized pressure gradient scale length a/Lp = -lref * (dp/dr) / (e*ne*te)."""
    c = constants_si()
    ap = calc_ak_from_grad_k(grad_p, c['e'] * ne * te, lref)
    return ap

def calc_2ion_ap_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, ane, ate, ne=None, te=None, lref=None):
    """Compute total normalized pressure gradient (a/Lp) for a two-ion system given both species' density and temperature gradients."""
    ane_temp = calc_ak_from_grad_k(ane, ne, lref) if ne is not None and lref is not None else copy.deepcopy(ane)
    ate_temp = calc_ak_from_grad_k(ate, te, lref) if ne is not None and lref is not None else copy.deepcopy(ate)
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    tinorma_temp = calc_tinorm_from_ti(tinorma, te) if te is not None else copy.deepcopy(tinorma)
    tinormb_temp = calc_tinorm_from_ti(tinormb, te) if te is not None else copy.deepcopy(tinormb)
    ania_temp = calc_ak_from_grad_k(ania, calc_ni_from_ninorm(ninorma_temp, ne, nscale=1.0), lref) if ne is not None and lref is not None else copy.deepcopy(ania)
    anib_temp = calc_ak_from_grad_k(anib, calc_ni_from_ninorm(ninormb_temp, ne, nscale=1.0), lref) if ne is not None and lref is not None else copy.deepcopy(anib)
    atia_temp = calc_ak_from_grad_k(atia, calc_ti_from_tinorm(tinorma_temp, te, tscale=1.0), lref) if te is not None and lref is not None else copy.deepcopy(atia)
    atib_temp = calc_ak_from_grad_k(atib, calc_ti_from_tinorm(tinormb_temp, te, tscale=1.0), lref) if te is not None and lref is not None else copy.deepcopy(atib)
    ap = ane_temp + ate_temp + ninorma_temp * tinorma_temp * (ania_temp + atia_temp) + ninormb_temp * tinormb_temp * (anib_temp + atib_temp)
    return ap

def calc_2ion_ap_with_1ion_and_gradient_quasineutrality(ania, atia, atib, ninorma, tinorma, tinormb, zia, zib, ane, ate, ne=None, te=None, lref=None):
    """Compute a/Lp for a two-ion system, deriving the second species density gradient via gradient quasineutrality."""
    ninorma_temp, ninormb = calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorma, zia, zib, ne=ne)
    ania_temp, anib = calc_2ion_ani_from_ani_and_gradient_quasineutrality(ania, zia, zib, ane, ninorma, ne=ne, lref=lref)
    ninorma_temp = calc_ni_from_ninorm(ninorma_temp, ne, nscale=1.0) if ne is not None else copy.deepcopy(ninorma_temp)
    ninormb_temp = calc_ni_from_ninorm(ninormb, ne, nscale=1.0) if ne is not None else copy.deepcopy(ninormb)
    ania_temp = calc_grad_k_from_ak(ania_temp, ninorma_temp, lref) if lref is not None else copy.deepcopy(ania_temp)
    anib_temp = calc_grad_k_from_ak(anib, ninormb_temp, lref) if lref is not None else copy.deepcopy(anib)
    ap = calc_2ion_ap_with_2ions(ania_temp, anib_temp, atia, atib, ninorma_temp, ninormb_temp, tinorma, tinormb, ane, ate, ne=ne, te=te, lref=lref)
    return ap

def calc_3ion_ap_with_3ions(ania, anib, anic, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, ane, ate, ne=None, te=None, lref=None):
    """Compute total normalized pressure gradient (a/Lp) for a three-ion system given all species' density and temperature gradients."""
    ninormc_temp = calc_ninorm_from_ni(ninormc, ne) if ne is not None else copy.deepcopy(ninormc)
    tinormc_temp = calc_tinorm_from_ti(tinormc, te) if te is not None else copy.deepcopy(tinormc)
    anic_temp = calc_ak_from_grad_k(anic, calc_ni_from_ninorm(ninormc_temp, ne, nscale=1.0), lref) if ne is not None and lref is not None else copy.deepcopy(anic)
    atic_temp = calc_ak_from_grad_k(atic, calc_ti_from_tinorm(tinormc_temp, te, tscale=1.0), lref) if te is not None and lref is not None else copy.deepcopy(atic)
    ap_2ion = calc_2ion_ap_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, ane, ate, ne=ne, te=te, lref=lref)
    ap = ap_2ion + ninormc_temp * tinormc_temp * (anic_temp + atic_temp)
    return ap

def calc_3ion_ap_with_2ions_and_gradient_quasineutrality(ania, anib, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, ane, ate, ne=None, te=None, lref=None):
    """Compute a/Lp for a three-ion system, inferring the third species density gradient via gradient quasineutrality."""
    ninorma_temp, ninormb_temp, ninormc = calc_3ion_ninorm_from_ninorm_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=ne)
    ania_temp, anib_temp, anic = calc_3ion_ani_from_ani_and_gradient_quasineutrality(ania, anib, zia, zib, zic, ane, ninorma, ninormb, ne=ne, lref=lref)
    ninorma_temp = calc_ni_from_ninorm(ninorma_temp, ne, nscale=1.0) if ne is not None else copy.deepcopy(ninorma_temp)
    ninormb_temp = calc_ni_from_ninorm(ninormb_temp, ne, nscale=1.0) if ne is not None else copy.deepcopy(ninormb_temp)
    ninormc_temp = calc_ni_from_ninorm(ninormc, ne, nscale=1.0) if ne is not None else copy.deepcopy(ninormc)
    ania_temp = calc_grad_k_from_ak(ania_temp, ninorma_temp, lref) if lref is not None else copy.deepcopy(ania_temp)
    anib_temp = calc_grad_k_from_ak(anib_temp, ninormb_temp, lref) if lref is not None else copy.deepcopy(anib_temp)
    anic_temp = calc_grad_k_from_ak(anic, ninormc_temp, lref) if lref is not None else copy.deepcopy(anic)
    ap = calc_3ion_ap_with_3ions(ania_temp, anib_temp, anic_temp, atia, atib, atic, ninorma_temp, ninormb_temp, ninormc_temp, tinorma, tinormb, tinormc, ane, ate, ne=ne, te=te, lref=lref)
    return ap

def calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(ania, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic, ane, ate, ne=None, te=None, lref=None):
    """Compute a/Lp for a three-ion system using one known density gradient, Zeff gradient, and gradient quasineutrality."""
    ninorma_temp, ninormb, ninormc = calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ninorma, zeff, zia, zib, zic, ne=ne)
    ania_temp, anib, anic = calc_3ion_ani_from_ani_azeff_and_gradient_quasineutrality(ania, azeff, zeff, zia, zib, zic, ane, ninorma, ne=ne, lref=lref)
    ninorma_temp = calc_ni_from_ninorm(ninorma_temp, ne, nscale=1.0) if ne is not None else copy.deepcopy(ninorma_temp)
    ninormb_temp = calc_ni_from_ninorm(ninormb, ne, nscale=1.0) if ne is not None else copy.deepcopy(ninormb)
    ninormc_temp = calc_ni_from_ninorm(ninormc, ne, nscale=1.0) if ne is not None else copy.deepcopy(ninormc)
    ania_temp = calc_grad_k_from_ak(ania_temp, ninorma_temp, lref) if lref is not None else copy.deepcopy(ania_temp)
    anib_temp = calc_grad_k_from_ak(anib, ninormb_temp, lref) if lref is not None else copy.deepcopy(anib)
    anic_temp = calc_grad_k_from_ak(anic, ninormc_temp, lref) if lref is not None else copy.deepcopy(anic)
    ap = calc_3ion_ap_with_3ions(ania_temp, anib_temp, anic_temp, atia, atib, atic, ninorma_temp, ninormb_temp, ninormc_temp, tinorma, tinormb, tinormc, ane, ate, ne=ne, te=te, lref=lref)
    return ap

def calc_2ion_grad_p_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, grad_ne, grad_te, ne, te, lref, norm_inputs=False):
    """Compute the physical total pressure gradient for a two-ion system given both species' gradients."""
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    lref_temp = None if norm_inputs else copy.deepcopy(lref)
    ap = calc_2ion_ap_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, grad_ne, grad_te, ne=ne_temp, te=te_temp, lref=lref_temp)
    grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    return grad_p

def calc_2ion_grad_p_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, grad_ne, grad_te, ne, te, lref, norm_inputs=False):
    """Compute the physical total pressure gradient for a two-ion system, inferring the second gradient via gradient quasineutrality."""
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    lref_temp = None if norm_inputs else copy.deepcopy(lref)
    ap = calc_2ion_ap_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, grad_ne, grad_te, ne=ne_temp, te=te_temp, lref=lref_temp)
    grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    return grad_p

def calc_3ion_grad_p_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, grad_ne, grad_te, ne, te, lref, norm_inputs=False):
    """Compute the physical total pressure gradient for a three-ion system given all species' gradients."""
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    lref_temp = None if norm_inputs else copy.deepcopy(lref)
    ap = calc_3ion_ap_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, grad_ne, grad_te, ne=ne_temp, te=te_temp, lref=lref_temp)
    grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    return grad_p

def calc_3ion_grad_p_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, grad_ne, grad_te, ne, te, lref, norm_inputs=False):
    """Compute the physical pressure gradient for a three-ion system, inferring the third gradient via gradient quasineutrality."""
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    lref_temp = None if norm_inputs else copy.deepcopy(lref)
    ap = calc_3ion_ap_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, grad_ne, grad_te, ne=ne_temp, te=te_temp, lref=lref_temp)
    grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    return grad_p

def calc_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, grad_ne, grad_te, ne, te, lref, norm_inputs=False):
    """Compute the physical pressure gradient for a three-ion system using one known density gradient, Zeff gradient, and gradient quasineutrality."""
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    lref_temp = None if norm_inputs else copy.deepcopy(lref)
    ap = calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, grad_ne, grad_te, ne=ne_temp, te=te_temp, lref=lref_temp)
    grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    return grad_p

def calc_zeff_from_2ion_ninorm_with_2ions(ninorma, ninormb, zia, zib, ne=None, ze=1.0):
    """Compute Zeff from two ion species with known normalized densities."""
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    zeff = (ninorma_temp * (zia ** 2) + ninormb_temp * (zib ** 2)) / ze
    return zeff

def calc_zeff_from_2ion_ninorm_with_1ion_and_quasineutrality(ninorma, zia, zib, ne=None, ze=1.0):
    """Compute Zeff for a two-ion system, deriving the second density via quasineutrality."""
    ninorma_temp, ninormb = calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorma, zia, zib, ne=ne)
    zeff = calc_zeff_from_2ion_ninorm_with_2ions(ninorma_temp, ninormb, zia, zib, ne=None, ze=ze)
    return zeff

def calc_zeff_from_3ion_ninorm_with_3ions(ninorma, ninormb, ninormc, zia, zib, zic, ne=None, ze=1.0):
    """Compute Zeff from three ion species with known normalized densities."""
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    ninormc_temp = calc_ninorm_from_ni(ninormc, ne) if ne is not None else copy.deepcopy(ninormc)
    zeff = (ninorma_temp * (zia ** 2) + ninormb_temp * (zib ** 2) + ninormc_temp * (zic ** 2)) / ze
    return zeff

def calc_zeff_from_3ion_ninorm_with_2ions_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=None, ze=1.0):
    """Compute Zeff for a three-ion system, inferring the third density via quasineutrality."""
    ninorma_temp, ninormb_temp, ninormc = calc_3ion_ninorm_from_ninorm_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=ne)
    zeff = calc_zeff_from_3ion_ninorm_with_3ions(ninorma_temp, ninormb_temp, ninormc, zia, zib, zic, ne=None, ze=ze)
    return zeff

def calc_azeff_from_2ion_ani_with_2ions(ania, anib, ninorma, ninormb, zia, zib, ane, ne=None, lref=None, ze=1.0):
    """Compute the normalized Zeff gradient (a/L_Zeff) from two ion species with known normalized density gradients."""
    ane_temp = calc_ak_from_grad_k(ane, ne, lref) if ne is not None and lref is not None else copy.deepcopy(ane)
    ania_temp = calc_ak_from_grad_k(ania, ninorma, lref) if lref is not None else copy.deepcopy(ania)
    anib_temp = calc_ak_from_grad_k(anib, ninormb, lref) if lref is not None else copy.deepcopy(anib)
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    zeff = calc_zeff_from_2ion_ninorm_with_2ions(ninorma_temp, ninormb_temp, zia, zib, ne=None)
    azeff = ((ania_temp * ninorma_temp * (zia ** 2) + anib_temp * ninormb_temp * (zib ** 2)) / (zeff * ze)) - ane_temp
    return azeff

def calc_azeff_from_2ion_ani_with_1ion_and_gradient_quasineutrality(ania, ninorma, zia, zib, ane, ne=None, lref=None, ze=1.0):
    """Compute a/L_Zeff for a two-ion system, inferring the second gradient via gradient quasineutrality."""
    ane_temp = calc_ak_from_grad_k(ane, ne, lref) if ne is not None and lref is not None else copy.deepcopy(ane)
    ninorma_temp, ninormb = calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorma, zia, zib, ne=ne)
    ania_temp, anib = calc_2ion_ani_from_ani_and_gradient_quasineutrality(ania, zia, zib, ane, ninorma, ne=ne, lref=lref)
    azeff = calc_azeff_from_2ion_ani_with_2ions(ania_temp, anib, ninorma_temp, ninormb, zia, zib, ane_temp, ne=None, lref=None, ze=ze)
    return azeff

def calc_azeff_from_3ion_ani_with_3ions(ania, anib, anic, ninorma, ninormb, ninormc, zia, zib, zic, ane, ne=None, lref=None, ze=1.0):
    """Compute a/L_Zeff from three ion species with known normalized density gradients."""
    ane_temp = calc_ak_from_grad_k(ane, ne, lref) if ne is not None and lref is not None else copy.deepcopy(ane)
    ania_temp = calc_ak_from_grad_k(ania, ninorma, lref) if lref is not None else copy.deepcopy(ania)
    anib_temp = calc_ak_from_grad_k(anib, ninormb, lref) if lref is not None else copy.deepcopy(anib)
    anic_temp = calc_ak_from_grad_k(anic, ninormc, lref) if lref is not None else copy.deepcopy(anic)
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    ninormc_temp = calc_ninorm_from_ni(ninormc, ne) if ne is not None else copy.deepcopy(ninormc)
    zeff = calc_zeff_from_3ion_ninorm_with_3ions(ninorma_temp, ninormb_temp, ninormc_temp, zia, zib, zic, ne=None)
    azeff = ((ania_temp * ninorma_temp * (zia ** 2) + anib_temp * ninormb_temp * (zib ** 2) + anic_temp * ninormc_temp * (zic ** 2)) / (zeff * ze)) - ane_temp
    return azeff

def calc_azeff_from_3ion_ani_with_2ions_and_gradient_quasineutrality(ania, anib, ninorma, ninormb, zia, zib, zic, ane, ne=None, lref=None, ze=1.0):
    """Compute a/L_Zeff for a three-ion system, inferring the third gradient via gradient quasineutrality."""
    ane_temp = calc_ak_from_grad_k(ane, ne, lref) if ne is not None and lref is not None else copy.deepcopy(ane)
    ninorma_temp, ninormb_temp, ninormc = calc_3ion_ninorm_from_ninorm_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=ne)
    ania_temp, anib_temp, anic = calc_3ion_ani_from_ani_and_gradient_quasineutrality(ania, anib, zia, zib, zic, ane, ninorma, ninormb, ne=ne, lref=lref)
    azeff = calc_azeff_from_3ion_ani_with_3ions(ania_temp, anib_temp, anic, ninorma_temp, ninormb_temp, ninormc, zia, zib, zic, ane_temp, ne=None, lref=None, ze=ze)
    return azeff

def calc_ne_from_beta_and_pnorm(beta, pnorm, btot, te):
    """Invert the beta definition to solve for electron density given beta and normalized pressure."""
    c = constants_si()
    ne = beta * (btot ** 2) / (2.0 * c['mu'] * c['e'] * te * pnorm)
    return ne

def calc_te_from_beta_and_pnorm(beta, pnorm, btot, ne):
    """Invert the beta definition to solve for electron temperature given beta and normalized pressure."""
    c = constants_si()
    te = beta * (btot ** 2) / (2.0 * c['mu'] * c['e'] * ne * pnorm)
    return te

def calc_btot_from_beta_and_pnorm(beta, pnorm, ne, te):
    """Solve for total magnetic field strength from beta and normalized pressure."""
    c = constants_si()
    btot = np.sqrt(2.0 * c['mu'] * c['e'] * ne * te * pnorm / beta)
    return btot

def calc_btot_from_beta_and_p(beta, p):
    """Solve for total magnetic field strength from beta and physical pressure."""
    c = constants_si()
    btot = np.sqrt(2.0 * c['mu'] * p / beta)
    return btot

def calc_beta_from_p(p, btot):
    """Compute plasma beta as the ratio of thermal pressure to magnetic pressure: beta = 2*mu0*p / B^2."""
    c = constants_si()
    beta = 2.0 * c['mu'] * p / (btot ** 2)
    return beta

def calc_beta_from_pnorm(pnorm, btot, ne, te):
    """Compute plasma beta from normalized pressure, electron density, and electron temperature."""
    c = constants_si()
    betae = 2.0 * c['mu'] * c['e'] * ne * te / (btot ** 2)
    beta = betae * pnorm
    return beta

def calc_ne_from_alpha_and_ap(alpha, ap, q, btot, te):
    """Invert the alpha definition to solve for electron density given alpha and normalized pressure gradient a/Lp."""
    c = constants_si()
    ne = alpha * (btot ** 2) / (2.0 * c['mu'] * (q ** 2) * c['e'] * te * ap)
    return ne

def calc_te_from_alpha_and_ap(alpha, ap, q, btot, ne):
    """Invert the alpha definition to solve for electron temperature given alpha and normalized pressure gradient a/Lp."""
    c = constants_si()
    te = alpha * (btot ** 2) / (2.0 * c['mu'] * (q ** 2) * c['e'] * ne * ap)
    return te

def calc_btot_from_alpha_and_ap(alpha, ap, q, ne, te):
    """Solve for total magnetic field from alpha and normalized pressure gradient a/Lp."""
    c = constants_si()
    btot = np.sqrt(2.0 * c['mu'] * (q ** 2) * c['e'] * ne * te * ap / alpha)
    return btot

def calc_btot_from_alpha_and_grad_p(alpha, grad_p, q, lref):
    """Solve for total magnetic field from alpha and the physical pressure gradient."""
    c = constants_si()
    btot = np.sqrt(2.0 * c['mu'] * (q ** 2) * -lref * grad_p / alpha)
    return btot

def calc_alpha_from_grad_p(grad_p, q, btot, lref):
    """Compute the MHD stability parameter alpha = -2*mu0*q^2*lref*(dp/dr) / B^2."""
    c = constants_si()
    alpha = -2.0 * c['mu'] * (q ** 2) * lref * grad_p / (btot ** 2)
    return alpha

def calc_alpha_from_ap(ap, q, btot, ne, te):
    """Compute alpha from the normalized pressure gradient a/Lp: alpha = q^2 * betae * ap."""
    c = constants_si()
    betae = 2.0 * c['mu'] * c['e'] * ne * te / (btot ** 2)
    alpha = q * q * betae * ap
    return alpha

def calc_alpha_from_2ion_ap_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, q, ane, ate, btot, ne, te, lref=None):
    """Compute alpha for a two-ion system given both species' normalized density and temperature gradients."""
    ne_temp = copy.deepcopy(ne) if lref is not None else None
    te_temp = copy.deepcopy(te) if lref is not None else None
    ap = calc_2ion_ap_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, ane, ate, ne=ne_temp, te=te_temp, lref=lref)
    alpha = calc_alpha_from_ap(ap, q, btot, ne, te)
    return alpha

def calc_alpha_from_2ion_ap_with_1ion_and_gradient_quasineutrality(ania, atia, atib, ninorma, tinorma, tinormb, zia, zib, q, ane, ate, btot, ne, te, lref=None):
    """Compute alpha for a two-ion system, inferring the second density gradient via gradient quasineutrality."""
    ne_temp = copy.deepcopy(ne) if lref is not None else None
    te_temp = copy.deepcopy(te) if lref is not None else None
    ap = calc_2ion_ap_with_1ion_and_gradient_quasineutrality(ania, atia, atib, ninorma, tinorma, tinormb, zia, zib, ane, ate, ne=ne_temp, te=te_temp, lref=lref)
    alpha = calc_alpha_from_ap(ap, q, btot, ne, te)
    return alpha

def calc_alpha_from_2ion_grad_p_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False):
    """Compute alpha from physical pressure gradients for a two-ion system with both species known."""
    grad_p = calc_2ion_grad_p_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, grad_ne, grad_te, ne, te, lref, norm_inputs=norm_inputs)
    alpha = calc_alpha_from_grad_p(grad_p, q, btot, lref)
    return alpha

def calc_alpha_from_2ion_grad_p_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False):
    """Compute alpha from physical pressure gradients for a two-ion system, inferring the second via gradient quasineutrality."""
    grad_p = calc_2ion_grad_p_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, grad_ne, grad_te, ne, te, lref, norm_inputs=norm_inputs)
    alpha = calc_alpha_from_grad_p(grad_p, q, btot, lref)
    return alpha

def calc_alpha_from_3ion_ap_with_3ions(ania, anib, anic, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, q, ane, ate, btot, ne, te, lref=None):
    """Compute alpha for a three-ion system given all species' normalized density and temperature gradients."""
    ne_temp = copy.deepcopy(ne) if lref is not None else None
    te_temp = copy.deepcopy(te) if lref is not None else None
    ap = calc_3ion_ap_with_3ions(ania, anib, anic, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, ane, ate, ne=ne_temp, te=te_temp, lref=lref)
    alpha = calc_alpha_from_ap(ap, q, btot, ne, te)
    return alpha

def calc_alpha_from_3ion_ap_with_2ions_and_gradient_quasineutrality(ania, anib, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, q, ane, ate, btot, ne, te, lref=None):
    """Compute alpha for a three-ion system, inferring the third density gradient via gradient quasineutrality."""
    ne_temp = copy.deepcopy(ne) if lref is not None else None
    te_temp = copy.deepcopy(te) if lref is not None else None
    ap = calc_3ion_ap_with_2ions_and_gradient_quasineutrality(ania, anib, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, ane, ate, ne=ne_temp, te=te_temp, lref=lref)
    alpha = calc_alpha_from_ap(ap, q, btot, ne, te)
    return alpha

def calc_alpha_from_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(ania, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic, q, ane, ate, btot, ne, te, lref=None):
    """Compute alpha for a three-ion system using one known density gradient, Zeff gradient, and gradient quasineutrality."""
    ne_temp = copy.deepcopy(ne) if lref is not None else None
    te_temp = copy.deepcopy(te) if lref is not None else None
    ap = calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(ania, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic, ane, ate, ne=ne_temp, te=te_temp, lref=lref)
    alpha = calc_alpha_from_ap(ap, q, btot, ne, te)
    return alpha

def calc_alpha_from_3ion_grad_p_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False):
    """Compute alpha from physical pressure gradients for a three-ion system with all species known."""
    grad_p = calc_3ion_grad_p_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, grad_ne, grad_te, ne, te, lref, norm_inputs=norm_inputs)
    alpha = calc_alpha_from_grad_p(grad_p, q, btot, lref)
    return alpha

def calc_alpha_from_3ion_grad_p_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False):
    """Compute alpha from physical pressure gradients for a three-ion system, inferring the third via gradient quasineutrality."""
    grad_p = calc_3ion_grad_p_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, grad_ne, grad_te, ne, te, lref, norm_inputs=norm_inputs)
    alpha = calc_alpha_from_grad_p(grad_p, q, btot, lref)
    return alpha

def calc_alpha_from_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False):
    """Compute alpha from physical pressure gradients for a three-ion system using one density gradient, Zeff gradient, and gradient quasineutrality."""
    grad_p = calc_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, grad_ne, grad_te, ne, te, lref, norm_inputs=norm_inputs)
    alpha = calc_alpha_from_grad_p(grad_p, q, btot, lref)
    return alpha

def calc_coulomb_logarithm_nrl_from_ne_and_te(ne, te):
    """Compute the NRL Coulomb logarithm for electron-electron collisions from ne (m^-3) and te (eV)."""
    cl = 15.2 - 0.5 * np.log(ne * 1.0e-20) + np.log(te * 1.0e-3)
    if cl.ndim == 0:
        cl = float(cl)
    return cl

def calc_nustar_nrl_from_ne_and_te(zeff, q, rmin, rmaj, ne, te):
    """Compute electron collisionality nu* using the NRL Coulomb logarithm."""
    c = constants_si()
    cl = calc_coulomb_logarithm_nrl_from_ne_and_te(ne, te)
    nt = (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    kk = (1.0e4 / 1.09) * q * rmaj * ((rmin / rmaj) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    nustar = cl * zeff * nt * kk
    return nustar

def calc_lognustar_from_nustar(nustar):
    """Convert collisionality nu* to its base-10 logarithm."""
    lognustar = np.log10(nustar)
    return lognustar

def calc_nustar_from_lognustar(lognustar):
    """Convert log10(nu*) back to collisionality nu*."""
    nustar = np.power(10.0, lognustar)
    return nustar

def calc_ne_from_nustar_nrl(nustar, zeff, q, rmin, rmaj, te):
    """Invert the NRL nu* formula to solve for electron density (requires numerical root finding)."""
    c = constants_si()
    eom = c['e'] / c['me']
    tb = q * rmaj * ((rmin / rmaj) ** (-1.5)) / ((eom * te) ** 0.5)
    kk = (1.0e4 / 1.09) * zeff * ((te * 1.0e-3) ** (-1.5))
    nu = nustar / (tb * kk)
    te_arr = ensure_numpy(te)
    nu_arr = ensure_numpy(nu)
    data = {'te': te_arr.flatten() * 1.0e-3, 'knu': nu_arr.flatten()}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    func_ne20 = lambda row: root_scalar(
        lambda ne: calc_coulomb_logarithm_nrl_from_ne_and_te(ne * 1.0e20, row['te'] * 1.0e3) * ne - row['knu'],
        x0=0.01,
        x1=1.0,
        maxiter=100,
    )
    sol_ne20 = rootdata.apply(func_ne20, axis=1)
    retry = sol_ne20.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_ne20_v2 = lambda row: root_scalar(
            lambda ne: calc_coulomb_logarithm_nrl_from_ne_and_te(ne * 1.0e20, row['te'] * 1.0e3) * ne - row['knu'],
            x0=1.0,
            x1=0.1,
            maxiter=100,
        )
        sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
    warnings.resetwarnings()
    ne_sol = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    ne = ensure_type_match(ne_sol.reshape(te_arr.shape), te)
    #logger.debug(f'<{calc_ne_from_nustar_nrl.__name__}>: data')
    #logger.debug(pd.DataFrame(data={'nustar': nustar, 'te': te, 'ne': ne}))
    return ne

def calc_te_from_nustar_nrl(nustar, zeff, q, rmin, rmaj, ne):
    """Invert the NRL nu* formula to solve for electron temperature (requires numerical root finding)."""
    c = constants_si()
    moe = c['me'] / c['e']
    kk = (10.0 ** 0.5) * (1.0e2 / 1.09) * zeff * q * rmaj * ((rmin / rmaj) ** (-1.5)) * (moe ** 0.5) * (ne * 1.0e-20)
    nu = nustar / kk
    ne_arr = ensure_numpy(ne)
    nu_arr = ensure_numpy(nu)
    data = {'ne': ne_arr.flatten() * 1.0e-20, 'knu': nu_arr.flatten()}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    func_te3 = lambda row: root_scalar(
        lambda te: calc_coulomb_logarithm_nrl_from_ne_and_te(row['ne'] * 1.0e20, te * 1.0e3) / (te ** 2) - row['knu'],
        x0=1.0,
        x1=0.1,
        maxiter=100,
    )
    sol_te3 = rootdata.apply(func_te3, axis=1)
    retry = sol_te3.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_te3_v2 = lambda row: root_scalar(
            lambda te: calc_coulomb_logarithm_nrl_from_ne_and_te(row['ne'] * 1.0e20, te * 1.0e3) / (te ** 2) - row['knu'],
            x0=0.01,
            x1=0.1,
            maxiter=100,
        )
        sol_te3.loc[retry] = rootdata.loc[retry].apply(func_te3_v2, axis=1)
    warnings.resetwarnings()
    te_sol = sol_te3.apply(lambda sol: 1.0e3 * sol.root).to_numpy()
    te = ensure_type_match(te_sol.reshape(ne_arr.shape), ne)
    #logger.debug(f'<{calc_te_from_nustar_nrl.__name__}>: data')
    #logger.debug(pd.DataFrame(data={'nustar': nustar, 'ne': ne, 'te': te}))
    return te

def calc_zeff_from_nustar_nrl(nustar, q, rmin, rmaj, ne, te):
    """Solve for Zeff given nu* and known ne, te using the NRL collisionality formula."""
    c = constants_si()
    cl = calc_coulomb_logarithm_nrl_from_ne_and_te(ne, te)
    nt = (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    kk = (1.0e4 / 1.09) * q * rmaj * ((rmin / rmaj) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    zeff = nustar / (cl * nt * kk)
    return zeff

def calc_rhos_from_ts_and_btot(ts, btot, _as, _zs=1.0):
    """Compute gyroradius rho_s = sqrt(m_s * T_s) / (|z_s| * e * B) for species s."""
    c = constants_si()
    rhos = (ts ** 0.5) * (((_as * c['u'] / c['e']) ** 0.5) / _zs) / btot
    return rhos

def calc_btot_from_rhos(rhos, ts, _as, _zs=1.0):
    """Invert the gyroradius formula to solve for total magnetic field B from rho_s and T_s."""
    c = constants_si()
    btot = (ts ** 0.5) * (((_as * c['u'] / c['e']) ** 0.5) / _zs) / rhos
    return btot

def calc_ts_from_rhos(rhos, btot, _as, _zs=1.0):
    """Invert the gyroradius formula to solve for temperature T_s from rho_s and B."""
    c = constants_si()
    te = ((rhos * btot * _zs) ** 2) / (_as * c['u'] / c['e'])
    return te

def calc_rhoref_from_te_and_btot(te, btot, ai):
    """Compute the reference gyroradius rho_ref = sqrt(m_i * T_e) / (e * B) used for normalization."""
    c = constants_si()
    rhoref = calc_rhos_from_ts_and_btot(te, btot, ai, _zs=1.0)
    return rhoref

def calc_btot_from_rhoref(rhoref, te, ai):
    """Solve for total magnetic field from the reference gyroradius and electron temperature."""
    btot = calc_btot_from_rhos(rhoref, te, ai, _zs=1.0)
    return btot

def calc_te_from_rhoref(rhoref, btot, ai):
    """Solve for electron temperature from the reference gyroradius and total magnetic field."""
    te = calc_ts_from_rhos(rhoref, btot, ai, _zs=1.0)
    return te

def calc_rhostar_from_rhoref(rhoref, lref):
    """Compute the normalized gyroradius rho* = rho_ref / lref."""
    rhostar = normalize(rhoref, lref)
    return rhostar

def calc_rhoref_from_rhostar(rhostar, lref):
    """Recover the reference gyroradius from rho* and the reference length: rho_ref = rho* * lref."""
    rhoref = unnormalize(rhostar, lref)
    return rhoref

def calc_rhostar_from_te_and_btot(te, btot, ai, lref):
    """Compute rho* = rho_ref / lref from electron temperature and magnetic field."""
    rhoref = calc_rhoref_from_te_and_btot(te, btot, ai)
    rhostar = calc_rhostar_from_rhoref(rhoref, lref)
    return rhostar

def calc_btot_from_rhostar(rhostar, te, ai, lref):
    """Solve for total magnetic field from rho* and electron temperature."""
    rhoref = calc_rhoref_from_rhostar(rhostar, lref)
    btot = calc_btot_from_rhoref(rhoref, te, ai)
    return btot

def calc_te_from_rhostar(rhostar, btot, ai, lref):
    """Solve for electron temperature from rho* and total magnetic field."""
    rhos = calc_rhoref_from_rhostar(rhostar, lref)
    te = calc_te_from_rhoref(rhos, btot, ai)
    return te

def calc_ne_from_alpha_and_rhostar(alpha, rhostar, ap, ai, zi, q, lref):
    """Solve for electron density simultaneously satisfying both the alpha and rho* constraints."""
    c = constants_si()
    mi = ai * c['u']
    qi = zi * c['e']
    prefactor = mi / (2.0 * c['mu'] * (qi ** 2) * (q ** 2))
    ne = prefactor * alpha / ((rhostar ** 2) * (lref ** 2) * ap)
    return ne

def calc_btot_from_alpha_and_rhostar(alpha, rhostar, ap, ai, zi, q, ne, te, lref):
    """Solve for total magnetic field simultaneously satisfying both the alpha and rho* constraints."""
    c = constants_si()
    mi = ai * c['u']
    qi = zi * c['e']
    prefactor = 2.0 * c['mu'] * (mi ** 0.5) / qi
    btot = (prefactor * (q ** 2) * ne * ((c['e'] * te) ** 1.5) * ap / (alpha * rhostar * lref)) ** (1.0 / 3.0)
    return btot

def calc_ne_from_beta_and_rhostar(beta, rhostar, pnorm, ai, zi, lref):
    """Solve for electron density simultaneously satisfying both the beta and rho* constraints."""
    c = constants_si()
    mi = ai * c['u']
    qi = zi * c['e']
    prefactor = mi / (2.0 * c['mu'] * (qi ** 2))
    ne = prefactor * beta / ((rhostar ** 2) * (lref ** 2) * pnorm)
    return ne

def calc_btot_from_beta_and_rhostar(beta, rhostar, pnorm, ai, zi, ne, te, lref):
    """Solve for total magnetic field simultaneously satisfying both the beta and rho* constraints."""
    c = constants_si()
    mi = ai * c['u']
    qi = zi * c['e']
    prefactor = 2.0 * c['mu'] * (mi ** 0.5) / qi
    bref = (prefactor * ne * ((c['e'] * te) ** 1.5) * pnorm / (beta * rhostar * lref)) ** (1.0 / 3.0)
    return bref

def calc_ne_from_nustar_nrl_alpha_and_ap(nustar, alpha, ap, zeff, q, btot, rmin, rmaj):
    """Solve for electron density satisfying nu* (NRL), alpha, and a/Lp simultaneously."""
    c = constants_si()
    kalp = 1.0e23 * 2.0 * c['mu'] * c['e'] * (q ** 2) * ap / (alpha * (btot ** 2))
    knu = (1.0e4 / 1.09) * ((1.0e-3 * c['me'] / c['e']) ** 0.5) * zeff * q * rmaj * ((rmin / rmaj) ** (-1.5))
    log_arr = ensure_numpy(-np.log(kalp))
    cst_arr = ensure_numpy(nustar / (knu * (kalp ** 2)))
    data = {'logterm': log_arr.flatten(), 'constant': cst_arr.flatten()}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    func_ne20 = lambda row: root_scalar(
        lambda ne: (15.2 + row['logterm'] - 1.5 * np.log(ne)) * (ne ** 3) - row['constant'],
        x0=0.01,
        x1=1.0,
        maxiter=100,
    )
    sol_ne20 = rootdata.apply(func_ne20, axis=1)
    retry = sol_ne20.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_ne20_v2 = lambda row: root_scalar(
            lambda ne: (15.2 + row['logterm'] - 1.5 * np.log(ne)) * (ne ** 3) - row['constant'],
            x0=1.0,
            x1=0.1,
            maxiter=100,
        )
        sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
    warnings.resetwarnings()
    ne_sol = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    ne = ensure_type_match(ne_sol.reshape(cst_arr.shape), nustar)
    #logger.debug(f'<{calc_ne_from_nustar_nrl_alpha_and_ap.__name__}>: data')
    #logger.debug(pd.DataFrame(data={'nustar': nustar, 'alpha': alpha, 'ap': ap, 'ne': ne}))
    return ne

def calc_te_from_nustar_nrl_alpha_and_ap(nustar, alpha, ap, zeff, q, btot, rmin, rmaj):
    """Solve for electron temperature satisfying nu* (NRL), alpha, and a/Lp simultaneously."""
    c = constants_si()
    kalp = 1.0e23 * 2.0 * c['mu'] * c['e'] * (q ** 2) * ap / (alpha * (btot ** 2))
    knu = (1.0e4 / 1.09) * ((1.0e-3 * c['me'] / c['e']) ** 0.5) * zeff * q * rmaj * ((rmin / rmaj) ** (-1.5))
    log_arr = ensure_numpy(-np.log(kalp))
    cst_arr = ensure_numpy(nustar * kalp / knu)
    data = {'logterm': log_arr.flatten(), 'constant': cst_arr.flatten()}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    func_te3 = lambda row: root_scalar(
        lambda te: (15.2 - 0.5 * row['logterm'] + 1.5 * np.log(te)) / (te ** 3) - row['constant'],
        x0=1.0,
        x1=0.1,
        maxiter=100,
    )
    sol_te3 = rootdata.apply(func_te3, axis=1)
    retry = sol_te3.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_te3_v2 = lambda row: root_scalar(
            lambda te: (15.2 - 0.5 * row['logterm'] + 1.5 * np.log(te)) / (te ** 3) - row['constant'],
            x0=0.01,
            x1=0.1,
            maxiter=100,
        )
        sol_te3.loc[retry] = rootdata.loc[retry].apply(func_te3_v2, axis=1)
    warnings.resetwarnings()
    te_sol = sol_te3.apply(lambda sol: 1.0e3 * sol.root).to_numpy()
    te = ensure_type_match(te_sol.reshape(cst_arr.shape), nustar)
    #logger.debug(f'<{calc_te_from_nustar_nrl_alpha_and_ap.__name__}>: data')
    #logger.debug(pd.DataFrame(data={'nustar': nustar, 'alpha': alpha, 'ap': ap, 'te': te}))
    return te

def calc_vref_from_te_and_aref(te, aref):
    """Compute the reference thermal velocity v_ref = sqrt(T_e / m_ref) for normalization."""
    c = constants_si()
    vref = (te * c['e'] / (c['u'] * aref)) ** 0.5
    return vref

def calc_te_from_vref(vref, aref):
    """Recover electron temperature from the reference thermal velocity and reference mass."""
    c = constants_si()
    te = (vref ** 2) * c['u'] * aref / c['e']
    return te

def calc_aref_from_vref(vref, te):
    """Infer the reference mass number from the reference thermal velocity and electron temperature."""
    c = constants_si()
    aref = c['e'] * te / ((vref ** 2) * c['u'])
    return aref

def calc_vths_from_ts(ts, _as):
    """Compute species thermal velocity v_th,s = sqrt(T_s / m_s)."""
    c = constants_si()
    vths = (2.0 * ts * c['e'] / (c['u'] * _as)) ** 0.5
    return vths

def calc_ts_from_vths(vths, _as):
    """Recover species temperature from thermal velocity: T_s = m_s * v_th,s^2."""
    c = constants_si()
    ts = (vths ** 2) * c['u'] * _as / (2.0 * c['e'])
    return ts

def calc_mach_from_u(u, cref):
    """Compute the Mach number as the ratio of flow velocity to reference sound speed."""
    mach = normalize(u, cref)
    return mach

def calc_u_from_mach(mach, cref):
    """Recover flow velocity from Mach number and reference sound speed: u = Mach * c_ref."""
    u = unnormalize(mach, cref)
    return u

def calc_machpar_from_machtor_and_toroidal_assumption_circular(machtor, q, epsilon, x):
    """Convert toroidal Mach number to parallel Mach number under the circular-geometry toroidal assumption."""
    btorbyb = ((q ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    machpar = machtor * btorbyb
    return machpar

def calc_aupar_from_autor_and_toroidal_assumption_circular(autor, machtor, s, q, epsilon, x):
    """Convert the normalized toroidal rotation gradient to parallel rotation gradient under the circular-geometry toroidal assumption."""
    btorbyb = ((q ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    bpolbyb = (((epsilon * x) ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    grad_btorbyb = btorbyb * (bpolbyb ** 2) * (s - 1.0) / (epsilon * x)
    aupar = autor * btorbyb - machtor * grad_btorbyb
    return aupar

def calc_machper_from_machtor_and_toroidal_assumption_circular(machtor, q, epsilon, x):
    """Convert toroidal Mach number to perpendicular (E×B) Mach number under the circular-geometry toroidal assumption."""
    bpolbyb = (((epsilon * x) ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    machper = machtor * bpolbyb
    return machper

def calc_auper_from_autor_and_toroidal_assumption_circular(autor, machtor, s, q, epsilon, x):
    """Convert the normalized toroidal rotation gradient to perpendicular rotation gradient under the circular-geometry toroidal assumption."""
    btorbyb = ((q ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    bpolbyb = (((epsilon * x) ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    grad_btorbyb = btorbyb * (bpolbyb ** 2) * (s - 1.0) / (epsilon * x)
    auper = autor * btorbyb - machtor * grad_btorbyb
    return auper

def calc_gammae_from_aupar_without_grad_dpi(aupar, q, epsilon):
    """Compute the ExB shearing rate from the parallel rotation gradient, neglecting the diamagnetic pressure term."""
    gammae = -(epsilon / q) * aupar
    return gammae

def calc_grad_dpi_from_gammae_machtor_and_autor(gammae, machtor, autor, q, r, ro):
    """Compute the diamagnetic pressure gradient term from the ExB rate and toroidal rotation quantities."""
    # Also no CI test
    return None

def calc_bunit_from_bref(bref, sfac):
    """Compute the flux-surface unit field B_unit = B_ref * s_fac from the reference field and shape factor."""
    bunit = normalize(bref, sfac)
    return bunit

def calc_bref_from_bunit(bunit, sfac):
    """Recover the reference field B_ref from B_unit and the flux-surface shape factor."""
    bref = unnormalize(bunit, sfac)
    return bref

def calc_lds_from_ns_and_ts(ns, ts, zs=1.0):
    """Compute the Debye length lambda_D,s for species s from its density and temperature."""
    c = constants_si()
    lds = (c['eps'] * ts / (c['e'] * ns * (zs ** 2))) ** 0.5
    return lds

def calc_ldsnorm_from_lds(lds, rhoref):
    """Normalize the Debye length by the reference gyroradius: lambda_D* = lambda_D / rho_ref."""
    ldsnorm = normalize(lds, rhoref)
    return ldsnorm

def calc_ldenorm_from_ne_te_and_rhoref(ne, te, rhoref, ze=1.0):
    """Compute the normalized electron Debye length from ne, te, and rho_ref."""
    lde = calc_lds_from_ns_and_ts(ne, te, zs=ze)
    ldenorm = calc_ldsnorm_from_lds(lde, rhoref)
    return ldenorm

def calc_invb90_from_t_and_z(ta, za, zb=None):
    """Compute 1/b_90 (inverse 90-degree deflection parameter) used in the general Coulomb logarithm."""
    c = constants_si()
    zb_temp = copy.deepcopy(zb) if zb is not None else copy.deepcopy(za)
    invb90 = (4.0 * np.pi * c['eps'] / c['e']) * ta / (za * zb_temp)
    return invb90

def calc_coulomb_logarithm_from_n_and_t(na, ta, za, zb=None):
    """Compute the Coulomb logarithm as ln(b_90 * lambda_D) from density, temperature, and charge."""
    lda = calc_lds_from_ns_and_ts(na, ta, zs=za)
    invb90 = calc_invb90_from_t_and_z(ta, za, zb=zb)
    cl = np.log(invb90 * lda)
    if cl.ndim == 0:
        cl = float(cl)
    return cl

def calc_nu_from_n_and_t(na, nb, ta, za, zb, ma, cla=False):
    """Compute the collision frequency nu_{ab} between species a (at temperature ta) and species b."""
    c = constants_si()
    factor = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    invb90 = calc_invb90_from_t_and_z(ta, za, zb=zb)
    cl = calc_coulomb_logarithm_from_n_and_t(na, ta, za, zb=(zb if not cla else za))
    nu = factor * nb * (c['e'] * ta / (c['u'] * ma)) ** 0.5 / (invb90 ** 2) * cl
    return nu

def calc_nuei_from_ne_te_and_zeff(ne, te, zeff, zi, ze=1.0, cle=False):
    """Compute the electron-ion collision frequency from ne, te, Zeff, and ion charge."""
    c = constants_si()
    factor = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    invb90 = calc_invb90_from_t_and_z(te, ze, zb=((zeff * ze) ** 0.5))
    cl = calc_coulomb_logarithm_from_n_and_t(ne, te, ze, zb=(zi if not cle else ze))
    nuei = factor * ne * (c['e'] * te / (c['me'])) ** 0.5 / (invb90 ** 2) * cl
    return nuei

def calc_nuee_from_nuei_and_zeff(nuei, zeff):
    """Convert electron-ion collision frequency to electron-electron frequency: nu_ee = nu_ei / Zeff."""
    nuee = normalize(nuee, zeff)
    return nuee

def calc_nuei_from_nuee_and_zeff(nuee, zeff):
    """Convert electron-electron collision frequency to electron-ion frequency: nu_ei = nu_ee * Zeff."""
    nuei = unnormalize(nuee, zeff)
    return nuei

def calc_nunorm_from_nu(nu, gref):
    """Normalize a collision frequency by the gyro-Bohm reference frequency: nu* = nu / g_ref."""
    nunorm = normalize(nu, gref)
    return nunorm

def calc_nueenorm_from_ne_and_te(ne, te, gref, ze=1.0):
    """Compute the normalized electron-electron collision frequency from ne and te."""
    c = constants_si()
    me = c['me'] / c['u']
    nuee = calc_nu_from_n_and_t(ne, ne, te, ze, ze, me)
    nueenorm = calc_nunorm_from_nu(nuee, gref)
    return nueenorm

def calc_nueinorm_from_ne_ni_and_te(ne, ni, te, zi, gref, ze=1.0, cle=False):
    """Compute the normalized electron-ion collision frequency from ne, ni, and te."""
    c = constants_si()
    me = c['me'] / c['u']
    nuei = calc_nu_from_n_and_t(ne, ni, te, ze, zi, me, cla=cle)
    nueinorm = calc_nunorm_from_nu(nuei, gref)
    return nueinorm

def calc_nueinorm_from_ne_te_and_zeff(ne, te, zeff, zi, gref, ze=1.0, cle=False):
    """Compute the normalized electron-ion collision frequency using Zeff instead of individual ion densities."""
    nuei = calc_nuei_from_ne_te_and_zeff(ne, te, zeff, zi, ze, cle=cle)
    nueinorm = calc_nunorm_from_nu(nuei, gref)
    return nueinorm

def calc_nuiinorm_from_ni_and_ti(nia, nib, tia, zia, zib, mia, gref):
    """Compute the normalized ion-ion collision frequency between species a and species b."""
    nuii = calc_nu_from_n_and_t(nia, nib, tia, zia, zib, mia)
    nuiinorm = calc_nunorm_from_nu(nuii, gref)
    return nuiinorm

def calc_te_from_betae_and_ldenorm(betae, ldenorm, sfac, aref, ze=1.0):
    """Solve for electron temperature from electron beta and normalized Debye length."""
    c = constants_si()
    bc = (ze ** 2) * c['u'] * aref / (2.0 * c['eps'] * c['e'] * c['mu'])
    te = bc * betae * (ldenorm ** 2) * (sfac ** 2)
    return te

def calc_ne_from_nueenorm(nueenorm, te, aref, lref, ze=1.0):
    """Solve for electron density from the normalized electron-electron collision frequency and te (numerical root finding)."""
    c = constants_si()
    f_nu = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    invb90 = calc_invb90_from_t_and_z(te, ze, zb=ze)
    nc = f_nu * lref * (c['u'] * aref / c['me']) ** 0.5
    nl = invb90 * (c['eps'] / c['e']) ** 0.5 * (te / (ze ** 2)) ** 0.5
    log_arr = ensure_numpy(np.log(nl))
    cst_arr = ensure_numpy(nueenorm * (invb90 ** 2) / nc)
    data = {'logterm': log_arr.flatten(), 'constant': cst_arr.flatten()}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    func_ne20 = lambda row: root_scalar(
        lambda ne: (row['logterm'] - 0.5 * np.log(ne * 1.0e20)) * (ne * 1.0e20) - row['constant'],
        x0=0.01,
        x1=1.0,
        maxiter=100,
    )
    sol_ne20 = rootdata.apply(func_ne20, axis=1)
    retry = sol_ne20.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_ne20_v2 = lambda row: root_scalar(
            lambda ne: (row['logterm'] - 0.5 * np.log(ne * 1.0e20)) * (ne * 1.0e20) - row['constant'],
            x0=1.0,
            x1=0.1,
            maxiter=100,
        )
        sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
    warnings.resetwarnings()
    ne_sol = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    ne = ensure_type_match(ne_sol.reshape(cst_arr.shape), nueenorm)
    #logger.debug(f'<{calc_ne_from_nueenorm.__name__}>: data')
    #logger.debug(pd.DataFrame(data={'nueenorm': nueenorm, 'te': te, 'ne': ne}))
    return ne

def calc_ne_from_nueenorm_and_betae(nueenorm, betae, bref, aref, lref):
    """Solve for electron density simultaneously satisfying the normalized nu_ee and betae constraints."""
    c = constants_si()
    f_nu = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    nc = f_nu * lref * (c['u'] * aref / c['me']) ** 0.5
    bs = (4.0 * np.pi * c['eps']) * betae * bref ** 2 / (2.0 * c['mu'] * (c['e'] ** 2))
    nc = f_nu * lref * (c['u'] * aref / c['me']) ** 0.5
    nl = (bs ** 1.5) / (4.0 * np.pi) ** 0.5
    log_arr = ensure_numpy(np.log(nl))
    cst_arr = ensure_numpy(1.0e-60 * nueenorm * (bs ** 2) / nc)   # 1e20 density factor moved here for numerical stability
    data = {'logterm': log_arr.flatten(), 'constant': cst_arr.flatten()}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    func_ne20 = lambda row: root_scalar(
        lambda ne: (row['logterm'] - 2.0 * np.log(ne * 1.0e20)) * (ne ** 3) - row['constant'],
        x0=0.01,
        x1=1.0,
        maxiter=100,
    )
    sol_ne20 = rootdata.apply(func_ne20, axis=1)
    retry = sol_ne20.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_ne20_v2 = lambda row: root_scalar(
            lambda ne: (row['logterm'] - 2.0 * np.log(ne * 1.0e20)) * (ne ** 3) - row['constant'],
            x0=1.0,
            x1=0.1,
            maxiter=100,
        )
        sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
    warnings.resetwarnings()
    ne_sol = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    ne = ensure_type_match(ne_sol.reshape(cst_arr.shape), nueenorm)
    #logger.debug(f'<{calc_ne_from_nueenorm_and_betae.__name__}>: data')
    #logger.debug(pd.DataFrame(data={'nueenorm': nueenorm, 'betae': betae, 'ne': ne}))
    return ne

def calc_te_from_betae(betae, ne, bref):
    """Solve for electron temperature from electron beta, electron density, and reference field."""
    te = calc_te_from_beta_and_pnorm(betae, 1.0, bref, ne)
    return te

def calc_bunit_from_ldenorm(ldenorm, ne, aref, ze=1.0):
    """Compute the unit magnetic field B_unit from the normalized Debye length, ne, and reference mass."""
    c = constants_si()
    bunit = (ne * (ze ** 2) * c['u'] * aref / c['eps']) ** 0.5 * ldenorm
    return bunit

def calc_nustar_from_ne_and_te(zeff, q, rmin, rmaj, ne, te):
    """Compute electron collisionality nu* using the general Coulomb logarithm (not NRL approximation)."""
    c = constants_si()
    cl = calc_coulomb_logarithm_from_n_and_t(ne, te, 1.0)
    nt = (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    kk = (1.0e4 / 1.09) * q * rmaj * ((rmin / rmaj) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    nustar = cl * zeff * nt * kk
    return nustar

def calc_ne_from_nustar(nustar, zeff, q, rmin, rmaj, te):
    """Invert the general nu* formula to solve for electron density (numerical root finding)."""
    c = constants_si()
    eom = c['e'] / c['me']
    tb = q * rmaj * ((rmin / rmaj) ** (-1.5)) / ((eom * te) ** 0.5)
    kk = (1.0e4 / 1.09) * zeff * ((te * 1.0e-3) ** (-1.5))
    nu = nustar / (tb * kk)
    te_arr = ensure_numpy(te)
    nu_arr = ensure_numpy(nu)
    data = {'te': te_arr.flatten() * 1.0e-3, 'knu': nu_arr.flatten()}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    func_ne20 = lambda row: root_scalar(
        lambda ne: calc_coulomb_logarithm_from_n_and_t(ne * 1.0e20, row['te'] * 1.0e3, 1.0) * ne - row['knu'],
        x0=0.01,
        x1=1.0,
        maxiter=100,
    )
    sol_ne20 = rootdata.apply(func_ne20, axis=1)
    retry = sol_ne20.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_ne20_v2 = lambda row: root_scalar(
            lambda ne: calc_coulomb_logarithm_from_n_and_t(ne * 1.0e20, row['te'] * 1.0e3, 1.0) * ne - row['knu'],
            x0=1.0,
            x1=0.1,
            maxiter=100,
        )
        sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
    warnings.resetwarnings()
    ne_sol = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    ne = ensure_type_match(ne_sol.reshape(te_arr.shape), nustar)
    #logger.debug(f'<{calc_ne_from_nustar.__name__}>: data')
    #logger.debug(pd.DataFrame(data={'nustar': nustar, 'te': te, 'ne': ne}))
    return ne

def calc_te_from_nustar(nustar, zeff, q, rmin, rmaj, ne):
    """Invert the general nu* formula to solve for electron temperature (numerical root finding)."""
    c = constants_si()
    moe = c['me'] / c['e']
    kk = (10.0 ** 0.5) * (1.0e2 / 1.09) * zeff * q * rmaj * ((rmin / rmaj) ** (-1.5)) * (moe ** 0.5) * (ne * 1.0e-20)
    nu = nustar / kk
    ne_arr = ensure_numpy(ne)
    nu_arr = ensure_numpy(nu)
    data = {'ne': ne_arr.flatten() * 1.0e-20, 'knu': nu_arr.flatten()}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    func_te3 = lambda row: root_scalar(
        lambda te: calc_coulomb_logarithm_from_n_and_t(row['ne'] * 1.0e20, te * 1.0e3, 1.0) / (te ** 2) - row['knu'],
        x0=1.0,
        x1=0.1,
        maxiter=100,
    )
    sol_te3 = rootdata.apply(func_te3, axis=1)
    retry = sol_te3.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_te3_v2 = lambda row: root_scalar(
            lambda te: calc_coulomb_logarithm_from_n_and_t(row['ne'] * 1.0e20, te * 1.0e3, 1.0) / (te ** 2) - row['knu'],
            x0=0.01,
            x1=0.1,
            maxiter=100,
        )
        sol_te3.loc[retry] = rootdata.loc[retry].apply(func_te3_v2, axis=1)
    warnings.resetwarnings()
    te_sol = sol_te3.apply(lambda sol: 1.0e3 * sol.root).to_numpy()
    te = ensure_type_match(te_sol.reshape(ne_arr.shape), nustar)
    #logger.debug(f'<{calc_te_from_nustar.__name__}>: data')
    #logger.debug(pd.DataFrame(data={'nustar': nustar, 'ne': ne, 'te': te}))
    return te

def calc_zeff_from_nustar(nustar, q, rmin, rmaj, ne, te):
    """Solve for Zeff from the general nu* formula given ne and te."""
    c = constants_si()
    cl = calc_coulomb_logarithm_from_n_and_t(ne, te, 1.0)
    nt = (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    kk = (1.0e4 / 1.09) * q * rmaj * ((rmin / rmaj) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    zeff = nustar / (cl * nt * kk)
    return zeff

def calc_flux_surface_values_from_mxh(rmin, rgeo, zgeo, kappa, drgeo, dzgeo, s_kappa, cos, sin, s_cos, s_sin):
    """Evaluate (R, Z, arc length element, |grad r|) on a flux surface described by MXH coefficients."""
    n_theta = 1001
    theta = np.linspace(-np.pi, np.pi, n_theta)
    if not isinstance(kappa, float):
        for d in range(kappa.ndim):
            theta = np.expand_dims(theta, axis=-1)
    a = copy.deepcopy(theta)
    a_t = np.ones_like(a)
    a_tt = np.zeros_like(a)
    a_r = np.zeros_like(a)
    for i in range(len(cos)):
        if i < len(sin):
            s = np.array([sin[i]]) if isinstance(sin[i], float) else sin[i]
            a += np.expand_dims(s, axis=0) * np.sin(float(i) * theta)
            a_t += np.expand_dims(s, axis=0) * float(i) * np.cos(float(i) * theta)
            a_tt += np.expand_dims(s, axis=0) * float(-i * i) * np.sin(float(i) * theta)
        if i < len(s_sin):
            s_s = np.array([s_sin[i]]) if isinstance(sin[i], float) else s_sin[i]
            a_r += np.expand_dims(s_s, axis=0) * np.sin(float(i) * theta)
        if i < len(cos):
            s = np.array([cos[i]]) if isinstance(cos[i], float) else cos[i]
            a += np.expand_dims(s, axis=0) * np.cos(float(i) * theta)
            a_t += np.expand_dims(s, axis=0) * float(-i) * np.sin(float(i) * theta)
            a_tt += np.expand_dims(s, axis=0) * float(-i * i) * np.cos(float(i) * theta)
        if i < len(s_cos):
            s_s = np.array([s_cos[i]]) if isinstance(s_cos[i], float) else s_cos[i]
            a_r += np.expand_dims(s_s, axis=0) * np.cos(float(i) * theta)
    r = np.expand_dims(rgeo, axis=0) + np.expand_dims(rmin, axis=0) * np.cos(a)
    r_t = np.expand_dims(-rmin, axis=0) * a_t * np.sin(a)
    #r_tt = np.expand_dims(-rmin.to_numpy(), axis=0) * (a_t**2 * np.cos(a) + a_tt * np.sin(a))
    r_r = np.expand_dims(drgeo, axis=0) + np.cos(a) - np.expand_dims(rmin, axis=0) * np.sin(a) * a_r
    z = np.expand_dims(zgeo, axis=0) + np.expand_dims(kappa * rmin, axis=0) * np.sin(theta)
    z_t = np.expand_dims(kappa * rmin, axis=0) * np.cos(theta)
    #z_tt = np.expand_dims(-kappa * rmin, axis=0) * np.sin(theta)
    z_r = np.expand_dims(dzgeo, axis=0) + np.expand_dims(kappa * (1.0 + s_kappa), axis=0) * np.sin(theta)
    l_t = (r_t ** 2 + z_t ** 2) ** 0.5
    j_r = r * (r_r * z_t - r_t * z_r)
    inv_j_r = 1.0 / np.where(np.isclose(j_r, 0.0), 0.001, j_r)
    grad_r = np.where(np.isclose(j_r, 0.0), 1.0, r * l_t * inv_j_r)
    return r, z, l_t, grad_r

def calc_vol_from_contour(r, z, rgeo):
    """Compute the enclosed plasma volume from a closed flux-surface contour (R, Z)."""
    xs = trapezoid(r, x=z, axis=0)
    vol = 2.0 * np.pi * rgeo * xs
    return vol, xs

def calc_b_from_flux_surface_values(r, grad_r, l_t, rmin, q):
    """Compute poloidal and total field magnitudes on a flux surface from MXH geometric quantities and q."""
    r_temp = copy.deepcopy(r)
    grad_r_temp = copy.deepcopy(grad_r)
    l_t_temp = copy.deepcopy(l_t)
    n_theta = r.shape[0]
    if np.all(r[0] == r[-1]) and np.all(grad_r[0] == grad_r[-1]) and np.all(l_t[0] == l_t[-1]):
        r_temp = r_temp[:-1]
        grad_r_temp = grad_r_temp[:-1]
        l_t_temp = l_t_temp[:-1]
        n_theta = n_theta - 1
    c = 2.0 * np.pi * np.sum(l_t_temp / (r_temp * grad_r_temp), axis=0)
    f = 2.0 * np.pi * rmin / (np.where(np.isclose(c, 0.0), 1.0, c) / float(n_theta))
    #f[..., 0] = 2.0 * f[..., 1] - f[..., 2]
    bt = np.expand_dims(f, axis=0) / r
    bp = np.expand_dims(rmin / q, axis=0) * grad_r / r
    b = bt ** 2 + bp ** 2
    return bt, bp, b

def calc_geo_from_flux_surface_values(r, grad_r, l_t, b, rmin, rgeo):
    """Compute the flux-surface metric factor g_t from field and MXH shape values."""
    r_v = np.expand_dims(rmin * rgeo, axis=0)
    g_t = r * b * l_t / (np.where(np.isclose(r_v, 0.0), 1.0, r_v) * grad_r)
    #g_t[..., 0] = 2.0 * g_t[..., 1] - g_t[..., 2]
    return g_t

def calc_grad_vol_from_flux_surface_values(r, l_t, grad_r):
    """Compute dV/dr and surface area by integrating over a discretized flux-surface contour."""
    r_temp = copy.deepcopy(r)
    grad_r_temp = copy.deepcopy(grad_r)
    l_t_temp = copy.deepcopy(l_t)
    n_theta = r.shape[0]
    if np.all(r[0] == r[-1]) and np.all(grad_r[0] == grad_r[-1]) and np.all(l_t[0] == l_t[-1]):
        r_temp = r_temp[:-1]
        grad_r_temp = grad_r_temp[:-1]
        l_t_temp = l_t_temp[:-1]
        n_theta = n_theta - 1
    c = 2.0 * np.pi * np.sum(l_t_temp / (r_temp * grad_r_temp), axis=0)
    sa = 2.0 * np.pi * np.sum(l_t_temp * r_temp, axis=0) * 2.0 * np.pi / float(n_theta)
    grad_vol = 2.0 * np.pi * np.where(np.isfinite(c), c, 0.0) / float(n_theta)
    return grad_vol, sa

def calc_flux_surface_average_k_from_b_and_geo(k, b, g_t):
    """Compute the flux-surface average <k> = <k/B * g_t> / <g_t/B> for a poloidal function k."""
    k_temp = copy.deepcopy(k)
    b_temp = copy.deepcopy(b)
    g_t_temp = copy.deepcopy(g_t)
    if np.all(k[0] == k[-1]) and np.all(b[0] == b[-1]) and np.all(g_t[0] == g_t[-1]):
        k_temp = k_temp[:-1]
        b_temp = b_temp[:-1]
        g_t_temp = g_t_temp[:-1]
    denom = np.sum(np.where(np.isfinite(g_t_temp), g_t_temp, 0.0) / b_temp, axis=0)
    #denom[..., 0] = 2.0 * denom[..., 1] - denom[..., 2]
    kfsa = np.sum(k_temp * g_t_temp / b_temp, axis=0) / denom
    return kfsa
