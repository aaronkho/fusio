import copy
import warnings
import logging
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import root_scalar

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
    if isinstance(val, number_types):
        return np.atleast_1d([val])
    elif isinstance(val, array_types):
        return np.atleast_1d(val)
    elif isinstance(val, class_types):
        return val.to_numpy()
    else:
        return val


def ensure_type_match(val, other):
    if isinstance(other, number_types) and val.size == 1:
        return other.__class__(val.item(0))
    elif isinstance(other, xarray_types):
        return other.__class__(coords=other.coords, data=val)
    else:
        return other.__class__(val)


e_si = 1.60218e-19       # C
u_si = 1.66054e-27       # kg
mu_si = 4.0e-7 * np.pi   # H/m
eps_si = 8.85419e-12     # F/m
me_si = 5.4858e-4 * u_si
mp_si = (1.0 + 7.2764e-3) * u_si


def constants_si():
    return {
        'e': e_si,
        'u': u_si,
        'mu': mu_si,
        'eps': eps_si,
        'me': me_si,
        'mp': mp_si,
    }


def define_ion_species(z=None, a=None, short_name=None, long_name=None, user_mass=False):

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
        tz = int(np.rint(z))
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
    val = norm * ref
    return val

def normalize(val, ref):
    norm = val / ref
    return norm

def calc_a_from_epsilon(epsilon, lref):
    a = unnormalize(epsilon, lref)
    return a

def calc_epsilon_from_a(a, lref):
    epsilon = normalize(a, lref)
    return epsilon

def calc_r_from_x(x, a):
    r = unnormalize(x, a)
    return r

def calc_x_from_r(r, a):
    x = normalize(r, a)
    return x

def calc_ni_from_ninorm(ninorm, ne, nscale=1.0e19):
    nref = nscale * ne
    ni = unnormalize(ninorm, nref)
    return ni

def calc_ti_from_tinorm(tinorm, te, tscale=1.0e3):
    tref = tscale * te
    ti = unnormalize(tinorm, tref)
    return ti

def calc_ninorm_from_ni(ni, ne):
    ninorm = normalize(ni, ne)
    return ninorm

def calc_tinorm_from_ti(ti, te):
    tinorm = normalize(ti, te)
    return tinorm

def calc_ak_from_grad_k(grad_k, k, lref):
    ak = -lref * grad_k / k
    return ak

def calc_grad_k_from_ak(ak, k, lref):
    grad_k = ak * k / -lref
    return grad_k

def calc_q_circular_from_bp(bp, r, bo, ro):
    q = r * bo / (ro * bp)
    return q

def calc_bp_from_q_circular(q, r, bo, ro):
    bp = r * bo / (ro * q)
    return bp

def calc_grad_q_circular_from_s(s, bp, bo, ro):
    grad_q = s * bo / (ro * bp)
    return grad_q

def calc_grad_q_from_s_and_q(s, q, r):
    grad_q = calc_grad_k_from_ak(s, q, -r)
    return grad_q

def calc_s_circular_from_grad_bp(grad_bp, bp, r):
    s = 1.0 - r * grad_bp / bp
    return s

def calc_grad_bp_from_grad_q_circular(grad_q, bp, r, bo, ro):
    grad_bp = (1.0 - ro * bp * grad_q / bo) * bp / r
    return grad_bp

def calc_grad_bp_from_s_circular(s, bp, r):
    grad_bp = (1.0 - s) * bp / r
    return grad_bp

def calc_ninorm_from_zeff_and_quasineutrality(zeff, zia, zib, zi_target, ninorma, ze=1.0):
    zze = (zib - zeff) * ze
    zza = (zib - zia) * zia
    zz_target = (zib - zi_target) * zi_target
    ninorm_target = (zze - ninorma * zza) / zz_target
    return ninorm_target

# def calc_ninorm_from_quasineutrality(zia, zib, zi_target, ninorma, ninormb, ze=1.0):
#    ninorm_target = (ze - ninorma * zia - ninormb * zib) / zi_target
#    return ninorm_target

def calc_ninorm_from_quasineutrality(zi, zi_target, ninorm, ze=1.0):
    ninorm_target = (ze - ninorm * zi) / zi_target
    return ninorm_target

def calc_2ion_ninorm_from_ninorm_and_quasineutrality(ni, zia, zib, ne=None):
    ninorma = calc_ninorm_from_ni(ni, ne) if ne is not None else copy.deepcopy(ni)
    ninormb = calc_ninorm_from_quasineutrality(zia, zib, ninorma)
    return ninorma, ninormb

def calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ni, zeff, zia, zib, zic, ne=None):
    ninorma = calc_ninorm_from_ni(ni, ne) if ne is not None else copy.deepcopy(ni)
    ninormb = calc_ninorm_from_zeff_and_quasineutrality(zeff, zia, zic, zib, ninorma)
    ninormc = calc_ninorm_from_quasineutrality(1.0, zic, ninorma * zia + ninormb * zib)
    return ninorma, ninormb, ninormc

def calc_3ion_ninorm_from_ninorm_and_quasineutrality(nia, nib, zia, zib, zic, ne=None):
    ninorma = calc_ninorm_from_ni(nia, ne) if ne is not None else copy.deepcopy(nia)
    ninormb = calc_ninorm_from_ni(nib, ne) if ne is not None else copy.deepcopy(nib)
    ninormc = calc_ninorm_from_quasineutrality(1.0, zic, ninorma * zia + ninormb * zib)
    return ninorma, ninormb, ninormc

def calc_ni_from_zeff_and_quasineutrality(zeff, zia, zib, zi_target, nia, ne):
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
    ninorm = calc_ninorm_from_ni(ni, ne)
    ninorm_target = calc_ninorm_from_quasineutrality(zi, zi_target, ninorm)
    ni_target = calc_ni_from_ninorm(ninorm_target, ne, nscale=1.0)
    return ni_target

def calc_2ion_ni_from_ni_and_quasineutrality(ni, zia, zib, ne, norm_inputs=False):
    nia = calc_ni_from_ninorm(ni, ne, nscale=1.0) if norm_inputs else copy.deepcopy(ni)
    nib = calc_ni_from_quasineutrality(zia, zib, nia, ne)
    return nia, nib

def calc_3ion_ni_from_ni_zeff_and_quasineutrality(ni, zeff, zia, zib, zic, ne, norm_inputs=False):
    nia = calc_ni_from_ninorm(ni, ne, nscale=1.0) if norm_inputs else copy.deepcopy(ni)
    nib = calc_ni_from_zeff_and_quasineutrality(zeff, zia, zic, zib, nia, ne)
    nic = calc_ni_from_quasineutrality(1.0, zic, nia * zia + nib * zib, ne)
    return nia, nib, nic

def calc_3ion_ni_from_ni_and_quasineutrality(nia, nib, zia, zib, zic, ne, norm_inputs=False):
    nia = calc_ni_from_ninorm(nia, ne, nscale=1.0) if norm_inputs else copy.deepcopy(nia)
    nib = calc_ni_from_ninorm(nib, ne, nscale=1.0) if norm_inputs else copy.deepcopy(nib)
    nic = calc_ni_from_quasineutrality(1.0, zic, nia * zia + nib * zib, ne)
    return nia, nib, nic

def calc_ani_from_azeff_and_gradient_quasineutrality(azeff, zeff, zia, zib, zi_target, ninorma, ninorm_target, ane, ania, ze=1.0):
    zze = (zeff - zib) * ze
    zza = (zib - zia) * zia
    zz_target = (zi_target - zib) * zi_target
    ani_target = (azeff * zeff * ze + ane * zze + ania * ninorma * zza) / (ninorm_target * zz_target)
    return ani_target

# def calc_ani_from_gradient_quasineutrality(zia, zib, zi_target, ninorma, ninormb, ninorm_target, ane, ania, anib, ze=1.0):
#     ani_target = (ane * ze - ninorma * ania * zia - ninormb * anib * zib) / (ninorm_target * zi_target)
#     return ani_target

def calc_ani_from_gradient_quasineutrality(zi, zi_target, ninorm, ninorm_target, ane, ani, ze=1.0):
    ani_target = (ane * ze - ninorm * ani * zi) / (ninorm_target * zi_target)
    return ani_target

def calc_2ion_ani_from_ani_and_gradient_quasineutrality(ani, zia, zib, ane, ninorm, ne=1.0, lref=None):
    nenorm = copy.deepcopy(ne) if lref is not None else 1.0
    ane = calc_ak_from_grad_k(ane, nenorm, lref) if lref is not None else copy.deepcopy(ane)
    ania = calc_ak_from_grad_k(ani, ninorm, lref) if lref is not None else copy.deepcopy(ani)
    ninorma, ninormb = calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorm, zia, zib, ne=nenorm)
    anib = calc_ani_from_gradient_quasineutrality(zia, zib, ninorma, ninormb, ane, ania)
    return ania, anib

def calc_3ion_ani_from_ani_azeff_and_gradient_quasineutrality(ani, azeff, zeff, zia, zib, zic, ane, ninorm, ne=1.0, lref=None):
    nenorm = copy.deepcopy(ne) if lref is not None else 1.0
    ane = calc_ak_from_grad_k(ane, nenorm, lref) if lref is not None else copy.deepcopy(ane)
    ania = calc_ak_from_grad_k(ani, ninorm, lref) if lref is not None else copy.deepcopy(ani)
    azeff_temp = calc_ak_from_grad_k(azeff, zeff, lref) if lref is not None else copy.deepcopy(azeff)
    ninorma, ninormb, ninormc = calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ninorm, zeff, zia, zib, zic, ne=nenorm)
    anib = calc_ani_from_azeff_and_gradient_quasineutrality(azeff_temp, zeff, zia, zic, zib, ninorma, ninormb, ane, ania)
    anic = calc_ani_from_gradient_quasineutrality(1.0, zic, 1.0, ninormc, ane, ninorma * ania * zia + ninormb * anib * zib)
    return ania, anib, anic

def calc_3ion_ani_from_ani_and_gradient_quasineutrality(ania, anib, zia, zib, zic, ane, ninorma, ninormb, ne=1.0, lref=None):
    nenorm = copy.deepcopy(ne) if lref is not None else 1.0
    ane = calc_ak_from_grad_k(ane, nenorm, lref) if lref is not None else copy.deepcopy(ane)
    ania_temp = calc_ak_from_grad_k(ania, ninorma, lref) if lref is not None else copy.deepcopy(ania)
    anib_temp = calc_ak_from_grad_k(anib, ninormb, lref) if lref is not None else copy.deepcopy(anib)
    ninorma_temp, ninormb_temp, ninormc = calc_3ion_ninorm_from_ninorm_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=nenorm)
    anic = calc_ani_from_gradient_quasineutrality(1.0, zic, 1.0, ninormc, ane, ninorma_temp * ania_temp * zia + ninormb_temp * anib_temp * zib)
    return ania_temp, anib_temp, anic

def calc_grad_ni_from_grad_zeff_and_gradient_quasineutrality(grad_zeff, zeff, zia, zib, zi_target, nia, ni_target, grad_nia, ne, grad_ne, lref):
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
    ninorm = calc_ninorm_from_ni(ni, ne)
    ninorm_target = calc_ninorm_from_ni(ni_target, ne)
    ane = calc_ak_from_grad_k(grad_ne, ne, lref)
    ani = calc_ak_from_grad_k(grad_ni, ni, lref)
    ani_target = calc_ani_from_gradient_quasineutrality(zi, zi_target, ninorm, ninorm_target, ane, ani)
    grad_ni_target = calc_grad_k_from_ak(ani_target, ni_target, lref)
    return grad_ni_target

def calc_2ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(grad_ni, zia, zib, ni, grad_ne, ne, lref, norm_inputs=False):
    nia, nib = calc_2ion_ni_from_ni_and_quasineutrality(ni, zia, zib, ne, norm_inputs=norm_inputs)
    grad_ne_temp = calc_grad_k_from_ak(grad_ne, ne, lref) if norm_inputs else copy.deepcopy(grad_ne)
    grad_nia = calc_grad_k_from_ak(grad_ni, nia, lref) if norm_inputs else copy.deepcopy(grad_ni)
    grad_nib = calc_grad_ni_from_gradient_quasineutrality(zia, zib, nia, nib, grad_nia, ne, grad_ne_temp, lref)
    return grad_nia, grad_nib

def calc_3ion_grad_ni_from_grad_ni_grad_zeff_and_gradient_quasineutrality(grad_ni, grad_zeff, zeff, zia, zib, zic, ni, grad_ne, ne, lref, norm_inputs=False):
    nia, nib, nic = calc_3ion_ni_from_ni_zeff_and_quasineutrality(ni, zeff, zia, zib, zic, ne, norm_inputs=norm_inputs)
    grad_ne_temp = calc_grad_k_from_ak(grad_ne, ne, lref) if norm_inputs else copy.deepcopy(grad_ne)
    grad_nia = calc_grad_k_from_ak(grad_ni, nia, lref) if norm_inputs else copy.deepcopy(grad_ni)
    grad_zeff_temp = calc_grad_k_from_ak(grad_zeff, zeff, lref) if norm_inputs else copy.deepcopy(grad_zeff)
    grad_nib = calc_grad_ni_from_grad_zeff_and_gradient_quasineutrality(grad_zeff_temp, zeff, zia, zic, zib, nia, nib, grad_nia, ne, grad_ne_temp, lref)
    grad_nic = calc_grad_ni_from_gradient_quasineutrality(1.0, zic, ne, nic, grad_nia * zia + grad_nib * zib, ne, grad_ne_temp, lref)
    return grad_nia, grad_nib, grad_nic

def calc_3ion_grad_ni_from_grad_ni_and_gradient_quasineutrality(grad_nia, grad_nib, zia, zib, zic, nia, nib, grad_ne, ne, lref, norm_inputs=False):
    nia_temp, nib_temp, nic = calc_3ion_ni_from_ni_and_quasineutrality(nia, nib, zia, zib, zic, ne, norm_inputs=norm_inputs)
    grad_ne_temp = calc_grad_k_from_ak(grad_ne, ne, lref) if norm_inputs else copy.deepcopy(grad_ne)
    grad_nia_temp = calc_grad_k_from_ak(grad_nia, nia_temp, lref) if norm_inputs else copy.deepcopy(grad_nia)
    grad_nib_temp = calc_grad_k_from_ak(grad_nib, nib_temp, lref) if norm_inputs else copy.deepcopy(grad_nib)
    grad_nic = calc_grad_ni_from_gradient_quasineutrality(1.0, zic, ne, nic, grad_nia_temp * zia + grad_nib_temp * zib, ne, grad_ne_temp, lref)
    return grad_nia_temp, grad_nib_temp, grad_nic

def calc_p_from_pnorm(pnorm, ne, te):
    c = constants_si()
    p = c['e'] * ne * te * pnorm
    return p

def calc_pnorm_from_p(p, ne, te):
    c = constants_si()
    pnorm = p / (c['e'] * ne * te)
    return pnorm

def calc_2ion_pnorm_with_2ions(ninorma, ninormb, tinorma, tinormb, ne=None, te=None):
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    tinorma_temp = calc_tinorm_from_ti(tinorma, te) if te is not None else copy.deepcopy(tinorma)
    tinormb_temp = calc_tinorm_from_ti(tinormb, te) if te is not None else copy.deepcopy(tinormb)
    pnorm = 1.0 + ninorma_temp * tinorma_temp + ninormb_temp * tinormb_temp
    return pnorm

def calc_2ion_pnorm_with_1ion_and_quasineutrality(ninorma, tinorma, tinormb, zia, zib, ne=None, te=None):
    ninorma_temp, ninormb = calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorma, zia, zib, ne=ne)
    pnorm = calc_2ion_pnorm_with_2ions(ninorma_temp, ninormb, tinorma, tinormb, ne=None, te=te)
    return pnorm

def calc_3ion_pnorm_with_3ions(ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, ne=None, te=None):
    ninormc_temp = calc_ninorm_from_ni(ninormc, ne) if ne is not None else copy.deepcopy(ninormc)
    tinormc_temp = calc_tinorm_from_ti(tinormc, te) if te is not None else copy.deepcopy(tinormc)
    pnorm_2ion = calc_2ion_pnorm_with_2ions(ninorma, ninormb, tinorma, tinormb, ne=ne, te=te)
    pnorm = pnorm_2ion + ninormc_temp * tinormc_temp
    return pnorm

def calc_3ion_pnorm_with_2ions_and_quasineutrality(ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, ne=None, te=None):
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    ninormc = calc_ninorm_from_quasineutrality(1.0, zic, ninorma_temp * zia + ninormb_temp * zib)
    pnorm = calc_3ion_pnorm_with_3ions(ninorma_temp, ninormb_temp, ninormc, tinorma, tinormb, tinormc, ne=None, te=te)
    return pnorm

def calc_3ion_pnorm_with_1ion_zeff_and_quasineutrality(ninorma, tinorma, tinormb, tinormc, zeff, zia, zib, zic, ne=None, te=None):
    ninorma_temp, ninormb, ninormc = calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ninorma, zeff, zia, zib, zic, ne=ne)
    pnorm = calc_3ion_pnorm_with_3ions(ninorma_temp, ninormb, ninormc, tinorma, tinormb, tinormc, ne=None, te=te)
    return pnorm

def calc_2ion_p_with_2ions(nia, nib, tia, tib, ne, te, norm_inputs=False):
    ninorma = calc_ninorm_from_ni(nia, ne) if not norm_inputs else copy.deepcopy(nia)
    ninormb = calc_ninorm_from_ni(nib, ne) if not norm_inputs else copy.deepcopy(nib)
    tinorma = calc_tinorm_from_ti(tia, te) if not norm_inputs else copy.deepcopy(tia)
    tinormb = calc_tinorm_from_ti(tib, te) if not norm_inputs else copy.deepcopy(tib)
    pnorm = calc_2ion_pnorm_with_2ions(ninorma, ninormb, tinorma, tinormb)
    p = calc_p_from_pnorm(pnorm, ne, te)
    return p

def calc_2ion_p_with_1ion_and_quasineutrality(nia, tia, tib, zia, zib, ne, te, norm_inputs=False):
    nia_temp, nib = calc_2ion_ni_from_ni_and_quasineutrality(nia, zia, zib, ne, norm_inputs=norm_inputs)
    tia_temp = calc_ti_from_tinorm(tia, te, tscale=1.0) if norm_inputs else copy.deepcopy(tia)
    tib_temp = calc_ti_from_tinorm(tib, te, tscale=1.0) if norm_inputs else copy.deepcopy(tib)
    p = calc_2ion_p_with_2ions(nia_temp, nib, tia_temp, tib_temp, ne, te, norm_inputs=False)
    return p

def calc_3ion_p_with_3ions(nia, nib, nic, tia, tib, tic, ne, te, norm_inputs=False):
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    pnorm = calc_3ion_pnorm_with_3ions(nia, nib, nic, tia, tib, tic, ne=ne_temp, te=te_temp)
    p = calc_p_from_pnorm(pnorm, ne, te)
    return p

def calc_3ion_p_with_2ions_and_quasineutrality(nia, nib, tia, tib, tic, zia, zib, zic, ne, te, norm_inputs=False):
    nia_temp, nib_temp, nic = calc_3ion_ni_from_ni_and_quasineutrality(nia, nib, zia, zib, zic, ne, norm_inputs=norm_inputs)
    tia_temp = calc_ti_from_tinorm(tia, te, tscale=1.0) if norm_inputs else copy.deepcopy(tia)
    tib_temp = calc_ti_from_tinorm(tib, te, tscale=1.0) if norm_inputs else copy.deepcopy(tib)
    tic_temp = calc_ti_from_tinorm(tic, te, tscale=1.0) if norm_inputs else copy.deepcopy(tic)
    p = calc_3ion_p_with_3ions(nia_temp, nib_temp, nic, tia_temp, tib_temp, tic_temp, ne, te, norm_inputs=False)
    return p

def calc_3ion_p_with_1ion_zeff_and_quasineutrality(nia, tia, tib, tic, zeff, zia, zib, zic, ne, te, norm_inputs=False):
    nia_temp, nib, nic = calc_3ion_ni_from_ni_zeff_and_quasineutrality(nia, zeff, zia, zib, zic, ne, norm_inputs=norm_inputs)
    tia_temp = calc_ti_from_tinorm(tia, te, tscale=1.0) if norm_inputs else copy.deepcopy(tia)
    tib_temp = calc_ti_from_tinorm(tib, te, tscale=1.0) if norm_inputs else copy.deepcopy(tib)
    tic_temp = calc_ti_from_tinorm(tic, te, tscale=1.0) if norm_inputs else copy.deepcopy(tic)
    p = calc_3ion_p_with_3ions(nia_temp, nib, nic, tia_temp, tib_temp, tic_temp, ne, te, norm_inputs=False)
    return p

def calc_grad_p_from_ap(ap, ne, te, lref):
    c = constants_si()
    grad_p = calc_grad_k_from_ak(ap, c['e'] * ne * te, lref)
    return grad_p

def calc_ap_from_grad_p(grad_p, ne, te, lref):
    c = constants_si()
    ap = calc_ak_from_grad_k(grad_p, c['e'] * ne * te, lref)
    return ap

def calc_2ion_ap_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, ane, ate, ne=None, te=None, lref=None):
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
    ninorma_temp, ninormb = calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorma, zia, zib, ne=ne)
    ania_temp, anib = calc_2ion_ani_from_ani_and_gradient_quasineutrality(ania, zia, zib, ane, ninorma, ne=ne, lref=lref)
    ninorma_temp = calc_ni_from_ninorm(ninorma_temp, ne, nscale=1.0) if ne is not None else copy.deepcopy(ninorma_temp)
    ninormb_temp = calc_ni_from_ninorm(ninormb, ne, nscale=1.0) if ne is not None else copy.deepcopy(ninormb)
    ania_temp = calc_grad_k_from_ak(ania_temp, ninorma_temp, lref) if lref is not None else copy.deepcopy(ania_temp)
    anib_temp = calc_grad_k_from_ak(anib, ninormb_temp, lref) if lref is not None else copy.deepcopy(anib)
    ap = calc_2ion_ap_with_2ions(ania_temp, anib_temp, atia, atib, ninorma_temp, ninormb_temp, tinorma, tinormb, ane, ate, ne=ne, te=te, lref=lref)
    return ap

def calc_3ion_ap_with_3ions(ania, anib, anic, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, ane, ate, ne=None, te=None, lref=None):
    ninormc_temp = calc_ninorm_from_ni(ninormc, ne) if ne is not None else copy.deepcopy(ninormc)
    tinormc_temp = calc_tinorm_from_ti(tinormc, te) if te is not None else copy.deepcopy(tinormc)
    anic_temp = calc_ak_from_grad_k(anic, calc_ni_from_ninorm(ninormc_temp, ne, nscale=1.0), lref) if ne is not None and lref is not None else copy.deepcopy(anic)
    atic_temp = calc_ak_from_grad_k(atic, calc_ti_from_tinorm(tinormc_temp, te, tscale=1.0), lref) if te is not None and lref is not None else copy.deepcopy(atic)
    ap_2ion = calc_2ion_ap_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, ane, ate, ne=ne, te=te, lref=lref)
    ap = ap_2ion + ninormc_temp * tinormc_temp * (anic_temp + atic_temp)
    return ap

def calc_3ion_ap_with_2ions_and_gradient_quasineutrality(ania, anib, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, ane, ate, ne=None, te=None, lref=None):
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
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    lref_temp = None if norm_inputs else copy.deepcopy(lref)
    ap = calc_2ion_ap_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, grad_ne, grad_te, ne=ne_temp, te=te_temp, lref=lref_temp)
    grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    return grad_p

def calc_2ion_grad_p_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, grad_ne, grad_te, ne, te, lref, norm_inputs=False):
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    lref_temp = None if norm_inputs else copy.deepcopy(lref)
    ap = calc_2ion_ap_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, grad_ne, grad_te, ne=ne_temp, te=te_temp, lref=lref_temp)
    grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    return grad_p

def calc_3ion_grad_p_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, grad_ne, grad_te, ne, te, lref, norm_inputs=False):
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    lref_temp = None if norm_inputs else copy.deepcopy(lref)
    ap = calc_3ion_ap_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, grad_ne, grad_te, ne=ne_temp, te=te_temp, lref=lref_temp)
    grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    return grad_p

def calc_3ion_grad_p_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, grad_ne, grad_te, ne, te, lref, norm_inputs=False):
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    lref_temp = None if norm_inputs else copy.deepcopy(lref)
    ap = calc_3ion_ap_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, grad_ne, grad_te, ne=ne_temp, te=te_temp, lref=lref_temp)
    grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    return grad_p

def calc_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, grad_ne, grad_te, ne, te, lref, norm_inputs=False):
    ne_temp = None if norm_inputs else copy.deepcopy(ne)
    te_temp = None if norm_inputs else copy.deepcopy(te)
    lref_temp = None if norm_inputs else copy.deepcopy(lref)
    ap = calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, grad_ne, grad_te, ne=ne_temp, te=te_temp, lref=lref_temp)
    grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    return grad_p

def calc_zeff_from_2ion_ninorm_with_2ions(ninorma, ninormb, zia, zib, ne=None, ze=1.0):
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    zeff = (ninorma_temp * (zia ** 2) + ninormb_temp * (zib ** 2)) / ze
    return zeff

def calc_zeff_from_2ion_ninorm_with_1ion_and_quasineutrality(ninorma, zia, zib, ne=None, ze=1.0):
    ninorma_temp, ninormb = calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorma, zia, zib, ne=ne)
    zeff = calc_zeff_from_2ion_ninorm_with_2ions(ninorma_temp, ninormb, zia, zib, ne=None, ze=ze)
    return zeff

def calc_zeff_from_3ion_ninorm_with_3ions(ninorma, ninormb, ninormc, zia, zib, zic, ne=None, ze=1.0):
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    ninormc_temp = calc_ninorm_from_ni(ninormc, ne) if ne is not None else copy.deepcopy(ninormc)
    zeff = (ninorma_temp * (zia ** 2) + ninormb_temp * (zib ** 2) + ninormc_temp * (zic ** 2)) / ze
    return zeff

def calc_zeff_from_3ion_ninorm_with_2ions_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=None, ze=1.0):
    ninorma_temp, ninormb_temp, ninormc = calc_3ion_ninorm_from_ninorm_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=ne)
    zeff = calc_zeff_from_3ion_ninorm_with_3ions(ninorma_temp, ninormb_temp, ninormc, zia, zib, zic, ne=None, ze=ze)
    return zeff

def calc_azeff_from_2ion_ani_with_2ions(ania, anib, ninorma, ninormb, zia, zib, ane, ne=None, lref=None, ze=1.0):
    ane_temp = calc_ak_from_grad_k(ane, ne, lref) if ne is not None and lref is not None else copy.deepcopy(ane)
    ania_temp = calc_ak_from_grad_k(ania, ninorma, lref) if lref is not None else copy.deepcopy(ania)
    anib_temp = calc_ak_from_grad_k(anib, ninormb, lref) if lref is not None else copy.deepcopy(anib)
    ninorma_temp = calc_ninorm_from_ni(ninorma, ne) if ne is not None else copy.deepcopy(ninorma)
    ninormb_temp = calc_ninorm_from_ni(ninormb, ne) if ne is not None else copy.deepcopy(ninormb)
    zeff = calc_zeff_from_2ion_ninorm_with_2ions(ninorma_temp, ninormb_temp, zia, zib, ne=None)
    azeff = ((ania_temp * ninorma_temp * (zia ** 2) + anib_temp * ninormb_temp * (zib ** 2)) / (zeff * ze)) - ane_temp
    return azeff

def calc_azeff_from_2ion_ani_with_1ion_and_gradient_quasineutrality(ania, ninorma, zia, zib, ane, ne=None, lref=None, ze=1.0):
    ane_temp = calc_ak_from_grad_k(ane, ne, lref) if ne is not None and lref is not None else copy.deepcopy(ane)
    ninorma_temp, ninormb = calc_2ion_ninorm_from_ninorm_and_quasineutrality(ninorma, zia, zib, ne=ne)
    ania_temp, anib = calc_2ion_ani_from_ani_and_gradient_quasineutrality(ania, zia, zib, ane, ninorma, ne=ne, lref=lref)
    azeff = calc_azeff_from_2ion_ani_with_2ions(ania_temp, anib, ninorma_temp, ninormb, zia, zib, ane_temp, ne=None, lref=None, ze=ze)
    return azeff

def calc_azeff_from_3ion_ani_with_3ions(ania, anib, anic, ninorma, ninormb, ninormc, zia, zib, zic, ane, ne=None, lref=None, ze=1.0):
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
    ane_temp = calc_ak_from_grad_k(ane, ne, lref) if ne is not None and lref is not None else copy.deepcopy(ane)
    ninorma_temp, ninormb_temp, ninormc = calc_3ion_ninorm_from_ninorm_and_quasineutrality(ninorma, ninormb, zia, zib, zic, ne=ne)
    ania_temp, anib_temp, anic = calc_3ion_ani_from_ani_and_gradient_quasineutrality(ania, anib, zia, zib, zic, ane, ninorma, ninormb, ne=ne, lref=lref)
    azeff = calc_azeff_from_3ion_ani_with_3ions(ania_temp, anib_temp, anic, ninorma_temp, ninormb_temp, ninormc, zia, zib, zic, ane_temp, ne=None, lref=None, ze=ze)
    return azeff

def calc_ne_from_beta_and_pnorm(beta, pnorm, btot, te):
    c = constants_si()
    ne = beta * (btot ** 2) / (2.0 * c['mu'] * c['e'] * te * pnorm)
    return ne

def calc_te_from_beta_and_pnorm(beta, pnorm, btot, ne):
    c = constants_si()
    te = beta * (btot ** 2) / (2.0 * c['mu'] * c['e'] * ne * pnorm)
    return te

def calc_btot_from_beta_and_pnorm(beta, pnorm, ne, te):
    c = constants_si()
    btot = np.sqrt(2.0 * c['mu'] * c['e'] * ne * te * pnorm / beta)
    return btot

def calc_btot_from_beta_and_p(beta, p):
    c = constants_si()
    btot = np.sqrt(2.0 * c['mu'] * p / beta)
    return btot

def calc_beta_from_p(p, btot):
    c = constants_si()
    beta = 2.0 * c['mu'] * p / (btot ** 2)
    return beta

def calc_beta_from_pnorm(pnorm, btot, ne, te):
    c = constants_si()
    betae = 2.0 * c['mu'] * c['e'] * ne * te / (btot ** 2)
    beta = betae * pnorm
    return beta

def calc_ne_from_alpha_and_ap(alpha, ap, q, btot, te):
    c = constants_si()
    ne = alpha * (btot ** 2) / (2.0 * c['mu'] * (q ** 2) * c['e'] * te * ap)
    return ne

def calc_te_from_alpha_and_ap(alpha, ap, q, btot, ne):
    c = constants_si()
    te = alpha * (btot ** 2) / (2.0 * c['mu'] * (q ** 2) * c['e'] * ne * ap)
    return te

def calc_btot_from_alpha_and_ap(alpha, ap, q, ne, te):
    c = constants_si()
    btot = np.sqrt(2.0 * c['mu'] * (q ** 2) * c['e'] * ne * te * ap / alpha)
    return btot

def calc_btot_from_alpha_and_grad_p(alpha, grad_p, q, lref):
    c = constants_si()
    btot = np.sqrt(2.0 * c['mu'] * (q ** 2) * -lref * grad_p / alpha)
    return btot

def calc_alpha_from_grad_p(grad_p, q, btot, lref):
    c = constants_si()
    alpha = -2.0 * c['mu'] * (q ** 2) * lref * grad_p / (btot ** 2)
    return alpha

def calc_alpha_from_ap(ap, q, btot, ne, te):
    c = constants_si()
    betae = 2.0 * c['mu'] * c['e'] * ne * te / (btot ** 2)
    alpha = q * q * betae * ap
    return alpha

def calc_alpha_from_2ion_ap_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, q, ane, ate, btot, ne, te, lref=None):
    ne_temp = copy.deepcopy(ne) if lref is not None else None
    te_temp = copy.deepcopy(te) if lref is not None else None
    ap = calc_2ion_ap_with_2ions(ania, anib, atia, atib, ninorma, ninormb, tinorma, tinormb, ane, ate, ne=ne_temp, te=te_temp, lref=lref)
    alpha = calc_alpha_from_ap(ap, q, btot, ne, te)
    return alpha

def calc_alpha_from_2ion_ap_with_1ion_and_gradient_quasineutrality(ania, atia, atib, ninorma, tinorma, tinormb, zia, zib, q, ane, ate, btot, ne, te, lref=None):
    ne_temp = copy.deepcopy(ne) if lref is not None else None
    te_temp = copy.deepcopy(te) if lref is not None else None
    ap = calc_2ion_ap_with_1ion_and_gradient_quasineutrality(ania, atia, atib, ninorma, tinorma, tinormb, zia, zib, ane, ate, ne=ne_temp, te=te_temp, lref=lref)
    alpha = calc_alpha_from_ap(ap, q, btot, ne, te)
    return alpha

def calc_alpha_from_2ion_grad_p_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False):
    grad_p = calc_2ion_grad_p_with_2ions(grad_nia, grad_nib, grad_tia, grad_tib, nia, nib, tia, tib, grad_ne, grad_te, ne, te, lref, norm_inputs=norm_inputs)
    alpha = calc_alpha_from_grad_p(grad_p, q, btot, lref)
    return alpha

def calc_alpha_from_2ion_grad_p_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False):
    grad_p = calc_2ion_grad_p_with_1ion_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, nia, tia, tib, zia, zib, grad_ne, grad_te, ne, te, lref, norm_inputs=norm_inputs)
    alpha = calc_alpha_from_grad_p(grad_p, q, btot, lref)
    return alpha

def calc_alpha_from_3ion_ap_with_3ions(ania, anib, anic, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, q, ane, ate, btot, ne, te, lref=None):
    ne_temp = copy.deepcopy(ne) if lref is not None else None
    te_temp = copy.deepcopy(te) if lref is not None else None
    ap = calc_3ion_ap_with_3ions(ania, anib, anic, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc, ane, ate, ne=ne_temp, te=te_temp, lref=lref)
    alpha = calc_alpha_from_ap(ap, q, btot, ne, te)
    return alpha

def calc_alpha_from_3ion_ap_with_2ions_and_gradient_quasineutrality(ania, anib, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, q, ane, ate, btot, ne, te, lref=None):
    ne_temp = copy.deepcopy(ne) if lref is not None else None
    te_temp = copy.deepcopy(te) if lref is not None else None
    ap = calc_3ion_ap_with_2ions_and_gradient_quasineutrality(ania, anib, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic, ane, ate, ne=ne_temp, te=te_temp, lref=lref)
    alpha = calc_alpha_from_ap(ap, q, btot, ne, te)
    return alpha

def calc_alpha_from_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(ania, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic, q, ane, ate, btot, ne, te, lref=None):
    ne_temp = copy.deepcopy(ne) if lref is not None else None
    te_temp = copy.deepcopy(te) if lref is not None else None
    ap = calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(ania, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic, ane, ate, ne=ne_temp, te=te_temp, lref=lref)
    alpha = calc_alpha_from_ap(ap, q, btot, ne, te)
    return alpha

def calc_alpha_from_3ion_grad_p_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False):
    grad_p = calc_3ion_grad_p_with_3ions(grad_nia, grad_nib, grad_nic, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic, grad_ne, grad_te, ne, te, lref, norm_inputs=norm_inputs)
    alpha = calc_alpha_from_grad_p(grad_p, q, btot, lref)
    return alpha

def calc_alpha_from_3ion_grad_p_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False):
    grad_p = calc_3ion_grad_p_with_2ions_and_gradient_quasineutrality(grad_nia, grad_nib, grad_tia, grad_tib, grad_tic, nia, nib, tia, tib, tic, zia, zib, zic, grad_ne, grad_te, ne, te, lref, norm_inputs=norm_inputs)
    alpha = calc_alpha_from_grad_p(grad_p, q, btot, lref)
    return alpha

def calc_alpha_from_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, q, grad_ne, grad_te, btot, ne, te, lref, norm_inputs=False):
    grad_p = calc_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(grad_nia, grad_tia, grad_tib, grad_tic, nia, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, grad_ne, grad_te, ne, te, lref, norm_inputs=norm_inputs)
    alpha = calc_alpha_from_grad_p(grad_p, q, btot, lref)
    return alpha

def calc_coulomb_logarithm_nrl_from_ne_and_te(ne, te):
    cl = 15.2 - 0.5 * np.log(ne * 1.0e-20) + np.log(te * 1.0e-3)
    if cl.ndim == 0:
        cl = float(cl)
    return cl

def calc_nustar_nrl_from_ne_and_te(zeff, q, rmin, rmaj, ne, te):
    c = constants_si()
    cl = calc_coulomb_logarithm_nrl_from_ne_and_te(ne, te)
    nt = (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    kk = (1.0e4 / 1.09) * q * rmaj * ((rmin / rmaj) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    nustar = cl * zeff * nt * kk
    return nustar

def calc_lognustar_from_nustar(nustar):
    lognustar = np.log10(nustar)
    return lognustar

def calc_nustar_from_lognustar(lognustar):
    nustar = np.power(10.0, lognustar)
    return nustar

def calc_ne_from_nustar_nrl(nustar, zeff, q, rmin, rmaj, te):
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
    c = constants_si()
    cl = calc_coulomb_logarithm_nrl_from_ne_and_te(ne, te)
    nt = (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    kk = (1.0e4 / 1.09) * q * rmaj * ((rmin / rmaj) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    zeff = nustar / (cl * nt * kk)
    return zeff

def calc_rhos_from_ts_and_btot(ts, btot, _as, _zs=1.0):
    c = constants_si()
    rhos = (ts ** 0.5) * (((_as * c['u'] / c['e']) ** 0.5) / _zs) / btot
    return rhos

def calc_btot_from_rhos(rhos, ts, _as, _zs=1.0):
    c = constants_si()
    btot = (ts ** 0.5) * (((_as * c['u'] / c['e']) ** 0.5) / _zs) / rhos
    return btot

def calc_ts_from_rhos(rhos, btot, _as, _zs=1.0):
    c = constants_si()
    te = ((rhos * btot * _zs) ** 2) / (_as * c['u'] / c['e'])
    return te

def calc_rhoref_from_te_and_btot(te, btot, ai):
    c = constants_si()
    rhoref = calc_rhos_from_ts_and_btot(te, btot, ai, _zs=1.0)
    return rhoref

def calc_btot_from_rhoref(rhoref, te, ai):
    btot = calc_btot_from_rhos(rhoref, te, ai, _zs=1.0)
    return btot

def calc_te_from_rhoref(rhoref, btot, ai):
    te = calc_ts_from_rhos(rhoref, btot, ai, _zs=1.0)
    return te

def calc_rhostar_from_rhoref(rhoref, lref):
    rhostar = normalize(rhoref, lref)
    return rhostar

def calc_rhoref_from_rhostar(rhostar, lref):
    rhoref = unnormalize(rhostar, lref)
    return rhoref

def calc_rhostar_from_te_and_btot(te, btot, ai, lref):
    rhoref = calc_rhoref_from_te_and_btot(te, btot, ai)
    rhostar = calc_rhostar_from_rhoref(rhoref, lref)
    return rhostar

def calc_btot_from_rhostar(rhostar, te, ai, lref):
    rhoref = calc_rhoref_from_rhostar(rhostar, lref)
    btot = calc_btot_from_rhoref(rhoref, te, ai)
    return btot

def calc_te_from_rhostar(rhostar, btot, ai, lref):
    rhos = calc_rhoref_from_rhostar(rhostar, lref)
    te = calc_te_from_rhoref(rhos, btot, ai)
    return te

def calc_ne_from_alpha_and_rhostar(alpha, rhostar, ap, ai, zi, q, lref):
    c = constants_si()
    mi = ai * c['u']
    qi = zi * c['e']
    prefactor = mi / (2.0 * c['mu'] * (qi ** 2) * (q ** 2))
    ne = prefactor * alpha / ((rhostar ** 2) * (lref ** 2) * ap)
    return ne

def calc_btot_from_alpha_and_rhostar(alpha, rhostar, ap, ai, zi, q, ne, te, lref):
    c = constants_si()
    mi = ai * c['u']
    qi = zi * c['e']
    prefactor = 2.0 * c['mu'] * (mi ** 0.5) / qi
    btot = (prefactor * (q ** 2) * ne * ((c['e'] * te) ** 1.5) * ap / (alpha * rhostar * lref)) ** (1.0 / 3.0)
    return btot

def calc_ne_from_beta_and_rhostar(beta, rhostar, pnorm, ai, zi, lref):
    c = constants_si()
    mi = ai * c['u']
    qi = zi * c['e']
    prefactor = mi / (2.0 * c['mu'] * (qi ** 2))
    ne = prefactor * beta / ((rhostar ** 2) * (lref ** 2) * pnorm)
    return ne

def calc_btot_from_beta_and_rhostar(beta, rhostar, pnorm, ai, zi, ne, te, lref):
    c = constants_si()
    mi = ai * c['u']
    qi = zi * c['e']
    prefactor = 2.0 * c['mu'] * (mi ** 0.5) / qi
    bref = (prefactor * ne * ((c['e'] * te) ** 1.5) * pnorm / (beta * rhostar * lref)) ** (1.0 / 3.0)
    return bref

def calc_ne_from_nustar_nrl_alpha_and_ap(nustar, alpha, ap, zeff, q, btot, rmin, rmaj):
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

def calc_vref_from_te_and_mref(te, aref):
    c = constants_si()
    vref = (te * c['e'] / (c['u'] * aref)) ** 0.5
    return vref

def calc_vths_from_ts(ts, _as):
    c = constants_si()
    vth = (2.0 * ts * c['e'] / (c['u'] * _as)) ** 0.5
    return vth

def calc_mach_from_u(u, cref):
    mach = normalize(u, cref)
    return mach

def calc_u_from_mach(mach, cref):
    u = unnormalize(mach, cref)
    return u

def calc_machpar_from_machtor_and_toroidal_assumption_circular(machtor, q, epsilon, x):
    btorbyb = ((q ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    machpar = machtor * btorbyb
    return machpar

def calc_aupar_from_autor_and_toroidal_assumption_circular(autor, machtor, s, q, epsilon, x):
    btorbyb = ((q ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    bpolbyb = (((epsilon * x) ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    grad_btorbyb = btorbyb * (bpolbyb ** 2) * (s - 1.0) / (epsilon * x)
    aupar = autor * btorbyb - machtor * grad_btorbyb
    return aupar

def calc_machper_from_machtor_and_toroidal_assumption_circular(machtor, q, epsilon, x):
    bpolbyb = (((epsilon * x) ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    machper = machtor * bpolbyb
    return machper

def calc_auper_from_autor_and_toroidal_assumption_circular(autor, machtor, s, q, epsilon, x):
    btorbyb = ((q ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    bpolbyb = (((epsilon * x) ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    grad_btorbyb = btorbyb * (bpolbyb ** 2) * (s - 1.0) / (epsilon * x)
    auper = autor * btorbyb - machtor * grad_btorbyb
    return auper

def calc_gammae_from_aupar_without_grad_dpi(aupar, q, epsilon):
    gammae = -(epsilon / q) * aupar
    return gammae

def calc_grad_dpi_from_gammae_machtor_and_autor(gammae, machtor, autor, q, r, ro):
    # Also no CI test
    return None

def calc_bunit_from_bref(bref, sfac):
    bunit = normalize(bref, sfac)
    return bunit

def calc_bref_from_bunit(bunit, sfac):
    bref = unnormalize(bunit, sfac)
    return bref

def calc_lds_from_ts_and_ns(ts, ns, zs):
    c = constants_si()
    lds = (c['eps'] * ts / (c['e'] * ns * (zs ** 2))) ** 0.5
    return lds

def calc_ldsnorm_from_lds(lds, rhos):
    ldsnorm = normalize(lds, rhos)
    return ldsnorm

def calc_ldenorm_from_te_ne_and_rhos(te, ne, rhos, ze=1.0):
    lde = calc_lds_from_ts_and_ns(te, ne, ze)
    ldenorm = calc_ldsnorm_from_lds(lde, rhos)
    return ldenorm

def calc_coulomb_logarithm_from_ne_and_te(ne, te, ze=1.0):
    c = constants_si()
    lda = calc_lds_from_ts_and_ns(te, ne, ze)
    inv_b90 = (4.0 * np.pi * c['eps'] / c['e']) * te / (ze * ze)
    cl = np.log(inv_b90 * lda)
    if cl.ndim == 0:
        cl = float(cl)
    return cl

def calc_nu_from_t_and_n(ta, na, nb, ma, za, zb):
    c = constants_si()
    factor = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    inv_b90 = (4.0 * np.pi * c['eps'] / c['e']) * ta / (za * zb)
    cl = calc_coulomb_logarithm_from_ne_and_te(na, ta, ze=za) + np.log(za / zb)
    nu = factor * nb * (c['e'] * ta / (c['u'] * ma)) ** 0.5 / (inv_b90 ** 2) * cl
    return nu

def calc_nuei_from_te_ne_and_zeff(te, ne, zeff, zi, ze=1.0):
    c = constants_si()
    factor = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    inv_b90 = (4.0 * np.pi * c['eps'] / c['e']) * te / (ze * ((zeff * ze) ** 0.5))
    cl = calc_coulomb_logarithm_from_ne_and_te(te, ne, ze=ze) + np.log(ze / zi)
    nuei = factor * ne * (c['e'] * te / (c['me'])) ** 0.5 / (inv_b90 ** 2) * cl
    return nuei

def calc_nunorm_from_nu(nu, gref):
    nunorm = normalize(nu, gref)
    return nunorm

def calc_nueenorm_from_te_and_ne(te, ne, gref, ze=1.0):
    c = constants_si()
    me = c['me'] / c['u']
    nuee = calc_nu_from_t_and_n(te, ne, ne, me, ze, ze)
    nueenorm = calc_nunorm_from_nu(nuee, gref)
    return nueenorm

def calc_nueinorm_from_te_ne_and_ni(te, ne, ni, zi, gref, ze=1.0):
    c = constants_si()
    me = c['me'] / c['u']
    nuei = calc_nu_from_t_and_n(te, ne, ni, me, ze, zi)
    nueinorm = calc_nunorm_from_nu(nuei, gref)
    return nueinorm

def calc_nueinorm_from_te_ne_and_zeff(te, ne, zeff, zi, gref, ze=1.0):
    nuei = calc_nuei_from_te_ne_and_zeff(te, ne, zeff, zi, ze)
    nueinorm = calc_nunorm_from_nu(nuei, gref)
    return nueinorm

def calc_nuiinorm_from_te_ne_and_ni(tia, nia, nib, mia, zia, zib, gref):
    nuii = calc_nu_from_t_and_n(tia, nia, nib, mia, zia, zib)
    nuiinorm = calc_nunorm_from_nu(nuii, gref)
    return nuiinorm

def calc_te_from_betae_and_ldenorm(betae, ldenorm, sfac, mref, ze=1.0):
    c = constants_si()
    bc = (ze ** 2) * c['u'] * mref / (2.0 * c['eps'] * c['e'] * c['mu'])
    te = bc * betae * (ldenorm ** 2) * (sfac ** 2)
    return te

def calc_ne_from_nueenorm(nueenorm, te, mref, lref, ze=1.0):
    c = constants_si()
    f_nu = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    inv_b90_ee = 4.0 * np.pi * (c['eps'] / c['e']) * te / (ze ** 2)
    nc = f_nu * lref * (c['u'] * mref / c['me']) ** 0.5
    nl = inv_b90_ee * (c['eps'] / c['e']) ** 0.5 * (te / (ze ** 2)) ** 0.5
    data = {'logterm': np.log(nl), 'constant': nueenorm * (inv_b90_ee ** 2) / nc}
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
    ne = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    logger.debug(f'<{calc_ne_from_nueenorm.__name__}>: data')
    logger.debug(pd.DataFrame(data={'nueenorm': nueenorm, 'te': te, 'ne': ne}))
    return ne

def calc_bunit_from_ldenorm(ldenorm, ne, mref, ze=1.0):
    c = constants_si()
    bunit = (ne * (ze ** 2) * c['u'] * mref / c['eps']) ** 0.5 * ldenorm
    return bunit

def calc_ne_from_nustar(nustar, zeff, q, r, ro, te):
    c = constants_si()
    eom = c['e'] / c['me']
    tb = q * ro * ((r / ro) ** (-1.5)) / ((eom * te) ** 0.5)
    kk = (1.0e4 / 1.09) * zeff * ((te * 1.0e-3) ** (-1.5))
    nu = nustar / (tb * kk)
    data = {'te': te * 1.0e-3, 'knu': nu}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    func_ne20 = lambda row: root_scalar(
        lambda ne: calc_coulomb_logarithm_from_ne_and_te(ne * 1.0e20, row['te'] * 1.0e3) * ne - row['knu'],
        x0=0.01,
        x1=1.0,
        maxiter=100,
    )
    sol_ne20 = rootdata.apply(func_ne20, axis=1)
    retry = sol_ne20.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_ne20_v2 = lambda row: root_scalar(
            lambda ne: calc_coulomb_logarithm_from_ne_and_te(ne * 1.0e20, row['te'] * 1.0e3) * ne - row['knu'],
            x0=1.0,
            x1=0.1,
            maxiter=100,
        )
        sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
    warnings.resetwarnings()
    ne = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    logger.debug(f'<{calc_ne_from_nustar.__name__}>: data')
    logger.debug(pd.DataFrame(data={'nustar': nustar, 'te': te, 'ne': ne}))
    return ne

def calc_te_from_nustar(nustar, zeff, q, r, ro, ne, verbose=0):
    c = constants_si()
    moe = c['me'] / c['e']
    kk = (10.0 ** 0.5) * (1.0e2 / 1.09) * zeff * q * ro * ((r / ro) ** (-1.5)) * (moe ** 0.5) * (ne * 1.0e-20)
    nu = nustar / kk
    data = {'ne': ne * 1.0e-20, 'knu': nu}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    func_te3 = lambda row: root_scalar(
        lambda te: calc_coulomb_logarithm_from_ne_and_te(row['ne'] * 1.0e20, te * 1.0e3) / (te ** 2) - row['knu'],
        x0=1.0,
        x1=0.1,
        maxiter=100,
    )
    sol_te3 = rootdata.apply(func_te3, axis=1)
    retry = sol_te3.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_te3_v2 = lambda row: root_scalar(
            lambda te: calc_coulomb_logarithm_from_ne_and_te(row['ne'] * 1.0e20, te * 1.0e3) / (te ** 2) - row['knu'],
            x0=0.01,
            x1=0.1,
            maxiter=100,
        )
        sol_te3.loc[retry] = rootdata.loc[retry].apply(func_te3_v2, axis=1)
    warnings.resetwarnings()
    te = sol_te3.apply(lambda sol: 1.0e3 * sol.root).to_numpy()
    logger.debug(f'<{calc_te_from_nustar.__name__}>: data')
    logger.debug(pd.DataFrame(data={'nustar': nustar, 'ne': ne, 'te': te}))
    return te

def calc_zeff_from_nustar(nustar, q, r, ro, ne, te):
    c = constants_si()
    cl = calc_coulomb_logarithm_from_ne_and_te(ne, te, ze=1.0)
    nt = (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    kk = (1.0e4 / 1.09) * q * ro * ((r / ro) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    zeff = nustar / (cl * nt * kk)
    return zeff

def calc_nustar(zeff, q, r, ro, ne, te):
    c = constants_si()
    cl = calc_coulomb_logarithm_from_ne_and_te(ne, te, ze=1.0)
    nt = (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    kk = (1.0e4 / 1.09) * q * ro * ((r / ro) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    nustar = cl * zeff * nt * kk
    return nustar

def calc_flux_surface_values_from_mxh(rmin, rgeo, zgeo, kappa, drgeo, dzgeo, s_kappa, cos, sin, s_cos, s_sin):
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
    xs = np.trapezoid(r, z, axis=0)
    vol = 2.0 * np.pi * rgeo * xs
    return vol, xs

def calc_b_from_flux_surface_values(r, grad_r, l_t, rmin, q):
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
    r_v = np.expand_dims(rmin * rgeo, axis=0)
    g_t = r * b * l_t / (np.where(np.isclose(r_v, 0.0), 1.0, r_v) * grad_r)
    #g_t[..., 0] = 2.0 * g_t[..., 1] - g_t[..., 2]
    return g_t

def calc_grad_vol_from_flux_surface_values(r, l_t, grad_r):
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