import copy
import logging
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

logger = logging.getLogger('fusio')

np_itypes = (np.int8, np.int16, np.int32, np.int64)
np_utypes = (np.uint8, np.uint16, np.uint32, np.uint64)
np_ftypes = (np.float16, np.float32, np.float64)

number_types = (float, int, np_itypes, np_utypes, np_ftypes)
array_types = (list, tuple, np.ndarray)
string_types = (str, np.str_)

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

def calc_q_from_bp(bp, r, bo, ro):
    q = r * bo / (ro * bp)
    return q

def calc_bp_from_q(q, r, bo, ro):
    bp = r * bo / (ro * q)
    return bp

def calc_grad_q_from_s(s, bp, bo, ro):
    grad_q = s * bo / (ro * bp)
    return grad_q

def calc_grad_q_from_s_and_q(s, q, r):
    grad_q = calc_grad_k_from_ak(s, q, -r)
    return grad_q

def calc_s_from_grad_bp(grad_bp, bp, r):
    s = 1.0 - r * grad_bp / bp
    return s

def calc_grad_bp_from_grad_q(grad_q, bp, r, bo, ro):
    grad_bp = (1.0 - ro * bp * grad_q / bo) * bp / r
    return grad_bp

def calc_grad_bp_from_s(s, bp, r):
    grad_bp = (1.0 - s) * bp / r
    return grad_bp

def calc_ninorm_from_zeff_and_quasineutrality(zeff, zia, zib, zi_target, ninorma, ze=1.0):
    zze = (zib - zeff) * ze
    zza = (zib - zia) * zia
    zz_target = (zib - zi_target) * zi_target
    ninorm_target = (zze - ninorma * zza) / zz_target
    return ninorm_target

def calc_ninorm_from_quasineutrality(zia, zib, zi_target, ninorma, ninormb, ze=1.0):
    ninorm_target = (ze - ninorma * zia - ninormb * zib) / zi_target
    return ninorm_target

def calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ni, zeff, zia, zib, zic, ne=None):
    ninorma = normalize(ni, ne) if ne is not None else copy.deepcopy(ni)
    ninormb = calc_ninorm_from_zeff_and_quasineutrality(zeff, zia, zic, zib, ninorma)
    ninormc = calc_ninorm_from_quasineutrality(zia, zib, zic, ninorma, ninormb)
    return ninorma, ninormb, ninormc

def calc_ni_from_zeff_and_quasineutrality(zeff, zia, zib, zi_target, nia, ne):
    ninorma = calc_ninorm_from_ni(nia, ne)
    ninorm_target = calc_ninorm_from_zeff_and_quasineutrality(zeff, zia, zib, zi_target, ninorma)
    ni_target = calc_ni_from_ninorm(ninorm_target, ne, nscale=1.0)
    return ni_target

def calc_ni_from_quasineutrality(zia, zib, zi_target, nia, nib, ne):
    ninorma = calc_ninorm_from_ni(nia, ne)
    ninormb = calc_ninorm_from_ni(nib, ne)
    ninorm_target = calc_ninorm_from_quasineutrality(zia, zib, zi_target, ninorma, ninormb)
    ni_target = calc_ni_from_ninorm(ninorm_target, ne, nscale=1.0)
    return ni_target

def calc_3ion_ni_from_ni_zeff_and_quasineutrality(ni, zeff, zia, zib, zic, ne, norm_inputs=False):
    nia = unnormalize(ni, ne) if norm_inputs else copy.deepcopy(ni)
    nib = calc_ni_from_zeff_and_quasineutrality(zeff, zia, zic, zib, nia, ne)
    nic = calc_ni_from_quasineutrality(zia, zib, zic, nia, nib, ne)
    return nia, nib, nic

def calc_ani_from_azeff_and_gradient_quasineutrality(azeff, zeff, zia, zib, zi_target, ninorma, ninorm_target, ane, ania, ze=1.0):
    zze = (zib - zeff) * ze
    zza = (zib - zia) * zia
    zz_target = (zib - zi_target) * zi_target
    ani_target = (azeff * zeff * ze + ane * zze - ania * zza * ninorma) / (ninorm_target * zz_target)
    return ani_target

def calc_ani_from_gradient_quasineutrality(zia, zib, zi_target, ninorma, ninormb, ninorm_target, ane, ania, anib, ze=1.0):
    ani_target = (ane * ze - ninorma * ania * zia - ninormb * anib * zib) / (ninorm_target * zi_target)
    return ani_target

def calc_3ion_ani_from_ani_azeff_and_gradient_quasineutrality(ani, azeff, zeff, zia, zib, zic, ane, ni, lref=None, ne=None):
    ania = calc_ak_from_grad_k(ani, ni, lref) if lref is not None else copy.deepcopy(ani)
    ninorma, ninormb, ninormc = calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ni, zeff, zia, zib, zic, ne)
    anib = calc_ani_from_azeff_and_gradient_quasineutrality(azeff, zeff, zia, zic, zib, ninorma, ninormb, ane, ania)
    anic = calc_ani_from_gradient_quasineutrality(zia, zib, zic, ninorma, ninormb, ninormc, ane, ania, anib)
    return ania, anib, anic

def calc_grad_ni_from_grad_zeff_and_gradient_quasineutrality(grad_zeff, zeff, zia, zib, zi_target, nia, ni_target, grad_ne, grad_nia, ne, lref):
    azeff = calc_ak_from_grad_k(grad_zeff, zeff, lref)
    ninorma = calc_ninorm_from_ni(nia, ne)
    ninorm_target = calc_ninorm_from_ni(ni_target, ne)
    ane = calc_ak_from_grad_k(grad_ne, ne, lref)
    ania = calc_ak_from_grad_k(grad_nia, nia, lref)
    ani_target = calc_ani_from_azeff_and_gradient_quasineutrality(azeff, zeff, zia, zib, zi_target, ninorma, ninorm_target, ane, ania)
    grad_ni_target = calc_grad_k_from_ak(ani_target, ni_target, lref)
    return grad_ni_target

def calc_grad_ni_from_gradient_quasineutrality(zia, zib, zi_target, nia, nib, ni_target, grad_ne, grad_nia, grad_nib, ne, lref):
    ninorma = calc_ninorm_from_ni(nia, ne)
    ninormb = calc_ninorm_from_ni(nib, ne)
    ninorm_target = calc_ninorm_from_ni(ni_target, ne)
    ane = calc_ak_from_grad_k(grad_ne, ne, lref)
    ania = calc_ak_from_grad_k(grad_nia, nia, lref)
    anib = calc_ak_from_grad_k(grad_nib, nib, lref)
    ani_target = calc_ani_from_gradient_quasineutrality(zia, zib, zi_target, ninorma, ninormb, ninorm_target, ane, ania, anib)
    grad_ni_target = calc_grad_k_from_ak(ani_target, ni_target, lref)
    return grad_ni_target

def calc_3ion_grad_ni_from_grad_ni_grad_zeff_and_gradient_quasineutrality(grad_ni, grad_zeff, zeff, zia, zib, zic, grad_ne, ni, ne, lref, norm_inputs=False):
    grad_ne_temp = calc_grad_k_from_ak(grad_ne, ne, lref) if norm_inputs else copy.deepcopy(grad_ne)
    grad_nia = calc_grad_k_from_ak(grad_ni, ni, lref) if norm_inputs else copy.deepcopy(grad_ni)
    grad_zeff_temp = calc_grad_k_from_ak(grad_zeff, zeff, lref) if norm_inputs else copy.deepcopy(grad_zeff)
    nia, nib, nic = calc_3ion_ni_from_ni_zeff_and_quasineutrality(ni, zeff, zia, zib, zic, ne, norm_inputs)
    grad_nib = calc_grad_ni_from_grad_zeff_and_gradient_quasineutrality(grad_zeff_temp, zeff, zia, zic, zib, nia, nib, grad_ne_temp, grad_nia, ne, lref)
    grad_nic = calc_grad_ni_from_gradient_quasineutrality(zia, zib, zic, nia, nib, nic, grad_ne_temp, grad_nia, grad_nib, ne, lref)
    return grad_nia, grad_nib, grad_nic

def calc_p_from_pnorm(pnorm, ne, te):
    c = constants_si()
    p = c['e'] * ne * te * pnorm
    return p

def calc_pnorm_from_p(p, ne, te):
    c= constants_si()
    pnorm = p / (c['e'] * ne * te)
    return pnorm

def calc_3ion_pnorm_with_3ions(ninorma, ninormb, ninormc, tinorma, tinormb, tinormc):
    pnorm = 1.0 + ninorma * tinorma + ninormb * tinormb + ninormc * tinormc
    return pnorm

def calc_3ion_pnorm_with_2ions_and_quasineutrality(ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic):
    ninormc = calc_ninorm_from_quasineutrality(zia, zib, zic, ninorma, ninormb)
    pnorm = calc_3ion_pnorm_with_3ions(ninorma, ninormb, ninormc, tinorma, tinormb, tinormc)
    return pnorm

def calc_3ion_pnorm_with_1ion_zeff_and_quasineutrality(ninorma, tinorma, tinormb, tinormc, zeff, zia, zib, zic):
    ninorma_temp, ninormb, ninormc = calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ninorma, zeff, zia, zib, zic)
    pnorm = calc_3ion_pnorm_with_3ions(ninorma_temp, ninormb, ninormc, tinorma, tinormb, tinormc)
    return pnorm

def calc_3ion_p_with_3ions(ne, nia, nib, nic, te, tia, tib, tic, norm_inputs=False):
    p = 0.0 * ne
    if norm_inputs:
        pnorm = calc_3ion_pnorm_with_3ions(nia, nib, nic, tia, tib, tic)
        p = calc_p_from_pnorm(pnorm, ne, te)
    else:
        p = ee * (ne * te + nia * tia + nib * tib + nic * tic)
    logger.debug(f'<{calc_3ion_p_with_3ions.__name__}>: p\n{p}\n')
    return p

def calc_3ion_p_with_2ions_and_quasineutrality(ne, nia, nib, te, tia, tib, tic, zia, zib, zic, norm_inputs=False):
    nic = (
        calc_ninorm_from_quasineutrality(zia, zib, zic, nia, nib)
        if norm_inputs else
        calc_ni_from_quasineutrality(zia, zib, zic, nia, nib, ne)
    )
    p = calc_3ion_p_with_3ions(ne, nia, nib, nic, te, tia, tib, tic, norm_inputs)
    return p

def calc_3ion_p_with_1ion_zeff_and_quasineutrality(ne, nia, te, tia, tib, tic, zeff, zia, zib, zic, norm_inputs=False):
    nia_temp, nib, nic = (
        calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(nia, zeff, zia, zib, zic)
        if norm_inputs else
        calc_3ion_ni_from_ni_zeff_and_quasineutrality(nia, zeff, zia, zib, zic, ne)
    )
    p = calc_3ion_p_with_3ions(ne, nia_temp, nib, nic, te, tia, tib, tic, norm_inputs)
    return p

def calc_zeff_from_3ion_ni_with_3ions(ne, nia, nib, nic, zia, zib, zic, norm_inputs=False):
    ninorma = copy.deepcopy(nia) if norm_inputs else calc_ninorm_from_ni(nia, ne)
    ninormb = copy.deepcopy(nib) if norm_inputs else calc_ninorm_from_ni(nib, ne)
    ninormc = copy.deepcopy(nic) if norm_inputs else calc_ninorm_from_ni(nic, ne)
    zeff = ninorma * (zia ** 2) + ninormb * (zib ** 2) + ninormc * (zic ** 2)
    return zeff

def calc_zeff_from_3ion_ni_with_2ions_and_quasineutrality(ne, nia, nib, zia, zib, zic, norm_inputs=False):
    nic = (
        calc_ninorm_from_quasineutrality(zia, zib, zic, nia, nib)
        if norm_inputs else
        calc_ni_from_quasineutrality(zia, zib, zic, nia, nib, ne)
    )
    zeff = calc_zeff_from_3ion_ni_with_3ions(ne, nia, nib, nic, zia, zib, zic, norm_inputs)
    return zeff

def calc_grad_p_from_ap(ap, ne, te, lref):
    c = constants_si()
    grad_p = calc_grad_k_from_ak(ap, c['e'] * ne * te, lref)
    return grad_p

def calc_ap_from_grad_p(grad_p, ne, te, lref):
    c = constants_si()
    ap = calc_ak_from_grad_k(grad_p, c['e'] * ne * te, lref)
    return ap

def calc_3ion_ap_with_3ions(ane, ania, anib, anic, ate, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc):
    ap = ane + ate + ninorma * tinorma * (ania + atia) + ninormb * tinormb * (anib + atib) + ninormc * tinormc * (anic + atic)
    logger.debug(f'<{calc_3ion_ap_with_3ions.__name__}>: ap\n{ap}\n')
    return ap

def calc_3ion_ap_with_2ions_and_gradient_quasineutrality(ane, ania, anib, ate, atia, atib, atic, ninorma, ninormb, tinorma, tinormb, tinormc, zia, zib, zic):
    ninormc = calc_ninorm_from_quasineutrality(zia, zib, zic, ninorma, ninormb)
    anic = calc_ani_from_gradient_quasineutrality(zia, zib, zic, ninorma, ninormb, ninormc, ane, ania, anib)
    ap = calc_3ion_ap_with_3ions(ane, ania, anib, anic, ate, atia, atib, atic, ninorma, ninormb, ninormc, tinorma, tinormb, tinormc)
    return ap

def calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(ane, ania, ate, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic):
    ninorma_temp, ninormb, ninormc = calc_3ion_ninorm_from_ninorm_zeff_and_quasineutrality(ninorma, zeff, zia, zib, zic)
    ania_temp, anib, anic = calc_3ion_ani_from_ani_azeff_and_gradient_quasineutrality(ania, azeff, zeff, zia, zib, zic, ane, ninorma_temp)
    ap = calc_3ion_ap_with_3ions(ane, ania_temp, anib, anic, ate, atia, atib, atic, ninorma_temp, ninormb, ninormc, tinorma, tinormb, tinormc)
    return ap

def calc_3ion_grad_p_with_3ions(grad_ne, grad_nia, grad_nib, grad_nic, grad_te, grad_tia, grad_tib, grad_tic, ne, nia, nib, nic, te, tia, tib, tic, lref=None):
    grad_p = 0.0 * ne
    if lref is not None:
        ap = calc_3ion_ap_with_3ions(grad_ne, grad_nia, grad_nib, grad_nic, grad_te, grad_tia, grad_tib, grad_tic, nia, nib, nic, tia, tib, tic)
        grad_p = calc_grad_p_from_ap(ap, ne, te, lref)
    else:
        c = constants_si()
        grad_p = c['e'] * (ne * grad_te + grad_ne * te + nia * grad_tia + grad_nia * tia + nib * grad_tib + grad_nib * tib + nic * grad_tic + grad_nic * tic)
    return grad_p

def calc_3ion_grad_p_with_2ions_and_gradient_quasineutrality(grad_ne, grad_nia, grad_nib, grad_te, grad_tia, grad_tib, grad_tic, ne, nia, nib, te, tia, tib, tic, lref=None):
    nic = (
        calc_ninorm_from_quasineutrality(zia, zib, zic, nia, nib)
        if lref is None else
        calc_ni_from_quasineutrality(zia, zib, zi_target, nia, nib, ne)
    )
    grad_nic = (
        calc_ani_from_gradient_quasineutrality(zia, zib, zic, nia, nib, nic, grad_ne, grad_nia, grad_nib)
        if lref is None else
        calc_grad_ni_from_gradient_quasineutrality(zia, zib, zic, nia, nib, nic, grad_ne, grad_nia, grad_nib, lref)
    )
    grad_p = calc_3ion_grad_p_with_3ions(grad_ne, grad_nia, grad_nib, grad_nic, grad_te, grad_tia, grad_tib, grad_tic, ne, nia, nib, nic, te, tia, tib, tic, lref)
    return grad_p

def calc_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(grad_ne, grad_nia, grad_te, grad_tia, grad_tib, grad_tic, ne, nia, te, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, lref=None):
    norm_inputs = True if lref is not None else False
    nia_temp, nib, nic = calc_3ion_ni_from_ni_zeff_and_quasineutrality(nia, zeff, zia, zib, zic, ne, norm_inputs)
    grad_nia_temp, grad_nib, grad_nic = calc_3ion_grad_ni_from_grad_ni_grad_zeff_and_gradient_quasineutrality(grad_nia, grad_zeff, zeff, zia, zib, zic, grad_ne, nia_temp, ne, lref)
    grad_p = calc_3ion_grad_p_with_3ions(grad_ne, grad_nia, grad_nib, grad_nic, grad_te, grad_tia, grad_tib, grad_tic, ne, nia, nib, nic, te, tia, tib, tic, lref)
    return grad_p

def calc_azeff_from_3ion_grad_ni_with_3ions(grad_ne, grad_nia, grad_nib, grad_nic, ne, nia, nib, nic, zia, zib, zic, lref=None, ze=1.0):
    norm_inputs = True if lref is not None else False
    ane = copy.deepcopy(grad_ne) if norm_inputs else calc_ak_from_grad_k(grad_ne, ne, lref)
    ania = copy.deepcopy(grad_nia) if norm_inputs else calc_ak_from_grad_k(grad_nia, nia, lref)
    anib = copy.deepcopy(grad_nib) if norm_inputs else calc_ak_from_grad_k(grad_nib, nib, lref)
    anic = copy.deepcopy(grad_nic) if norm_inputs else calc_ak_from_grad_k(grad_nic, nic, lref)
    zeff = calc_zeff_from_3ion_ni_with_3ions(ne, nia, nib, nic, zia, zib, zic, norm_inputs)
    ninorma = copy.deepcopy(nia) if norm_inputs else calc_ninorm_from_ni(nia, ne)
    ninormb = copy.deepcopy(nib) if norm_inputs else calc_ninorm_from_ni(nib, ne)
    ninormc = copy.deepcopy(nic) if norm_inputs else calc_ninorm_from_ni(nic, ne)
    azeff = ane - (ninorma * ania * (zia ** 2) + ninormb * anib * (zib ** 2) + ninormc * anic * (zic ** 2)) / (ze * zeff) 
    return azeff

def calc_azeff_from_3ion_grad_ni_with_2ions_and_gradient_quasineutrality(grad_ne, grad_nia, grad_nib, ne, nia, nib, zia, zib, zic, lref=None):
    norm_inputs = True if lref is not None else False
    nic = (
        calc_ninorm_from_quasineutrality(zia, zib, zic, nia, nib)
        if norm_inputs else
        calc_ni_from_quasineutrality(zia, zib, zic, nia, nib, ne)
    )
    grad_nic = (
        calc_ani_from_gradient_quasineutrality(zia, zib, zic, nia, nib, nic, grad_ne, grad_nia, grad_nib)
        if norm_inputs else
        calc_grad_ni_from_gradient_quasineutrality(zia, zib, zic, nia, nib, nic, grad_ne, grad_nia, grad_nib, ne, lref)
    )
    azeff = calc_azeff_from_3ion_grad_ni_with_3ions(grad_ne, grad_nia, grad_nib, grad_nic, ne, nia, nib, nic, zia, zib, zic, lref)
    return azeff

def calc_ne_from_beta_and_pnorm(beta, te, bref, pnorm):
    c = constants_si()
    ne = beta * (bref ** 2) / (2.0 * c['mu'] * te * pnorm)
    return ne

def calc_te_from_beta_and_pnorm(beta, ne, bref, pnorm):
    c = constants_si()
    te = beta * (bref ** 2) / (2.0 * c['mu'] * ne * pnorm)
    return te

def calc_bo_from_beta_and_pnorm(beta, ne, te, pnorm):
    c = constants_si()
    bo = np.sqrt(2.0 * c['mu'] * ne * te * pnorm / beta)
    return bo

def calc_bo_from_beta_and_p(beta, p):
    c = constants_si()
    bo = np.sqrt(2.0 * c['mu'] * p / beta)
    return bo

def calc_beta_from_p(p, bref):
    c = constants_si()
    beta = 2.0 * c['mu'] * p / (bref ** 2)
    return beta

def calc_beta_from_pnorm(pnorm, bref, ne, te):
    c = constants_si()
    betae = 2.0 * c['mu'] * c['e'] * ne * te / (bref ** 2)
    beta = betae * pnorm
    return beta

def calc_ne_from_alpha_and_ap(alpha, q, te, bref, ap):
    c = constants_si()
    ne = alpha * bref * bref / (2.0 * c['mu'] * (q ** 2) * c['e'] * te * ap)
    return ne

def calc_te_from_alpha_and_ap(alpha, q, ne, bref, ap):
    c = constants_si()
    te = alpha * bref * bref / (2.0 * c['mu'] * (q ** 2) * c['e'] * ne * ap)
    return te

def calc_bo_from_alpha_and_ap(alpha, q, ne, te, ap):
    c = constants_si()
    bo = np.sqrt(2.0 * c['mu'] * (q ** 2) * c['e'] * ne * te * ap / alpha)
    return bo

def calc_bo_from_alpha_and_grad_p(alpha, q, lref, grad_p):
    c = constants_si()
    bo = np.sqrt(2.0 * c['mu'] * (q ** 2) * lref * -grad_p / alpha)
    return bo

def calc_alpha_from_grad_p(grad_p, q, bref, lref):
    c = constants_si()
    alpha = -2.0 * c['mu'] * (q ** 2) * lref * grad_p / (bref ** 2)
    return alpha

def calc_alpha_from_ap(ap, q, bref, ne, te):
    c = constants_si()
    betae = 2.0 * c['mu'] * c['e'] * ne * te / (bref * bref)
    alpha = q * q * betae * ap
    return alpha

def calc_alpha_from_grad_zeff(grad_zeff, zeff, zia, zib, zic, grad_ne, grad_nia, grad_te, grad_tia, grad_tib, grad_tic, ne, nia, te, tia, tib, tic, q, bref, lref):
    grad_p = calc_3ion_grad_p_with_1ion_grad_zeff_and_gradient_quasineutrality(grad_ne, grad_nia, grad_te, grad_tia, grad_tib, grad_tic, ne, nia, te, tia, tib, tic, grad_zeff, zeff, zia, zib, zic, lref)
    alpha = calc_alpha_from_grad_p(grad_p, q, bref, lref)
    return alpha

def calc_alpha_from_azeff(azeff, zeff, zia, zib, zic, ane, ania, ate, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, q, bref, ne, te):
    ap = calc_3ion_ap_with_1ion_azeff_and_gradient_quasineutrality(ane, ania, ate, atia, atib, atic, ninorma, tinorma, tinormb, tinormc, azeff, zeff, zia, zib, zic)
    alpha = calc_alpha_from_ap(ap, q, bref, ne, te)
    return alpha

def calc_ne_from_nustar(nustar, zeff, q, r, ro, te):
    c = constants_si()
    eom = c['e'] / c['me']
    tb = q * ro * ((r / ro) ** (-1.5)) / ((eom * te) ** 0.5)
    kk = (1.0e4 / 1.09) * zeff * ((te * 1.0e-3) ** (-1.5))
    nu = nustar / (tb * kk)
    data = {'te': te * 1.0e-3, 'knu': nu}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
    func_ne20 = lambda row: root_scalar(
        lambda ne: (15.2 + np.log(row['te']) - 0.5 * np.log(ne)) * ne - row['knu'],
        x0=0.01,
        x1=1.0,
        maxiter=100,
    )
    sol_ne20 = rootdata.apply(func_ne20, axis=1)
    retry = sol_ne20.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_ne20_v2 = lambda row: root_scalar(
            lambda ne: (15.2 + np.log(row['te']) - 0.5 * np.log(ne)) * ne - row['knu'],
            x0=1.0,
            x1=0.1,
            maxiter=100,
        )
        sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
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
    func_te3 = lambda row: root_scalar(
        lambda te: (15.2 - 0.5 * np.log(row['ne']) + np.log(te)) / (te ** 2) - row['knu'],
        x0=1.0,
        x1=0.1,
        maxiter=100,
    )
    sol_te3 = rootdata.apply(func_te3, axis=1)
    retry = sol_te3.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_te3_v2 = lambda row: root_scalar(
            lambda te: (15.2 - 0.5 * np.log(row['ne']) + np.log(te)) / (te ** 2) - row['knu'],
            x0=0.01,
            x1=0.1,
            maxiter=100,
        )
        sol_te3.loc[retry] = rootdata.loc[retry].apply(func_te3_v2, axis=1)
    te = sol_te3.apply(lambda sol: 1.0e3 * sol.root).to_numpy()
    logger.debug(f'<{calc_te_from_nustar.__name__}>: data')
    logger.debug(pd.DataFrame(data={'nustar': nustar, 'ne': ne, 'te': te}))
    return te

def calc_zeff_from_nustar(nustar, q, r, ro, ne, te):
    c = constants_si()
    nt = (15.2 - 0.5 * np.log(ne * 1.0e-20) + np.log(te * 1.0e-3)) * (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    kk = (1.0e4 / 1.09) * q * ro * ((r / ro) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    zeff = nustar / (nt * kk)
    return zeff

def calc_nustar(zeff, q, r, ro, ne, te):
    c = constants_si()
    nt = (15.2 - 0.5 * np.log(ne * 1.0e-20) + np.log(te * 1.0e-3)) * zeff * (ne * 1.0e-20) / ((te * 1.0e-3) ** 2)
    kk = (1.0e4 / 1.09) * q * ro * ((r / ro) ** (-1.5)) * ((1.0e-3 * c['me'] / c['e']) ** 0.5)
    nustar = nt * kk
    return nustar

def calc_lognustar_from_nustar(nustar):
    lognustar = np.log10(nustar)
    return lognustar

def calc_nustar_from_lognustar(lognustar):
    nustar = np.power(10.0, lognustar)
    return nustar

def calc_bo_from_rhostar(rhostar, ai, zi, a, te):
    c = constants_si()
    bo = (((ai * c['mp'] / c['e']) ** 0.5) / zi) * (te ** 0.5) / (rhostar * a)
    return bo

def calc_te_from_rhostar(rhostar, ai, zi, a, bo):
    c = constants_si()
    te = ((zi * rhostar * a * bo) ** 2) / (ai * c['mp'] / c['e'])
    return te

def calc_rhostar(ai, zi, a, te, bo):
    c = constants_si()
    rhostar = ((ai * c['mp'] / c['ee']) ** 0.5 / zi) * (te ** 0.5) / (bo * a)
    return rhostar

def calc_ne_from_alpha_and_rhostar(alpha, rhostar, ai, zi, q, a, ap):
    c = constants_si()
    mi = ai * c['mp']
    qi = zi * c['e']
    prefactor = mi / (2.0 * c['mu'] * (qi ** 2) * (q ** 2))
    ne = prefactor * alpha / ((rhostar ** 2) * (a ** 2) * ap)
    return ne

def calc_bo_from_alpha_and_rhostar(alpha, rhostar, ai, zi, q, a, ne, te, ap):
    c = constants_si()
    mi = ai * c['mp']
    qi = zi * c['e']
    prefactor = 2.0 * c['mu'] * (mi ** 0.5) / qi
    bo = (prefactor * (q ** 2) * ne * ((c['e'] * te), 1.5) * ap / (alpha * rhostar * a)) ** (1.0 / 3.0)
    return bo

def calc_ne_from_beta_and_rhostar(beta, rhostar, ai, zi, q, a, pnorm):
    c = constants_si()
    mi = ai * c['mp']
    qi = zi * c['e']
    prefactor = mi / (2.0 * c['mu'] * (qi ** 2))
    ne = prefactor * beta / ((rhostar ** 2) * (a ** 2) * pnorm)
    return ne

def calc_bo_from_beta_and_rhostar(beta, rhostar, ai, zi, q, a, ne, te, pnorm):
    c = constants_si()
    mi = ai * c['mp']
    qi = zi * c['e']
    prefactor = 2.0 * c['mu'] * (mi ** 0.5) / qi
    bo = (prefactor * ne * ((c['e'] * te) ** 1.5) * pnorm / (beta * rhostar * a)) ** (1.0 / 3.0)
    return bo

def calc_ne_from_nustar_alpha_and_ap(nustar, alpha, zeff, q, r, ro, bo, ap):
    c = constants_si()
    kalp = 1.0e23 * 2.0 * c['mu'] * c['e'] * (q ** 2) * ap / (alpha * (bo ** 2))
    knu = (1.0e4 / 1.09) * ((1.0e-3 * c['me'] / c['e']) ** 0.5) * zeff * q * ro * ((r / ro) ** (-1.5))
    data = {'logterm': -np.log(kalp), 'constant': nustar / (knu * kalp * kalp)}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
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
    ne = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    logger.debug(f'<{calc_ne_from_nustar_alpha_and_ap.__name__}>: data')
    logger.debug(pd.DataFrame(data={'nustar': nustar, 'alpha': alpha, 'ap': ap, 'ne': ne}))
    return ne

def calc_te_from_nustar_alpha_and_ap(nustar, alpha, zeff, q, r, ro, bo, ap, verbose=0):
    c = constants_si()
    kalp = 1.0e23 * 2.0 * c['mu'] * c['e'] * (q ** 2) * ap / (alpha * (bo ** 2))
    knu = (1.0e4 / 1.09) * ((1.0e-3 * c['me'] / c['e']) ** 0.5) * zeff * q * ro * ((r / ro) ** (-1.5))
    data = {'logterm': -np.log(kalp), 'constant': nustar * kalp / knu}
    rootdata = pd.DataFrame(data)
    logger.debug(rootdata)
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
    te = sol_te3.apply(lambda sol: 1.0e3 * sol.root).to_numpy()
    logger.debug(f'<{calc_te_from_nustar_alpha_and_ap.__name__}>: data')
    logger.debug(pd.DataFrame(data={'nustar': nustar, 'alpha': alpha, 'ap': ap, 'te': te}))
    return te

def calc_machpar_from_machtor_and_puretor(machtor, q, epsilon, x):
    btorbyb = ((q ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    machpar = machtor * btorbyb
    return machpar

def calc_aupar_from_autor_and_puretor(autor, machtor, s, q, epsilon, x):
    btorbyb = ((q ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    bpolbyb = (((epsilon * x) ** 2) / ((q ** 2) + ((epsilon * x) ** 2))) ** 0.5
    grad_btorbyb = btorbyb * (bpolbyb ** 2) * (s - 1.0) / (epsilon * x)
    aupar = autor * btorbyb - machtor * grad_btorbyb
    return aupar

def calc_gammae_from_aupar_without_grad_dpi(aupar, q, epsilon):
    gammae = -(epsilon / q) * aupar
    return gammae

def calc_grad_dpi_from_gammae_machtor_and_autor(gammae, machtor, autor, q, r, ro):
    return None

def calc_bunit_from_bo(bo, sfac):
    bunit = normalize(bo, sfac)
    return bunit

def calc_bo_from_bunit(bunit, sfac):
    bo = unnormalize(bunit, sfac)
    return bo

def calc_rhos_from_ts_ms_and_b(ts, ms, b):
    c = constants_si()
    rhos = (ts * c['u'] * ms / c['e']) ** 0.5 / b
    return rhos

def calc_vsound_from_te_and_mref(te, mref):
    c = constants_si()
    vsound = (c['e'] * te / (c['u'] * mref)) ** 0.5
    return vsound

def calc_vtherms_from_ts_and_ms(ts, ms):
    c = constants_si()
    vths = (2.0 * c['e'] * ts / (c['u'] * ms)) ** 0.5
    return vths

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

def calc_nu_from_t_and_n(ta, na, nb, ma, za, zb):
    c = constants_si()
    lda = calc_lds_from_ts_and_ns(ta, na, za)
    factor = 0.5 * np.pi * (2.0 * np.pi) ** 0.5
    inv_b90 = (4.0 * np.pi * c['eps'] / c['e']) * ta / (za * zb)
    nu = factor * nb * (c['e'] * ta / (c['u'] * ma)) ** 0.5 / (inv_b90 ** 2) * np.log(inv_b90 * lda)
    return nu

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
    func_ne20 = lambda row: root_scalar(
        lambda ne: (row['logterm'] - 0.5 * np.log(1.0e20 * ne)) * 1.0e20 * ne - row['constant'],
        x0=0.01,
        x1=1.0,
        maxiter=100,
    )
    sol_ne20 = rootdata.apply(func_ne20, axis=1)
    retry = sol_ne20.apply(lambda sol: not sol.converged)
    if np.any(retry):
        func_ne20_v2 = lambda row: root_scalar(
            lambda ne: (row['logterm'] - 0.5 * np.log(1.0e20 * ne)) * 1.0e20 * ne - row['constant'],
            x0=1.0,
            x1=0.1,
            maxiter=100,
        )
        sol_ne20.loc[retry] = rootdata.loc[retry].apply(func_ne20_v2, axis=1)
    ne = sol_ne20.apply(lambda sol: 1.0e20 * sol.root).to_numpy()
    return ne

def calc_bunit_from_ldenorm(ldenorm, ne, mref, ze=1.0):
    c = constants_si()
    bunit = (ne * (ze ** 2) * c['u'] * mref / c['eps']) ** 0.5 * ldenorm
    return bunit