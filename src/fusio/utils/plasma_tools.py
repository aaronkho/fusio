import copy
import numpy as np

np_itypes = (np.int8, np.int16, np.int32, np.int64)
np_utypes = (np.uint8, np.uint16, np.uint32, np.uint64)
np_ftypes = (np.float16, np.float32, np.float64)

number_types = (float, int, np_itypes, np_utypes, np_ftypes)
array_types = (list, tuple, np.ndarray)
string_types = (str, np.str_)


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

