import copy
import numpy as np

def define_cocos(cocos_number):
    # Default dictionary returns COCOS=1
    sign_dict = {
        'eBp': 0,   # Normalization of flux by 2 pi
        'sBp': 1,   # Increasing or decreasing flux from axis
        'scyl': 1,  # Handedness of cylindrical coordinates
        'spol': 1,  # Handedness of poloidal coordinates
        'srel': 1,  # Swapping of handedness between cylindrical and poloidal
    }
    if cocos_number < 0:
        cocos_number = -cocos_number
        sign_dict['srel'] = -1
    if cocos_number > 10:
        cocos_number -= 10
        sign_dict['eBp'] = 1
    if cocos_number in [3, 4, 7, 8]:
        sign_dict['sBp'] = -1
    if cocos_number in [2, 4, 6, 8]:
        sign_dict['scyl'] = -1
    if cocos_number in [3, 4, 5, 6]:
        sign_dict['spol'] = -1
    return sign_dict


def define_cocos_converter(cocos_in, cocos_out):
    in_dict = define_cocos(cocos_in)
    out_dict = define_cocos(cocos_out)
    for key in out_dict:
        if key == 'eBp':
            out_dict[key] -= in_dict[key]
        else:
            out_dict[key] *= in_dict[key]
    return out_dict


def determine_cocos(sign_dict):
    cocos_number = 0  # Signifies unknown
    fcomplete = True
    for var in ['eBp', 'sBp', 'scyl', 'spol', 'srel']:
        if var not in sign_dict:
            fcomplete = False
    if fcomplete:
        cocos_number = 1
        if sign_dict['sBp'] * sign_dict['spol'] < 0:
            cocos_number += 4
        if sign_dict['sBp'] < 0:
            cocos_number += 2
        if sign_dict['scyl'] == 0:
            print('Ambiguous cylindrical direction, assuming ccw from top')
        elif sign_dict['scyl'] < 0:
            cocos_number += 1
        if sign_dict['eBp'] < 0:
            print('Ambiguous per radian specification, assuming not per radian')
        elif sign_dict['eBp'] > 0:
            cocos_number += 10
        if sign_dict['srel'] == 0:
            print('Ambiguous relative coordinate handedness, assuming all right-handed')
        if sign_dict['srel'] < 0:
            cocos_number = -cocos_number
    return cocos_number


def detect_cocos(eqdsk):
    sign_dict = {}
    sIp = int(np.sign(eqdsk['cpasma'])) if 'cpasma' in eqdsk else 0
    sBt = int(np.sign(eqdsk['bcentr'])) if 'bcentr' in eqdsk else 0
    if sIp != 0 and sBt != 0:
        sign_dict['scyl'] = 0
        sign_dict['eBp'] = -1
        sign_dict['srel'] = 0
        if 'sibdry' in eqdsk and 'simagx' in eqdsk:
            sign_dict['sBp'] = int(np.sign(eqdsk['sibdry'] - eqdsk['simagx'])) * sIp
        if 'qpsi' in eqdsk:
            sign_dict['spol'] = int(np.sign(eqdsk['qpsi'][-1])) * sIp * sBt
    return determine_cocos(sign_dict)


def convert_cocos(eqdsk, cocos_in, cocos_out, bt_sign_out=None, ip_sign_out=None):
    out = {
        'nx': eqdsk.get('nx', None),
        'ny': eqdsk.get('ny', None),
        'rdim': eqdsk.get('rdim', None),
        'zdim': eqdsk.get('zdim', None),
        'rcentr': eqdsk.get('rcentr', None),
        'bcentr': eqdsk.get('bcentr', None),
        'rleft': eqdsk.get('rleft', None),
        'zmid': eqdsk.get('zmid', None),
        'rmagx': eqdsk.get('rmagx', None),
        'zmagx': eqdsk.get('zmagx', None),
        'cpasma': eqdsk.get('cpasma', None),
    }
    sign_dict = define_cocos_converter(cocos_in, cocos_out)
    sIp = sign_dict['scyl']
    sBt = sign_dict['scyl']
    if 'bcentr' in eqdsk:
        out['bcentr'] = copy.deepcopy(eqdsk['bcentr']) * sBt
        if bt_sign_out is not None:
            out['bcentr'] *= np.sign(out['bcentr']) * np.sign(bt_sign_out)
            sBt *= int(np.sign(out['bcentr']) * np.sign(bt_sign_out))
    if 'cpasma' in eqdsk:
        out['cpasma'] = copy.deepcopy(eqdsk['cpasma']) * sIp
        if ip_sign_out is not None:
            out['cpasma'] *= np.sign(out['cpasma']) * np.sign(ip_sign_out)
            sIp *= int(np.sign(out['cpasma']) * np.sign(ip_sign_out))
    if 'simagx' in eqdsk:
        out['simagx'] = copy.deepcopy(eqdsk['simagx']) * np.power(2.0 * np.pi, sign_dict['eBp']) * sign_dict['sBp'] * sIp
    if 'sibdry' in eqdsk:
        out['sibdry'] = copy.deepcopy(eqdsk['sibdry']) * np.power(2.0 * np.pi, sign_dict['eBp']) * sign_dict['sBp'] * sIp
    if 'fpol' in eqdsk:
        out['fpol'] = copy.deepcopy(eqdsk['fpol']) * sBt
    if 'pres' in eqdsk:
        out['pres'] = copy.deepcopy(eqdsk['pres'])
    if 'ffprime' in eqdsk:
        out['ffprime'] = copy.deepcopy(eqdsk['ffprime']) * np.power(2.0 * np.pi, -sign_dict['eBp']) * sign_dict['sBp'] * sIp
    if 'pprime' in eqdsk:
        out['pprime'] = copy.deepcopy(eqdsk['pprime']) * np.power(2.0 * np.pi, -sign_dict['eBp']) * sign_dict['sBp'] * sIp
    if 'psi' in eqdsk:
        out['psi'] = copy.deepcopy(eqdsk['psi']) * np.power(2.0 * np.pi, sign_dict['eBp']) * sign_dict['sBp'] * sIp
    if 'qpsi' in eqdsk:
        out['qpsi'] = copy.deepcopy(eqdsk['qpsi']) * sign_dict['spol'] * sIp * sBt
    if 'rlim' in eqdsk and 'zlim' in eqdsk:
        out['rlim'] = copy.deepcopy(eqdsk['rlim'])
        out['zlim'] = copy.deepcopy(eqdsk['zlim'])
    if 'rbdry' in eqdsk and 'zbdry' in eqdsk:
        out['rbdry'] = copy.deepcopy(eqdsk['rbdry'])
        out['zbdry'] = copy.deepcopy(eqdsk['zbdry'])
    return out

