import logging
import copy
from pathlib import Path
from ..classes.io import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
from numpy.typing import ArrayLike, NDArray
import numpy as np
from scipy.integrate import quad  # type: ignore[import-untyped]
import contourpy
from shapely import Point, Polygon  # type: ignore[import-untyped]

logger = logging.getLogger('fusio')


def define_cocos(cocos_number: int) -> MutableMapping[str, int]:
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


def define_cocos_converter(cocos_in: int, cocos_out: int) -> MutableMapping[str, int]:
    in_dict = define_cocos(cocos_in)
    out_dict = define_cocos(cocos_out)
    for key in out_dict:
        if key == 'eBp':
            out_dict[key] -= in_dict[key]
        else:
            out_dict[key] *= in_dict[key]
    return out_dict


def determine_cocos(sign_dict: MutableMapping[str, int]) -> int:
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


def detect_cocos(eqdsk: MutableMapping[str, Any]) -> int:
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


def convert_cocos(eqdsk: MutableMapping[str, Any], cocos_in: int, cocos_out: int, bt_sign_out: int | None = None, ip_sign_out: int | None = None) -> MutableMapping[str, Any]:
    out = {
        'nr': eqdsk.get('nr', None),
        'nz': eqdsk.get('nz', None),
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
            sBt *= int(np.sign(out['bcentr']) * np.sign(bt_sign_out))
            out['bcentr'] *= np.sign(out['bcentr']) * np.sign(bt_sign_out)
    if 'cpasma' in eqdsk:
        out['cpasma'] = copy.deepcopy(eqdsk['cpasma']) * sIp
        if ip_sign_out is not None:
            sIp *= int(np.sign(out['cpasma']) * np.sign(ip_sign_out))
            out['cpasma'] *= np.sign(out['cpasma']) * np.sign(ip_sign_out)
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
        out['nlim'] = copy.deepcopy(eqdsk['nlim'])
        out['rlim'] = copy.deepcopy(eqdsk['rlim'])
        out['zlim'] = copy.deepcopy(eqdsk['zlim'])
    if 'rbdry' in eqdsk and 'zbdry' in eqdsk:
        out['nbdry'] = copy.deepcopy(eqdsk['nbdry'])
        out['rbdry'] = copy.deepcopy(eqdsk['rbdry'])
        out['zbdry'] = copy.deepcopy(eqdsk['zbdry'])
    return out


def trace_flux_surfaces(r: NDArray, z: NDArray, psi: NDArray, levels: NDArray, axis: Sequence[float] | None = None) -> MutableMapping[float, Any]:
    check = Point(axis) if isinstance(axis, (list, tuple, np.ndarray)) else Point(np.mean(r), np.mean(z))
    cg_psi = contourpy.contour_generator(r, z, psi)
    contours = {}
    for level in levels:
        vertices = cg_psi.create_contour(level)
        for i in range(len(vertices)):
            if vertices[i] is not None:
                polygon = Polygon(np.array(vertices[i]))
                if polygon.contains(check):
                    contours[float(level)] = vertices[i].copy()
                    break
    return contours


def calculate_mxh_coefficients(r: NDArray, z: NDArray, n: int = 5) -> Sequence[Sequence[float]]:

    z = np.roll(z, -np.argmax(r))
    r = np.roll(r, -np.argmax(r))
    if z[1] < z[0]: # reverses array so that theta increases
        z = np.flip(z)
        r = np.flip(r)

    # compute bounding box for each flux surface
    rmin = 0.5 * (np.nanmax(r) - np.nanmin(r))
    kappa = 0.5 * (np.nanmax(z) - np.nanmin(z)) / rmin
    r0 = 0.5 * (np.nanmax(r) + np.nanmin(r))
    z0 = 0.5 * (np.max(z) + np.min(z))
    bbox = [r0, rmin, z0, kappa]

    # solve for polar angles
    # need to use np.clip to avoid floating-point precision errors
    theta_r = np.arccos(np.clip(((r - r0) / rmin), -1, 1))
    theta = np.arcsin(np.clip(((z - z0) / rmin / kappa), -1, 1))

    # Find the continuation of theta and theta_r to [0,2pi]
    theta_r_cont = np.copy(theta_r)
    theta_cont = np.copy(theta)

    max_theta = np.argmax(theta)
    min_theta = np.argmin(theta)
    max_theta_r = np.argmax(theta_r)
    min_theta_r = np.argmin(theta_r)

    theta_cont[:max_theta] = theta_cont[:max_theta]
    theta_cont[max_theta:max_theta_r] = np.pi - theta[max_theta:max_theta_r]
    theta_cont[max_theta_r:min_theta] = np.pi - theta[max_theta_r:min_theta]
    theta_cont[min_theta:] = 2.0 * np.pi + theta[min_theta:]

    theta_r_cont[:max_theta] = theta_r_cont[:max_theta]
    theta_r_cont[max_theta:max_theta_r] = theta_r[max_theta:max_theta_r]
    theta_r_cont[max_theta_r:min_theta] = 2.0 * np.pi - theta_r[max_theta_r:min_theta]
    theta_r_cont[min_theta:] = 2.0 * np.pi - theta_r[min_theta:]

    theta_r_cont = theta_r_cont - theta_cont
    theta_r_cont[-1] = theta_r_cont[0]

    # Fourier decompose to find coefficients
    c = [0.0] * (n + 1)
    s = [0.0] * (n + 1)

    def f_theta_r(theta):
        return np.interp(theta, theta_cont, theta_r_cont)

    for i in range(n + 1):
        s[i] = quad(f_theta_r, 0, 2.0 * np.pi, weight='sin', wvar=i)[0] / np.pi
        c[i] = quad(f_theta_r, 0, 2.0 * np.pi, weight='cos', wvar=i)[0] / np.pi

    c[0] /= 2

    return c, s, bbox


def read_eqdsk(path: str | Path) -> MutableMapping[str, Any]:
    ''' Read an eqdsk file '''

    def _sep_eq_line(line, float_width=16, floats_per_line=5, sep=' '):
        ''' Split a eqdsk-style line and inserts seperator characters '''
        splitted = [line[num*float_width:(num+1)*float_width] for num in range(floats_per_line)]
        separate = sep.join(splitted)
        return separate

    def _read_chunk(lines, length, floats_per_line=5):
        num_lines = int(np.ceil(length / floats_per_line))
        vals = []
        for line in lines[:num_lines]:
            sep = _sep_eq_line(line)
            vals.append(np.fromstring(sep, sep=' '))
        del lines[:num_lines]
        return vals

    lines = []
    if isinstance(path, (str, Path)):
        geqdsk_path = Path(path)
        if geqdsk_path.is_file():
            with open(geqdsk_path, 'r') as ff:
                lines = ff.readlines()

    data: MutableMapping[str, Any] = {}
    if len(lines) > 0:

        data['case'] = lines[0][:48].strip()
        header = lines.pop(0)[48:].split()

        # Read sizes of arrays/vectors
        data['idum'] = int(header[0])
        data['nr'] = int(header[1])
        data['nz'] = int(header[2])

        # Read singles
        data['rdim'], data['zdim'], data['rcentr'], data['rleft'], data['zmid'] = np.fromstring(_sep_eq_line(lines.pop(0)), sep=' ')
        data['rmagx'], data['zmagx'], data['simagx'], data['sibdry'], data['bcentr'] = np.fromstring(_sep_eq_line(lines.pop(0)), sep=' ')
        data['cpasma'], data['simagx2'], _, data['rmagx2'], _ = np.fromstring(_sep_eq_line(lines.pop(0)), sep=' ')
        data['zmagx2'], _, data['sibdry2'], _, _ = np.fromstring(_sep_eq_line(lines.pop(0)), sep=' ')

        # Check if duplicate fields are equal
        for base in ['simagx', 'sibdry', 'rmagx', 'zmagx']:
            if not data[base] == data.pop(base + '2'):
                raise Exception("Dual values for '{!s}' not equal!".format(base))

        # Read 1D array blocks
        for name in ['fpol', 'pres', 'ffprime', 'pprime']:
            data[name] = np.concatenate(_read_chunk(lines, data['nr']))

        # Read psi map
        data['psi'] = np.concatenate(_read_chunk(lines, data['nr'] * data['nz']))
        data['psi'] = data['psi'].reshape((data['nz'], data['nr']))

        # Read q-profile
        data['qpsi'] = np.concatenate(_read_chunk(lines, data['nr']))

        # Read sizes of boundary vector and limiter vector
        cheader = lines.pop(0)
        data['nbdry'] = int(cheader[:5])
        data['nlim'] = int(cheader[5:])

        # Read boundary vector
        if data['nbdry'] > 0:
            boundary = _read_chunk(lines, data['nbdry'] * 2)
            boundary = np.concatenate(boundary).reshape((data['nbdry'], 2))
            data['rbdry'] = boundary[:, 0]
            data['zbdry'] = boundary[:, 1]
        else:
            data['rbdry'] = None
            data['zbdry'] = None

        # Read limiter vector
        if data['nlim'] > 0:
            limiter = _read_chunk(lines, data['nlim'] * 2)
            limiter = np.concatenate(limiter).reshape((data['nlim'], 2))
            data['rlim'] = limiter[:, 0]
            data['zlim'] = limiter[:, 1]
        else:
            data['rlim'] = None
            data['zlim'] = None

    return data


def write_eqdsk(data: MutableMapping[str, Any], path: str | Path) -> None:

    if isinstance(path, (str, Path)) and isinstance(data, dict):
        geqdsk_path = Path(path)
        if geqdsk_path.exists():
            print(f'{geqdsk_path} exists, overwriting file with EQDSK file!')
        geqdsk_path.parent.mkdir(parents=True, exist_ok=True)

        gcase = data['case'] if 'case' in data else ''
        if len(gcase) > 48:
            gcase = gcase[:48]
        idum = data['idum'] if 'idum' in data else 0

        # Write sizes of arrays/vectors
        dstr = '%-48s%4d%4d%4d\n' % (gcase, idum, data['nr'], data['nz'])

        # Write singles
        dstr += '%16.9E%16.9E%16.9E%16.9E%16.9E\n' % (data['rdim'], data['zdim'], data['rcentr'], data['rleft'], data['zmid'])
        dstr += '%16.9E%16.9E%16.9E%16.9E%16.9E\n' % (data['rmagx'], data['zmagx'], data['simagx'], data['sibdry'], data['bcentr'])
        dstr += '%16.9E%16.9E%16.9E%16.9E%16.9E\n' % (data['cpasma'], data['simagx'], 0.0, data['rmagx'], 0.0)
        dstr += '%16.9E%16.9E%16.9E%16.9E%16.9E\n' % (data['zmagx'], 0.0, data['sibdry'], 0.0, 0.0)

        # Write 1D array blocks
        for name in ['fpol', 'pres', 'ffprime', 'pprime']:
            for ii in range(data['nr']):
                dstr += '%16.9E' % (data[name][ii])
                if (ii + 1) % 5 == 0 and (ii + 1) != len(data[name]):
                    dstr += '\n'
            dstr += '\n'

        # Write psi map
        kk = 0
        for ii in range(data['nz']):
            for jj in range(data['nr']):
                dstr += '%16.9E' % (data['psi'][ii, jj])
                if (kk + 1) % 5 == 0 and (kk + 1) != data['nr'] * data['nz']:
                    dstr += '\n'
                kk = kk + 1
        dstr += '\n'

        # Read q-profile
        for ii in range(len(data['qpsi'])):
            dstr += '%16.9E' % (data['qpsi'][ii])
            if (ii + 1) % 5 == 0 and (ii + 1) != len(data['qpsi']):
                dstr += '\n'
        dstr += '\n'

        nbdry = data.get('nbdry')
        rbdry = data.get('rbdry')
        zbdry = data.get('zbdry')
        if nbdry is None or rbdry is None or zbdry is None:
            nbdry = 0
            rbdry = []
            zbdry = []
        nlim = data.get('nlim')
        rlim = data.get('rlim')
        zlim = data.get('zlim')
        if nlim is None or rlim is None or zlim is None:
            nlim = 0
            rlim = []
            zlim = []

        dstr += '%5d%5d\n' % (nbdry, nlim)
        kk = 0
        for ii in range(nbdry):
            dstr += '%16.9E' % (rbdry[ii])
            if (kk + 1) % 5 == 0 and (ii + 1) != nbdry:
                dstr += '\n'
            kk = kk + 1
            dstr += '%16.9E' % (zbdry[ii])
            if (kk + 1) % 5 == 0 and (ii + 1) != nbdry:
                dstr += '\n'
            kk = kk + 1
        dstr += '\n'
        kk = 0
        for ii in range(nlim):
            dstr += '%16.9E' % (rlim[ii])
            if (kk + 1) % 5 == 0 and (kk + 1) != nlim:
                dstr += '\n'
            kk = kk + 1
            dstr += '%16.9E' % (zlim[ii])
            if (kk + 1) % 5 == 0 and (kk + 1) != nlim:
                dstr += '\n'
            kk = kk + 1
        dstr += '\n'

        with open(geqdsk_path, 'w') as ff:
            ff.write(dstr)

        print(f'Output EQDSK file saved as {geqdsk_path}!')

