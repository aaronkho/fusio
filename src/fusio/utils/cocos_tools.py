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

