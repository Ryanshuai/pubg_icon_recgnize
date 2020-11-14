# [y1, x1, y2, x2]

def get_pos(name):
    name_tuple = crop_position[name]
    if len(name_tuple) == 2:
        y, x = name_tuple
        return y, x, 64, 64
    y1, x1, y2, x2 = name_tuple
    return y1, x1, y2 - y1, x2 - x1


crop_position = {
    'fire_mode': [1330, 1648, 1364, 1676],
    'in_tab': [132, 926, 155, 974],
    'posture': [1005, 705, 1048, 743],
    # 'in_scope': [1669, 1179, 1766, 1208],

    'gun1_name': [130, 2245, 165, 2250 + 64],
    'gun1_scope': [152, 2554],
    'gun1_muzzle': [330, 2191],
    'gun1_grip': [330, 2327],
    # 'gun1_magazine': [330, 2474],
    'gun1_butt': [330, 2756],

    'gun2_name': [436, 2250],
    'gun2_scope': [458, 2554],
    'gun2_muzzle': [636, 2191],
    'gun2_grip': [636, 2327],
    # 'gun2_magazine': [636, 2474],
    'gun2_butt': [636, 2756]
}
