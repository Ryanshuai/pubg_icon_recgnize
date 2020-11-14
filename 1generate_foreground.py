import os
import cv2
import numpy as np
import random
from positions import get_pos


def get_white_shield(im, min_rgb=235):
    shield_rgb = np.where(im > min_rgb, 1, 0)
    shield = shield_rgb[:, :, 0] & shield_rgb[:, :, 1] & shield_rgb[:, :, 2]
    return shield[:, :, np.newaxis].astype(np.uint8)


def get_icon(name, screen, white_thr):
    y, x, h, w = get_pos(name)
    icon = screen[y:y + h, x:x + w]
    shield = get_white_shield(icon, white_thr)
    icon *= shield
    return np.concatenate((icon, shield * 255), axis=-1)


if __name__ == '__main__':
    def generate_icon(pos_name, names, white_thr, input_root, output_root):
        for name in names:
            input_dir = os.path.join(input_root, name)
            output_dir = os.path.join(output_root, name)
            os.makedirs(output_dir, exist_ok=True)
            for file in os.listdir(input_dir):
                screen = cv2.imread(os.path.join(input_dir, file))
                icon = get_icon(pos_name, screen, white_thr)
                cv2.imwrite(os.path.join(output_dir, file), icon)


    gun1_name = ['m416', 'scar', 'g36c', 'qbz', 'm249', 'aug', 'm762', 'akm', 'mk14', 'groza', 'uzi', 'tommy',
                 'vss', 'pp19', 'ump45', 'vector', 'mk47', 'slr', 's1897', 'mini14', 'awm', 's686', 'win94', 'dbs',
                 'm24', '98k', 'qbu', 'sks', 'mp5k', 's12k', 'dp28', 'm16', ]
    fire_mode = ["single", "burst2", "burst3", "full"]
    in_tab = ["in_tab"]

    input_root = "1screens"
    output_root = "2foreground"

    generate_icon('gun1_name', gun1_name, 240, input_root, output_root)
    generate_icon('fire_mode', fire_mode, 230, input_root, output_root)
    generate_icon('in_tab', in_tab, 235, input_root, output_root)
