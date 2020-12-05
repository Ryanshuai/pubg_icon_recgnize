import cv2
import numpy as np
import random


def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


def generate_icon_foreground(icon_4c):
    h0, w0, c = icon_4c.shape
    icon_4c = cv2.copyMakeBorder(icon_4c, h0, h0, w0, w0, cv2.BORDER_CONSTANT, value=0)

    size_factor = random.uniform(0.9, 1.11)
    icon_4c = cv2.resize(icon_4c, (0, 0), fx=size_factor, fy=size_factor)
    x, y = random.randint(-3, 3), random.randint(-3, 3)
    icon_4c = translate(icon_4c, x, y)
    h, w, c = icon_4c.shape

    dh, dw = int(abs(h - h0) // 2), int(abs(w - w0) // 2)
    icon_4c = icon_4c[dh:h0 + dh, dw:w0 + dw, :]
    return icon_4c[:, :, :3], icon_4c[:, :, [3]]


class Generater:
    def __init__(self, in_icon_dir):
        input_background_dir = "1screens/background"
        # input_background_dir = "1screens/outside"
        background_paths = os.listdir(input_background_dir)
        self.background_screens = [cv2.imread(os.path.join(input_background_dir, path)) for path in background_paths]

        icon_paths = os.listdir(in_icon_dir)
        self.icons = [cv2.imread(os.path.join(in_icon_dir, path), cv2.IMREAD_UNCHANGED) for path in icon_paths]
        self.h, self.w, c = self.icons[0].shape

    def generate_background(self):
        idx = random.randint(0, len(self.background_screens) - 1)
        screen = self.background_screens[idx][100:1400, 610:]
        y = random.randint(0, screen.shape[0] - self.h - 1)
        x = random.randint(0, screen.shape[1] - self.w - 1)
        icon_background = screen[y:y + self.h, x:x + self.w]
        return icon_background

    def generate_icon(self):
        background = self.generate_background()

        idx = random.randint(0, len(self.icons) - 1)
        icon_4c = self.icons[idx]
        icon, mask = generate_icon_foreground(icon_4c)
        img_mix = icon + background * (1 - mask // 255)
        # cv2.imshow("img_mix", img_mix)
        # cv2.waitKey()
        return img_mix


if __name__ == '__main__':
    import os


    def generate_one_dateset(file_name, item_names, train_set_num=500, in_root="2foreground"):
        for item_name in item_names:
            in_dir = os.path.join(in_root, item_name)
            gen = Generater(in_dir)
            out_dir = os.path.join(file_name, "train", item_name)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(train_set_num):
                im = gen.generate_icon()
                cv2.imwrite(os.path.join(out_dir, str(i) + ".png"), im)

            out_dir = os.path.join(file_name, "val", item_name)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(100):
                im = gen.generate_icon()
                cv2.imwrite(os.path.join(out_dir, str(i) + ".png"), im)

        out_dir = os.path.join(file_name, "train", "0background")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(train_set_num):
            im = gen.generate_background()
            cv2.imwrite(os.path.join(out_dir, str(i) + ".png"), im)

        out_dir = os.path.join(file_name, "val", "0background")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(100):
            im = gen.generate_background()
            cv2.imwrite(os.path.join(out_dir, str(i) + ".png"), im)


    pos_name_dict = {
        'gun_scope': ['x1h', 'x1r', 'x2', 'x3', 'x4', 'x6', 'x8', 'x15', ],
        'gun_muzzle': ['m_com_ar', 'm_com_sm', 'm_fla_ar', 'm_fla_sm', 'm_sup_ar'],
        'gun_grip': ['g_ang', 'g_hal', 'g_las', 'g_lig', 'g_thu', 'g_ver'],
        'gun_butt': ['sto'],

        'gun_name': ['m416', 'scar', 'g36c', 'qbz', 'm249', 'aug', 'm762', 'akm', 'mk14', 'groza', 'uzi', 'tommy',
                     'vss', 'pp19', 'ump45', 'vector', 'mk47', 'slr', 's1897', 'mini14', 'awm', 's686', 'win94', 'dbs',
                     'm24', '98k', 'qbu', 'sks', 'mp5k', 's12k', 'dp28', 'm16', ],
        'fire_mode': ["single", "burst2", "burst3", "full"],
        'in_tab': ["in_tab"],
    }

    # for pos_name, name_list in pos_name_dict.items():
    #     generate_one_dateset(pos_name), name_list)

    generate_one_dateset('dataSets/gun_butt', ['sto', ], train_set_num=500)
    # generate_one_dateset('in_tab', ['in_tab', ],train_set_num=2500)
    # generate_one_dateset('gun_scope', ['x1h', 'x1r', 'x2', 'x3', 'x4', 'x6', 'x8', 'x15', ])
    # generate_one_dateset('gun_grip', ['g_ang', 'g_hal', 'g_las', 'g_lig', 'g_thu', 'g_ver'])
    # generate_one_dateset('gun_name',['m416', 'scar', 'g36c', 'qbz', 'm249', 'aug', 'm762', 'akm', 'mk14', 'groza', 'uzi', 'tommy',
    #                  'vss', 'pp19', 'ump45', 'vector', 'mk47', 'slr', 's1897', 'mini14', 'awm', 's686', 'win94', 'dbs',
    #                  'm24', '98k', 'qbu', 'sks', 'mp5k', 's12k', 'dp28', 'm16', ])
