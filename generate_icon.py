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
    def __init__(self, icon_dir):
        print(icon_dir)
        os.makedirs(icon_dir, exist_ok=True)
        input_background_dir = "original_data/background_screen"
        background_paths = os.listdir(input_background_dir)
        self.background_screens = [cv2.imread(os.path.join(input_background_dir, path)) for path in background_paths]

        icon_paths = os.listdir(icon_dir)
        self.icons = [cv2.imread(os.path.join(icon_dir, path), cv2.IMREAD_UNCHANGED) for path in icon_paths]
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
        img_mix = np.clip(img_mix, 0, 255)
        # cv2.imshow("img_mix", img_mix)
        # cv2.waitKey()
        return img_mix


if __name__ == '__main__':
    import os

    output_root_dir = "pytorch_dataset/train"
    os.makedirs(output_root_dir, exist_ok=True)
    input_gun_icon_dir = "original_data/train_gun_icon"
    for name_dir in os.listdir(input_gun_icon_dir):
        icon_dir = os.path.join(input_gun_icon_dir, name_dir)
        gen = Generater(icon_dir)

        for i in range(500):
            im = gen.generate_icon()
            output_dir = os.path.join(output_root_dir, name_dir)
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, str(i) + ".png"), im)
        for i in range(500):
            im = gen.generate_background()
            output_dir = os.path.join(output_root_dir, "background")
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, str(i) + ".png"), im)

    output_root_dir = "pytorch_dataset/val"
    os.makedirs(output_root_dir, exist_ok=True)
    for name_dir in os.listdir(input_gun_icon_dir):
        icon_dir = os.path.join(input_gun_icon_dir, name_dir)
        gen = Generater(icon_dir)

        for i in range(100):
            im = gen.generate_icon()
            output_dir = os.path.join(output_root_dir, name_dir)
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, str(i) + ".png"), im)
        for i in range(100):
            im = gen.generate_background()
            output_dir = os.path.join(output_root_dir, "background")
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, str(i) + ".png"), im)
