import cv2
import numpy as np
import random

img_size = 100


def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


def generate_icon_background(screen):
    screen = screen[100:1400, 610:]
    y = random.randint(0, screen.shape[0] - img_size - 1)
    x = random.randint(0, screen.shape[1] - img_size - 1)
    icon_background = screen[y:y + img_size, x:x + img_size]
    return icon_background


def generate_gun_icon_frontground(icon_4c):
    h, w, c = icon_4c.shape
    icon_4c = cv2.resize(icon_4c, (int(h * 2), int(w * 2)))
    h, w, c = icon_4c.shape
    pad_h, pad_w = max(0, (img_size - h) // 2), max(0, (img_size - w) // 2)
    icon_4c = cv2.copyMakeBorder(icon_4c, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)

    # depth = random.randint(0, 5)
    # kernel = np.ones((depth, depth), np.uint8)
    # erosion = cv2.erode(icon_4c[:, :, 3], kernel, iterations=1)
    # icon_4c = icon_4c * (erosion // 255)[:, :, np.newaxis]

    size_factor = random.uniform(0.9, 1.05)
    h, w, c = icon_4c.shape
    icon_4c = cv2.resize(icon_4c, (int(h * size_factor), int(w * size_factor)))

    x, y = random.randint(-10, 10), random.randint(-10, 10)
    icon_4c = translate(icon_4c, x, y)

    half_size = img_size // 2
    icon_4c = cv2.copyMakeBorder(icon_4c, half_size, half_size, half_size, half_size, cv2.BORDER_CONSTANT, value=0)
    h, w, c = icon_4c.shape
    icon_4c = icon_4c[h // 2 - half_size: h // 2 + half_size, w // 2 - half_size: w // 2 + half_size]
    return icon_4c[:, :, :3], icon_4c[:, :, [3]]


def generate_fire_mode_frontground():
    pass


def generate_position_frontground():
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import os

    input_gun_icon_dir = "train_gun_icon"
    input_background_dir = "background_screen"

    output_dir = "icon_dataset/train"
    os.makedirs(output_dir, exist_ok=True)
    output_number = 500
    for name in os.listdir(input_gun_icon_dir):
        print(name)
        img_path = os.path.join(input_gun_icon_dir, name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        background_paths = os.listdir(input_background_dir)
        background_screens = [cv2.imread(os.path.join(input_background_dir, path)) for path in background_paths]
        for i in range(0, output_number):
            gun_icon, mask = generate_gun_icon_frontground(img)

            background_screen = background_screens[random.randint(0, len(background_paths) - 1)]
            background = generate_icon_background(background_screen)
            img_mix = gun_icon + background * (1 - mask // 255)

            os.makedirs(os.path.join(output_dir, name[:-4]), exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, name[:-4], str(i) + ".png"), img_mix)

    background_paths = os.listdir(input_background_dir)
    background_screens = [cv2.imread(os.path.join(input_background_dir, path)) for path in background_paths]

    for i in range(0, output_number):
        background_screen = background_screens[random.randint(0, len(background_paths) - 1)]
        background = generate_icon_background(background_screen)
        os.makedirs(os.path.join(output_dir, "nothing"), exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "nothing", str(i) + ".png"), background)

    output_dir = "icon_dataset/test"
    os.makedirs(output_dir, exist_ok=True)
    output_number = 100
    for name in os.listdir(input_gun_icon_dir):
        print(name)
        img_path = os.path.join(input_gun_icon_dir, name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        background_paths = os.listdir(input_background_dir)
        background_screens = [cv2.imread(os.path.join(input_background_dir, path)) for path in background_paths]
        for i in range(0, output_number):
            background_screen = background_screens[random.randint(0, len(background_paths) - 1)]
            background = generate_icon_background(background_screen)

            gun_icon, mask = generate_gun_icon_frontground(img)
            img_mix = gun_icon + background * (1 - mask // 255)

            os.makedirs(os.path.join(output_dir, name[:-4]), exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, name[:-4], str(i) + ".png"), img_mix)

    background_paths = os.listdir(input_background_dir)
    background_screens = [cv2.imread(os.path.join(input_background_dir, path)) for path in background_paths]

    for i in range(0, output_number):
        background_screen = background_screens[random.randint(0, len(background_paths) - 1)]
        background = generate_icon_background(background_screen)
        os.makedirs(os.path.join(output_dir, "nothing"), exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "nothing", str(i) + ".png"), background)
