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

    # depth = random.randint(0, 4)
    # kernel = np.ones((depth, depth), np.uint8)
    # erosion = cv2.erode(icon_4c[:, :, 3], kernel, iterations=1)
    # icon_4c = icon_4c * (erosion // 255)[:, :, np.newaxis]

    size_factor = random.uniform(0.9, 1.15)
    h, w, c = icon_4c.shape
    icon_4c = cv2.resize(icon_4c, (int(h * size_factor), int(w * size_factor)))

    x, y = random.randint(-3, 3), random.randint(-3, 3)
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


def generate_dataset(output_dir, output_number):
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(input_gun_icon_dir):
        print(name)
        img_path = os.path.join(input_gun_icon_dir, name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        background_paths = os.listdir(input_background_dir)
        background_screens = [cv2.imread(os.path.join(input_background_dir, path)) for path in background_paths]
        os.makedirs(os.path.join(output_dir, name[:-4]), exist_ok=True)
        for i in range(0, output_number):
            gun_icon, mask = generate_gun_icon_frontground(img)

            background_screen = background_screens[random.randint(0, len(background_paths) - 1)]
            background = generate_icon_background(background_screen)
            img_mix = gun_icon + background * (1 - mask // 255)

            # radius = random.randint(1, 4) * 2 + 1
            # img_mix = cv2.GaussianBlur(img_mix, (radius, radius), 0)
            # canny = cv2.Canny(img_mix, 10, 30)

            # cv2.imshow("img_mix", img_mix)
            # cv2.imshow("canny", canny)
            # cv2.waitKey()

            img_mix = cv2.resize(img_mix, (224, 224))
            cv2.imwrite(os.path.join(output_dir, name[:-4], str(i) + ".png"), img_mix)

    background_paths = os.listdir(input_background_dir)
    background_screens = [cv2.imread(os.path.join(input_background_dir, path)) for path in background_paths]

    os.makedirs(os.path.join(output_dir, "background"), exist_ok=True)
    for i in range(0, output_number):
        background_screen = background_screens[random.randint(0, len(background_paths) - 1)]
        background = generate_icon_background(background_screen)
        # radius = random.randint(1, 4) * 2 + 1
        # background = cv2.GaussianBlur(bac kground, (radius, radius), 0)
        # canny = cv2.Canny(background, 10, 30)
        background = cv2.resize(background, (224, 224))
        cv2.imwrite(os.path.join(output_dir, "background", str(i) + ".png"), background)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import os

    input_gun_icon_dir = "original_data/train_gun_icon"
    input_background_dir = "original_data/background_screen"

    generate_dataset("pytorch_dataset/train", 500)
    generate_dataset("pytorch_dataset/val", 100)

    output_dir = "pytorch_dataset/train"
    output_number = 500
