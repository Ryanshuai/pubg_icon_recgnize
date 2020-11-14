import os
import cv2
import numpy as np
from shutil import copyfile, copy


def get_white_shield(im, min_rgb=235):
    shield_rgb = np.where(im > min_rgb, 1, 0)
    shield = shield_rgb[:, :, 0] & shield_rgb[:, :, 1] & shield_rgb[:, :, 2]
    return shield


# for im_name in os.listdir("in_tab_screen"):
#     im_path = os.path.join("in_tab_screen", im_name)
#     im = cv2.imread(im_path)
#     im_crap = im[132:155, 926:974, :]
#     im_shied = get_white_shield(im_crap, 225)
#     im_shied = im_shied.astype(np.uint8)
#     im_shied = im_shied[:, :, np.newaxis]
#     im_crap *= im_shied
#
#     rgba = np.concatenate((im_crap, im_shied * 255), axis=-1)
#
#     cv2.imshow("rgba", rgba)
#     cv2.waitKey()
#
#     outName = "in_tab"
#     out_dir = "train_gun_icon/" + outName
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, im_name)
#     cv2.imwrite(out_path, rgba)
#
#     out_dir = "test_gun_icon/" + outName
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, im_name)
#     cv2.imwrite(out_path, rgba)

root_dir = "screen"
for name in ["single_screen", "burst2_screen", "burst3_screen", "full_screen"]:
    for im_name in os.listdir(name):
        im_path = os.path.join(root_dir, name, im_name)
        print(im_path)
        im = cv2.imread(im_path)
        im_crap = im[1330:1364, 1648:1676, :]
        im_shied = get_white_shield(im_crap, 230)
        im_shied = im_shied[:, :, np.newaxis]
        im_crap *= im_shied.astype(np.uint8)

        rgba = np.concatenate((im_crap, im_shied * 255), axis=-1)
        rgba = rgba.clip(0, 255)
        rgba = rgba.astype(np.uint8)

        cv2.imshow("rgba", rgba)
        cv2.waitKey()

        outName = name.split("_")[0]
        out_dir = "train_gun_icon/" + outName
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, im_name)
        cv2.imwrite(out_path, rgba)

        out_dir = "test_gun_icon/" + outName
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, im_name)
        cv2.imwrite(out_path, rgba)

