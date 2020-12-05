import os
import cv2
import numpy as np

# dir = "in_tab_real/train/in_tab"
#
# i = 0
# while i < len(os.listdir(dir)):
#     path_i = os.path.join(dir, os.listdir(dir)[i])
#     im_i = cv2.imread(path_i)
#     print("____________________________________________")
#     print(path_i)
#     name_list = os.listdir(dir)
#     # for j in range(i + 1, len(name_list)):
#     for j in range(i + 1, min(i + 10, len(name_list))):
#         path_j = os.path.join(dir, name_list[j])
#         im_j = cv2.imread(path_j)
#         if np.max(abs(im_j - im_i)) == 0:
#             os.remove(path_j)
#             print("remove:", path_j)
#     i += 1

root = "gun_scope_real/train/"
for x in os.listdir(root):
    dire = os.path.join(root, x)
    name_list = os.listdir(dire)
    length = len(name_list)
    for i in range(length - 1, -1, -1):
        src_path = os.path.join(dire, name_list[i])
        print(src_path)
        dst_path = os.path.join(dire, str(i) + ".png")
        os.rename(src_path, dst_path)
