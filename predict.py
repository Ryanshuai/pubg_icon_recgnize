from torchvision import transforms
import torch

from PIL import Image
import os
import numpy as np

from net import VGG

model = VGG(len(os.listdir('pytorch_dataset/train')))
model.load_state_dict(torch.load('loss_0.001207__acc_5.000000.pth.tar'))
model.eval()

i_name = ["ang", "burst2", "burst3", "com_ar", "com_sm", "fla_ar", "fla_sm", "full", "hal", "in_tab", "las",
          "lig", "single", "sto", "sup_ar", "thu", "ver", "x1h", "x1r", "x2", "x3", "x4", "x6", "x8", "x15", ]

preprocess = transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor()])


def image2name(im):
    assert isinstance(im, Image.Image)
    im = preprocess(im)
    input_batch = im.unsqueeze(0)  # create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)
    idx = int(np.argmax(output[0]))
    # print(idx)
    return i_name[idx]


if __name__ == '__main__':

    test_dir = "pytorch_dataset/test"
    for image_name in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_name)
        im = Image.open(image_path).convert('RGB')

        name = image2name(im)

        print(image_name + "----->" + name)
