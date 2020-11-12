from PIL import Image
from torchvision import transforms
import os
import torch
import numpy as np

from net import vgg11_bn
from idx_name import name

test_dir = "pytorch_dataset/test"

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
model = vgg11_bn()
model.load_state_dict(torch.load('loss_5.000000__acc_0.001125.pth.tar'))
model.eval()

for image_name in os.listdir(test_dir):
    image_path = os.path.join(test_dir, image_name)
    input_image = Image.open(image_path)

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)
    idx = int(np.argmax(output[0]))
    print(image_name + "----->" + str(idx) + " : " + str(name[idx]))
