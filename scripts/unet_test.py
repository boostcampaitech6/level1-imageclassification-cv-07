import os
from glob import glob

from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

import torch
import albumentations as A

from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model

data_path = '../data/eval/images/'
img_path = sorted(glob(data_path + '*'))

model = create_model("Unet_2020-07-20")
device = torch.device('cuda')
model.to(device)
model.eval()

transform = A.Compose([A.Normalize(p=1)], p=1)


seg_path = []
for path in tqdm(img_path):
    image = load_rgb(path)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x.to(device))[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    final = image.copy()
    for i in range(3):
        final[:, :, i] = mask * image[:, :, i]

    final_img = Image.fromarray(final)
    directory, file_n = os.path.split(path)
    directory = directory.replace('det_images', 'seg_images')
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, file_n)
    seg_path.append(save_path)
    final_img.save(save_path)
