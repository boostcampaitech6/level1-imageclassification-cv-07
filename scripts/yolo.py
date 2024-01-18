import os
import shutil
from glob import glob

from ultralytics import YOLO

model = YOLO("./yolov8x.pt")

data_list = sorted(glob('../data/train/*/*/*'))
batch_size = 7

for i in range(0, len(data_list), batch_size):
    data = data_list[i:i+batch_size]
    directory, filename = os.path.split(data[0])
    model.predict(
        source=data,
        save_crop=True,
        max_det=1,
        classes=0,
        project=directory,
        name='crop'
    )

data_list = sorted(glob('../data/train/*/*/*/*/*/*'))
for file_path in data_list:
    directory, file_name = os.path.split(file_path)
    directory = directory.replace('images', 'det_images')
    directory = directory.replace('crop/crops/person', '')
    if not os.path.exists(directory):
        os.makedirs(directory)
    result_path = os.path.join(directory, file_name)
    shutil.copy(file_path, result_path)
