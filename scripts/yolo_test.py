import os
from glob import glob
import shutil

from ultralytics import YOLO

model = YOLO("./yolov8x.pt")

data_list = sorted(glob('../data/eval/images/*'))

for i in range(0, len(data_list), 100):
    data = data_list[i:i+100]
    directory, filename = os.path.split(data[0])
    model.predict(
        source=data,
        save_crop=True,
        max_det=1,
        classes=0,
        project=directory,
        name='crop'
    )

data_list = sorted(glob('../data/eval/*/*/*/*/*'))
for file_path in data_list:
    file_name = os.path.basename(file_path)
    if not os.path.exists('../data/eval/det_images'):
        os.makedirs('../data/eval/det_images')
    result_path = os.path.join('../data/eval/det_images', file_name)
    shutil.copy(file_path, result_path)
