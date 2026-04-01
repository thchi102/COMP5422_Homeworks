import os
import shutil
from glob import glob

SRC = "VehicleClassificationDataset"
DST = "VehicleClassificationDataset_small"
N = 100

script_dir = os.path.dirname(os.path.abspath(__file__))

for split in os.listdir(os.path.join(script_dir, SRC)):
    split_src = os.path.join(script_dir, SRC, split)
    if not os.path.isdir(split_src):
        continue
    for class_name in os.listdir(split_src):
        class_src = os.path.join(split_src, class_name)
        if not os.path.isdir(class_src):
            continue
        class_dst = os.path.join(script_dir, DST, split, class_name)
        os.makedirs(class_dst, exist_ok=True)
        images = sorted(glob(os.path.join(class_src, "*.jpg")))[:N]
        for img_path in images:
            shutil.copy(img_path, class_dst)
        print(f"Copied {len(images)} images -> {os.path.relpath(class_dst, script_dir)}")
