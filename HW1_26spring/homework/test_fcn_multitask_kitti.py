import torch
import numpy as np

from .models import FCN_MT, load_model
from .utils import load_kitti_data, DenseVisualization
from . import dense_transforms
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for both segmentation and depth estimation tasks 
          on the provided images of the KITTI dataset
    """
    model = load_model('fcn_mt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_loader = load_kitti_data("kitti_test_samples", batch_size=1)

    model.eval()
    i = 0
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            output_seg, output_depth = model(images)

            image, depth_pred, seg_pred = DenseVisualization(
                images[0], output_depth[0], output_seg[0]
            ).__visualizeitem__()
            image.save(f"fcn_mt_kitti_results/input/input_mt_kitti_{i}.png")
            depth_pred.save(f"fcn_mt_kitti_results/depth_pred/depth_pred_mt_kitti_{i}.png")
            seg_pred.save(f"fcn_mt_kitti_results/seg_pred/seg_pred_mt_kitti_{i}.png")
            i += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
