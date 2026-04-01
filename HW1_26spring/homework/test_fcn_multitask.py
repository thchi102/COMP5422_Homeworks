import torch
import numpy as np

from .models import FCN_MT, load_model
from .utils import load_dense_data, ConfusionMatrix, DenseVisualization
from . import dense_transforms
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt
from homework.train_fcn_multitask import masked_depth_error


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for both segmentation and depth estimation tasks
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task
    Hint: use DepthError for you to calculate rel, a1, a2, and a3 for the depth estimation task. 
    """
    model = load_model('fcn_mt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    conf_matrix = ConfusionMatrix(size=19)
    test_loader = load_dense_data("DenseCityscapesDataset/val", batch_size=1)
    model.eval()
    i = 0
    with torch.no_grad():
        for (images, labels_seg, labels_depth) in test_loader:
            images = images.to(device)
            labels_seg = labels_seg.to(device)
            labels_depth = labels_depth.to(device)
            output_seg, output_depth = model(images)
            conf_matrix.add(output_seg.argmax(dim=1), labels_seg)
            # if i < 16:
            #     # DenseVisualization(img, depth, seg); optional depth_gt, seg_gt
            #     image, depth_pred, seg_pred, depth_gt, seg_gt = DenseVisualization(
            #         images[0],
            #         output_depth[0],
            #         output_seg[0],
            #         labels_depth[0],
            #         labels_seg[0],
            #     ).__visualizeitem__()
            #     image.save(f"fcn_mt_results/input/input_mt_{i}.png")
            #     depth_pred.save(f"fcn_mt_results/depth_pred/depth_pred_mt_{i}.png")
            #     seg_pred.save(f"fcn_mt_results/seg_pred/seg_pred_mt_{i}.png")
            #     depth_gt.save(f"fcn_mt_results/depth_gt/depth_gt_mt_{i}.png")
            #     seg_gt.save(f"fcn_mt_results/seg_gt/seg_gt_mt_{i}.png")
            #     i += 1
            
    abs_rel, a1, a2, a3 = masked_depth_error(output_depth, labels_depth)
    accuracy = conf_matrix.global_accuracy
    mIoU = conf_matrix.iou
    print(f"Accuracy: {accuracy}")
    print(f"mIoU: {mIoU}")
    print(f"Abs Rel: {abs_rel}")
    print(f"A1: {a1}")
    print(f"A2: {a2}")
    print(f"A3: {a3}")

    return accuracy, mIoU


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
