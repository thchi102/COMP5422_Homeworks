import torch
import numpy as np

from .models import FCN_ST, load_model, load_model_custom
from .utils import load_dense_data, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt
from PIL import Image

CITYSCAPES_PALETTE = [
            128, 64,  128,   # 0  road
            244, 35,  232,   # 1  sidewalk
            70,  70,  70,    # 2  building
            102, 102, 156,   # 3  wall
            190, 153, 153,   # 4  fence
            153, 153, 153,   # 5  pole
            250, 170, 30,    # 6  traffic light
            220, 220, 0,     # 7  traffic sign
            107, 142, 35,    # 8  vegetation
            152, 251, 152,   # 9  terrain
            70,  130, 180,   # 10 sky
            220, 20,  60,    # 11 person
            255, 0,   0,     # 12 rider
            0,   0,   142,   # 13 car
            0,   0,   70,    # 14 truck
            0,   60,  100,   # 15 bus
            0,   80,  100,   # 16 train
            0,   0,   230,   # 17 motorcycle
            119, 11,  32,    # 18 bicycle
        ]

def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your single-task model, and perform evaluation for the segmentation task
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task
     
    """
    model = load_model('fcn_st')
    # model = load_model_custom('fcn_st', 'fcn_st_best_0.4073.th')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    conf_matrix = ConfusionMatrix(size=19)
    test_loader = load_dense_data("DenseCityscapesDataset/val", batch_size=1)
    model.eval()
    i = 0
    with torch.no_grad():
        for (images, labels, _) in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            conf_matrix.add(outputs.argmax(dim=1), labels)
            
            # if i < 16:
            #     # visualize the input and prediction
            #     input_img = images[0].cpu().numpy().transpose(1, 2, 0)  # CxHxW -> HxWxC
            #     pred_mask = outputs.argmax(dim=1)[0].cpu().numpy()
            #     gt_mask = labels[0].cpu().numpy()
            #     gt_mask[gt_mask == 255] = 19
                
            #     # Undo torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #     mean = np.array([0.485, 0.456, 0.406])
            #     std = np.array([0.229, 0.224, 0.225])
            #     input_img_unnorm = (input_img * std) + mean
            #     input_img_unnorm = (input_img_unnorm * 255).astype(np.uint8)
            #     cityscapes_palette = CITYSCAPES_PALETTE + [0] * (768 - len(CITYSCAPES_PALETTE))
            #     # Save input image
            #     input_image_unnorm = Image.fromarray(input_img_unnorm)
            #     input_image_unnorm.save(f"fcn_st_results/input/input_st_{i}.png")
            #     # Save ground truth mask
            #     gt_mask_image = Image.fromarray(gt_mask.astype(np.uint8))
            #     gt_mask_image.putpalette(cityscapes_palette)
            #     gt_mask_image.save(f"fcn_st_results/gt/gt_st_{i}.png")
            #     # Save prediction mask
            #     pred_mask_image = Image.fromarray(pred_mask.astype(np.uint8))
            #     pred_mask_image.putpalette(cityscapes_palette)
            #     pred_mask_image.save(f"fcn_st_results/pred/pred_st_{i}.png")
            #     i+=1
    
    accuracy = conf_matrix.global_accuracy
    mIoU = conf_matrix.iou
    print(f"Accuracy: {accuracy}")
    print(f"mIoU: {mIoU}")
    return accuracy, mIoU


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
