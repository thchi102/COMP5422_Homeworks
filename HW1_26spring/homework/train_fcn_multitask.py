import torch
import numpy as np

from .models import FCN_MT, save_model
from .utils import load_dense_data, ConfusionMatrix, DepthError
from . import dense_transforms
import torch.utils.tensorboard as tb

from tqdm import tqdm
import math


def masked_depth_loss(pred: torch.Tensor, target: torch.Tensor, criterion_none: torch.nn.Module):
    """
    Depth loss on valid pixels only. Divide by `scale` so targets are ~O(1) and
    SmoothL1 gradients are comparable to segmentation CE (raw depth in Cityscapes npy is often ~10²).
    """
    if target.dim() == 3:
        target = target.unsqueeze(1).float()
    else:
        target = target.float()
    pred = pred.float()
    valid_mask = target > 0
    return (criterion_none(pred, target) * valid_mask).float().sum() / torch.nonzero(valid_mask, as_tuple=False).size(0)

def masked_depth_error(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.cpu().float().detach().numpy()
    target = target.unsqueeze(1).cpu().float().detach().numpy()

    pred = pred[target>0]
    target = target[target>0]
    abs_rel, a1, a2, a3 = DepthError(target, pred).compute_errors
    
    return abs_rel, a1, a2, a3

def train(args):
    from os import path
    model = FCN_MT()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here
    Hint: validation during training: use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: Use dense_transforms for data augmentation. If you found a good data augmentation parameters for the CNN, use them here too. 
    Hint: Use the log function below to debug and visualize your model
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # load data
    train_loader = load_dense_data("DenseCityscapesDataset/train", batch_size=32)
    valid_loader = load_dense_data("DenseCityscapesDataset/val", batch_size=32)
    loss_weights = torch.tensor(
        [
            3.29,     # 0 road
            21.9,     # 1 sidewalk
            4.68,     # 2 building
            121.32,   # 3 wall
            266.84,   # 4 fence
            117.6,    # 5 pole
            1022.23,  # 6 traffic light
            205.68,   # 7 traffic sign
            6.13,     # 8 vegetation
            118.81,   # 9 terrain
            35.17,    # 10 sky
            168.36,   # 11 person
            460.62,   # 12 rider
            15.53,    # 13 car
            272.62,   # 14 truck
            501.94,   # 15 bus
            3536.12,  # 16 train
            2287.91,  # 17 motorcycle
            140.32,   # 18 bicycle
        ]
    ).to(device)
    # loss_weights = loss_weights/loss_weights.sum()

    # loss and optimizer
    criterion_seg = torch.nn.CrossEntropyLoss(weight=loss_weights, ignore_index=255)
    criterion_depth = torch.nn.L1Loss(reduction="none")
    depth_scale = 512.0
    loss_lambda = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    max_grad_norm = 5.0
    # train
    num_epochs = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_loss_seg = 0
        train_loss_depth = 0
        train_accuracy = 0
        train_total = 0
        train_abs_rel = 0
        train_a1 = 0
        train_a2 = 0
        train_a3 = 0
        
        train_tqdm = tqdm(train_loader, desc=f"Train: Epoch {epoch+1}/{num_epochs}")
        conf_matrix_train_seg = ConfusionMatrix(size=19)
        for i, (images, labels_seg, labels_depth) in enumerate(train_tqdm):
            images = images.to(device)
            labels_seg = labels_seg.to(device)
            labels_depth = labels_depth.to(device)
            # forward pass
            outputs_seg, outputs_depth = model(images)
            CELoss = criterion_seg(outputs_seg.float(), labels_seg)
            depth_l1 = masked_depth_loss(
                outputs_depth, labels_depth, criterion_depth
            )
            L1Loss = loss_lambda * depth_l1
            loss = CELoss + L1Loss
            train_loss += loss.item()
            train_loss_seg += CELoss.item()
            train_loss_depth += L1Loss.item()
            train_total += 1
            conf_matrix_train_seg.add(outputs_seg.argmax(dim=1), labels_seg)

            abs_rel, a1, a2, a3 = masked_depth_error(outputs_depth, labels_depth)
            train_abs_rel += abs_rel
            train_a1 += a1
            train_a2 += a2
            train_a3 += a3

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        # log 
        if train_logger is not None:
            train_logger.add_scalar('train/loss', train_loss/train_total, global_step=epoch)
            train_logger.add_scalar('train/abs_rel', train_abs_rel/train_total, global_step=epoch)
            train_logger.add_scalar('train/a1', train_a1/train_total, global_step=epoch)
            train_logger.add_scalar('train/a2', train_a2/train_total, global_step=epoch)
            train_logger.add_scalar('train/a3', train_a3/train_total, global_step=epoch)
            train_logger.add_scalar('train/accuracy_seg', conf_matrix_train_seg.global_accuracy, global_step=epoch)
            train_logger.add_scalar('train/iou', conf_matrix_train_seg.iou, global_step=epoch)
            train_logger.add_scalar('train/loss_seg', train_loss_seg/train_total, global_step=epoch)
            train_logger.add_scalar('train/loss_depth', train_loss_depth/train_total, global_step=epoch)

        # validation
        model.eval()
        valid_loss = 0
        valid_loss_seg = 0
        valid_loss_depth = 0
        valid_accuracy = 0
        valid_total = 0
        valid_abs_rel = 0
        valid_a1 = 0
        valid_a2 = 0
        valid_a3 = 0

        valid_tqdm = tqdm(valid_loader, desc=f"Valid: Epoch {epoch+1}/{num_epochs}")
        conf_matrix_valid_seg = ConfusionMatrix(size=19)
        with torch.no_grad():
            for i, (images, labels_seg, labels_depth) in enumerate(valid_tqdm):
                images = images.to(device)
                labels_seg = labels_seg.to(device)
                labels_depth = labels_depth.to(device)
                outputs_seg, outputs_depth = model(images)
                CELoss = criterion_seg(outputs_seg.float(), labels_seg)
                depth_l1 = masked_depth_loss(
                    outputs_depth, labels_depth, criterion_depth
                )
                L1Loss = loss_lambda * depth_l1
                loss = CELoss + L1Loss
                valid_loss += loss.item()
                valid_loss_seg += CELoss.item()
                valid_loss_depth += L1Loss.item()
                valid_total += 1
                conf_matrix_valid_seg.add(outputs_seg.argmax(dim=1), labels_seg)

                abs_rel, a1, a2, a3 = masked_depth_error(outputs_depth, labels_depth)
                valid_abs_rel += abs_rel
                valid_a1 += a1
                valid_a2 += a2
                valid_a3 += a3
        mean_valid_loss = valid_loss / valid_total
        # log
        if valid_logger is not None:
            valid_logger.add_scalar('valid/loss', mean_valid_loss, global_step=epoch)
            valid_logger.add_scalar('valid/abs_rel', valid_abs_rel/valid_total, global_step=epoch)
            valid_logger.add_scalar('valid/a1', valid_a1/valid_total, global_step=epoch)
            valid_logger.add_scalar('valid/a2', valid_a2/valid_total, global_step=epoch)
            valid_logger.add_scalar('valid/a3', valid_a3/valid_total, global_step=epoch)
            valid_logger.add_scalar('valid/accuracy_seg', conf_matrix_valid_seg.global_accuracy, global_step=epoch)
            valid_logger.add_scalar('valid/iou', conf_matrix_valid_seg.iou, global_step=epoch)
            valid_logger.add_scalar('valid/loss_seg', valid_loss_seg/valid_total, global_step=epoch)
            valid_logger.add_scalar('valid/loss_depth', valid_loss_depth/valid_total, global_step=epoch)
        if train_logger is not None:
            train_logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        scheduler.step()

        print(f"""Epoch {epoch+1}/{num_epochs} 
                  Train Loss: {train_loss/train_total} 
                  Train Accuracy Seg: {conf_matrix_train_seg.global_accuracy} 
                  Train IoU: {conf_matrix_train_seg.iou} 
                  Train Loss Seg: {train_loss_seg/train_total} 
                  Train Loss Depth: {train_loss_depth/train_total}
                  Train Abs Rel: {train_abs_rel/train_total}
                  Train A1: {train_a1/train_total}
                  Train A2: {train_a2/train_total}
                  Train A3: {train_a3/train_total}""")
        print(f"""Epoch {epoch+1}/{num_epochs} 
                  Valid Loss: {mean_valid_loss} 
                  Valid Accuracy Seg: {conf_matrix_valid_seg.global_accuracy} 
                  Valid IoU: {conf_matrix_valid_seg.iou} 
                  Valid Loss Seg: {valid_loss_seg/valid_total} 
                  Valid Loss Depth: {valid_loss_depth/valid_total}
                  Valid Abs Rel: {valid_abs_rel/valid_total}
                  Valid A1: {valid_a1/valid_total}
                  Valid A2: {valid_a2/valid_total}
                  Valid A3: {valid_a3/valid_total}""")

    save_model(model)

def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
