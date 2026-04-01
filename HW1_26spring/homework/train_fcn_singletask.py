import torch
import numpy as np

from .models import FCN_ST, save_model, save_model_custom
from .utils import load_dense_data, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
from tqdm import tqdm
import copy

def train(args):
    from os import path
    model = FCN_ST()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: Use dense_transforms for data augmentation. If you found a good data augmentation parameters for the CNN, use them here too.
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
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
    conf_matrix = ConfusionMatrix(size=19)
    # load data
    train_loader = load_dense_data("DenseCityscapesDataset/train", batch_size=16)
    valid_loader = load_dense_data("DenseCityscapesDataset/val", batch_size=16)

    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights, ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # train
    num_epochs = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    save_interval = 50
    best_checkpoint = None
    best_iou_train = 0
    best_accuracy_train = 0
    best_iou_valid = 0
    best_accuracy_valid = 0
    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_total = 0

        train_tqdm = tqdm(train_loader, desc=f"Train: Epoch {epoch}/{num_epochs}")
        conf_matrix_train = ConfusionMatrix(size=19)
        for i, (images, labels, _) in enumerate(train_tqdm):
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(images)
            loss = criterion(outputs.float(), labels)
            train_loss += loss.item()
            train_total += 1
            conf_matrix_train.add(outputs.argmax(dim=1), labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # log 
        if train_logger is not None:
            train_logger.add_scalar('train/loss', train_loss/train_total, global_step=epoch)
            train_logger.add_scalar('train/accuracy', conf_matrix_train.global_accuracy, global_step=epoch)
            train_logger.add_scalar('train/iou', conf_matrix_train.iou, global_step=epoch)
            train_logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step=epoch)

        # validation
        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        valid_total = 0

        valid_tqdm = tqdm(valid_loader, desc=f"Valid: Epoch {epoch}/{num_epochs}")
        conf_matrix_valid = ConfusionMatrix(size=19)
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(valid_tqdm):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.float(), labels)
                valid_loss += loss.item()
                valid_total += 1
                conf_matrix_valid.add(outputs.argmax(dim=1), labels)
        mean_valid_loss = valid_loss / valid_total
        # log
        if valid_logger is not None:
            valid_logger.add_scalar('valid/loss', mean_valid_loss, global_step=epoch)
            valid_logger.add_scalar('valid/accuracy', conf_matrix_valid.global_accuracy, global_step=epoch)
            valid_logger.add_scalar('valid/iou', conf_matrix_valid.iou, global_step=epoch)
        scheduler.step()

        if conf_matrix_valid.iou > best_iou_valid:
            best_iou_train = conf_matrix_train.iou
            best_accuracy_train = conf_matrix_train.global_accuracy
            best_iou_valid = conf_matrix_valid.iou
            best_accuracy_valid = conf_matrix_valid.global_accuracy
            best_checkpoint = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss/train_total} Train Accuracy: {conf_matrix_train.global_accuracy} Train IoU: {conf_matrix_train.iou}")
        print(f"Epoch {epoch+1}/{num_epochs} Valid Loss: {mean_valid_loss} Valid Accuracy: {conf_matrix_valid.global_accuracy} Valid IoU: {conf_matrix_valid.iou}")
        conf_matrix_train.save_confusion_matrix("confusion_matrix_fcn_train.png")
        conf_matrix_valid.save_confusion_matrix("confusion_matrix_fcn_valid.png")

    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint)
    print(f"Best train accuracy: {best_accuracy_train}")
    print(f"Best train mIoU: {best_iou_train}")
    print(f"Best valid accuracy: {best_accuracy_valid}")
    print(f"Best valid mIoU: {best_iou_valid}")

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
