from .models import CNNClassifier, save_model, SoftmaxCrossEntropyLoss
from .utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
import torchvision
import torch.utils.tensorboard as tb
from tqdm import tqdm
import copy



def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    """
    Your code here
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # load data
    train_loader = load_data("VehicleClassificationDataset/train_subset", batch_size=64)
    valid_loader = load_data("VehicleClassificationDataset/validation_subset", batch_size=2)

    # loss and optimizer
    criterion = SoftmaxCrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.resnet50.layer4.parameters(), "lr": 5e-5},
        {"params": model.resnet50.fc.parameters(), "lr": 2e-4},
    ], weight_decay=2e-4)

    # train
    num_epochs = 15
    best_val = 0
    best_checkpoint = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_total = 0

        train_tqdm = tqdm(train_loader, desc=f"Train: Epoch {epoch+1}/{num_epochs}")
        conf_matrix_train = ConfusionMatrix(size=6)
        for i, (images, labels) in enumerate(train_tqdm):
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            conf_matrix_train.add(outputs.argmax(dim=1), labels)
            train_total += 1
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # log 
        train_logger.add_scalar('train/loss', train_loss/train_total, global_step=epoch)
        train_logger.add_scalar('train/accuracy', conf_matrix_train.global_accuracy, global_step=epoch)

        # validation
        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        valid_total = 0

        valid_tqdm = tqdm(valid_loader, desc=f"Valid: Epoch {epoch+1}/{num_epochs}")
        conf_matrix_valid = ConfusionMatrix(size=6)
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_tqdm):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                conf_matrix_valid.add(outputs.argmax(dim=1), labels)
                valid_total += 1
        # log
        valid_logger.add_scalar('valid/loss', valid_loss/valid_total, global_step=epoch)
        valid_logger.add_scalar('valid/accuracy', conf_matrix_valid.global_accuracy, global_step=epoch)

        print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss/train_total} Train Accuracy: {conf_matrix_train.global_accuracy} Valid Loss: {valid_loss/valid_total} Valid Accuracy: {conf_matrix_valid.global_accuracy}")
        if valid_accuracy/valid_total > best_val:
            best_val = valid_accuracy/valid_total
            best_checkpoint = copy.deepcopy(model.state_dict())
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint)
    # save model
    # print(f"Best train accuracy: {best_train}")
    # print(f"Best validation accuracy: {best_val}")
    save_model(model)

    with open("cnn_best_metrics.txt", "w", encoding="utf-8") as mf:
        mf.write(f"Best train accuracy: {float(best_train)}\n")
        mf.write(f"Best validation accuracy: {float(best_val)}\n")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
