from .models import CNNClassifier, save_model, SoftmaxCrossEntropyLoss
from .utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
import torchvision
import torch.utils.tensorboard as tb
from tqdm import tqdm



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
    # load data
    train_loader = load_data("HW1_26spring/VehicleClassificationDataset/train_subset", batch_size=64)
    valid_loader = load_data("HW1_26spring/VehicleClassificationDataset/validation_subset", batch_size=64)

    # loss and optimizer
    criterion = SoftmaxCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # train 
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            train_logger.add_scalar('train/loss', loss.item(), global_step=i)
        # validation
        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(valid_loader)):
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_logger.add_scalar('valid/loss', loss.item(), global_step=i)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
