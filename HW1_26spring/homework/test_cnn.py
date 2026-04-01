from .models import CNNClassifier, load_model
from .utils import ConfusionMatrix, load_data
import torch
import torchvision
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    model = CNNClassifier()

    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for the vehicle classification task
    Hint: use the ConfusionMatrix for you to calculate accuracy
    """
    class_names = ['Bicycle', 'Car', 'Taxi', 'Bus', 'Truck', 'Van']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('cnn')
    model.to(device)
    conf_matrix = ConfusionMatrix(size=6)
    test_loader = load_data("VehicleClassificationDataset/validation_subset", batch_size=1)
    model.eval()
    with torch.no_grad():
        for (images, labels) in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                conf_matrix.add(outputs.argmax(dim=1), labels)

    accuracy = conf_matrix.global_accuracy
    conf_matrix.save_confusion_matrix("confusion_matrix_cnn.png", class_names)
    print(f"Accuracy: {accuracy}")
    return accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
