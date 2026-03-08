import torch

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        """
        dropout_rate = 0.5
        num_classes = 6
        weights = ResNet50_Weights.DEFAULT
        self.resnet50 = resnet50(weights=weights)
        self.resnet50.fc = nn.Sequential(
            nn.Linear(self.resnet50.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes),
        )
        self.transform = weights.transforms()
        # raise NotImplementedError('CNNClassifier.__init__') 

    def forward(self, x):
        """
        Your code here
        """
        x = self.transform(x)
        x = self.resnet50(x)
        return x
        # raise NotImplementedError('CNNClassifier.forward') 


class FCN_ST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The Single-Task FCN needs to output segmentation maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN_ST.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding CNNClassifier
              convolution
        """
        raise NotImplementedError('FCN_ST.forward')


class FCN_MT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The Multi-Task FCN needs to output both segmentation and depth maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN_MT.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation
        @return: torch.Tensor((B,1,H,W)), 1 is one channel for depth estimation
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        raise NotImplementedError('FCN_MT.forward')


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):

        """
        Your code here
        Hint: inputs (prediction scores), targets (ground-truth labels)
        Hint: Implement a Softmax-CrossEntropy loss for classification
        Hint: return loss, F.cross_entropy(inputs, targets)
        """
        max_per_sample, _ = inputs.max(dim=1, keepdim=True)
        inputs = inputs - max_per_sample
        exp_inputs = torch.exp(inputs)
        sum_exp_inputs = exp_inputs.sum(dim=1, keepdim=True)
        log_probs = -torch.log(exp_inputs / sum_exp_inputs)

        loss = log_probs[:, targets].mean()
        return loss
        # raise NotImplementedError('SoftmaxCrossEntropyLoss.__init__')


model_factory = {
    'cnn': CNNClassifier,
    'fcn_st': FCN_ST,
    'fcn_mt': FCN_MT
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
