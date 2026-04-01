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
        dropout_rate = 0.4
        num_classes = 6
        weights = ResNet50_Weights.DEFAULT
        self.resnet50 = resnet50(weights=weights)
        self.resnet50.fc = nn.Sequential(
            nn.Linear(self.resnet50.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes),
        )

        for param in self.resnet50.parameters():
            param.requires_grad = False
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True
        for param in self.resnet50.layer4.parameters():
            param.requires_grad = True
        # raise NotImplementedError('CNNClassifier.__init__') 

    def forward(self, x):
        """
        Your code here
        """
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
        self.num_classes = 19
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        self.stem = nn.Sequential(*list(backbone.children())[:3])
        self.max_pool1 = list(backbone.children())[3]
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, bias=False),
        )

        self.pool1 = nn.Conv2d(1024, self.num_classes, kernel_size=1, bias=False)
        self.pool2 = nn.Conv2d(512, self.num_classes, kernel_size=1, bias=False)

        # self.deconv1 = nn.Sequential(
        #     nn.Conv2d(19, 19, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(19),
        #     nn.ReLU(inplace=True),
        # )
        # self.deconv2 = nn.Sequential(
        #     nn.Conv2d(19, 19, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(19),
        #     nn.ReLU(inplace=True),
        # )

        self.refine = nn.Sequential(
            nn.Conv2d(self.num_classes, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding CNNClassifier
              convolution
        """
        H, W = x.size()[-2:]
        x1 = self.stem(x)
        x2 = self.layer1(self.max_pool1(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        score = self.classifier(x5)
        score = F.interpolate(score, size=x4.shape[2:], mode="bilinear", align_corners=False)
        score = score + self.pool1(x4)
        # score = self.deconv1(score)
        score = F.interpolate(score, size=x3.shape[2:], mode="bilinear", align_corners=False)
        score = score + self.pool2(x3)
        # score = self.deconv2(score)
        score = F.interpolate(score, size=x2.shape[2:], mode="bilinear", align_corners=False)
        score = F.interpolate(score, size=x1.shape[2:], mode="bilinear", align_corners=False)
        score = F.interpolate(score, size=(H, W), mode="bilinear", align_corners=False)
        score = self.refine(score)
        return score


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

        self.num_classes = 19
        single_task_model = load_model('fcn_st')
        self.stem = single_task_model.stem
        self.max_pool1 = single_task_model.max_pool1
        self.layer1 = single_task_model.layer1
        self.layer2 = single_task_model.layer2
        self.layer3 = single_task_model.layer3
        self.layer4 = single_task_model.layer4

        self.seg_classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, bias=False),
        )

        self.seg_pool1 = nn.Conv2d(1024, self.num_classes, kernel_size=1, bias=False)
        self.seg_pool2 = nn.Conv2d(512, self.num_classes, kernel_size=1, bias=False)

        self.seg_refine = nn.Sequential(
            nn.Conv2d(self.num_classes, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False),
        )

        self.depth_pool1 = nn.Conv2d(1024, 1, kernel_size=1, bias=False)
        self.depth_pool2 = nn.Conv2d(512, 1, kernel_size=1, bias=False)

        self.depth_classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 1, kernel_size=1, bias=False),
        )
        self.depth_refine = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=False),
        )

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
        H, W = x.size()[-2:]
        x1 = self.stem(x)
        x2 = self.layer1(self.max_pool1(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        seg_score = self.seg_classifier(x5)
        seg_score = F.interpolate(seg_score, size=x4.shape[2:], mode="bilinear", align_corners=False)
        seg_score = seg_score + self.seg_pool1(x4)
        seg_score = F.interpolate(seg_score, size=x3.shape[2:], mode="bilinear", align_corners=False)
        seg_score = seg_score + self.seg_pool2(x3)
        seg_score = F.interpolate(seg_score, size=x2.shape[2:], mode="bilinear", align_corners=False)
        seg_score = F.interpolate(seg_score, size=x1.shape[2:], mode="bilinear", align_corners=False)
        seg_score = F.interpolate(seg_score, size=(H, W), mode="bilinear", align_corners=False)
        seg_score = self.seg_refine(seg_score)

        depth_score = self.depth_classifier(x5)
        depth_score = F.interpolate(depth_score, size=x4.shape[2:], mode="bilinear", align_corners=False)
        depth_score = depth_score + self.depth_pool1(x4)
        depth_score = F.interpolate(depth_score, size=x3.shape[2:], mode="bilinear", align_corners=False)
        depth_score = depth_score + self.depth_pool2(x3)
        depth_score = F.interpolate(depth_score, size=x2.shape[2:], mode="bilinear", align_corners=False)
        depth_score = F.interpolate(depth_score, size=x1.shape[2:], mode="bilinear", align_corners=False)
        depth_score = F.interpolate(depth_score, size=(H, W), mode="bilinear", align_corners=False)
        depth_score = self.depth_refine(depth_score)

        return seg_score, depth_score


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

        loss = log_probs[torch.arange(len(targets)), targets].mean()
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

def save_model_custom(model, name):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % name))
    raise ValueError("model type '%s' not supported!" % str(type(model)))
    


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r

def load_model_custom(model, name):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % name), map_location='cpu'))
    return r
