import os
import torch

import numpy as np

from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

from . import dense_transforms
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


class VehicleClassificationDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: load your data from provided dataset (VehicleClassificationDataset) to train your designed model
        """
        # e.g., Bicycle 0, Car 1, Taxi 2, Bus 3, Truck 4, Van 5
        self.data = []
        self.label = []
        self.classes = {
            'Bicycle': 0,
            'Car': 1,
            'Taxi': 2,
            'Bus': 3,
            'Truck': 4,
            'Van': 5,
        }

        # transform
        if dataset_path.endswith('train_subset'):
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # load data
        for class_name, class_id in self.classes.items():
            class_path = os.path.join(dataset_path, class_name)
            for image_path in glob(os.path.join(class_path, '*.jpg')):
                self.data.append(image_path)
                self.label.append(class_id)
        # raise NotImplementedError('VehicleClassificationDataset.__init__')
         

    def __len__(self):
        """
        Your code here
        """
        return len(self.data)
        # raise NotImplementedError('VehicleClassificationDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        Hint: generate samples for training
        Hint: return image, and its image-level class label
        """
        image = Image.open(self.data[idx])
        image = self.transform(image)
        label = torch.tensor(self.label[idx])
        return (image, label)
        # raise NotImplementedError('VehicleClassificationDataset.__getitem__')


class DenseCityscapesDataset(Dataset):
    """
    HINT:
    Before translating the disparity into real depth, 
    we need to load the disparity from .npy files correctly. 
    
    Example:
        value = np.load('0.npy') [:,:,0]
        disparity = (value * 65535 - 1) / 256
        depth = (Baseline * focal_length) / disparity 
    
    According to the readme of the CityScape dataset 
    (https://github.com/mcordts/cityscapesScripts/blob/master/README.md#dataset-structure),
    we need to load disparity with (float(p)-1.) / 256. The *65535 operation is 
    because we provide data in the .npy format, not the original 16-bit png. 
    Please also note that there are some invalid depth values (not positive)
    in the ground truths caused by sensors. They should be masked out in the 
    training and evaluation. 

    """
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        """
        Your code here
        """
        self.image = []
        self.semantic_GT = []
        self.depth_GT = []
        if dataset_path.endswith('train'):
            self.transform = dense_transforms.Compose3([
                dense_transforms.ColorJitter3(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                dense_transforms.RandomGrayscale3(p=0.1),
                dense_transforms.ToTensor3(),
                dense_transforms.RandomHorizontalFlip3(0.5),
                dense_transforms.RandomVerticalFlip3(0.5),
                dense_transforms.Normalize3(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = dense_transforms.Compose3([
                dense_transforms.ToTensor3(),
                dense_transforms.Normalize3(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.baseline = 0.222384
        self.focal_length = 2273.82

        for filename in os.listdir(os.path.join(dataset_path, 'image')):
            self.image.append(os.path.join(dataset_path, 'image', filename))
            self.semantic_GT.append(os.path.join(dataset_path, 'label', filename))
            self.depth_GT.append(os.path.join(dataset_path, 'depth', filename))
        # raise NotImplementedError('DenseCityscapesDataset.__init__')

    def __len__(self):

        """
        Your code here
        """
        return len(self.image)
        # raise NotImplementedError('DenseCityscapesDataset.__len__')

    def __getitem__(self, idx):

        """
        Hint: generate samples for training
        Hint: return image, semantic_GT, and depth_GT
        """
        image = np.load(self.image[idx]).astype(np.float32)
        image = Image.fromarray((image*255).astype(np.uint8))

        semantic_GT = np.load(self.semantic_GT[idx])
        semantic_GT[semantic_GT < 0] = 255

        depth_arr = np.load(self.depth_GT[idx])
        disparity_value = depth_arr[:, :, 0].astype(np.float64)
        disparity = (disparity_value * 65535.0 - 1.0) / 256.0
        bf = float(self.baseline * self.focal_length)
        depth_GT = bf / disparity
        depth_GT[depth_GT < 0] = 0.0

        image, semantic_GT, depth_GT = self.transform(image, semantic_GT, depth_GT)
        return image, semantic_GT, depth_GT
        # raise NotImplementedError('DenseCityscapesDataset.__getitem__')
    

class DenseKITTIDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        """
        Your code here
        """
        self.images = []
        
        for filename in os.listdir(dataset_path):
            self.images.append(os.path.join(dataset_path, filename))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        """
        Your code here
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Your code here
        """
        image = Image.open(self.images[idx])

        image = self.transform(image)
        return image

class DenseVisualization():
    def __init__(self, img, depth, segmentation, depth_gt=None, seg_gt=None):
        self.img = img
        self.depth = depth
        self.segmentation = segmentation
        self.depth_gt = depth_gt
        self.seg_gt = seg_gt
        # Match cityscapesScripts viewer (cityscapesViewer.py): plasma + disparity norm [3, 100]
        self._depth_vis_bf = 0.222384 * 2273.82

        self.CITYSCAPES_PALETTE = [
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

    @staticmethod
    def _take_batch0(t):
        if isinstance(t, torch.Tensor) and t.dim() == 4:
            return t[0]
        return t

    def _depth_to_rgb_pil(self, depth):
        if isinstance(depth, torch.Tensor):
            d = depth.detach().cpu().float().squeeze()
            while d.dim() > 2:
                d = d.squeeze(0)
            d_np = d.numpy()
        else:
            d_np = np.squeeze(np.asarray(depth, dtype=np.float64))
        valid = np.isfinite(d_np) & (d_np > 0)
        disp = np.zeros_like(d_np, dtype=np.float64)
        disp[valid] = self._depth_vis_bf / np.maximum(d_np[valid], 1e-6)
        cmap = plt.cm.plasma
        norm = mcolors.Normalize(vmin=3.0, vmax=100.0)
        rgba = np.asarray(cmap(norm(np.nan_to_num(disp, nan=0.0))))
        rgba[~valid] = (0.0, 0.0, 0.0, 1.0)
        depth_rgb = (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(depth_rgb, mode="RGB")

    def _tensor_to_seg_hw(self, seg):
        if isinstance(seg, torch.Tensor):
            s = seg.detach().cpu()
            if s.dim() == 3 and s.shape[0] > 1:
                s = s.argmax(dim=0)
            elif s.dim() == 3:
                s = s.squeeze(0)
            return s.long().numpy()
        return np.asarray(seg)

    def _seg_hw_to_palette_rgb(self, seg_hw):
        """Map Cityscapes ignore (255) to void index 19 (black); clip train IDs to 0–19."""
        m = np.asarray(seg_hw, dtype=np.int64)
        m = np.where(m == 255, 19, m)
        m = np.clip(m, 0, 19).astype(np.uint8)
        pal = self.CITYSCAPES_PALETTE + [0] * (768 - len(self.CITYSCAPES_PALETTE))
        pil = Image.fromarray(m, mode="P")
        pil.putpalette(pal)
        return pil.convert("RGB")

    def __visualizeitem__(self):
        """
        Your code here
        Hint: you can visualize your model predictions and save them into images. 
        """
        if isinstance(self.img, torch.Tensor):
            x = self.img.detach().cpu().float()
            if x.dim() == 4:
                x = x[0]
            if x.shape[0] == 3:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                x = (x * std + mean).clamp(0.0, 1.0)
                rgb = (x.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            elif x.dim() == 2:
                g = (x.numpy() * 255.0).round().astype(np.uint8)
                rgb = np.stack([g, g, g], axis=-1)
            else:
                rgb = (x.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        else:
            arr = np.asarray(self.img)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            rgb = arr if arr.ndim == 3 else np.stack([arr] * 3, axis=-1)
        image_pil = Image.fromarray(rgb, mode="RGB")

        depth_pil = self._depth_to_rgb_pil(self.depth)
        seg_pil = self._seg_hw_to_palette_rgb(self._tensor_to_seg_hw(self.segmentation))

        out = [image_pil, depth_pil, seg_pil]
        if self.depth_gt is not None:
            out.append(self._depth_to_rgb_pil(self.depth_gt))
        if self.seg_gt is not None:
            out.append(self._seg_hw_to_palette_rgb(self._tensor_to_seg_hw(self.seg_gt)))
        return tuple(out)


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = VehicleClassificationDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseCityscapesDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_kitti_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseKITTIDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)
    
    def save_confusion_matrix(self, save_path, class_names):
        import matplotlib.pyplot as plt
        class_names = class_names or [f'Class {i}' for i in range(self.size)]
        matrix = self.matrix.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap='Blues')
        plt.colorbar(im)

        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)

        for i in range(self.size):
            for j in range(self.size):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center')

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(save_path)

        plt.close(fig)


class DepthError(object):
    def __init__(self, gt, pred):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.gt = gt
        self.pred = pred

    @property
    def compute_errors(self):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((self.gt / self.pred), (self.pred / self.gt))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        # rmse = (self.gt - self.pred) ** 2
        # rmse = np.sqrt(rmse.mean())

        # rmse_log = (np.log(self.gt) - np.log(self.pred)) ** 2
        # rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(self.gt - self.pred) / self.gt)

        # sq_rel = np.mean(((self.gt - self.pred) ** 2) / self.gt)

        return abs_rel, a1, a2, a3
