import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as F


def to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
    return zeros.scatter(scatter_dim, y_tensor.to(torch.int64), 1)


class RandomRotation(object):
    def __init__(self, degrees, seed=1):
        self.degrees = (-degrees, degrees)
        random.seed(seed)

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, False, False, None)


class PaperDataset(torch.utils.data.Dataset):
    def __init__(self, training=True, transform=None):
        self.size = 84 # 800 84
        if training==True:
            features = np.load(f'/home/soonwook34/jongwon/pretrain-gnns/MnistSimpleCNN/data/PAPER/processed/features_train_{self.size}.npy')
            labels = np.load(f'/home/soonwook34/jongwon/pretrain-gnns/MnistSimpleCNN/data/PAPER/processed/labels_train_{self.size}.npy')
        else:
            features = np.load(f'/home/soonwook34/jongwon/pretrain-gnns/MnistSimpleCNN/data/PAPER/processed/features_test_{self.size}.npy')
            labels = np.load(f'/home/soonwook34/jongwon/pretrain-gnns/MnistSimpleCNN/data/PAPER/processed/labels_test_{self.size}.npy')

        features = np.reshape(features, (-1, self.size, self.size, 3))#.astype(np.float32)
        labels = labels.astype(np.int)
        self.x_data = features
        self.y_data = labels
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = np.array(self.x_data)[idx].reshape(self.size, self.size, 3)
        x = Image.fromarray(x)
        y = torch.tensor(np.array(self.y_data[idx]))

        if self.transform:
            x = self.transform(x)
        x = transforms.ToTensor()(np.array(x)/255)
        return x.to(torch.float32), y, idx

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

