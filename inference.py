from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

from utils.general import image_padding


class LoadImages:
    def __init__(self, source, transform=None, suffix=None):
        self.source = Path(source) if isinstance(source, str) else source
        self.suffix = ['.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'] \
            if suffix is None else suffix
        self.files = []
        self.name = []

        if self.source.is_dir():
            for img_path in self.source.iterdir():
                if img_path.suffix in self.suffix:
                    img = image_padding(str(img_path), (150, 150))
                    self.files.append(img)
                    self.name.append(img_path.stem)
        elif self.source.is_file() and self.source.suffix in self.suffix:
            img = image_padding(str(source), (150, 150))
            self.files.append(img)
            self.name.append(source.stem)
        else:
            raise TypeError("Source must be directory or image!")

        self.n = len(self.files)
        self.transform = transform

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.n:
            raise StopIteration

        img = self.files[self.count]
        name = self.name[self.count]
        if self.transform:
            img = self.transform(img)
        img = img.unsqueeze(0)
        self.count += 1
        s = f'image {name} {self.count}/{self.n} '

        return img, s

    def __len__(self):
        return self.n


def inference(weight, source):
    # 读取模型
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(weight)
    model.to(DEVICE)

    # 读取数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataloader = LoadImages(source, transform=transform)

    model.eval()
    res = []
    with torch.no_grad():
        for img, s in dataloader:
            img = Variable(img).to(DEVICE)
            output = model(img)
            pred = nn.Softmax(dim=1)(output)
            pred = pred.view(1, -1).cpu().numpy()
            output = 1 if pred[0, 1] > 0.2 else 0
            res.append(output)
            print(s + str(output))

    res = np.array(res)
    acc = np.sum(res == 0) / len(res)
    print(acc)


if __name__ == '__main__':
    weight = 'output-01-26-184710/best_epoch_weights.pt'
    source = '51-2/淋巴'
    inference(weight, source)
