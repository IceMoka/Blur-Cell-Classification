from pathlib import Path

import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.general import image_padding


class CellImageDataset(Dataset):
    def __init__(self, source, transform=None, aug_label=None):
        self.samples = source
        self.transform = transform
        self.aug_label = aug_label

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        img = self.samples[item][0]
        label = self.samples[item][1]

        if self.transform and (not self.aug_label or label in self.aug_label):
            img = self.transform(img)
        img = self.base_transform(img)

        return img, label


class DataLoaderFactory:
    """
    binary：
        1（正类）：不可分类的图片
        0（负类）：'lymphoma', 'mature lymph', 'mature single-core', 'neutrophilic myelocyte', 'premyelocyte'
    multi-class：
        0：lymphoma
        1：mature lymph
        2：mature single-core
        3：neutrophilic myelocyte
        4：premyelocyte
    """

    def __init__(self, source, mode):
        self.mode = mode
        self.category = ['lymphoma', 'mature lymph', 'mature single-core',
                         'neutrophilic myelocyte', 'premyelocyte', 'others']

        source = Path(source) if isinstance(source, str) else source
        path_list, label_list = [], []
        for path in source.glob('**/*.jpg'):
            label = self.category.index(path.parent.name)
            if self.mode != 'binary' and label == len(self.category) - 1:
                continue
            path_list.append(str(path))
            label_list.append(label)
        self.cell_dataframe = pd.DataFrame(data={
            'path': path_list,
            'label': label_list
        })

    def __call__(self, transform=None, aug_label=None, train_test_ratio=0.7):
        train_samples, test_samples = [], []
        category_num = len(self.category) if self.mode == 'binary' else len(self.category) - 1
        for i in range(category_num):
            single_category = self.cell_dataframe[self.cell_dataframe['label'] == i]
            single_category_train = single_category.sample(frac=train_test_ratio)
            single_category_test = pd.concat([single_category, single_category_train]).drop_duplicates(keep=False)

            if self.mode == 'binary':
                label = 0 if i in list(range(category_num - 1)) else 1
            else:
                label = i
            for row in range(single_category_train.shape[0]):
                image = image_padding(single_category_train.iloc[row, 0], (112, 112))
                train_samples.append((image, label))
            for row in range(single_category_test.shape[0]):
                image = image_padding(single_category_test.iloc[row, 0], (112, 112))
                test_samples.append((image, label))

        dataset_train = CellImageDataset(train_samples, transform=transform, aug_label=aug_label)
        dataset_test = CellImageDataset(test_samples, aug_label=aug_label)
        return dataset_train, dataset_test
