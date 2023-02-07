import argparse
import datetime
import os
from pathlib import Path

from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import models

from utils.dataloader import DataLoaderFactory
from utils.loss import BCEFocalLoss, MultiClassFocalLoss
from utils.scheduler import get_lr_scheduler, set_optimizer_lr
from utils.logger import Logger


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='dir')
    parser.add_argument('--save_dir', type=str, default='', help='save to project')
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_decay_type", type=str, default='cos', help='cos or other')
    parser.add_argument("--lr", type=int, default=1e-2)
    parser.add_argument("--min_lr", type=int, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--aug_label", type=list, default=[1])
    opt = parser.parse_args()

    return opt


def main(opt):
    train(**vars(opt))


def train(source,
          save_dir,
          n_classes,
          batch_size,
          lr_decay_type,
          lr,
          min_lr,
          epochs,
          aug_label):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = 'binary' if n_classes == 2 else 'multi-class'

    if save_dir:
        save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
        if not save_dir.exists():
            save_dir.mkdir()
    else:
        cur_time = datetime.datetime.now().strftime('%m-%d-%H%M%S')
        save_dir = Path('output-' + cur_time)
        save_dir.mkdir()

    # 数据增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    ])

    # 读取数据
    dataloader_factory = DataLoaderFactory(source, mode=mode)
    dataset_train, dataset_test = dataloader_factory(transform=transform, aug_label=aug_label, train_test_ratio=0.8)

    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # 实例化模型
    criterion = BCEFocalLoss() if mode == 'binary' else MultiClassFocalLoss()
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=n_classes)
    model.to(DEVICE)
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = get_lr_scheduler(lr_decay_type, lr, min_lr, epochs)
    # 实例化LossHistory
    log_dir = save_dir / 'log'
    logger = Logger(log_dir)

    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).to(DEVICE), Variable(target).to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()  # 清空过往梯度
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度更新网络参数
            print_loss = loss.data.item()
            sum_loss += print_loss
            if (batch_idx + 1) % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(data),
                                                                               batch_size * len(train_loader),
                                                                               100. * (batch_idx + 1) / len(
                                                                                   train_loader), loss.item()))
        avg_loss = sum_loss / len(train_loader)
        print('Epoch: {}\tLoss: {}'.format(epoch, avg_loss))
        set_optimizer_lr(optimizer, scheduler, epoch)

        model.eval()
        test_loss = 0
        test_pred = torch.tensor([])
        test_target = torch.tensor([])

        with torch.no_grad():
            for data, target in test_loader:
                data, target = Variable(data).to(DEVICE), Variable(target).to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                pred = nn.Softmax(dim=1)(output)
                test_pred = torch.cat((test_pred, pred.cpu()), dim=0)
                test_target = torch.cat((test_target, target.cpu()), dim=0)
                print_loss = loss.data.item()
                test_loss += print_loss
            avg_test_loss = test_loss / len(test_loader)
            test_target, test_pred = test_target.numpy(), test_pred.numpy()
            if mode == 'binary':
                auc = metrics.roc_auc_score(test_target, test_pred[:, 1])
            else:
                auc = metrics.roc_auc_score(test_target, test_pred, multi_class='ovo')
            print('\nVal set: Average loss: {:.4f}\tAUC: {:.4f}\n'.format(avg_test_loss, auc))

        logger.append_loss(avg_loss, avg_test_loss)

        if epoch % 50 == 0 and epoch != epochs:
            torch.save(model, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pt" % (
                epoch, avg_loss, avg_test_loss)))

        if len(logger.val_loss) <= 1 or avg_test_loss <= min(logger.val_loss):
            logger.update_roc(test_pred, test_target)
            print('Save best model to best_epoch_weights.pt\n')
            torch.save(model, os.path.join(save_dir, "best_epoch_weights.pt"))

    logger.loss_plot()
    logger.roc_plot()
    print('Save last model to last_epoch_weights.pt\n')
    torch.save(model, os.path.join(save_dir, "last_epoch_weights.pt"))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
