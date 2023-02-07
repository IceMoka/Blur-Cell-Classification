from pathlib import Path

import scipy.signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize


class Logger:
    def __init__(self, log_dir, loss_lw=2, roc_lw=2, roc_colors=None):
        self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
        self.losses, self.val_loss = [], []
        self.roc_label, self.roc_prob = None, None

        self.loss_lw = loss_lw
        self.roc_lw = roc_lw
        self.roc_colors = ['navy', 'darkorange', 'deeppink', 'aqua', 'cornflowerblue', 'olivedrab'] \
            if roc_colors is None else roc_colors

        if not self.log_dir.exists():
            self.log_dir.mkdir()

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(str(self.log_dir / "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(str(self.log_dir / "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

    def update_roc(self, pred, target):
        self.roc_label = target
        self.roc_prob = pred

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=self.loss_lw, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=self.loss_lw, label='val loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--',
                     linewidth=self.loss_lw,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                     linewidth=self.loss_lw,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(str(self.log_dir / "train_period_loss.png"))

    def roc_plot(self):
        n_classes = len(np.unique(self.roc_label))

        if n_classes <= 2:
            fpr, tpr, thresholds = roc_curve(self.roc_label, self.roc_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            optimal_index = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_index]
            print(f'Best threshold to binary-class is {optimal_threshold:.2f}\n')

            roc_data = np.concatenate((self.roc_label.reshape((-1, 1)), self.roc_prob), axis=1)
            roc_df = pd.DataFrame(roc_data, columns=['label'] +
                                                    ['prob' + str(i) for i in range(n_classes)])

            plt.figure()
            plt.plot(fpr, tpr, color=self.roc_colors[0], lw=self.roc_lw,
                     label='ROC curve of positive sample (AUC = {:0.2f})'.format(roc_auc))
        else:
            self.roc_label = label_binarize(self.roc_label, classes=[i for i in range(n_classes)])
            roc_data = np.concatenate((self.roc_label, self.roc_prob), axis=1)
            roc_df = pd.DataFrame(roc_data, columns=[str(i) for i in range(n_classes)] +
                                                    ['prob' + str(i) for i in range(n_classes)])

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(self.roc_label[:, i], self.roc_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # micro
            fpr["micro"], tpr["micro"], _ = roc_curve(self.roc_label.ravel(), self.roc_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            # macro
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            plt.figure()
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], color=self.roc_colors[i], lw=self.roc_lw,
                         label='ROC curve of class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
                     color='red', linestyle=':', linewidth=4)
            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
                     color='darkmagenta', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=self.roc_lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Graph of Receiver operating characteristic to {}'.
                  format('binary-class' if n_classes == 2 else 'multi-class'))
        plt.legend(loc="lower right")

        plt.savefig(str(self.log_dir / "train_best_roc.png"))
        roc_df.to_excel(str(self.log_dir / 'roc_data_results.xlsx'), index=False)
