from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class Measure(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    @torch.no_grad()
    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> float:
        """"""
        from sklearn import metrics
        metric = self.metric
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()

        if metric in ['ap', 'auc']:
            if y_pred.dim() == 1:
                y_pred = y_pred.sigmoid()
            else:
                y_pred = F.softmax(y_pred, dim=1)[:, 1]
        else:
            if y_true.dim() == 1:
                if y_pred.dim() == 1:
                    y_pred = (y_pred > 0.5).long()
                else:
                    y_pred = y_pred.argmax(-1)
            else:
                # multi-classes
                y_pred = (y_pred > 0).float()

        if metric in ['auc', 'acc']:
            y_true = y_true.to(torch.long)
        else:
            y_true = y_true.to(torch.float)

        if metric == 'acc':
            metric_fn = metrics.accuracy_score
        elif metric == 'micro-f1':
            metric_fn = partial(metrics.f1_score, average='micro')
        elif metric == 'macro-f1':
            metric_fn = partial(metrics.f1_score, average='macro')
        elif metric == 'ap':
            metric_fn = metrics.average_precision_score
        elif metric == 'auc':
            metric_fn = metrics.roc_auc_score
        else:
            raise ValueError(f'Unsupported metric {metric}')

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return metric_fn(y_true, y_pred)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.metric})'
