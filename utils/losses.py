import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfConfidMSELoss(nn.modules.loss._Loss):
    def __init__(self):
        self.nb_classes = 2
        self.weighting = 10
        super().__init__()

    def forward(self, input_pred, input_confidence, target):
        probs = F.softmax(input_pred, dim=1)
        confidence = torch.sigmoid(input_confidence).squeeze()
        # Apply optional weighting
        weights = 100* torch.ones_like(target).type(torch.FloatTensor).cuda()
        weights[(probs.argmax(dim=1) != target)] *= self.weighting
        labels_hot = one_hot_embedding(target, self.nb_classes).cuda()
        # Segmentation special case
        labels_hot = labels_hot.permute(0, 3, 1, 2)

        # print((probs * labels_hot))
        # print((probs * labels_hot).sum(dim=1))
        # print(confidence)
        loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2
        return torch.mean(loss)


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]