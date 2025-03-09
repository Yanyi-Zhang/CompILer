import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        batch_size, topk, length, c = x.size()
        return (
            F.avg_pool1d(
                x.clamp(min=eps).pow(p).permute(0, 3, 2, 1).contiguous().view(batch_size, length * c, topk),
                kernel_size=topk,
            )
            .pow(1.0 / p)
            .view(batch_size, length, c)
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )
