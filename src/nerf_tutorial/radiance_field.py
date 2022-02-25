# from https://github.com/ALBERT-Inc/blog_nerf/blob/master/NeRF.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


class RadianceField(nn.Module):
    """Radiance Field Functions.
    This is ``F_Theta`` in the paper.
    """

    def __init__(self, input_ch, middle_ch, dim_former=256, dim_latter=128):
        super(RadianceField, self).__init__()
        self.layer0 = nn.Linear(input_ch, dim_former)
        self.layer1 = nn.Linear(dim_former, dim_former)
        self.layer2 = nn.Linear(dim_former, dim_former)
        self.layer3 = nn.Linear(dim_former, dim_former)
        self.layer4 = nn.Linear(dim_former, dim_former)
        self.layer5 = nn.Linear(dim_former+input_ch, dim_former)
        self.layer6 = nn.Linear(dim_former, dim_former)
        self.layer7 = nn.Linear(dim_former, dim_former)
        self.sigma = nn.Linear(dim_former, 1)
        self.layer8 = nn.Linear(dim_former, dim_former)
        self.layer9 = nn.Linear(dim_former+middle_ch, dim_latter)
        self.layer10 = nn.Linear(dim_latter, dim_latter)
        self.layer11 = nn.Linear(dim_latter, dim_latter)
        self.layer12 = nn.Linear(dim_latter, dim_latter)
        self.rgb = nn.Linear(dim_latter, 3)

        self.apply(_init_weights)

    def forward(self, x, d):
        """Apply function.
        Args:
            x (tensor, [batch_size, 3]): Points on rays.
            d (tensor, [batch_size, 3]): Direction of rays.
        Returns:
            rgb (tensor, [batch_size, 3]): Emitted color.
            sigma (tensor, [batch_size, 1]): Volume density.
        """

        # forward
        h = F.relu(self.layer0(x))
        h = F.relu(self.layer1(h))
        h = F.relu(self.layer2(h))
        h = F.relu(self.layer3(h))
        h = F.relu(self.layer4(h))
        h = torch.cat([h, x], axis=1)
        h = F.relu(self.layer5(h))
        h = F.relu(self.layer6(h))
        h = F.relu(self.layer7(h))
        sigma = F.softplus(self.sigma(h))
        h = self.layer8(h)
        h = torch.cat([h, d], axis=1)
        h = F.relu(self.layer9(h))
        h = F.relu(self.layer10(h))
        h = F.relu(self.layer11(h))
        h = F.relu(self.layer12(h))
        rgb = torch.sigmoid(self.rgb(h))

        return rgb, sigma
