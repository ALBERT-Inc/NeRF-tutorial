import torch
import torch.nn as nn
import numpy as np


class Intrinsic(nn.Module):
    """Intrinsic parameter class.

    Args:
        image_wh (list[int]/torch.tensor/numpy.array):
                image size of training data.
        focals (list[float]/torch.tensor/numpy.array):
                focal lengths in [fx, fy] style.
        cxcy (list[float]/torch.tensor/numpy.array):
                image center coordinates in [cx, cy] style.
        fixed, normalize_focals: 
                see NeRFConfig class for more details.
    """

    def __init__(self,
                 image_wh,
                 focals=None,
                 cxcy=None,
                 fixed=False,
                 normalize_focals=True):
        super().__init__()

        self.register_buffer(
            "image_wh", torch.tensor(image_wh, dtype=torch.float32)
        )
        self.normalize_focals = normalize_focals
        W, H = image_wh

        if focals is None:
            if normalize_focals:
                self.fs = nn.Parameter(
                    torch.ones(2, dtype=torch.float32), requires_grad=True
                )
            else:
                self.fs = nn.Parameter(
                    torch.tensor(image_wh, dtype=torch.float32),
                    requires_grad=True
                )
        else:
            assert normalize_focals is False

            if fixed:
                self.register_buffer(
                    "fs", torch.tensor(focals, dtype=torch.float32)
                )
            else:
                self.fs = nn.Parameter(
                    torch.tensor(focals, dtype=torch.float32),
                    requires_grad=True
                )

        if cxcy is None:
            cxcy = torch.tensor([W*0.5, H*0.5], dtype=torch.float32)
        else:
            cxcy = torch.tensor(cxcy, dtype=torch.float32)
        self.register_buffer("cxcy", cxcy)

    def get_cam_pixels(self, device=None):
        W, H = self.image_wh

        v, u = np.mgrid[:H, :W].astype(np.float32)
        u = torch.tensor(u, dtype=torch.float32, device=device)
        v = torch.tensor(v, dtype=torch.float32, device=device)

        if self.normalize_focals:
            fs = (
                self.fs * torch.tensor(self.image_wh,
                                       dtype=torch.float32,
                                       device=device)
            )
        else:
            fs = self.fs
        cxcy = self.cxcy

        _x = (u - cxcy[0]) / fs[0]
        _y = (v - cxcy[1]) / fs[1]
        _z = torch.ones_like(_x, device=device)
        _w = torch.ones_like(_x, device=device)

        xyzw = torch.stack([_x, _y, _z, _w], dim=2)
        return xyzw.reshape(-1, 4)
