# slightly modified of https://github.com/ALBERT-Inc/blog_nerf/blob/master/NeRF.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
from .radiance_field import RadianceField
from .nerf_utils import (
    position_encode, position_encode_barf,
    split_ray, sample_coarse, sample_fine, _pcpdf, ray
)


class NeRFConfig(object):
    """Config class for NeRF.

    Args:
        dim_former (int): hidden dim in nerf network before sigma.
        dim_latter (int): hidden dim in nerf network after sigma.
        t_n (float): nearest distance of rendering range.
        t_f (float): farthest distance of rendering range.
        L_x (int): frequency size of `coordinate` positional encoding.
        L_d (int): frequency size of `direction` positional encoding.
        N_c (int): bin size of integral in nerf `coarse` rendering.
        N_f (int): bin size of integral in nerf `fine` rendering.
        c_bg (tuple[float]): background color of training images.
        fine_network (bool): whether to use fine network.
        normalize_focals (bool): whether to hold focal length value
                as aspect to the image size.
        intrinsic_fixed (bool): whether to adjust intrinsic parameters
                with network training.
        extrinsic_fixed (bool): whether to adjust extrinsic parameters
                with network training.
        extrinsic_transf (torch.tensor): matrix for transform coord-systems,
                like right-handed to left-handed.
        barf (bool): whether to use coarse-to-fine registration in BARF.
        barf_start_epoch (int): start epoch of coarse-to-fine registration.
        barf_end_epoch (int): end epoch of coarse-to-fine registration
        nerfmm (bool): whether to joint-optimize with camera parameters.
    """

    def __init__(self,
                dim_former=256,
                dim_latter=128,
                t_n=0.,
                t_f=1.,
                L_x=10,
                L_d=4,
                N_c=64,
                N_f=128,
                c_bg=None,
                fine_network=True,
                normalize_focals=False,
                intrinsic_fixed=True,
                extrinsic_fixed=True,
                extrinsic_transf=None,
                barf=False,
                barf_start_epoch=400,
                barf_end_epoch=800,
                nerfmm=False):

        if (not nerfmm) and (extrinsic_transf is not None):
            msg = '`extrinsic_transf` should be None when `nerfmm` is False'
            raise ValueError(msg)

        self.dim_former = dim_former
        self.dim_latter = dim_latter
        self.t_n = t_n
        self.t_f = t_f
        self.L_x = L_x
        self.L_d = L_d
        self.N_c = N_c
        self.N_f = N_f
        self.c_bg = c_bg
        self.fine_network = fine_network
        self.barf = barf
        self.barf_start_epoch = barf_start_epoch
        self.barf_end_epoch = barf_end_epoch

        self.normalize_focals = normalize_focals
        self.intrinsic_fixed = intrinsic_fixed

        self.nerfmm = nerfmm
        self.extrinsic_transf = extrinsic_transf
        self.extrinsic_fixed = extrinsic_fixed

    def nerf_kwargs(self):
        """get NeRF-class-initialization kwargs.
        """
        
        return {
            "dim_former": self.dim_former,
            "dim_latter": self.dim_latter,
            "t_n": self.t_n,
            "t_f": self.t_f,
            "L_x": self.L_x,
            "L_d": self.L_d,
            "N_c": self.N_c,
            "N_f": self.N_f,
            "c_bg": self.c_bg,
            "fine_network": self.fine_network,
            "barf": self.barf,
        }

    def intrinsic_kwargs(self):
        """get Intrinsic-class-initialization kwargs.
        """
        
        return {
            "normalize_focals": self.normalize_focals,
            "fixed": self.intrinsic_fixed,
        }

    def extrinsic_kwargs(self):
        """get Extrinsic-class-initialization kwargs.
        """
        
        if self.nerfmm:
            extrinsic_kwargs = {
                "fixed": self.extrinsic_fixed,
                "transf": self.extrinsic_transf,
            }
        else:
            extrinsic_kwargs = {
                "fixed": self.extrinsic_fixed,
            }

        return extrinsic_kwargs


class NeRF(nn.Module):
    """rendering class of Neural Radiance Fields
    
    Args: see the document of NeRFConfig class for more details.
    """

    def __init__(self,
                 dim_former=256,
                 dim_latter=128,
                 t_n=0.,
                 t_f=1.,
                 L_x=10,
                 L_d=4,
                 N_c=128,
                 N_f=128,
                 c_bg=(1., 1., 1.),
                 barf=False,
                 rf=RadianceField,
                 fine_network=False):
        super().__init__()
        self.t_n = t_n
        self.t_f = t_f
        self.L_x = L_x
        self.L_d = L_d
        self.N_c = N_c
        self.N_f = N_f
        self.c_bg = c_bg

        if barf:
            self.register_buffer(
                "alpha", torch.tensor(0., dtype=torch.float32))

            self.pe = position_encode_barf
            input_ch = 6 * L_x + 3
            middle_ch = 6 * L_d + 3
        else:
            self.pe = position_encode
            input_ch = 6 * L_x
            middle_ch = 6 * L_d

        self.rf_c = rf(
            input_ch=input_ch, middle_ch=middle_ch,
            dim_former=dim_former, dim_latter=dim_latter)

        if fine_network:
            self.rf_f = rf(
                input_ch=input_ch, middle_ch=middle_ch,
                dim_former=dim_former, dim_latter=dim_latter)

        self.fine_network = fine_network

    def _device(self):
        return next(self.parameters()).device

    def _rgb_and_weight(self, o, d, t, N, network):
        batch_size = o.shape[0]

        x = ray(o, d, t)
        x = x.view(batch_size, N, -1)
        d = d[:, None].repeat(1, N, 1)

        x = x.view(batch_size * N, -1)
        d = d.view(batch_size * N, -1)

        # forward.
        rgb, sigma = self.radiance_field(x, d, network=network)

        rgb = rgb.view(batch_size, N, -1)
        sigma = sigma.view(batch_size, N, -1)

        delta = F.pad(t[:, 1:] - t[:, :-1], (0, 1), mode='constant', value=1e8)
        mass = sigma[..., 0] * delta
        mass = F.pad(mass, (1, 0), mode='constant', value=0.)

        alpha = 1. - torch.exp(- mass[:, 1:])
        T = torch.exp(- torch.cumsum(mass[:, :-1], dim=1))
        w = T * alpha
        return rgb, w

    def forward(self, o, d, only_coarse=False):
        batch_size = o.shape[0]
        device = o.device

        partitions = split_ray(self.t_n, self.t_f, self.N_c, batch_size)
        _t_c = sample_coarse(partitions)
        t_c = torch.tensor(_t_c)
        t_c = t_c.to(device)

        rgb_c, w_c = self._rgb_and_weight(
            o, d, t_c, self.N_c, network="coarse")
        C_c = torch.sum(w_c[..., None]*rgb_c, dim=1)

        if self.c_bg is not None:
            bg = torch.tensor(self.c_bg, device=device, dtype=torch.float32)
            bg = bg.view(1, 3)
            C_c += (1. - torch.sum(w_c, axis=1, keepdims=True)) * bg

        if self.fine_network and (not only_coarse):
            _w_c = w_c.detach().cpu().numpy()
            t_f = sample_fine(partitions, _w_c, _t_c, self.N_f)
            t_f = torch.tensor(t_f)
            t_f = t_f.to(device)

            rgb_f, w_f = self._rgb_and_weight(
                o, d, t_f, self.N_f + self.N_c, network="fine")
            C_f = torch.sum(w_f[..., None]*rgb_f, dim=1)
            if self.c_bg is not None:
                C_f += (1. - torch.sum(w_f, axis=1, keepdims=True)) * bg

            output = [C_c, C_f]
        else:
            output = [C_c]

        return output

    def radiance_field(self, x, d, network="coarse"):
        if network == "coarse":
            rf = self.rf_c
        elif network == "fine":
            assert hasattr(self, "rf_f")
            rf = self.rf_f
        else:
            msg = "`network` must be `fine` or `coarse`."
            raise ValueError(msg)

        if hasattr(self, "alpha"):
            x = self.pe(x, self.L_x, self.alpha)
            d = self.pe(d, self.L_d, self.alpha)
        else:
            x = self.pe(x, self.L_x)
            d = self.pe(d, self.L_d)

        rgb, sigma = rf(x, d)

        return rgb, sigma


class NeRFLoss(nn.Module):
    """Loss for NeRF training.

    Args:
        nerf (torch.nn.Module): nerf model.
        intrinsic (torch.nn.Module): intrinsic parameters.
        extrinsic (torch.nn.Module): extrinsic parameters.
    """

    def __init__(self, nerf, intrinsic=None, extrinsic=None):
        super().__init__()
        self.nerf = nerf
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        
    def forward(self, inputs):
        device = self.nerf._device()
        
        if (self.intrinsic is not None) and (self.extrinsic is not None):
            C, perm_in_img, img_ids = inputs
            _d = self.intrinsic.get_cam_pixels(device=device)[perm_in_img]
            o, d = self.extrinsic.cam2world(_d, img_ids)
        else:
            C, o, d = inputs
            o = torch.tensor(o, dtype=torch.float32, device=device)
            d = torch.tensor(d, dtype=torch.float32, device=device)
            
        C = torch.tensor(C, dtype=torch.float32, device=device)
        
        output = self.nerf(o, d)
        
        loss = 0.
        for C_pred in output:
            loss += F.mse_loss(C_pred, C)
        return loss
