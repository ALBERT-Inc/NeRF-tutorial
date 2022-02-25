import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R


def rotvec2matrix(rotvec):
    """Convert rotation vector to rotatin matrix.
    Args:
        rotvec (Tensor, [N, 3] or [3,]): Rotation vector to convert to rotatin
            matrix.
    Returns:
        matrix (Tensorm [N, 3, 3] or [3, 3]): Rotation matrix.
    """
    stacked = True
    if rotvec.dim() == 1:
        rotvec = rotvec[None]
        stacked = False

    _theta = torch.norm(rotvec, dim=1, keepdim=True)
    theta = torch.max(_theta, torch.full_like(_theta, 1e-16))
    k = rotvec / theta
    
    k_x = k[:, 0]
    k_y = k[:, 1]
    k_z = k[:, 2]
    k_0 = torch.zeros_like(k_x)
    K = torch.stack([torch.stack([ k_0, -k_z,  k_y], axis=1),  # NOQA
                     torch.stack([ k_z,  k_0, -k_x], axis=1),  # NOQA
                     torch.stack([-k_y,  k_x,  k_0], axis=1)],
                    axis=1)

    sin_theta = torch.sin(theta)[:, None]
    cos_theta = torch.cos(theta)[:, None]
    I = torch.eye(3, device=rotvec.device)[None]  # NOQA

    matrix = I + sin_theta * K + (1 - cos_theta) * K @ K

    if not stacked:
        matrix = matrix[0]
    return matrix


class RVecExtrinsic(nn.Module):
    """Extrinsic parameter class with Rotation Vector.
    
    Args:
        image_num (int): the number of training images.
        poses (numpy.array): pose matrices with (N, 4, 4) shape.
        fixed, transf: see NeRFConfig class for more details.
    """

    def __init__(self,
                 image_num,
                 poses=None,
                 fixed=False,
                 transf=None):

        super().__init__()

        self.register_buffer(
            "pose", torch.tensor(poses, dtype=torch.float32))

        if transf is None:
            self.register_buffer("transf", torch.tensor([
                [  1,  0,  0,  0],
                [  0,  1,  0,  0],
                [  0,  0,  1,  0],
                [  0,  0,  0,  1],
            ], dtype=torch.float32)[None])
        else:
            assert transf.shape == (4, 4)
            self.register_buffer("transf", transf[None])

        if poses is None:
            self.translation = nn.Parameter(
                torch.zeros((image_num, 3), dtype=torch.float32),
                requires_grad=True
            )
            self.rotation = nn.Parameter(
                torch.zeros((image_num, 3), dtype=torch.float32),
                requires_grad=True
            )
        else:
            translation = poses[:, :3, 3]
            rotation = [(p@self.transf[0].numpy())[:3, :3] for p in poses]
            rotation = np.stack([
                R.from_matrix(r).as_rotvec() for r in rotation])

            if fixed:
                self.register_buffer(
                    "translation",
                    torch.tensor(translation, dtype=torch.float32)
                )
                self.register_buffer(
                    "rotation",
                    torch.tensor(rotation, dtype=torch.float32)
                )
            else:
                self.translation = nn.Parameter(
                    torch.tensor(translation, dtype=torch.float32),
                    requires_grad=True
                )
                self.rotation = nn.Parameter(
                    torch.tensor(rotation, dtype=torch.float32),
                    requires_grad=True
                )

    def cam2world(self, xyzw, image_ids):
        """transforms from image-plane coordinates to world cooordinates.
        
        Args:
            xyzw (torch.tensor): pixels in image-plane coorinates.
                    this takes (N, 4) shape, 4 means (x, y, 1, 1).
            image_ids (torch.tensor): image ids corresponds to `xyzw`.
                    this takes (N, ) shape.
        Returns:
            o (torch.tensor): camera origins in world coordinate.
                this takes (W*H, 3) shape.
            d (torch.tensor): camera directions in world coordinate.
                this takes (W*H, 3) shape.
        """

        _o = np.zeros((len(xyzw), 4), dtype=np.float32)
        _o[:, 3] = 1.
        _o = torch.tensor(_o, dtype=torch.float32, device=xyzw.device)

        o = self.translation[image_ids]
        r = self.rotation[image_ids]
        r = rotvec2matrix(r)

        bottom = torch.tensor(
            [[[0, 0, 0, 1]]], dtype=torch.float32, device=o.device)

        pose = torch.cat([
            torch.cat([r, o[..., None]], dim=2),
            bottom.repeat(len(r), 1, 1)
        ], dim=1)
        pose = torch.bmm(pose, self.transf.repeat(len(pose), 1, 1))

        d = torch.bmm(pose, xyzw[..., None])[:, :, 0][:, :3]
        o = torch.bmm(pose, _o[..., None])[:, :, 0][:, :3]
        d = d - o
        d = d / torch.norm(d, dim=1, keepdim=True)
        return o, d
    
    def __getitem__(self, idx):
        if not isinstance(idx, list):
            idx = [idx]
        
        o = self.translation[idx]
        r = self.rotation[idx]
        r = rotvec2matrix(r)

        bottom = torch.tensor(
            [[[0, 0, 0, 1]]], dtype=torch.float32, device=o.device)

        pose = torch.cat([
            torch.cat([r, o[..., None]], dim=2),
            bottom.repeat(len(r), 1, 1)
        ], dim=1)
        pose = torch.bmm(pose, self.transf.repeat(len(pose), 1, 1))
        return pose


class PoseExtrinsic(nn.Module):
    """Extrinsic parameter class with Rotation Matrix.
    
    Args:
        image_num (int): the number of training images.
        pose (numpy.array/torch.tensor): 
                pose matrices with (N, 4, 4) shape.
        fixed: see NeRFConfig class for more details.
    """

    def __init__(self, image_num, pose, fixed=False):
        super().__init__()
        if fixed:
            self.register_buffer(
                "pose",
                torch.tensor(pose, dtype=torch.float32)
            )
        else:
            self.pose = nn.Parameter(
                torch.tensor(pose, dtype=torch.float32),
                requires_grad=True
            )

    def cam2world(self, xyzw, image_ids):
        """transforms from image-plane coordinates to world cooordinates.
        
        Args:
            xyzw (torch.tensor): pixels in image-plane coorinates.
                    this takes (N, 4) shape, 4 means (x, y, 1, 1).
            image_ids (torch.tensor): image ids corresponds to `xyzw`.
                    this takes (N, ) shape.
        Returns:
            o (torch.tensor): camera origins in world coordinate.
                this takes (W*H, 3) shape.
            d (torch.tensor): camera directions in world coordinate.
                this takes (W*H, 3) shape.
        """

        pose = self.pose[image_ids]

        _o = np.zeros((len(xyzw), 4), dtype=np.float32)
        _o[:, 3] = 1.
        _o = torch.tensor(_o, dtype=torch.float32, device=xyzw.device)

        d = torch.bmm(pose, xyzw[..., None])[:, :, 0][:, :3]
        o = torch.bmm(pose, _o[..., None])[:, :, 0][:, :3]
        d = d - o
        d = d / torch.norm(d, dim=1, keepdim=True)
        return o, d

    def __getitem__(self, idx):
        if not isinstance(idx, list):
            idx = [idx]
        return self.pose[idx]
