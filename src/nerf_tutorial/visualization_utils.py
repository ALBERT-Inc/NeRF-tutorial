import io
import math
import torch
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

from PIL import Image
from functools import partial
from .nerf_utils import render_nerf
from scipy.spatial.transform import Rotation as R


def plot_cameras(ax, poses, f, c, screen_size, label,
                 alpha=0.3, c_cam_point="#00aa00", c_cam_line="#0a0000"):
    """plot cameras in the world-coordinate system respect to the poses.

    Args:
        ax (matplotlib.Axes): matplotlib axes to plot.
        poses (numpy.array): camera poses with (N, 4, 4) shape.
        f (list[float]): focal lengths in [fx, fy] style.
        c (list[float]): image center coordinates in [cx, cy] style.
        screen_size (list[float]): image size in [W, H] style.
        label (str): label of plotted lines.
        alpha (float): alpha value of plots.
        c_cam_point (str): color code of camera point.
        c_cam_line (str): color code of camera trajectory.
    """

    W, H = screen_size

    # make grid
    v, u = torch.meshgrid(torch.arange(800), torch.arange(1200))
    u = u.to(torch.float32)
    v = v.to(torch.float32)

    _x = (u - c[0]*0.5) / f[0] * 0.5
    _y = (v - c[1]*0.5) / f[1] * 0.5
    _z = torch.ones_like(u) * 1
    xyz = torch.stack([_x, _y, _z], dim=2)

    # get corner of screen in camera-coordinates
    lxty = xyz[0, 0]
    lxby = xyz[-1, 0]
    rxty = xyz[0, -1]
    rxby = xyz[-1, -1]
    scr_coords = torch.stack([lxty, lxby, rxty, rxby], axis=0)

    scr_coords_w = []
    for p in torch.tensor(poses):
        t = p[:, -1][None][:, :3]
        r = p[:, :-1][None].repeat(len(scr_coords), 1, 1)
        r = r.permute(0, 2, 1)[:, :, :3]

        p = torch.bmm(scr_coords[:, None], r)[:, 0]
        p = p + t

        scr_coords_w.append(p)

        to_lxty = torch.stack([t[0], p[0]], dim=0).T
        to_lxby = torch.stack([t[0], p[1]], dim=0).T
        to_rxty = torch.stack([t[0], p[2]], dim=0).T
        to_rxby = torch.stack([t[0], p[3]], dim=0).T
        scr_square = p[[0, 1, 3, 2, 0]].T

        ax.scatter(
            t[0][0], t[0][1], t[0][2],
            color=c_cam_point, s=30, alpha=1.
        )
        ax.plot(
            to_lxty[0], to_lxty[1], to_lxty[2],
            color=c_cam_point, alpha=alpha
        )
        ax.plot(
            to_lxby[0], to_lxby[1], to_lxby[2],
            color=c_cam_point, alpha=alpha
        )
        ax.plot(
            to_rxty[0], to_rxty[1], to_rxty[2],
            color=c_cam_point, alpha=alpha
        )
        ax.plot(
            to_rxby[0], to_rxby[1], to_rxby[2],
            color=c_cam_point, alpha=alpha
        )
        line, = ax.plot(
            scr_square[0], scr_square[1], scr_square[2],
            color=c_cam_line, alpha=alpha
        )

    line.set_label(label)


class IpywidgetsRenderer(object):
    """NeRF Rendering class with Ipywidgets.

    Args:
        nerf (torch.nn.Module): nerf model.
        camera_parameters (dict): dict of camera parameters.
        value (float): move/rotate value.
        only_coarse (bool): whether to use 
                            coarse network for rendering.
    """

    def __init__(self, nerf, camera_parameters, 
                 value=0.1, only_coarse=False):
        self.nerf = nerf
        self.camera_parameters = camera_parameters
        self.value = value
        self.only_coarse = only_coarse
        self.movedirs2positions = {
            "UP": "header1",
            "FORWARD": "header2",
            "LEFT": "left",
            "RIGHT": "right",
            "DOWN": "footer1",
            "BACKWARD": "footer2",
        }
        self.rotdirs2positions = {
            "UP": "header",
            "LEFT": "left",
            "RIGHT": "right",
            "DOWN": "footer",
        }

        move_items, rotate_items = [], []
        for k, v in self.movedirs2positions.items():
            move_button = widgets.Button(
                description='Move {}'.format(k),
                layout=widgets.Layout(width='auto', grid_area=v)
            )
            move_button.on_click(partial(self._move_button_clicked, k))
            move_items.append(move_button)

        for k, v in self.rotdirs2positions.items():
            rotate_button = widgets.Button(
                description='Rotate {}'.format(k),
                layout=widgets.Layout(width='auto', grid_area=v)
            )
            rotate_button.on_click(partial(self._rotate_button_clicked, k))
            rotate_items.append(rotate_button)

        m_controller_layout = widgets.Layout(
            width='35%',
            grid_template_rows='auto auto auto',
            grid_template_columns='50% 50%',
            grid_template_areas='''
                    "header1 header2"
                    "left right"
                    "footer1 footer2"
                    '''
        )
        r_controller_layout = widgets.Layout(
            width='35%',
            grid_template_rows='auto auto auto',
            grid_template_columns='50% 50%',
            grid_template_areas='''
                    "header header"
                    "left right"
                    "footer footer"
                    '''
        )
        self.image = widgets.Image(
            value=self._render_nerf(
                self.nerf, self.camera_parameters,
                only_coarse=self.only_coarse),
            format='png',
            width='70%'
        )
        self.controller = widgets.HBox([
            widgets.GridBox(children=move_items, layout=m_controller_layout),
            widgets.GridBox(children=rotate_items, layout=r_controller_layout)
        ])
        viewer = widgets.VBox([self.image, self.controller])
        display(viewer)

    def _render_nerf(self, nerf, camera_parameters, only_coarse=False):
        Cs = render_nerf(nerf, camera_parameters, only_coarse=only_coarse)
        img = Image.fromarray((Cs*255).astype(np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        return img_bytes

    def _move_button_clicked(self, direction, e):
        device = self.camera_parameters["device"]
        move_value = self.value

        if direction == "UP":
            move_vec = torch.tensor(
                [0, 1, 0, 1], dtype=torch.float32, device=device)
            move_value *= -1.
        elif direction == "LEFT":
            move_vec = torch.tensor(
                [1, 0, 0, 1], dtype=torch.float32, device=device)
            move_value *= -1.
        elif direction == "RIGHT":
            move_vec = torch.tensor(
                [1, 0, 0, 1], dtype=torch.float32, device=device)
        elif direction == "DOWN":
            move_vec = torch.tensor(
                [0, 1, 0, 1], dtype=torch.float32, device=device)
        elif direction == "FORWARD":
            move_vec = torch.tensor(
                [0, 0, 1, 1], dtype=torch.float32, device=device)
        elif direction == "BACKWARD":
            move_vec = torch.tensor(
                [0, 0, 1, 1], dtype=torch.float32, device=device)
            move_value *= -1.
        else:
            raise ValueError("move direction is invalid")

        o_vec = torch.tensor(
            [0, 0, 0, 1], dtype=torch.float32, device=device)
        vecs = torch.stack([move_vec, o_vec], dim=0)
        pose = self.camera_parameters["pose"].repeat(len(vecs), 1, 1)
        _ds = torch.bmm(pose, vecs[..., None])[:, :, 0][:, :3]
        d = _ds[0] - _ds[1]
        d = d / torch.norm(d)

        move_matrix = torch.eye(4, device=device)
        move_matrix[:3, 3] = d * move_value
        self.camera_parameters["pose"] = \
            torch.bmm(move_matrix[None], self.camera_parameters["pose"])
        self.image.value = \
            self._render_nerf(
                self.nerf, self.camera_parameters,
                only_coarse=self.only_coarse)

    def _rotate_button_clicked(self, direction, e):
        device = self.camera_parameters["device"]

        rot_value = self.value
        if direction == "UP":
            r_axis = torch.tensor(
                [1, 0, 0, 1], dtype=torch.float32, device=device)
        elif direction == "LEFT":
            r_axis = torch.tensor(
                [0, 1, 0, 1], dtype=torch.float32, device=device)
            rot_value *= -1.
        elif direction == "RIGHT":
            r_axis = torch.tensor(
                [0, 1, 0, 1], dtype=torch.float32, device=device)
        elif direction == "DOWN":
            r_axis = torch.tensor(
                [1, 0, 0, 1], dtype=torch.float32, device=device)
            rot_value *= -1.
        else:
            raise ValueError("move direction is invalid")

        o_vec = torch.tensor(
            [0, 0, 0, 1], dtype=torch.float32, device=device)
        vecs = torch.stack([r_axis, o_vec], dim=0)
        pose = self.camera_parameters["pose"].repeat(len(vecs), 1, 1)
        _ds = torch.bmm(pose, vecs[..., None])[:, :, 0][:, :3]

        r_axis = _ds[0] - _ds[1]
        r_axis = r_axis / torch.norm(r_axis)
        r_axis = r_axis.detach().cpu().numpy()

        rot_matrix = torch.eye(4, device=device)
        rot_matrix[:3, :3] = torch.tensor(
            R.from_rotvec(math.pi * rot_value * r_axis).as_matrix(),
            device=device
        )
        self.camera_parameters["pose"][0, :3, :3] = \
            torch.bmm(
                rot_matrix[None],
                self.camera_parameters["pose"]
            )[0, :3, :3]
        self.image.value = \
            self._render_nerf(
                self.nerf, self.camera_parameters,
                only_coarse=self.only_coarse)
