import os
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R


def extract_camera_info(file_path):
    """extract camera information from colmap `cameras.txt` file.

    Args:
        file_path (str): path to `cameras.txt` file.
    Returns:
        cameras (list): list of camera informations.

        (example): [{'CAMERA_ID': '1',
                     'MODEL': 'OPENCV',
                     'WIDTH': 3000,
                     'HEIGHT': 2000,
                     'cmtx': array([[2.06903e+03, 0.00000e+00, 1.50000e+03],
                                    [0.00000e+00, 2.06827e+03, 1.00000e+03],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]]),
                     'dist': array([ 0.00222674, -0.00248576,  0.00103703,  0.00043848])},
                    {'CAMERA_ID': '2', ...}]
    """

    cameras = []

    with open(file_path) as fr:
        fr.readline()
        keys = fr.readline().strip().replace(" ", "")[1:].split(",")
        pks = keys[-1]
        keys = keys[:-1]
        line = fr.readline().strip()
        num_data = int(line.replace(" ", "")[1:].split(":")[-1])

        for idx in range(num_data):
            values = fr.readline().strip().split(" ")
            dict_values = {}
            for k, v in zip(keys, values):
                if k in ["WIDTH", "HEIGHT"]:
                    v = int(v)
                dict_values[k] = v

            # reference: https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
            if dict_values["MODEL"] == "OPENCV":
                # opencv: fx, fy, cx, cy, k1, k2, p1, p2
                pvs = [float(v) for v in values[len(keys):]]
                K = np.array([
                    [pvs[0],     0., pvs[2]],
                    [    0., pvs[1], pvs[3]],
                    [    0.,     0.,     1.],
                ])
                dist_coeffs = np.array(pvs[4:])
                dict_values["cmtx"] = K
                dict_values["dist"] = dist_coeffs
            else:
                raise NotImplementedError

            cameras.append(dict_values)

    return cameras


def extract_image_info(file_path):
    """extract image　information from colmap `images.txt` file.

    Args:
        file_path (str): path to `images.txt` file.
    Returns:
        images (list): list of image informations.

        (example): [{'CAMERA_ID': '1',
                     'NAME': 'DSC_0007.JPG',
                     'pose': array([[ 0.98186656,  0.12955013, -0.13840096, -3.13399   ],
                                    [-0.14768339,  0.98045887, -0.1299616 ,  0.237271  ],
                                    [ 0.11885991,  0.14804447,  0.98181218,  4.55247   ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])},
                    {'CAMERA_ID': '3', ...}]
    """

    images = []

    with open(file_path) as fr:
        fr.readline()
        keys = fr.readline().strip().replace(" ", "")[1:].split(",")
        fr.readline()
        line = fr.readline().strip()
        num_data = int(line.replace(" ", "")[1:].split(":")[1].split(",")[0])

        for idx in range(num_data):
            values = fr.readline().strip().split(" ")
            dict_values = {}
            rotation_q = []
            translation = []
            for k, v in zip(keys, values):
                if k in ["QW", "QX", "QY", "QZ"]:
                    rotation_q.append(float(v))
                elif k in ["TX", "TY", "TZ"]:
                    translation.append(float(v))
                else:
                    dict_values[k] = v

            quat = np.array([rotation_q[idx] for idx in [1, 2, 3, 0]])
            rotation = R.from_quat(quat)
            translation = np.array(translation)

            rotation = rotation.as_matrix()
            t = translation[:, None]
            pose = np.concatenate([rotation, t], axis=-1)
            pose = np.concatenate([pose, np.array([0, 0, 0, 1])[None]], axis=0)
            dict_values["pose"] = pose

            images.append(dict_values)

            fr.readline()

    return images


def undistort_image(image_info, camera_infos, image_dir, size=None):
    """undistort image using camera's distortion coefficients.

    Args:
        image_info (dict): dictionary of camera information.

            (example): {'CAMERA_ID': '1',
                        'NAME': 'DSC_0007.JPG',
                        'pose': array([[ 0.98186656,  0.12955013, -0.13840096, -3.13399   ],
                                       [-0.14768339,  0.98045887, -0.1299616 ,  0.237271  ],
                                       [ 0.11885991,  0.14804447,  0.98181218,  4.55247   ],
                                       [ 0.        ,  0.        ,  0.        ,  1.        ]])}

        camera_infos (list): list of camera informations.

            (example): [{'CAMERA_ID': '1',
                     'MODEL': 'OPENCV',
                     'WIDTH': 3000,
                     'HEIGHT': 2000,
                     'cmtx': array([[2.06903e+03, 0.00000e+00, 1.50000e+03],
                                    [0.00000e+00, 2.06827e+03, 1.00000e+03],
                                    [0.00000e+00, 0.00000e+00, 1.00000e+00]]),
                     'dist': array([ 0.00222674, -0.00248576,  0.00103703,  0.00043848])},
                    {'CAMERA_ID': '2', ...}]

        image_dir (str): path to image directory.
        size (tuple): image size to resize, like (resize_width, resize_height).
    Returns:
        img_undist (PIL.Image.Image): PIL image after undistort and resize.
        new_camera_matrix (numpy.array): camera intrinsic matrix after undistort and resize.

    """

    file_name = image_info["NAME"]
    camera_id = image_info["CAMERA_ID"]

    img = Image.open(os.path.join(image_dir, file_name))

    w, h = img.size
    camera_info = [
        ci for ci in camera_infos
        if ci["CAMERA_ID"] == camera_id
    ][0]

    camera_matrix = camera_info["cmtx"]
    dist_coef = camera_info["dist"]
    new_camera_matrix, area_to_crop = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coef, (w, h), 1, (w, h))

    new_img = cv2.undistort(
        np.array(img), camera_matrix, dist_coef, None, new_camera_matrix)

    x, y, w, h = area_to_crop
    img_undist = Image.fromarray(new_img[y:y+h, x:x+w])

    if size is not None:
        img_undist = img_undist.resize(size)
        new_w, new_h = size

        camera_matrix_scaler = np.array([
            [ new_w/w,       1, new_w/w],
            [       1, new_h/h, new_h/h],
            [       1,       1,       1],
        ], dtype=np.float32)
        new_camera_matrix = new_camera_matrix * camera_matrix_scaler

    return img_undist, new_camera_matrix


def extract_points3D(file_path, image_infos):
    """extract reconstructed 3d-points from colmap's `points3D.txt` file.

    Args:
        file_path (str): path to `points3D.txt` file.
        image_infos (list): list of image informations extracted by `extract_image_info` func.
    Returns:
        xyzs (numpy.array): coordinates of points. this takes (num_points, 3) shape.
        rgbs (numpy.array): colors of points. this takes (num_points, 3) shape.
        visibility_matrix (numpy.array): correspondence of 3d-point and image.
                if element is 1, the 3d-point is in the image.
                else (0), the 3d-point is NOT in the image.
                this matrix takes (num_points, num_images) shape.
    """

    xyzs, rgbs = [], []
    num_points_header = "Number of points: "
    imageid2index = {
        info["IMAGE_ID"]: idx
        for idx, info in enumerate(image_infos)
    }

    with open(file_path) as fr:
        for i in range(2):
            fr.readline()

        # read num_points
        line = fr.readline().rstrip().split(", ")[0]
        start_idx = line.find(num_points_header)
        num_points = int(line[start_idx+len(num_points_header):])
        visibility_matrix = np.zeros((num_points, len(image_infos)))

        for lid, line in enumerate(fr):
            line = line.rstrip().split(" ")
            xyzs.append(list(map(float, line[1:4])))
            rgbs.append(list(map(int, line[4:7])))

            visible_ids = np.array([
                imageid2index[image_id] for image_id in line[8::2]
            ])
            visibility_matrix[lid, visible_ids] = 1

    xyzs = np.array(xyzs, dtype=np.float32)
    rgbs = np.array(rgbs, dtype=np.float32)

    return xyzs, rgbs, visibility_matrix


def calc_bds(xyzs, poses, visibility_matrix):
    """calculate requirements of rendering range.

    Args:
        xyzs (numpy.array): coordinates of points. this takes (num_points, 3) shape.
        poses (numpy.array): pose matrices. this takes (N_images, 4, 4) shape.
        visibility_matrix (numpy.array): correspondence of 3d-point and image.
                if element is 1, the 3d-point is in the image.
                else (0), the 3d-point is NOT in the image.
                this matrix takes (num_points, num_images) shape.
    Returns:
        bds_min (float): minimum distance for rendering.
        bds_max (float): maximum distance for rendering.
    """

    xyzs = xyzs[:, None].transpose(2, 0, 1)
    poses = poses.transpose(1, 2, 0)

    # calc all distance between cameras and points.
    distance = xyzs - poses[:3, 3:4]
    distance_in_z = (distance * poses[:3, 2:3]).sum(axis=0)

    # use distance correspond to camera.
    distance_in_z = distance_in_z[visibility_matrix == 1]

    bds_min = np.percentile(distance_in_z, .1)
    bds_max = np.percentile(distance_in_z, 99.9)

    return bds_min, bds_max
