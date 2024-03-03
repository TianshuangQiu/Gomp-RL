from autolab_core import DepthImage, CameraIntrinsics, RigidTransform, PointCloud
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
import pyvista as pv
import cv2
from copy import deepcopy
import open3d as o3d
import pdb
import torch
from matplotlib import pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import cowsay
from datetime import datetime

BIN_LIMIT = np.array([[105, 65], [490, 260]])

# cam_intr = CameraIntrinsics.load("realsense/realsense.intr")
# cam_extr = RigidTransform.load("realsense/realsense_to_world.tf")

pointcloud = PointCloud(
    np.load("scene_3/view.npy"),
    frame="ur5",
)

data = pointcloud.data

img = cv2.imread("scene_3/color.png")
print(data[2].argsort())

crop = deepcopy(data.T)
crop = crop.reshape((480, 640, -1))
crop = crop[BIN_LIMIT[0][1] : BIN_LIMIT[1][1], BIN_LIMIT[0][0] : BIN_LIMIT[1][0]]
# plotter = pv.Plotter()
# plotter.add_points(crop.reshape(-1, 3))
# plotter.show()
img_original = deepcopy(img)
crop_img = img[BIN_LIMIT[0][1] : BIN_LIMIT[1][1], BIN_LIMIT[0][0] : BIN_LIMIT[1][0]]

device = torch.device("cuda:5")
print("running on ", device)
start = datetime.now()
sam = sam_model_registry["vit_h"](checkpoint="assets/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamPredictor(sam)
mask_generator.set_image(img_original)
masks, quality, _ = mask_generator.predict(
    point_coords=BIN_LIMIT,
    point_labels=np.array([1, 1]),
)

current_mask = masks[np.argmax(quality)]
current_mask[:, : BIN_LIMIT[0][0]] = True
current_mask[: BIN_LIMIT[0][1]] = True
current_mask[:, BIN_LIMIT[1][0] :] = True
current_mask[BIN_LIMIT[1][1] :] = True


def check_validity():
    valid_full = np.where(~current_mask)
    valid_crop = crop[valid_full[0] - BIN_LIMIT[0][1], valid_full[1] - BIN_LIMIT[0][0]]
    tallest_valid = np.argsort(valid_crop[:, -1])[-1]
    idx = np.unravel_index(tallest_valid, BIN_LIMIT[1] - BIN_LIMIT[0])
    return valid_full[1][idx[1]], valid_full[0][idx[0]]


pick_point = check_validity()
masks, quality, _ = mask_generator.predict(
    point_coords=np.array([pick_point]),
    point_labels=np.array([1]),
)
mask = masks[np.argmax(quality)]
img[current_mask] = 0
img[mask] = [255, 0, 0]
current_mask = np.logical_or(current_mask, mask)

cowsay.cow(f"LFGGG {datetime.now()-start}")
