"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Isaac Gym Graphics Example
--------------------------
This example demonstrates the use of several graphics operations of Isaac
Gym, including the following
- Load Textures / Create Textures from Buffer
- Apply Textures to rigid bodies
- Create Camera Sensors
  * Static location camera sensors
  * Camera sensors attached to a rigid body
- Retrieve different types of camera images

Requires Pillow (formerly PIL) to write images from python. Use `pip install pillow`
 to get Pillow.
"""


import os
import numpy as np
import pdb
from numpy import sqrt
from isaacgym import gymapi
from isaacgym import gymutil
from PIL import Image as im
from datetime import datetime
from autolab_core.rigid_transformations import RigidTransform
from autolab_core import DepthImage, PointCloud, Point
from autolab_core.orthographic_intrinsics import OrthographicIntrinsics
import shutil
import json
import itertools
from scipy.spatial.transform import Rotation as R
import math


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


current_run_dict = {"time": str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}

# acquire the gym interface
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Graphics Example",
    headless=True,
    custom_parameters=[],
)

# get default params
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.04
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

# sim_params.use_gpu_pipeline = False
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# create sim
sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
)
if sim is None:
    print("*** Failed to create sim")
    quit()

# Create a default ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

if not args.headless:
    # create viewer using the default camera properties
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError("*** Failed to create viewer")

# set up the env grid
num_envs = 1
spacing = 2.5
num_per_row = int(sqrt(num_envs))
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
current_run_dict["num_envs"] = num_envs

asset_root = "assets"
bin_options = gymapi.AssetOptions()
bin_options.use_mesh_materials = True
bin_options.vhacd_enabled = True
bin_options.fix_base_link = True

# load bin asset
bin_asset_file = "urdf/custom/cardboardbin.urdf"
print("Loading asset '%s' from '%s'" % (bin_asset_file, asset_root))
bin_asset = gym.load_asset(sim, asset_root, bin_asset_file, bin_options)
bin_position = (0.0254, -0.55, 0)
bin_pose = gymapi.Transform()
bin_pose.p = gymapi.Vec3(bin_position[0], bin_position[1], bin_position[2])
bin_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

vase_asset_file = "urdf/custom/vase.urdf"
print("Loading asset '%s' from '%s'" % (vase_asset_file, asset_root))
vase_options = gymapi.AssetOptions()
vase_options.use_mesh_materials = True
vase_options.vhacd_enabled = True
mesh_asset = gym.load_asset(sim, asset_root, vase_asset_file, vase_options)
v_pose = gymapi.Transform()
v_pose.p = gymapi.Vec3(0.0254, -0.55, 0.2)
v_pose.r = gymapi.Quat.from_euler_zyx(0, np.pi / 2, 0)


def visualize_depth(image_array):
    # -inf implies no depth value, set it to zero. output will be black.
    image_array[image_array == -np.inf] = 0

    # clamp depth image to 10 meters to make output image human friendly
    image_array[image_array < -10] = -10

    # flip the direction so near-objects are light and far objects are dark
    normalized_depth = -255.0 * (image_array / np.min(image_array + 1e-4))
    return normalized_depth


def deproject_point(
    cam_width,
    cam_height,
    pixel: np.ndarray,
    depth_buffer,
    seg_buffer,
    view,
    proj,
    none_okay=True,
):
    vinv = np.linalg.inv(view)
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    if none_okay:
        depth_buffer[seg_buffer == 0] = -10001
        if depth_buffer[pixel] < -10000:
            return None
    centerU = cam_width / 2
    centerV = cam_height / 2
    u = -(pixel[:, 1] - centerU) / (cam_width)  # image-space coordinate
    v = (pixel[:, 0] - centerV) / (cam_height)  # image-space coordinate
    # d = depth_buffer[pixel]  # depth buffer value
    d = depth_buffer.reshape(-1)
    X2 = np.array([d * fu * u, d * fv * v, d, np.ones_like(d)])  # deprojection vector
    p2 = X2.T * vinv  # Inverse camera view to get world coordinates
    return p2[:, :3]


def downsample(x, poolh, poolw, strideh, stridew, func=np.max):
    out = np.zeros(
        (1 + (x.shape[0] - poolh) // strideh, 1 + (x.shape[1] - poolw) // stridew)
    )
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = func(
                x[i * strideh : i * strideh + poolh, j * stridew : j * stridew + poolw]
            )
    return out


def compute_capsule(size_array):
    longest_side = np.argmax(size_array)
    smallest_side = np.argmin(size_array)
    middle_side = 3 - longest_side - smallest_side

    grasp_point = np.zeros(shape=(2, 3))
    grasp_point[0][longest_side] = size_array[longest_side] / 2
    grasp_point[1][longest_side] = -size_array[longest_side] / 2

    radius = np.sqrt(
        (size_array[smallest_side] / 2) ** 2 + (size_array[middle_side] / 2) ** 2
    )
    return (grasp_point, radius)


def compute_vertices(size_array):
    x, y, z = size_array
    x_end = [x / 2, -x / 2]
    y_end = [y / 2, -y / 2]
    z_end = [z / 2, -z / 2]

    vtx = np.array([list(v) for v in itertools.product(x_end, y_end, z_end)])
    return vtx


def transform_pts(arg_tuple):
    tsfm, points = arg_tuple
    tsfm_pt = np.hstack([points, np.ones((points.shape[0], 1))])
    tsfm_pt = tsfm @ tsfm_pt.T

    return np.round(tsfm_pt.T[:, :-1], 5)


def h_downsample(pt_cloud, min_pt, max_pt):
    min_x, min_y = min_pt[0, 0], min_pt[0, 1]
    max_x, max_y = max_pt[0, 0], max_pt[0, 1]
    height_map = np.zeros((30, 40))
    xy_grid = np.zeros((31, 41, 2))

    grid_counter = 0
    min_val = np.min(pt_cloud[:, 2])
    min_val = max(min_val, -0.40)  # prevents -inf
    for y in np.linspace(min_y, max_y + 0.001, 31):
        for x in np.linspace(min_x, max_x + 0.001, 41):
            idx = np.unravel_index(grid_counter, (31, 41))
            xy_grid[idx] = [x, y]
            grid_counter += 1

    for i in np.arange(1200):
        idx = np.unravel_index(i, (30, 40))
        x_min = xy_grid[idx[0], idx[1], 0]
        x_max = xy_grid[idx[0], idx[1] + 1, 0]
        y_min = xy_grid[idx[0], idx[1], 1]
        y_max = xy_grid[idx[0] + 1, idx[1], 1]

        mask = (
            (pt_cloud[:, 0] < x_max)
            & (pt_cloud[:, 0] >= x_min)
            & (pt_cloud[:, 1] < y_max)
            & (pt_cloud[:, 1] >= y_min)
        )
        heights = pt_cloud[mask.nonzero()[0]][:, 2]
        if len(heights) == 0:
            height_map[idx] = min_val
        else:
            height_map[idx] = np.max(heights)

    return height_map[::-1]


# Create environments
actor_handles = [[]]
camera_handles = [[]]
envs = []

# cam_prop = CameraIntrinsics(
#     None,
#     386.5911865234375,
#     386.2191467285156,
#     318.52899169921875,
#     236.66403198242188,
#     0,
#     480,
#     640,
# )
fov = 2 * np.arctan2(640, 2 * 386.5911865234375) * 180 / np.pi
with open("cfg/boxes.json", "r") as r:
    box_cfg = json.load(r)
# box_cfg = box_cfg[:3] + box_cfg[4:6] + [box_cfg[8]]
box_cfg = [box_cfg[8]]
# create environments
for i in range(num_envs):
    actor_handles.append([])
    segmentation_id = 1
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    floor = gym.create_box(sim, 1, 1, 0.05, gymapi.AssetOptions())
    floor_handle = gym.create_actor(env, floor, bin_pose, "floor", i)
    gym.set_rigid_body_color(
        env, floor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 0)
    )

    # set a gray color for bin
    dark_color = gymapi.Vec3(0.8, 0.8, 0.8)

    mesh_handle = gym.create_actor(env, mesh_asset, v_pose, "mesh", i)
    actor_handles[i].append(mesh_handle)
    bin_props = gym.get_actor_rigid_shape_properties(env, mesh_handle)
    gym.set_actor_rigid_shape_properties(env, mesh_handle, bin_props)
    gym.set_rigid_body_color(
        env, mesh_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, dark_color
    )
    segmentation_id += 1

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0)

    # Create cameras in each environment
    camera_handles.append([])
    camera_properties = gymapi.CameraProperties()

    camera_properties.width = 640
    camera_properties.height = 480
    camera_properties.horizontal_fov = fov
    camera_properties.use_collision_geometry = False

    # Set a fixed position and look-target for the first camera
    # position and target location are in the coordinate frame of the environment
    h1 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_transform = gymapi.Transform()
    camera_transform.p = gymapi.Vec3(-0.093956, -0.381233, 1.06595 + 0.36)
    rotation_matrix = (
        R.from_euler("z", 90, degrees=True).as_matrix()
        @ R.from_euler("y", 90, degrees=True).as_matrix()
        @ R.from_euler("x", 180, degrees=True).as_matrix()
        @ np.linalg.inv(
            np.array(
                [
                    [0.999346, -0.001493, -0.036126],
                    [-0.002240, -0.999785, -0.020626],
                    [-0.036087, 0.020693, -0.999134],
                ]
            )
        )
    )
    tf = RigidTransform(rotation_matrix)
    quat = tf.quaternion
    camera_transform.r = gymapi.Quat(quat[1], quat[2], quat[3], quat[0])
    gym.set_camera_transform(h1, envs[i], camera_transform)
    camera_handles[i].append(h1)

    h2 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_position = gymapi.Vec3(2.5, -0.4, 1.2)
    camera_target = gymapi.Vec3(0, -0.4, 1)
    gym.set_camera_location(h2, envs[i], camera_position, camera_target)
    camera_handles[i].append(h2)

    overhead_cam = gym.create_camera_sensor(envs[i], camera_properties)
    overhead_tsfm = gymapi.Transform()
    overhead_tsfm.p = gymapi.Vec3(0.0254, -0.56, 1.06595 + 0.36)
    # overhead_tsfm.p = gymapi.Vec3(0.02, -0.56, 1.06595 + 0.36)
    overhead_r = (
        R.from_euler("z", 90, degrees=True).as_matrix()
        @ R.from_euler("y", 90, degrees=True).as_matrix()
    )
    tf = RigidTransform(overhead_r)
    quat = tf.quaternion
    overhead_tsfm.r = gymapi.Quat(quat[1], quat[2], quat[3], quat[0])
    gym.set_camera_transform(overhead_cam, envs[i], overhead_tsfm)
    camera_handles[i].append(overhead_cam)


if os.path.exists("graphics_images"):
    shutil.rmtree("graphics_images")
    os.mkdir("graphics_images")

# if os.path.exists("poses"):
#     shutil.rmtree("poses")
#     os.mkdir("poses")

frame_count = 0

sideways_frame = -1
obj_handle = [None] * num_envs
objects_picked = 0
dead_envs = np.array([False] * num_envs)
view_matrix = np.matrix(gym.get_camera_view_matrix(sim, env, camera_handles[0][0]))
projection_matrix = np.matrix(
    gym.get_camera_proj_matrix(sim, env, camera_handles[0][0])
)
# Main simulation loop
while True:
    # step the physics simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # communicate physics to graphics system
    gym.step_graphics(sim)
    # render the camera sensors
    gym.render_all_camera_sensors(sim)

    if frame_count > 200:
        for i in range(num_envs):

            for j in range(0, 3):
                # The gym utility to write images to disk is recommended only for RGB images.
                rgb_filename = f"graphics_images/rgb_env{i}_cam{j}_frame{str(frame_count).zfill(4)}.png"
                gym.write_camera_image_to_file(
                    sim,
                    envs[i],
                    camera_handles[i][j],
                    gymapi.IMAGE_COLOR,
                    rgb_filename,
                )

            depth_image = gym.get_camera_image(
                sim, envs[i], camera_handles[i][0], gymapi.IMAGE_DEPTH
            )
            seg_image = gym.get_camera_image(
                sim, envs[i], camera_handles[i][0], gymapi.IMAGE_SEGMENTATION
            )
            normalized_depth = visualize_depth(depth_image)
            # Convert to a pillow image and write it to disk
            normalized_depth_image = im.fromarray(
                normalized_depth.astype(np.uint8), mode="L"
            )
            normalized_depth_image.save(
                f"graphics_images/depth_env{i}_cam{0}_frame{str(frame_count).zfill(4)}.png"
            )
            p_list = np.array(np.unravel_index(np.arange(640 * 480), (480, 640))).T
            p_list = p_list.reshape((480, 640, 2))
            pt_cloud = deproject_point(
                640,
                480,
                p_list.reshape((-1, 2)),
                depth_image,
                seg_image,
                view_matrix,
                projection_matrix,
                none_okay=False,
            )
            np.savetxt(
                "depth/point_cloud.txt",
                pt_cloud,
                fmt="%10.5f",
            )
            np.savetxt(
                "depth/proj.txt",
                projection_matrix,
                fmt="%10.5f",
            )
            np.savetxt(
                "depth/view.txt",
                view_matrix,
                fmt="%10.5f",
            )

    if not args.headless:
        # render the viewer
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time to sync viewer with
        # simulation rate. Not necessary in headless.
        gym.sync_frame_time(sim)

        # Check for exit condition - user closed the viewer window
        if gym.query_viewer_has_closed(viewer):
            break

    elif frame_count >= 201:
        break

    frame_count = frame_count + 1

    print(frame_count, datetime.now())

# with open(
#     "poses/run_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".json", "w"
# ) as w:
#     json.dump(current_run_dict, w, cls=NumpyEncoder)
print("Done")

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
