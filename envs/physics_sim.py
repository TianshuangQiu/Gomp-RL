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

def visualize_segmentation(image_array, seg_list):
    output = np.zeros(shape=(image_array.shape[0], image_array.shape[1], 3))
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            if image_array[i][j] > 0:
                output[i][j] = seg_list[image_array[i][j]]
            else:
                output[i][j] = np.array([0, 0, 0])
    return output


current_run_dict = {"time": str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}

# acquire the gym interface
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Graphics Example",
    headless=True,
    custom_parameters=[{"name":"--num_envs", "type":int, "default":1}, {"name":"--prefix", "type":str, "default":""}],
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
num_envs = args.num_envs
spacing = 5
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
bin_position = (-0.308, 0.51, 0)
bin_pose = gymapi.Transform()
bin_pose.p = gymapi.Vec3(bin_position[0], bin_position[1], bin_position[2])
bin_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

min_point = np.array([[-0.816, 0.23]])
max_point = np.array([[0.2, 0.51]])


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
            
    np.nan_to_num(height_map, copy=False, nan=min_val)

    return height_map[::-1]

def remove_box(env_handle, obj_handle):
    print(f"Putting away {obj_handle}")
    rm_state = gym.get_actor_rigid_body_states(
        env_handle, obj_handle, gymapi.STATE_ALL
    )
    # pdb.set_trace()
    rm_state["pose"]["p"].fill((-2, -2, 0.1))
    rm_state["vel"]["linear"].fill((0, 0, 0))
    gym.set_actor_rigid_body_states(
        env_handle, obj_handle, rm_state, gymapi.STATE_ALL
    )


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
fov = 2 * np.arctan2(640, 2 * 384.31756591796875) * 180 / np.pi
with open("cfg/boxes.json", "r") as r:
    box_cfg = json.load(r)
# box_cfg = box_cfg[:3] + box_cfg[4:6] + [box_cfg[8]]
# box_cfg = [box_cfg[8]]
# create environments
for i in range(num_envs):
    actor_handles.append([])
    segmentation_id = 1
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    current_run_dict[i] = {}
    current_run_dict[i]["alias"] = []
    current_run_dict[i]["poses"] = []
    current_run_dict[i]["sizes"] = []
    current_run_dict[i]["vertices"] = []
    current_run_dict[i]["color"] = []

    shuffle = np.random.choice(box_cfg, size=np.random.randint(25, 35), replace=True)
    for obj in shuffle:
        sizes = np.array(obj["dim"])
        current_run_dict[i]["sizes"].append(sizes)
        current_run_dict[i]["vertices"].append(compute_vertices(sizes))
        current_run_dict[i]["color"].append(obj["color"])
        current_run_dict[i]["alias"].append(obj["alias"])

    # set a gray color for bin
    dark_color = gymapi.Vec3(0.8, 0.8, 0.8)

    bin_handle = gym.create_actor(
        env, bin_asset, bin_pose, "bin", i, segmentationId=segmentation_id
    )
    actor_handles[i].append(bin_handle)
    bin_props = gym.get_actor_rigid_shape_properties(env, bin_handle)
    bin_props[0].restitution = 0.05
    bin_props[0].compliance = 0.05
    gym.set_actor_rigid_shape_properties(env, bin_handle, bin_props)
    gym.set_rigid_body_color(
        env, bin_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, dark_color
    )
    segmentation_id += 1

    # create jenga tower
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0)


    for count, box_size in enumerate(shuffle):
        if count % 3 == 0:
            pose.p.z += 0.3 + 0.02 * np.random.random()
            pose.p.x = bin_position[0]
            pose.p.y = bin_position[1]
            # pose.p.y = bin_position[1] + 0.05 * np.random.normal()
        elif count % 3 == 1:
            pose.p.x = bin_position[0] + 0.2 + 0.1 * np.random.random()
            pose.p.y = bin_position[1] + 0.1 * np.random.normal()
        else:
            pose.p.x = bin_position[0] - 0.2 - 0.1 * np.random.random()
            pose.p.y = bin_position[1] + 0.1 * np.random.normal()
        pose.r = gymapi.Quat.from_euler_zyx(
            np.random.random() * 90, np.random.random() * 90, np.random.random() * 90
        )
        box_options = gymapi.AssetOptions()
        box_options.density = 100
        box = gym.create_box(
            sim,
            current_run_dict[i]["sizes"][count][0],
            current_run_dict[i]["sizes"][count][1],
            current_run_dict[i]["sizes"][count][2],
            box_options,
        )
        box_ptr = gym.create_actor(
            env, box, pose, "box", i, segmentationId=segmentation_id
        )
        actor_handles[i].append(box_ptr)
        segmentation_id += 1

        color = gymapi.Vec3(
            current_run_dict[i]["color"][count][0],
            current_run_dict[i]["color"][count][1],
            current_run_dict[i]["color"][count][2],
        )
        gym.set_rigid_body_color(
            env, box_ptr, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
        )
        box_props = gym.get_actor_rigid_shape_properties(env, box_ptr)
        # box_props[0].restitution = 0
        # box_props[0].compliance = 0
        box_props[0].restitution = 0.1 + 0.05 * np.random.random()
        box_props[0].compliance = 0.01 + 0.01 * np.random.random()
        gym.set_actor_rigid_shape_properties(env, box_ptr, box_props)

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
    camera_transform.p = gymapi.Vec3(-0.12, 0.29, 1.37)
    rotation_matrix = (R.from_euler("z", 90, degrees=True).as_matrix()
        @ R.from_euler("y", -90, degrees=True).as_matrix()
        # @ R.from_euler("x", 180, degrees=True).as_matrix()
        @ np.array(
                [
                    [-1, 0, 0],
                    [ 0, 0.98480775, 0.17364818],
                    [0, 0.17364818, -0.98480775],
                ]
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
    overhead_tsfm.p = gymapi.Vec3(-0.12, 0.29, 1.06595 + 0.36)
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


os.makedirs("poses", exist_ok=True)
os.makedirs("depth", exist_ok=True)

frame_count = 0

sideways_frame = -1
obj_handle = [None] * num_envs
objects_picked = 0
dead_envs = np.array([False] * num_envs)
view_matrix = None

# Main simulation loop
while True:
    # step the physics simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # communicate physics to graphics system
    gym.step_graphics(sim)
    # render the camera sensors
    gym.render_all_camera_sensors(sim)
    # pdb.set_trace()

    if frame_count > 0 :
        for i in range(num_envs):
            # Get bin state
            state = gym.get_actor_rigid_body_states(
                envs[i], actor_handles[i][0], gymapi.STATE_ALL
            )
            original_position = state["pose"]["p"].copy()
            state["pose"]["p"].fill((5, 5, 5))
            if not gym.set_actor_rigid_body_states(
                envs[i], actor_handles[i][0], state, gymapi.STATE_ALL
            ):
                pdb.set_trace()
            gym.step_graphics(sim)
            # render the camera sensors
            gym.render_all_camera_sensors(sim)
            for j in range(0, 1):
                # The gym utility to write images to disk is recommended only for RGB images.
                rgb_filename = f"graphics_images/boxless_rgb_env{i}_cam{j}_frame{str(frame_count).zfill(4)}.png"
                gym.write_camera_image_to_file(
                    sim,
                    envs[i],
                    camera_handles[i][j],
                    gymapi.IMAGE_COLOR,
                    rgb_filename,
                )

            state["pose"]["p"] = original_position
            if not gym.set_actor_rigid_body_states(
                envs[i], actor_handles[i][0], state, gymapi.STATE_ALL
            ):
                pdb.set_trace()
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
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
            down_sampled_depth = downsample(depth_image, 15, 15, 15, 15)[1:-1, 1:-1]
            normalized_depth = visualize_depth(down_sampled_depth)
            # Convert to a pillow image and write it to disk
            dsd = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
            dsd.save(
                f"graphics_images/downsampled_depth_env{i}_cam{0}_frame{str(frame_count).zfill(4)}.jpg"
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
    if frame_count == 121:
        for i in range(num_envs):
            # Check box inside bin
            for handle in range(1, len(actor_handles[i])):
                pose = gym.get_actor_rigid_body_states(
                    envs[i], actor_handles[i][handle], gymapi.STATE_ALL
                )["pose"]
                pos = np.array(
                        [pose["p"]["x"], pose["p"]["y"], pose["p"]["z"]]
                    ).reshape(3)
                rot = np.array(
                    [
                        pose["r"]["w"],
                        pose["r"]["x"],
                        pose["r"]["y"],
                        pose["r"]["z"],
                    ]
                ).reshape(4)
                rot_mat = RigidTransform.rotation_from_quaternion(rot)
                tsfm = RigidTransform(rot_mat, pos)
                pts = transform_pts((tsfm.matrix, current_run_dict[i]["vertices"][handle-1]))[:, :2]
                if np.all(pts[:, 0] < min_point[0, 0]) or np.all(pts[:, 0] > max_point[0, 0]) or np.all(pts[:, 1] < min_point[0, 1]) or np.all(pts[:, 1] > max_point[0, 1]):
                    remove_box(envs[i], actor_handles[i][handle])

    if frame_count > 200 and frame_count < 1000:
        if frame_count < sideways_frame:
            for i, env in enumerate(envs):
                if dead_envs[i]:
                    continue
                if obj_handle[i] is not None:
                    gym.apply_body_forces(
                        env,
                        obj_handle[i],
                        gymapi.Vec3(0, 0, 50),
                        None,
                        gymapi.ENV_SPACE,
                    )

        elif sideways_frame < 0:
            # communicate physics to graphics system
            gym.step_graphics(sim)
            # render the camera sensors
            gym.render_all_camera_sensors(sim)
            if np.all(dead_envs):
                break
            for i, env in enumerate(envs):
                if dead_envs[i]:
                    continue
                # save object poses
                poses = []
                for h in actor_handles[i]:
                    state = gym.get_actor_rigid_body_states(
                        envs[i], h, gymapi.STATE_ALL
                    )
                    pose = state["pose"]
                    pos = np.array(
                        [pose["p"]["x"], pose["p"]["y"], pose["p"]["z"]]
                    ).reshape(3)
                    rot = np.array(
                        [
                            pose["r"]["w"],
                            pose["r"]["x"],
                            pose["r"]["y"],
                            pose["r"]["z"],
                        ]
                    ).reshape(4)
                    rot_mat = RigidTransform.rotation_from_quaternion(rot)
                    tsfm = RigidTransform(rot_mat, pos)
                    poses.append(np.round(tsfm.matrix, 5))

                current_run_dict[i]["poses"] = poses[1:]
                # get segmentation/ depth images
                depth_image = gym.get_camera_image(
                    sim, envs[i], camera_handles[i][0], gymapi.IMAGE_DEPTH
                )
                seg_image = gym.get_camera_image(
                    sim, envs[i], camera_handles[i][0], gymapi.IMAGE_SEGMENTATION
                )
                # write_depth = depth_image[::-1, ::-1] + 1.06595 + 0.36
                # write_depth[write_depth < 0] = 0
                # write_depth -= 0.36
                curr_time = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                projection_matrix = np.matrix(
                    gym.get_camera_proj_matrix(sim, env, camera_handles[i][0])
                )
                if view_matrix is None:
                    view_matrix = np.matrix(
                        gym.get_camera_view_matrix(sim, env, camera_handles[i][0])
                    )

                # print("computing projection plane")
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
                pt_cloud += np.array([[0, 0, -0.40]])

                # max_point = pt_cloud[639, :-1]
                # min_point = pt_cloud[479 * 640, :-1]
                header = (
                    np.array2string(
                        min_point,
                        formatter={"float_kind": lambda x: "%10.5f" % x},
                    )[2:-2]
                    + "\n"
                    + np.array2string(
                        max_point,
                        formatter={"float_kind": lambda x: "%10.5f" % x},
                    )[2:-2]
                    + "\n"
                    + "   30\n   40"
                )
                write_depth = h_downsample(pt_cloud, min_point, max_point)
                valid_pixels = np.where(seg_image > 1)
                num_valid = len(valid_pixels[0])
                if num_valid == 0:
                    dead_envs[i] = True
                    break
                else:
                    valid_pixel_array = np.vstack(valid_pixels).T
                    valid_point_idx = np.ravel_multi_index(np.vstack(valid_pixels),(480,640))
                    pt_cloud_valid = pt_cloud[valid_point_idx]
                    top_pt = np.argmax(pt_cloud_valid[:, 2])
                    # Depth and RGB are flipped
                    pixel = valid_pixel_array[top_pt]
                    pixel = (pixel[0], pixel[1])

                # seg ID starts at 1, bin takes up a spot
                print(f"top box found at {pixel}, targeting {seg_image[pixel] - 2}")
                target_obj_idx = seg_image[pixel] - 2
                # segmentation_colors = []
                # for tmp in range(np.max(seg_image) + 2):
                #     if tmp == seg_image[pixel]:
                #         segmentation_colors.append(np.array([0, 0.5, 0.9]))
                #     else:
                #         segmentation_colors.append(np.array([1, 1 ,0]))
                # vis_seg = visualize_segmentation(seg_image, segmentation_colors) * 256
                # # Convert to a pillow image and write it to disk
                # vis_seg_image = im.fromarray(vis_seg.astype(np.uint8), mode="RGB")
                # vis_seg_image.save(
                #     f"seg_env{i}_cam{j}_frame{str(frame_count).zfill(4)}.jpg"
                # )
                # pdb.set_trace()

                pos = pt_cloud_valid[top_pt]
                # pos = pt_cloud[pixel[0] * 540 + pixel[1]]
                gym.apply_body_forces(
                    env,
                    actor_handles[i][target_obj_idx+1],
                    gymapi.Vec3(0, 0, 70),
                    None,
                    gymapi.ENV_SPACE,
                )
                obj_handle[i] = actor_handles[i][target_obj_idx+1]

                sideways_frame = frame_count + 100

                ##SAVE SHIT BEGINS HERE###
                np.savetxt(
                    f"depth/depth_{args.prefix}_" + curr_time + f"_env{i}_frame{frame_count}.txt",
                    write_depth + np.random.normal(0, 0.01, size=write_depth.shape),
                    fmt="%10.5f",
                    header=header,
                    comments="",
                )
               
                np.savetxt(
                    f"poses/vert_{args.prefix}_" + curr_time + f"_env{i}_frame{frame_count}.txt",
                    transform_pts(
                        (
                            current_run_dict[i]["poses"][target_obj_idx],
                            current_run_dict[i]["vertices"][target_obj_idx],
                        )
                    )
                    + np.array([0, 0, -0.40]),
                    fmt="%10.5f",
                    header="   8 3",
                    footer=current_run_dict[i]["alias"][target_obj_idx],
                    comments="",
                )
                # pdb.set_trace()
                # with open(
                #     "logs/run_" + curr_time + f"_env{i}_frame{frame_count}.json",
                #     "w",
                # ) as w:
                #     json.dump(current_run_dict[i], w, cls=NumpyEncoder)
                ##SAVE SHIT ENDS HERE###
                del pt_cloud

        elif sideways_frame == frame_count:
            # Putting obj aside
            for i, env in enumerate(envs):
                if obj_handle[i] is not None:
                    print(f"Putting away {obj_handle[i]}")
                    rm_state = gym.get_actor_rigid_body_states(
                        envs[i], obj_handle[i], gymapi.STATE_ALL
                    )
                    # pdb.set_trace()
                    rm_state["pose"]["p"].fill((-2, -2, 0.7))
                    rm_state["vel"]["linear"].fill((0, 0, 0))
                    gym.set_actor_rigid_body_states(
                        envs[i], obj_handle[i], rm_state, gymapi.STATE_ALL
                    )

        else:
            sideways_frame = -1
            objects_picked += 1
            obj_handle = [None] * num_envs

    elif frame_count >= 1000:
        # for i in range(num_envs):
        #     for j in range(0, 2):
        #         # Retrieve image data directly. Use this for Depth, Segmentation, and Optical Flow images
        #         # Here we retrieve a depth image, normalize it to be visible in an
        #         # output image and then write it to disk using Pillow
        #         depth_image = gym.get_camera_image(
        #             sim, envs[i], camera_handles[i][j], gymapi.IMAGE_DEPTH
        #         )
        #         normalized_depth = visualize_depth(depth_image)
        #         # Convert to a pillow image and write it to disk
        #         normalized_depth_image = im.fromarray(
        #             normalized_depth.astype(np.uint8), mode="L"
        #         )
        #         normalized_depth_image.save(
        #             f"graphics_images/depth_env{i}_cam{j}_frame{str(frame_count).zfill(4)}.jpg"
        #         )

        #         seg_image = gym.get_camera_image(
        #             sim, envs[i], camera_handles[i][j], gymapi.IMAGE_SEGMENTATION
        #         )
        #         vis_seg = visualize_segmentation(seg_image, segmentation_colors) * 256
        #         # Convert to a pillow image and write it to disk
        #         vis_seg_image = im.fromarray(vis_seg.astype(np.uint8), mode="RGB")
        #         vis_seg_image.save(
        #             f"graphics_images/seg_env{i}_cam{j}_frame{str(frame_count).zfill(4)}.jpg"
        #         )
        break

    frame_count = frame_count + 1

    # print(frame_count, datetime.now())

# with open(
#     "poses/run_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".json", "w"
# ) as w:
#     json.dump(current_run_dict, w, cls=NumpyEncoder)
print("Done")

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
