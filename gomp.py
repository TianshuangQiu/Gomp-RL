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

sim_params.use_gpu_pipeline = False
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
gym.add_ground(sim, gymapi.PlaneParams())

if not args.headless:
    # create viewer using the default camera properties
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError("*** Failed to create viewer")

# set up the env grid
num_envs = 9
spacing = 2.0
num_per_row = int(sqrt(num_envs))
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Load ball assets
asset_root = "../../assets"
asset_file = "urdf/ball.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# load bin asset
bin_asset_file = "urdf/tray/traybox.urdf"
print("Loading asset '%s' from '%s'" % (bin_asset_file, asset_root))
bin_asset = gym.load_asset(sim, asset_root, bin_asset_file)
bin_pose = gymapi.Transform()
bin_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Load textures from file. Loads all .jpgs from the specified directory as textures
texture_files = os.listdir("../../assets/textures/")
texture_handles = []
for file in texture_files:
    if file.endswith(".jpg"):
        h = gym.create_texture_from_file(
            sim, os.path.join("../../assets/textures/", file)
        )
        if h == gymapi.INVALID_HANDLE:
            print("Couldn't load texture %s" % file)
        else:
            texture_handles.append(h)


# Create environments
actor_handles = [[]]
camera_handles = [[]]
envs = []

# create environments
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    # create ball pyramid
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    n = 4
    radius = 0.2
    ball_spacing = 2.5 * radius
    min_coord = -0.5 * (n - 1) * ball_spacing
    y = min_coord + 1
    bin_handle = gym.create_actor(
        env,
        bin_asset,
        bin_pose,
        "bin",
        0,
        0,
    )
    gym.set_actor_scale(env, bin_handle, 4)
    gym.set_rigid_body_color(
        env,
        bin_handle,
        0,
        gymapi.MESH_VISUAL_AND_COLLISION,
        gymapi.Vec3(0.24, 0.35, 0.8),
    )
    gym.set_rigid_body_segmentation_id(env, bin_handle, 0, 0)
    while n > 0:
        z = min_coord
        for j in range(n):
            x = min_coord
            for k in range(n):
                if np.random.random() > 0.5:
                    pose.p = gymapi.Vec3(x, 1.5 + y, z)
                    # Set up collision filtering.
                    # Everything should collide.
                    # Put all actors in the same group, with filtering mask set to 0 (no filtering).
                    collision_group = 0
                    collision_filter = 0

                    ahandle = gym.create_actor(
                        env, asset, pose, None, collision_group, collision_filter
                    )
                    gym.set_rigid_body_color(
                        env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
                    )
                    gym.set_rigid_body_segmentation_id(env, ahandle, 0, 1)

                x += ball_spacing
            z += ball_spacing
        y += ball_spacing
        n -= 1
        min_coord = -0.5 * (n - 1) * ball_spacing

    # Create 2 cameras in each environment, one which views the origin of the environment
    # and one which is attached to the 0th body of the 0th actor and moves with that actor
    camera_handles.append([])
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 720
    camera_properties.height = 480

    # Set a fixed position and look-target for the first camera
    # position and target location are in the coordinate frame of the environment
    h1 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_position = gymapi.Vec3(0, 4, 0)
    camera_target = gymapi.Vec3(0.00001, 0, 0.00001)
    # gym.set_light_parameters(
    #     sim, 0, gymapi.Vec3(1, 1, 1), gymapi.Vec3(1, 1, 1), gymapi.Vec3(1, -1, 0)
    # )
    # pdb.set_trace()
    gym.set_camera_location(h1, envs[i], camera_position, camera_target)
    camera_handles[i].append(h1)

    h2 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_position = gymapi.Vec3(4, 4, 4)
    camera_target = gymapi.Vec3(0, 0, 0)
    gym.set_camera_location(h2, envs[i], camera_position, camera_target)
    camera_handles[i].append(h2)

    # # Attach camera 2 to the first rigid body of the first actor in the environment, which
    # # is the ball. The camera offset is relative to the position of the actor, the camera_rotation
    # # is relative to the global coordinate frame, not the actor's rotation
    # # In even envs cameras are will be following rigid body position and orientation,
    # # in odd env only the position
    # h2 = gym.create_camera_sensor(envs[i], camera_properties)
    # camera_offset = gymapi.Vec3(1, 0, -1)
    # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(135))
    # actor_handle = gym.get_actor_handle(envs[i], 0)
    # body_handle = gym.get_actor_rigid_body_handle(envs[i], actor_handle, 0)

    # gym.attach_camera_to_body(
    #     h2,
    #     envs[i],
    #     body_handle,
    #     gymapi.Transform(camera_offset, camera_rotation),
    #     gymapi.FOLLOW_TRANSFORM,
    # )
    # camera_handles[i].append(h2)


# Assign textures to each actor by assigning a different texture to each rigid body within each actor. With
# only balls, each actor has only one rigit body
textures_applied = 0
for i in range(num_envs):
    actor_count = gym.get_actor_count(envs[i])
    for j in range(actor_count):
        actor_handle = gym.get_actor_handle(envs[i], j)
        num_bodies = gym.get_actor_rigid_body_count(envs[i], actor_handle)
        for b in range(num_bodies):
            texture_index = np.mod(textures_applied, len(texture_handles))
            gym.set_rigid_body_texture(
                envs[i],
                actor_handle,
                b,
                gymapi.MESH_VISUAL_AND_COLLISION,
                texture_handles[texture_index],
            )
            textures_applied = textures_applied + 1


if os.path.exists("graphics_images"):
    import shutil

    shutil.rmtree("graphics_images")
    os.mkdir("graphics_images")
frame_count = 0

# Main simulation loop
while True:
    # step the physics simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # communicate physics to graphics system
    gym.step_graphics(sim)

    # render the camera sensors
    gym.render_all_camera_sensors(sim)

    if frame_count > -1 and np.mod(frame_count, 200) == 0:
        for i in range(num_envs):
            for j in range(0, 2):
                # The gym utility to write images to disk is recommended only for RGB images.
                rgb_filename = f"graphics_images/rgb_env{i}_cam{j}_frame{str(frame_count).zfill(4)}.png"
                gym.write_camera_image_to_file(
                    sim,
                    envs[i],
                    camera_handles[i][j],
                    gymapi.IMAGE_COLOR,
                    rgb_filename,
                )

                # Retrieve image data directly. Use this for Depth, Segmentation, and Optical Flow images
                # Here we retrieve a depth image, normalize it to be visible in an
                # output image and then write it to disk using Pillow
                depth_image = gym.get_camera_image(
                    sim, envs[i], camera_handles[i][j], gymapi.IMAGE_DEPTH
                )
                # pdb.set_trace()

                # -inf implies no depth value, set it to zero. output will be black.
                depth_image[depth_image == -np.inf] = 0

                # clamp depth image to 10 meters to make output image human friendly
                depth_image[depth_image < -10] = -10

                # flip the direction so near-objects are light and far objects are dark
                normalized_depth = -255.0 * (depth_image / np.min(depth_image + 1e-4))

                # Convert to a pillow image and write it to disk
                normalized_depth_image = im.fromarray(
                    normalized_depth.astype(np.uint8), mode="L"
                )
                normalized_depth_image.save(
                    f"graphics_images/depth_env{i}_cam{j}_frame{str(frame_count).zfill(4)}.jpg"
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
        if frame_count > 1500:
            break

    frame_count = frame_count + 1

    print(frame_count, datetime.now())

print("Done")

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
