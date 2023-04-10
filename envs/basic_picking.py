import isaacgym
import torch
from isaacgym import gymtorch, gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask

import os
import numpy as np
from typing import Dict, Any, Tuple, List, Set
import pdb


class PickingEnv(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg
        self.num_obs = 30 * 40
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.num_envs = self.cfg["num_envs"]
        self.max_episode_length = self.cfg["max_episode_length"]
        self.cfg["env"]["numActions"] = 1

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)

        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = "../../assets"
        bin_options = gymapi.AssetOptions()
        bin_options.use_mesh_materials = True
        bin_options.vhacd_enabled = True
        bin_options.fix_base_link = True

        # load bin asset
        bin_asset_file = "urdf/custom/cardboardbin.urdf"
        print("Loading asset '%s' from '%s'" % (bin_asset_file, asset_root))
        bin_asset = self.gym.load_asset(
            self.sim, asset_root, bin_asset_file, bin_options
        )
        bin_position = (0.0254, -0.55, 0)
        bin_pose = gymapi.Transform()
        bin_pose.p = gymapi.Vec3(bin_position[0], bin_position[1], bin_position[2])
        bin_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        self.envs = []
        self.box_handles = []
        self.env_box_sizes = []
        self.camera_handles = []
        self.view_matrix = None
        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            current_handles = []

            self.envs.append(env_ptr)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return super().step(actions)

    def deproject_point(
        self,
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
        X2 = np.array(
            [d * fu * u, d * fv * v, d, np.ones_like(d)]
        )  # deprojection vector
        p2 = X2.T * vinv  # Inverse camera view to get world coordinates
        return p2[:, :3]

    @staticmethod
    def downsample(x, poolh, poolw, strideh, stridew, func=np.max):
        out = np.zeros(
            (1 + (x.shape[0] - poolh) // strideh, 1 + (x.shape[1] - poolw) // stridew)
        )
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = func(
                    x[
                        i * strideh : i * strideh + poolh,
                        j * stridew : j * stridew + poolw,
                    ]
                )
        return out


p = PickingEnv()
