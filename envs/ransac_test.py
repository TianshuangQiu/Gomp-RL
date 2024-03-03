from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
import pdb
from datetime import datetime


def compute_grasp(point_cloud):
    o3d_cloud = o3d.t.geometry.PointCloud()
    o3d_cloud.point.positions = o3d.core.Tensor(point_cloud)
    plane_model, inliers = o3d_cloud.segment_plane(
        distance_threshold=0.01, ransac_n=10, num_iterations=1000
    )

    plane_model, inliers = plane_model.numpy(), inliers.numpy()
    grasp_normal = plane_model[:-1]
    grasp_normal /= np.linalg.norm(grasp_normal)
    grasp_point = np.average(point_cloud[inliers], axis=0)

    return (grasp_point, grasp_normal), np.delete(point_cloud, inliers, axis=0)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

points = np.loadtxt("depth/point_cloud.txt")

points += np.random.normal(scale=0.005, size=points.shape)

print(points.shape)
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, color="r")
ax.set_aspect("equal", adjustable="box")


first_grasp, remaining = compute_grasp(points)
ax.scatter(remaining[:, 0], remaining[:, 1], remaining[:, 2], s=0.5, color="g")
center, direction = first_grasp
first_grasp = np.vstack([center, center + direction * 0.1])
ax.plot(
    first_grasp[:, 0], first_grasp[:, 1], first_grasp[:, 2], marker="o", linewidth=2
)

second_grasp, remaining = compute_grasp(remaining)
ax.scatter(remaining[:, 0], remaining[:, 1], remaining[:, 2], s=0.5, color="b")
center, direction = second_grasp
second_grasp = np.vstack([center, center + direction * 0.1])
ax.plot(
    second_grasp[:, 0], second_grasp[:, 1], second_grasp[:, 2], marker="o", linewidth=2
)

third_grasp, remaining = compute_grasp(remaining)
center, direction = third_grasp
third_grasp = np.vstack([center, center + direction * 0.1])
ax.plot(
    third_grasp[:, 0], third_grasp[:, 1], third_grasp[:, 2], marker="o", linewidth=2
)


plt.show()
