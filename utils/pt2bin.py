import trimesh
import numpy as np
import pdb


points = np.array([[3, 3, 3], [0, 3, 3], [0, 0, 2], [3, 0, 2]])
deep_length = 4


def pt_2_cube(points, deep_length):
    side1 = points[0] - points[1]
    side2 = points[1] - points[2]

    normal = np.cross(side1, side2)
    if normal[2] > 0:
        normal *= -1

    normal = normal / np.linalg.norm(normal) * deep_length

    bottom_layer = points - normal

    all_points = np.vstack([points, bottom_layer])
    return all_points


mesh = trimesh.convex.convex_hull(pt_2_cube(points, deep_length))
file_str = trimesh.exchange.obj.export_obj(mesh)
with open("mesh.obj", "w") as w:
    w.write(file_str)
