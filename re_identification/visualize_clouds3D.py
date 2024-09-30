import open3d as o3d
import numpy as np
import yaml
from datasetgcn import *
from lossconnection import LossConnection
from tqdm import tqdm

import typer
cli = typer.Typer()

def apply(pcd, T, min_x, max_x, offset=None, mask=False):
    pcd.transform(T)
    
    if mask:
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        mask1 = np.logical_and(points[:, 0]>=min_x, points[:, 0]<=max_x)
        #mask2 = np.logical_and(points[:, 1]>=0.5, points[:, 1]<=1.785)
        mask2 = np.logical_and(points[:, 1]>=0.5, points[:, 1]<=1.745)
        mask = np.logical_and(mask1, mask2)
        pcd.points = o3d.utility.Vector3dVector(points[mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    R = pcd.get_rotation_matrix_from_xyz((0, 0, -np.pi / 3))
    rotc = pcd.get_center()
    pcd.rotate(R, center=rotc)
    if offset is not None:
        pcd.translate(offset)

    return pcd, rotc

def apply_to_P(P, T, rot_center, offset=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    ##pcd.transform(T)
    R = pcd.get_rotation_matrix_from_xyz((0, 0, -np.pi / 3))
    pcd.rotate(R, center=rot_center)
    if offset is not None:
        pcd.translate(offset)
    return np.array(pcd.points)

def main(
        datapath: str = typer.Option(
        ...,
        "--data",
        help="data path ()",
    ),
    iou: float = typer.Option(
        ...,
        "--iou",
        help="IoU threshold",
    ),
):
    transformation_path = os.path.join(datapath, "transformations.yaml")
    
    with open(transformation_path, "r") as stream:
        try:
            transformations = yaml.safe_load(stream)["transformations"]
        except yaml.YAMLError as exc:
            print(exc)

    gt_08 = np.asarray(transformations["gt_08"])
    gt_14 = np.asarray(transformations["gt_14"])
    gt_21 = np.asarray(transformations["gt_21"])

    min_x = 27.73
    max_x = 28.0

    straw_path = os.path.join(datapath, f"14_21")
    data_21_gt   = Strawberries(f"{straw_path}/strawberries_21", os.path.join(straw_path, "selections_2.json"), gt_21, min_x=min_x, max_x=max_x)
    data_14_test = Strawberries(f"{straw_path}/strawberries_14", os.path.join(straw_path, "selections_1.json"), gt_14, min_x=min_x, max_x=max_x)
    conn_gt_test = LossConnection(os.path.join(straw_path, "connections.json"), data_14_test, data_21_gt)
    
    straw_path = os.path.join(datapath, f"08_14")
    data_14_gt   = Strawberries(f"{straw_path}/strawberries_14", os.path.join(straw_path, "selections_2.json"), gt_14, min_x=min_x, max_x=max_x)
    data_08_gt   = Strawberries(f"{straw_path}/strawberries_08", os.path.join(straw_path, "selections_1.json"), gt_08, min_x=min_x, max_x=max_x)
    conn_gt      = LossConnection(os.path.join(straw_path, "connections.json"), data_08_gt, data_14_gt)

    cloud_path = os.path.join(datapath, "reduced_08_14_1.ply")
    pcd08 = o3d.io.read_point_cloud(cloud_path)
    pcd08, rotc08 = apply(pcd08, gt_08, min_x, max_x, mask=True)


    off08_14 = [0, 0.151, 0]
    #off08_14 = [0, 0.15, 0]
    cloud_path = os.path.join(datapath, "reduced_08_14_2.ply")
    pcd14 = o3d.io.read_point_cloud(cloud_path)
    pcd14, rotc14 = apply(pcd14, gt_14, min_x, max_x, off08_14, mask=True)


    off14_21 = [0, 0.32, 0]
    #off14_21 = [0, 0.3, 0]
    cloud_path = os.path.join(datapath, "reduced_14_21_2.ply")
    pcd21 = o3d.io.read_point_cloud(cloud_path)
    pcd21, rotc21 = apply(pcd21, gt_21, min_x, max_x, off14_21, mask=True)


    points, lines  = [], []
    mask14 = np.zeros(data_14_gt.centers.shape[0], dtype=bool)

    P08      = apply_to_P(data_08_gt.centers,   gt_08, rotc08, None)
    P14      = apply_to_P(data_14_gt.centers,   gt_14, rotc14, off08_14)
    P14_test = apply_to_P(data_14_test.centers, gt_14, rotc14, off08_14)
    P21      = apply_to_P(data_21_gt.centers,   gt_21, rotc21, off14_21)

    for k_n in tqdm(conn_gt.connections_keys):
        k_p = conn_gt.connections_keys[k_n]

        c_14 = P14[data_14_gt.keys.index(k_n)]
        c_08 = P08[data_08_gt.keys.index(k_p)]

        mask14[data_14_gt.keys.index(k_n)] = True

        points.append(c_14)
        points.append(c_08)
        lines.append([len(points)-2, len(points)-1])

    colors = [[0, 0.5, 0] for i in range(len(lines))]

    line_set_14 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set_14.colors = o3d.utility.Vector3dVector(colors)

    spheres_14 = []
    spheres_14_c = []
    spheres_14_r = []

    for idx in range(len(data_14_gt.keys)):
        if not mask14[idx]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=data_14_gt.fruit_radius[idx])
            sphere.translate(P14[idx])
            sphere.paint_uniform_color([0.1, 0.1, 0.7])
            spheres_14.append(sphere)
            spheres_14_c.append(sphere.get_center())
            spheres_14_r.append(data_14_gt.fruit_radius[idx])


    ################ 21 to 14

    points, lines  = [], []
    mask21 = np.zeros(P21.shape[0], dtype=bool)

    for k_n in tqdm(conn_gt_test.connections_keys):
        k_p = conn_gt_test.connections_keys[k_n]

        c_21 = P21[data_21_gt.keys.index(k_n)]
        c_14 = P14_test[data_14_test.keys.index(k_p)]

        mask21[data_21_gt.keys.index(k_n)] = True

        points.append(c_21)
        points.append(c_14)
        lines.append([len(points)-2, len(points)-1])

    colors = [[0, 0.5, 0] for i in range(len(lines))]

    line_set_21 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set_21.colors = o3d.utility.Vector3dVector(colors)

    spheres_21 = []
    spheres_21_c = []
    spheres_21_r = []

    for idx in range(len(data_21_gt.keys)):
        if not mask21[idx]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(data_21_gt.fruit_radius[idx], 1e-4))
            sphere.translate(P21[idx])
            sphere.paint_uniform_color([0.1, 0.1, 0.7])
            spheres_21.append(sphere)
            spheres_21_c.append(sphere.get_center())
            spheres_21_r.append(data_21_gt.fruit_radius[idx])


    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=3200, height=1600, visible=True)
    view_ctl = visualizer.get_view_control()
    view_ctl.set_up([-0.0002971969814756552, -0.014997288869575557, 0.99988749017102752])  # set the negative direction of the y-axis as the up direction
    view_ctl.set_front([-0.011999257536550094, -0.9998154952548014, -0.01499977556391714 ])  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat([28.447480121135829, 1.0, 0.58518670083317892 ])  # set the original point as the center point of the window
    view_ctl.set_zoom(0.02)
    #param = view_ctl.convert_to_pinhole_camera_parameters()
    #o3d.io.write_pinhole_camera_parameters("view_motiv.json", param)
    visualizer.add_geometry(pcd08)
    visualizer.add_geometry(pcd14)
    visualizer.add_geometry(pcd21)
    visualizer.add_geometry(line_set_14)
    visualizer.add_geometry(line_set_21)

    for s in spheres_14:
        visualizer.add_geometry(s)
    for s in spheres_21:
        visualizer.add_geometry(s)

    visualizer.run()

if __name__ == "__main__":
    typer.run(main)
