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
    # pcd.rotate(R, center=rotc)
    if offset is not None:
        pcd.translate(offset)

    return pcd, rotc

def apply_to_P(P, T, rot_center, offset=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    ##pcd.transform(T)
    R = pcd.get_rotation_matrix_from_xyz((0, 0, -np.pi / 3))
    # pcd.rotate(R, center=rot_center)
    if offset is not None:
        pcd.translate(offset)
    return np.array(pcd.points)

def get_inst_mask(data, coords, data_centers_modified):
    inst_mask = np.zeros(coords.shape[0], dtype=np.int32)

    for k, c, r in tqdm(zip(data.keys, data_centers_modified, data.fruit_radius)):
        if k==0:
            continue
        mask = np.linalg.norm(coords-c, axis=1) <= r
        inst_mask[mask] = k
        #break
    return inst_mask

def mask2colors(colors, inst_mask):
    u = np.unique(inst_mask)
    np.random.seed(0)
    table = np.random.uniform(0.1, 1.0, (u.max()+1, 3))
    table[0, :] = 0

    newcolors = table[inst_mask]
    newcolors = np.where(newcolors==0, colors/2, newcolors)
    return newcolors

def homokeys(data_from, data_next, conn):
    keys_from = data_from.keys
    keys_next = data_next.keys

    new_ids = np.zeros(len(keys_from), dtype=np.int32)
    offset_id = 5000

    for i, k_from in enumerate(keys_from):
        if conn.connmatrix[i+1, 0] == 1: # no match
            new_ids[i] = offset_id
            offset_id += 1
        else:
            new_ids[i] = keys_next[conn.connmatrix[i+1, :].argmax()-1]
    return new_ids

def main(
        datapath: str = typer.Option(
        ...,
        "--data",
        help="data path",
    ),
    iou: float = typer.Option(
        -1.0,
        "--iou",
        help="IoU threshold. If negative, load GT annotations.",
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

    min_x = 28.5 #27.6
    max_x = 29.3 #28.5

    if iou<0:
        straw_path = os.path.join(datapath, f"14_21")
    else:
        straw_path = os.path.join(datapath, f"14_21_inst@{iou}")

    data_21_test   = Strawberries(f"{straw_path}/strawberries_21", os.path.join(straw_path, "selections_2.json"), gt_21, min_x=min_x, max_x=max_x)
    data_14_test = Strawberries(f"{straw_path}/strawberries_14", os.path.join(straw_path, "selections_1.json"), gt_14, min_x=min_x, max_x=max_x)
    conn_gt_test = LossConnection(os.path.join(straw_path, "connections.json"), data_14_test, data_21_test)
    
    straw_path = os.path.join(datapath, f"08_14")
    data_14_gt   = Strawberries(f"{straw_path}/strawberries_14", os.path.join(straw_path, "selections_2.json"), gt_14, min_x=min_x, max_x=max_x)
    data_08_gt   = Strawberries(f"{straw_path}/strawberries_08", os.path.join(straw_path, "selections_1.json"), gt_08, min_x=min_x, max_x=max_x)
    conn_gt      = LossConnection(os.path.join(straw_path, "connections.json"), data_08_gt, data_14_gt)

    cloud_path = os.path.join(datapath, "reduced_08_14_1.ply")
    pcd08 = o3d.io.read_point_cloud(cloud_path)
    pcd08, rotc08 = apply(pcd08, gt_08, min_x, max_x, mask=True)


    # off08_14 = [0, 0.151, 0]
    off08_14 = [0, 0.25, 0]
    cloud_path = os.path.join(datapath, "reduced_08_14_2.ply")
    pcd14 = o3d.io.read_point_cloud(cloud_path)
    pcd14, rotc14 = apply(pcd14, gt_14, min_x, max_x, off08_14, mask=True)


    # off14_21 = [0, 0.32, 0]
    off14_21 = [0, 0.5, 0]
    cloud_path = os.path.join(datapath, "reduced_14_21_2.ply")
    pcd21 = o3d.io.read_point_cloud(cloud_path)
    pcd21, rotc21 = apply(pcd21, gt_21, min_x, max_x, off14_21, mask=True)


    points, lines  = [], []
    mask14 = np.zeros(data_14_gt.centers.shape[0], dtype=bool)

    P08      = apply_to_P(data_08_gt.centers,   gt_08, rotc08, None)
    P14      = apply_to_P(data_14_gt.centers,   gt_14, rotc14, off08_14)
    P14_test = apply_to_P(data_14_test.centers, gt_14, rotc14, off08_14)
    P21      = apply_to_P(data_21_test.centers,   gt_21, rotc21, off14_21)

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

    pcd14_points = np.array(pcd14.points)
    pcd14_colors = np.array(pcd14.colors)

    for idx in range(len(data_14_gt.keys)):
        if not mask14[idx]:
            mask = np.linalg.norm(pcd14_points - P14[idx], axis=-1) <= max(data_14_gt.fruit_radius[idx], 1e-4)
            pcd14_colors[mask] = [0.1, 0.1, 0.7]

    pcd14.colors = o3d.utility.Vector3dVector(pcd14_colors)

    ################ 21 to 14

    points, lines  = [], []
    mask21 = np.zeros(P21.shape[0], dtype=bool)

    for k_n in tqdm(conn_gt_test.connections_keys):
        k_p = conn_gt_test.connections_keys[k_n]

        c_21 = P21[data_21_test.keys.index(k_n)]
        c_14 = P14_test[data_14_test.keys.index(k_p)]

        mask21[data_21_test.keys.index(k_n)] = True

        points.append(c_21)
        points.append(c_14)
        lines.append([len(points)-2, len(points)-1])

    colors = [[0, 0.5, 0] for i in range(len(lines))]

    line_set_21 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set_21.colors = o3d.utility.Vector3dVector(colors)


    singlepcd = False
    coloredsinglepcd = True

    if singlepcd:
        pcd21_points = np.array(pcd21.points)
        pcd21_colors = np.array(pcd21.colors)

        for idx in range(len(data_21_test.keys)):
            if not mask21[idx]:
                mask = np.linalg.norm(pcd21_points - P21[idx], axis=-1) <= max(data_21_test.fruit_radius[idx], 1e-4)
                pcd21_colors[mask] = [0.1, 0.1, 0.7]
        
        pcd21.colors = o3d.utility.Vector3dVector(pcd21_colors)
    elif coloredsinglepcd:
        pcd21_points = np.array(pcd21.points)
        pcd21_colors = np.array(pcd21.colors)

        inst_mask = get_inst_mask(data_21_test, pcd21_points, P21)
        pcd21.colors = o3d.utility.Vector3dVector(mask2colors(pcd21_colors, inst_mask))

        pcd14_points = np.array(pcd14.points)
        pcd14_colors = np.array(pcd14.colors)
        new_keys14_test = homokeys(data_14_test, data_21_test, conn_gt_test)
        data_14_test.keys = new_keys14_test

        inst_mask = get_inst_mask(data_14_test, pcd14_points, P14)
        pcd14.colors = o3d.utility.Vector3dVector(mask2colors(pcd14_colors, inst_mask))

        with open("instsegm_gcnconv_2_minxmaxx.pickle", "rb") as handle:
            import pickle
            metrics = pickle.load(handle)
            metrics["gt_keys"] -= 1
            mask = metrics["mask"]
            target_p = metrics["hungpreds"]
            
            # tpm, wpm, tn, fp, fn  = 0, 1, 2, 3, 4 | RGB
            colors = [[0, 255, 0], [255, 0, 0], [0, 255, 0], [255, 165, 0], [255, 0, 0]]
            colors = np.array(colors)/255.0
            
            points, lines = [], []
            line_color = []
            for i in range(mask.shape[0]):

                if mask[i] in [0, 1, 3]: # tpm, wpm, fp
                    c_21 = P21[i]
                    c_14 = P14_test[target_p[i]-1]

                    mask21[data_21_test.keys.index(k_n)] = True

                    points.append(c_21)
                    points.append(c_14)
                    lines.append([len(points)-2, len(points)-1])
                    line_color.append(colors[mask[i]])

            line_set_21 = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(lines),
                )
            line_set_21.colors = o3d.utility.Vector3dVector(line_color)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=3200, height=1600, visible=True)
    # visualizer.add_geometry(pcd08)
    visualizer.add_geometry(pcd14)
    visualizer.add_geometry(pcd21)
    # visualizer.add_geometry(line_set_14)
    visualizer.add_geometry(line_set_21)

    visualizer.get_render_option().load_from_json("view_animate3D.json")
    param = o3d.io.read_pinhole_camera_parameters("view_animate3D_pinhole2.json")
    visualizer.get_view_control().convert_from_pinhole_camera_parameters(param, True)

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(-1.5, 0.0)
        return False

    visualizer.register_animation_callback(rotate_view)
    visualizer.run()
    # visualizer.get_render_option().save_to_json("view_animate3D.json")
    # param = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters("view_animate3D_pinhole2.json", param)


if __name__ == "__main__":
    typer.run(main)
