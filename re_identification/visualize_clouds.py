import open3d as o3d
import numpy as np
import yaml
from datasetgcn import *
from lossconnection import LossConnection
import pickle
import cv2

import typer
cli = typer.Typer()



class VisualizeClouds:
    def __init__(self, datapath, iou):

        self.datapath = datapath
        self.iou      = iou

        print(self.datapath
        )

        transformation_path = os.path.join(self.datapath, "transformations.yaml")
        with open(transformation_path, "r") as stream:
            try:
                transformations = yaml.safe_load(stream)["transformations"]
            except yaml.YAMLError as exc:
                print(exc)

        self.gt_08 = np.asarray(transformations["gt_08"])
        self.gt_14 = np.asarray(transformations["gt_14"])
        self.gt_21 = np.asarray(transformations["gt_21"])


    def get_inst_mask(self, data, coords):
        inst_mask = np.zeros(coords.shape[0], dtype=np.int32)

        for k, c, r in tqdm(zip(data.keys, data.centers, data.fruit_radius)):
            if k==0:
                continue
            mask = np.linalg.norm(coords-c, axis=1) <= r
            inst_mask[mask] = k
            #break
        return inst_mask

    def mask2colors(self, colors, inst_mask):
        u = np.unique(inst_mask)
        np.random.seed(0)
        table = np.random.uniform(0.1, 1.0, (u.max()+1, 3))
        table[0, :] = 0

        newcolors = table[inst_mask]
        newcolors = np.where(newcolors==0, colors/2, newcolors)
        return newcolors

    def homokeys(self, data_from, data_next, conn):
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


    def produce_14gt(self, also_mask=True):
        cloud_path = os.path.join(self.datapath, "reduced_08_14_2.ply")
        pcd = o3d.io.read_point_cloud(cloud_path)
        pcd.transform(self.gt_14)
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)

        straw_path = os.path.join(self.datapath, f"14_21")

        data_14_gt = Strawberries(f"{straw_path}/strawberries_14", os.path.join(straw_path, "selections_1.json"), self.gt_14, min_x=27.58, max_x=29.45)
        data_21_gt = Strawberries(f"{straw_path}/strawberries_21", os.path.join(straw_path, "selections_2.json"), self.gt_21, min_x=27.58, max_x=29.45)
        conn_gt    = LossConnection(os.path.join(straw_path, "connections.json"), data_14_gt, data_21_gt)

        new_keys = self.homokeys(data_14_gt, data_21_gt, conn_gt)
        data_14_gt.keys = new_keys.tolist()
        inst_14_gt_mask = self.get_inst_mask(data_14_gt, points)
        pcd.colors = o3d.utility.Vector3dVector(self.mask2colors(colors, inst_14_gt_mask))

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=3200, height=1600, visible=True)
        visualizer.add_geometry(pcd)
        view_ctl = visualizer.get_view_control()
        
        param = view_ctl.convert_to_pinhole_camera_parameters()
        view_ctl.convert_from_pinhole_camera_parameters(param, True)
        visualizer.run()

        o3d.io.write_pinhole_camera_parameters("view2.json", param)

        visualizer.capture_screen_image("14gt.png")

        if also_mask:
            camera = view_ctl.convert_to_pinhole_camera_parameters()
            A = camera.intrinsic.intrinsic_matrix
            T = camera.extrinsic[:3, :]
            P = np.ones((data_14_gt.centers.shape[0], 4))
            P[:, :3] = data_14_gt.centers
            P = A@(T@(P.T))
            P /= P[2, :]
            P = np.round(P.T).astype(int)[:, :2]  ## [[u,v]]

            recipe = {}
            for i in range(P.shape[0]):
                recipe[data_14_gt.keys[i]] = P[i]
            with open("recipe_14gtmask.pickle", "wb") as handle:
                pickle.dump(recipe, handle)

            img = cv2.imread("14gt.png", cv2.IMREAD_UNCHANGED)
            for c in P:
                img = cv2.circle(img, (c[0], c[1]), 10, [255, 0, 0], 3)
            cv2.imshow("just a check, not saving this image", img)
            cv2.waitKey(0)


    def produce_21pred(self, also_mask=True):
        cloud_path = os.path.join(self.datapath, "reduced_14_21_2.ply")
        pcd = o3d.io.read_point_cloud(cloud_path)
        pcd.transform(self.gt_21)
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)

        straw_path = os.path.join(self.datapath, f"14_21_inst@{self.iou}")
        data_21_pred = Strawberries(f"{straw_path}/strawberries_21", os.path.join(straw_path, "selections_2.json"), self.gt_21, min_x=27.58, max_x=29.45)

        inst_21_pred_mask = self.get_inst_mask(data_21_pred, points)
        pcd.colors = o3d.utility.Vector3dVector(self.mask2colors(colors, inst_21_pred_mask))

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=3200, height=1600, visible=True)
        visualizer.add_geometry(pcd)
        view_ctl = visualizer.get_view_control()
        
        param = o3d.io.read_pinhole_camera_parameters("view2.json")
        view_ctl.convert_from_pinhole_camera_parameters(param, True)
        visualizer.run()
        visualizer.capture_screen_image("21pred.png")
        camera = view_ctl.convert_to_pinhole_camera_parameters()

        A = camera.intrinsic.intrinsic_matrix
        T = camera.extrinsic[:3, :]

        P = np.ones((data_21_pred.centers.shape[0], 4))
        P[:, :3] = data_21_pred.centers
        P = A@(T@(P.T))
        P /= P[2, :]
        P = np.round(P.T).astype(int)[:, :2]  ## [[u,v]]

        recipe = {}
        for i in range(P.shape[0]):
            recipe[data_21_pred.keys[i]] = P[i]
        with open("recipe_21predmask.pickle", "wb") as handle:
            pickle.dump(recipe, handle)

        img = cv2.imread("21pred.png", cv2.IMREAD_UNCHANGED)
        for c in P:
            img = cv2.circle(img, (c[0], c[1]), 10, [255, 0, 0], 3)
        cv2.imshow("just a check, not saving this image", img)
        cv2.waitKey(0)


    def draw_matches(self):

        straw_path = os.path.join(self.datapath, f"14_21_inst@{self.iou}")
        data_14_gt   = Strawberries(f"{straw_path}/strawberries_14", os.path.join(straw_path, "selections_1.json"), self.gt_14, min_x=27.58, max_x=29.45)
        data_21_pred = Strawberries(f"{straw_path}/strawberries_21", os.path.join(straw_path, "selections_2.json"), self.gt_21, min_x=27.58, max_x=29.45)
        conn_gt    = LossConnection(os.path.join(straw_path, "connections.json"), data_14_gt, data_21_pred)
        new_keys14 = self.homokeys(data_14_gt, data_21_pred, conn_gt)
        #data_14_gt.newkeys = new_keys#.tolist()

        img14 = cv2.imread("14gt.png",   cv2.IMREAD_UNCHANGED)
        img21 = cv2.imread("21pred.png", cv2.IMREAD_UNCHANGED)

        h, _, _ = img14.shape

        a = 1.7*(h//5)
        b = 2.5*(h//4)

        a=int(a)
        b=int(b)
        img14 = img14[a:b, :, :]
        img21 = img21[a:b, :, :]

        h2, w2, _ = img14.shape

        print(img14.shape, img21.shape)

        with open("recipe_14gtmask.pickle", "rb") as handle:
            recipe14 = pickle.load(handle)

        with open("recipe_21predmask.pickle", "rb") as handle:
            recipe21 = pickle.load(handle)

        with open("instsegm_gcnconv_2.pickle", "rb") as handle:
            metrics = pickle.load(handle)
            metrics["gt_keys"] -= 1

        gt_argmax = metrics["gt_keys"]
        hungpreds = metrics["hungpreds"]
        mask = metrics["mask"]

        canvas = np.zeros((h2+h2, w2, 3), img14.dtype)
        canvas[0:h2, :, :] = img21
        canvas[h2:, :, :]  = img14

        # tpm, wpm, tn, fp, fn  = 0, 1, 2, 3, 4

        colors = [[0, 255, 0], [0, 0, 255], [0, 255, 0], [0, 165, 255], [0, 0, 255]]

        #print(new_keys14)
        #print(recipe14.keys())

        for i, k in enumerate(recipe21.keys()):
            s = recipe21[k]

            if hungpreds[i]==0:
                pred_matching_key = 0
            else:
                pred_matching_key = new_keys14[hungpreds[i]-1]

            color = colors[mask[i]]

            #if k in recipe14:
            if pred_matching_key==0:# or pred_matching_key>=5000:
                y = s[1] - a
                canvas = cv2.circle(canvas, (s[0], y), 15, color, 2)
            elif pred_matching_key>=5000:
                scam = new_keys14.tolist().index(pred_matching_key)
                pred_matching_key = list(recipe14.keys())[scam]
                e = recipe14[pred_matching_key]
                s[1] -= a
                e[1] += h2 - a
                canvas = cv2.line(canvas, s, e, color, 2)
            else:
                e = recipe14[pred_matching_key]
                s[1] -= a
                e[1] += h2 - a
                canvas = cv2.line(canvas, s, e, color, 2)


        cv2.imshow("p", canvas)
        cv2.waitKey(0)


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
    visclouds = VisualizeClouds(datapath, iou)
    # visclouds.produce_14gt(also_mask=True)
    # visclouds.produce_21pred(also_mask=True)
    visclouds.draw_matches()


if __name__ == "__main__":
    typer.run(main)
