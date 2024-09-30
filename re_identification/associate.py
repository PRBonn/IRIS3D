import numpy as np
import open3d as o3d
import yaml
import json
from datasetgcn import *
from tqdm import tqdm

import typer
cli = typer.Typer()

@cli.command()
def main(
    datapath: str = typer.Option(
        ...,
        "--data",
        help="data path ()",
    ),
    iou_th: float = typer.Option(
        ...,
        "--iou",
        help="IoU threshold for associating to ground truth (to build connections)",
    ),
):
	print("producing with IoU threshold:", iou_th)

	with open(f"{datapath}/transformations.yaml") as stream:
		try:
			transformations = yaml.safe_load(stream)["transformations"]
		except yaml.YAMLError as exc:
			print(exc)

	gt_08 = np.asarray(transformations["gt_08"])
	gt_14 = np.asarray(transformations["gt_14"])
	gt_21 = np.asarray(transformations["gt_21"])


	## load big cloud
	path_to_cloud = f"{datapath}/reduced_14_21_2.ply"
	# o3d load cloud
	pcd = o3d.io.read_point_cloud(path_to_cloud)
	pcd.transform(gt_21)
	points = np.array(pcd.points)
	colors = np.array(pcd.colors)

	## load gt inst segm labels
	## use Strawberry class
	test_next = Strawberries(f"{datapath}/14_21/strawberries_21", os.path.join(datapath, "14_21/selections_2.json"), gt_21, min_x=27.58, max_x=29.45)


	## load predicted inst segm labels
	path_to_pred_instances = "../instance_segmentation/mink_pan/predicted_instances.npy"
	instances = np.fromfile(path_to_pred_instances, dtype=np.int32)
	assert instances.shape[0] == points.shape[0], str(instances.shape[0]) + " " + str(points.shape[0])

	unique_insts, count = np.unique(instances, return_counts=True)
	print("found", len(unique_insts), "unique predicted instances (including idx 0)")

	selections = {}
	unassigned_id = max(test_next.keys) + 1000

	def get_sphere_volume(r):
		return 4/3*np.pi*(r**3)
	def get_intersection_volume(c0, c1, r0, r1):
		d = np.linalg.norm(c0-c1)
		if d>= (r0+r1):
			return 0
		if d<=abs(r0-r1):
			return get_sphere_volume(min(r0, r1))
		return np.pi * (r0+r1-d)**2 * (d**2 + 2*d*(r0+r1) - 3*(r0-r1)**2) / 12.0 / d
	def get_iou(c0, c1, r0, r1):
		intersection = get_intersection_volume(c0, c1, r0, r1)
		union = get_sphere_volume(r0) + get_sphere_volume(r1) - intersection
		return intersection/union

	pred_keys = {}

	for u in tqdm(unique_insts, total=unique_insts.shape[0]):
		if u ==0:
			continue
		mask = instances == u
		pts_u = points[mask]
		clr_u = colors[mask]
		center = pts_u.mean(axis=0)
		assert center.shape[0] == 3, "something wrong with np mean axis"
		radius = np.linalg.norm(pts_u - center, axis=1, ord=2).max()
		if radius>0:
			pred_keys[u] = [center, radius, clr_u.mean(axis=0), clr_u.std(axis=0)]


	pred_keys_list = list(pred_keys.keys())
	mask_alreadyassigned = np.zeros(len(pred_keys_list), dtype=bool)
	mask_newid           = np.zeros(len(pred_keys_list), dtype=int)

	for k in tqdm(test_next.keys, total=len(test_next.keys)):
		ious_with_pred_spheres = np.zeros(len(pred_keys_list))

		kidx = test_next.keys.index(k)
		c = test_next.centers[kidx]
		r = test_next.fruit_radius[kidx]

		for pre_idx, prk in enumerate(pred_keys_list):
			iou = get_iou(c, pred_keys[prk][0], r, pred_keys[prk][1])
			ious_with_pred_spheres[pre_idx] = iou
		
		ious_with_pred_spheres[mask_alreadyassigned] = -1
		best_idx = np.argmax(ious_with_pred_spheres)
		best_iou = ious_with_pred_spheres[best_idx]

		if best_iou > iou_th:
			mask_newid[best_idx] = k
			mask_alreadyassigned[best_idx]=True


	print("assigned: ", mask_alreadyassigned.sum(), "/", len(pred_keys_list))
	print("gt", len(test_next.keys))

	for i in range(mask_alreadyassigned.shape[0]):
		predk = pred_keys_list[i]
		k = mask_newid[i]

		center_ = np.ones((1, 4))
		center_[0, :3] = pred_keys[predk][0]
		center = np.matmul(np.linalg.inv(gt_21), center_.T).T[0, :3]

		if k>0:
			selections[int(k)] = {
				"id": int(k),
				"center": center.tolist(),
				"radius": pred_keys[predk][1],
				"color_mean": pred_keys[predk][2].tolist(),
				"color_std": pred_keys[predk][3].tolist(),
				"color_display": np.random.uniform(3, 0, 1).tolist(),
				"bestiou": best_iou,
			}
		else:
			selections[int(unassigned_id)] = {
				"id": int(unassigned_id),
				"center": center.tolist(),
				"radius": pred_keys[predk][1],
				"color_mean": pred_keys[predk][2].tolist(),
				"color_std": pred_keys[predk][3].tolist(),
				"color_display": np.random.uniform(3, 0, 1).tolist(),
				"bestiou": best_iou,
			}
			unassigned_id += 1


	outpath = f"{datapath}/14_21_inst@{iou_th}"
	os.makedirs(outpath, exist_ok=True)

	os.system(f'ln -s {datapath}/14_21/cloud_1.ply {os.path.join(outpath, "cloud_1.ply")}')
	os.system(f'ln -s {datapath}/14_21/cloud_2.ply {os.path.join(outpath, "cloud_2.ply")}')
	os.system(f'ln -s {datapath}/14_21/connections.json {os.path.join(outpath, "connections.json")}')
	os.system(f'ln -s {datapath}/14_21/selections_1.json {os.path.join(outpath, "selections_1.json")}')
	os.system(f'ln -s {datapath}/14_21/strawberries_14_more {os.path.join(outpath, "strawberries_14_more")}')
	os.makedirs(os.path.join(outpath, "strawberries_21_more"), exist_ok=True)


	print("dumping to json", end="...")
	with open(os.path.join(outpath, "selections_2.json"), "w") as outfile:
		json.dump(selections, outfile, indent = 4)
	print("done! finished.")

if __name__ == "__main__":
    cli()