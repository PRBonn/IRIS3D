import open3d as o3d
import json
from tqdm import tqdm
import numpy as np
import os
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

    basepath = f"{datapath}/14_21_inst@{iou_th}"

    os.makedirs(f"{basepath}/strawberries_21_more", exist_ok=True)

    annopath = f"{basepath}/selections_2.json"
    with open(annopath) as f:
        annotations = json.load(f)

    path = f"{basepath}/cloud_2.ply"
    pcd = o3d.io.read_point_cloud(path)

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for k in tqdm(annotations, total=len(annotations.keys())):
        _, idxs, _ = kdtree.search_radius_vector_3d(annotations[k]['center'], 0.05)
        strawberry = pcd.select_by_index(idxs)
        o3d.io.write_point_cloud(f"{basepath}/strawberries_21_more/straw_{str(k).zfill(3)}.ply", strawberry)

if __name__ == "__main__":
    cli()