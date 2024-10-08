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
    # iou_th: float = typer.Option(
    #     ...,
    #     "--iou",
    #     help="IoU threshold for associating to ground truth (to build connections)",
    # ),
    n: int = typer.Option(
        ...,
        "--n",
        help="IoU threshold for associating to ground truth (to build connections)",
    ),
):

    basepath = f"{datapath}"

    os.makedirs(f"{basepath}/peppers_{n}", exist_ok=True)

    annopath = f"{basepath}/selections_{n}.json"
    with open(annopath) as f:
        annotations = json.load(f)

    path = f"{basepath}/map_{n}.ply"
    pcd = o3d.io.read_point_cloud(path)

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for k in tqdm(annotations, total=len(annotations.keys())):
        _, idxs, _ = kdtree.search_radius_vector_3d(annotations[k]['center'], annotations[k]['radius'])
        pepper = pcd.select_by_index(idxs)
        o3d.io.write_point_cloud(f"{basepath}/peppers_{n}/pepper_{str(k).zfill(3)}.ply", pepper)

if __name__ == "__main__":
    cli()