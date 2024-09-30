import os
import subprocess
from os.path import join

#import click
import torch
import yaml
from easydict import EasyDict as edict
from mink_pan.datasets.deploy_dataset import SemanticDatasetModule
from mink_pan.models.model import MinkPan

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import numpy as np
import open3d as o3d
import random
from mink_pan.utils.evaluate_panoptic import PanopticKittiEvaluator
import typer
cli = typer.Typer()

@cli.command()
def main(
	modelpath: str = typer.Option(
        ...,
        "--modelpath",
        help="data path ()",
    ),
):
	model_cfg = edict(
		yaml.safe_load(open("config/model.yaml"))
	)
	backbone_cfg = edict(
		yaml.safe_load(open("config/backbone.yaml"))
	)
	cfg = edict({**model_cfg, **backbone_cfg})


	data = SemanticDatasetModule(cfg)
	model = MinkPan(cfg)
	model.to(torch.device("cuda"))
	model.eval()

	inseval = PanopticKittiEvaluator(cfg.STRAWBERRIES)

	w = torch.load(modelpath, map_location="cpu")
	model.load_state_dict(w["encoder_state_dict"], strict=False)
	tb_logger = pl_loggers.TensorBoardLogger(
		"experiments/" + cfg.EXPERIMENT.ID, default_hp_metric=False
	)

	data.setup()
	model.cluster.set_ids(data.things_ids)

	cont = 0

	keys = [
		"pt_coord",
		"feats",
		"sem_label",
		"ins_label",
		"offset",
		"foreground",
		"totmask",
		"mid_x",
		"token",
	]

	min_x = 27.58
	max_x = 29.45
	ext = 0.3

	coords   = np.array(data.test_pan_set.dataset.pcds[0].points)
	full_ins = np.zeros(coords.shape[0], dtype=np.int32)
	full_offset = np.zeros_like(coords)

	A = min_x
	B = A + ext
	eps = 0.05

	while True:
		x = data.test_pan_set.__getitem__(A, B)
		x = {keys[i]: list(j) for i, j in enumerate(zip(x))}


		with torch.no_grad():
			sem_logits, offsets, ins_feat = model(x)
			sem_pred, ins_pred = model.inference(x, sem_logits, offsets)

		vA = A if A <= min_x else A + eps
		vB = B if B >= max_x else B - eps
		print(f"{ A:6.3f} : { B:6.3f} / { max_x:6.3f}")

		inseval.update(sem_pred, ins_pred, x)

		unique_pr = np.unique(ins_pred[0])
		if 0 in unique_pr:
			unique_pr = np.delete(unique_pr, 0)

		ins_offset = full_ins.max()

		MAXX = 0

		for k in unique_pr:
			kth_inst_mask = ins_pred[0] == k
			kth_inst_pts_x  = x["pt_coord"][0][kth_inst_mask][:, 0] + x["mid_x"][0]

			mask = np.logical_and(kth_inst_pts_x>=vA, kth_inst_pts_x<vB)

			if mask.all():
				idxs = np.arange(0, full_ins.shape[0], 1)[x["totmask"][0]][kth_inst_mask]
				full_ins[idxs] = k + ins_offset
				full_offset[idxs, :] = offsets[0][kth_inst_mask, :].detach().cpu().numpy()
				maxx = kth_inst_pts_x.max()
				if maxx>MAXX:
					MAXX = maxx

		A = MAXX - eps
		B = A + ext
		if B>max_x:
			break

	full_ins.tofile("predicted_instances.npy")
	#full_offset.tofile("predicted_offsets.npy")

	inseval.print_results()
	inseval.print_fp_fn()

if __name__ == "__main__":
    cli()