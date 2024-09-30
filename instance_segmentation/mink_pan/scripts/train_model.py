import os
import subprocess
from os.path import join

#import click
import torch
import yaml
from easydict import EasyDict as edict
from mink_pan.datasets.daniel_dataset import SemanticDatasetModule
from mink_pan.models.model import MinkPan
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

#@click.command()
#@click.option("--w", type=str, default=None, required=False, help="weights to load")
#@click.option(
#    "--ckpt",
#    type=str,
#    default=None,
#    required=False,
#    help="checkpoint to resume training",
#)
#@click.option("--nuscenes", is_flag=True)
#@click.option("--mini", is_flag=True, help="use mini split for nuscenes")
#@click.option(
#    "--seq",
#    type=int,
#    default=None,
#    required=False,
#    help="use a single sequence for train and val",
#)
#@click.option(
#    "--id", type=str, default=None, required=False, help="set id of the experiment"
#)
def main(w=None, ckpt=None, nuscenes=False, mini=False, seq=None, id=None):
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg})
    cfg.git_commit_version = str(
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
    )

    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    if mini and nuscenes:
        cfg.NUSCENES.MINI = True
    if seq:
        cfg.TRAIN.ONLY_SEQ = seq
    if id:
        cfg.EXPERIMENT.ID = id

    data = SemanticDatasetModule(cfg)
    model = MinkPan(cfg)
    if w:
        w = torch.load(w, map_location="cpu")
        model.load_state_dict(w["state_dict"], strict=False)

    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID, default_hp_metric=False
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    iou_ckpt = ModelCheckpoint(
        monitor="metrics/iou",
        filename=cfg.EXPERIMENT.ID + "_{epoch:02d}_{iou:.2f}",
        mode="max",
        save_last=True,
    )
#    pq_ckpt = ModelCheckpoint(
#        monitor="metrics/iou",
#        filename=cfg.EXPERIMENT.ID + "_{epoch:02d}_{pq:.2f}",
#        mode="max",
#        save_last=True,
#    )

    trainer = Trainer(
        # num_sanity_val_steps=0,
        #gpus=cfg.TRAIN.N_GPUS,
        accelerator="auto",
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor, iou_ckpt],#, pq_ckpt],
        # track_grad_norm=2,
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        # overfit_batches=0.0001,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        #resume_from_checkpoint=ckpt,
    )

    ###### Learning rate finder
    # import matplotlib.pyplot as plt
    # lr_finder = trainer.tuner.lr_find(model,data,min_lr=1e-8,max_lr=1e-2, num_training=10000)
    # fig = lr_finder.plot(suggest=True)
    # plt.savefig('lr_finder')

    quit()

    trainer.fit(model, data)


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


if __name__ == "__main__":
    main()
