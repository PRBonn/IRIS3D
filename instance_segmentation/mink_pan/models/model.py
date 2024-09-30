import mink_pan.utils.plot as pl
import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
from mink_pan.models.backbone import MinkEncoderDecoder
from mink_pan.models.loss import InsLoss, SemLoss
from mink_pan.utils.clustering import Clustering
from mink_pan.utils.evaluate_panoptic import PanopticKittiEvaluator
from pytorch_lightning.core.module import LightningModule
from torch.utils.tensorboard import SummaryWriter
import random
import open3d as o3d



class MinkPan(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))
        self.cfg = hparams

        backbone = MinkEncoderDecoder(hparams.BACKBONE)
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(backbone)
        self.sem_head = SemHead(hparams)
        self.ins_head = InsHead(hparams)

        self.sem_loss = SemLoss(hparams.LOSS.SEM.WEIGHTS)
        self.ins_loss = InsLoss()
        self.cluster = Clustering(hparams.POST)
        self.evaluator = PanopticKittiEvaluator(hparams.STRAWBERRIES)
        self.freezeModules()

        #print("loading model", end=" ...")
        #checkpoint = torch.load(f"model@0000.pt", map_location=torch.device("cuda"))
        #self.load_state_dict(checkpoint['encoder_state_dict'])
        #print("done!")
        self.epoch = 0
        self.steps = 0

        self.n_scans = 0

    def freezeModules(self):
        freeze_dict = {
            "BACKBONE": self.backbone,
            "SEM_HEAD": self.sem_head,
            "INS_HEAD": self.ins_head,
        }
        print("Frozen modules: ", self.cfg.TRAIN.FREEZE_MODULES)
        for module in self.cfg.TRAIN.FREEZE_MODULES:
            for param in freeze_dict[module].parameters():
                param.requires_grad = False

    def forward(self, x):
        feats, in_field = self.backbone(x)
        sem_logits = self.sem_head(feats, in_field)
        offsets, ins_feat = self.ins_head(feats, in_field)
        return sem_logits, offsets, ins_feat

    def getLoss(self, x, logits, offsets):
        logits = torch.cat(logits)
        labs = torch.from_numpy(np.concatenate(x["sem_label"])).to(logits.device)
        loss = self.sem_loss(logits, labs.view(-1))
        foreground = [
            torch.from_numpy(f).to(offsets[0].device) for f in x["foreground"]
        ]
        gt_offsets = [torch.from_numpy(o).to(offsets[0].device) for o in x["offset"]]
        ins_loss = self.ins_loss(offsets, gt_offsets, foreground)
        loss["ins"] = 10*ins_loss

        return loss

    def training_step(self, x: dict, idx):
        sem_logits, offsets, ins_feat = self(x)
        loss_dict = self.getLoss(x, sem_logits, offsets)

        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)

        total_loss = sum(loss_dict.values())
        self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        torch.cuda.empty_cache()
        self.steps += 1
        return total_loss

    def validation_step(self, x: dict, idx):

        if "EVALUATE" in self.cfg:
            self.evaluation_step(x, idx)
            return
        sem_logits, offsets, ins_feat = self(x)

        loss_dict = self.getLoss(x, sem_logits, offsets)


#        sem_logits_gt = torch.zeros((sem_logits[0].shape[0], 2)).cuda()
#        sem_logits_gt[x["sem_label"][0]==0, 0] = 1.0
#        sem_logits_gt[x["sem_label"][0]==1, 1] = 1.0

        sem = torch.argmax(sem_logits[0], dim=1).cpu().numpy()
        if sem[sem==1].shape[0]>(sem.shape[0]/5):
            print("skipping validation")
            self.skipped=True
            return 100
        self.skipped=False

        #sem_pred, ins_pred = self.inference(x, [sem_logits_gt], offsets) #sem_logits, offsets)
        sem_pred, ins_pred = self.inference(x, sem_logits, offsets) #sem_logits, offsets)


        print("unique groundtruth insts", np.unique(x["ins_label"][0]))
        print("unique predicted   insts", np.unique(ins_pred[0]))

        inst = ins_pred[0]
        #inst = x["ins_label"][0]
        u = np.unique(inst)
        table = np.random.uniform(0.1, 1.0, (u.max()+1, 3))
        table[0, :] = 0
        idxcol = table[inst]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x["pt_coord"][0])
        pcd.colors = o3d.utility.Vector3dVector(idxcol)
        o3d.visualization.draw(pcd)

        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)

        total_loss = sum(loss_dict.values())
        self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.evaluator.update(sem_pred, ins_pred, x)
        torch.cuda.empty_cache()

        coords = x["pt_coord"][0]

        coords[x["foreground"][0]] = coords[x["foreground"][0]] + x["offset"][0][x["foreground"][0]]#offsets[0][x["foreground"][0]].detach().cpu().numpy()

        colors = np.zeros_like(coords)

        colors[x["foreground"][0]] = [1, 0 ,0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw(pcd)

        coords = x["pt_coord"][0]

        coords[x["foreground"][0]] = coords[x["foreground"][0]] + offsets[0][x["foreground"][0]].detach().cpu().numpy()

        colors = np.zeros_like(coords)

        colors[x["foreground"][0]] = [0, 0, 1]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw(pcd)

        return total_loss

    def on_validation_epoch_end(self):#, outputs):
        if self.skipped:
            return
        self.log(
            "metrics/pq",
            self.evaluator.get_mean_pq(),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )
        self.log(
            "metrics/iou",
            self.evaluator.get_mean_iou(),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )
        self.log(
            "metrics/rq",
            self.evaluator.get_mean_rq(),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )
        self.evaluator.reset()

        print("saving model")
        torch.save({'encoder_state_dict': self.state_dict()}, 
                    f"model@{str(self.epoch).zfill(4)}.pt")

    def evaluation_step(self, x: dict, idx):
        sem_logits, offsets, ins_feat = self(x)
        sem_pred, ins_pred = self.inference(x, sem_logits, offsets)

        if "VIS_PRED" in self.cfg:
            pl.plot_instances(x["pt_coord"], ins_pred)
            color_map = self.trainer.datamodule.color_map
            pl.plot_semantics(x["pt_coord"], sem_pred, color_map)
        else:
            self.evaluator.update(sem_pred, ins_pred, x)

        if "SAVE_VAL" in self.cfg:
            pl.plot_instances(x["pt_coord"], ins_pred, save=True, n=self.n_scans)
            color_map = self.trainer.datamodule.color_map
            pl.plot_semantics(
                x["pt_coord"], sem_pred, color_map, save=True, n=self.n_scans
            )
            self.n_scans += 1

    def test_step(self, x: dict, idx):
        pass

    def setup(self, stage=None):
        ids = self.trainer.datamodule.things_ids
        self.cluster.set_ids(ids)

    def inference(self, x, sem_logits, offsets):
        sem_pred = []
        ins_pred = []
        for i in range(len(sem_logits)):
            sem = torch.argmax(sem_logits[i], dim=1).cpu().numpy()
            ins = self.cluster(sem, offsets[i], x["pt_coord"][i])
            sem_pred.append(sem)
            ins_pred.append(ins)
        return sem_pred, ins_pred

    def sem_inference(self, sem_logits):
        sem_pred = []
        for i in range(len(sem_logits)):
            sem = torch.argmax(sem_logits[i], dim=1).cpu().numpy()
            sem_pred.append(sem)
        return sem_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.TRAIN.LR)
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=self.cfg.TRAIN.STEP, gamma=self.cfg.TRAIN.DECAY
        # )
        # return [optimizer], [scheduler]
        return [optimizer]


## Heads
class SemHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_cls = int(cfg[cfg.MODEL.DATASET].NUM_CLASSES)
        out = int(cfg.BACKBONE.CHANNELS[-1] * cfg.BACKBONE.CR)
        self.conv = ME.MinkowskiConvolution(
            out, n_cls, kernel_size=3, dilation=1, stride=1, dimension=3
        )

    def forward(self, x, in_field):
        logits = self.conv(x)
        ## vox2points
        logits = logits.slice(in_field)
        logits = logits.decomposed_features
        return logits


class InsHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        out = int(cfg.BACKBONE.CHANNELS[-1] * cfg.BACKBONE.CR)
        init = int(cfg.BACKBONE.CHANNELS[0] * cfg.BACKBONE.CR)

        self.res = cfg.BACKBONE.RESOLUTION

        self.conv1 = ME.MinkowskiConvolution(
            out, out, kernel_size=3, dilation=1, stride=1, dimension=3
        )
        self.bn1 = ME.MinkowskiBatchNorm(out)
        self.act1 = ME.MinkowskiLeakyReLU(True)
        self.conv2 = ME.MinkowskiConvolution(
            out, 2 * init, kernel_size=3, dilation=1, stride=1, dimension=3
        )
        self.bn2 = ME.MinkowskiBatchNorm(2 * init)
        self.act2 = ME.MinkowskiLeakyReLU(True)
        self.conv3 = ME.MinkowskiConvolution(
            2 * init, init, kernel_size=3, dilation=1, stride=1, dimension=3
        )
        self.bn3 = ME.MinkowskiBatchNorm(init)
        self.act3 = ME.MinkowskiLeakyReLU(True)

        self.offset = nn.Sequential(
            nn.Linear(init + 3, init, bias=True), nn.BatchNorm1d(init), nn.ReLU()
        )
        self.offset_linear = nn.Linear(init, 3, bias=True)

    def forward(self, x, in_field):
        out = self.conv1(x)
        out = self.act1(self.bn1(out))
        out = self.conv2(out)
        out = self.act2(self.bn2(out))
        out = self.conv3(out)
        out = self.act3(self.bn3(out))
        ## vox2points
        out = out.slice(in_field)
        feats = out.decomposed_features
        coors = [c * self.res for c in out.decomposed_coordinates]

        offsets = [
            self.offset_linear(self.offset(torch.cat((f, c), dim=1)))
            for f, c in zip(feats, coors)
        ]
        return offsets, feats
