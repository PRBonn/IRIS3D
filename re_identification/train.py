import json
from tqdm import tqdm
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import itertools
import pickle

from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment

from datasetgcn import *
from model_zoo import *
from matchingloss import MatchingLoss

from lossconnection import LossConnection
from utils_metrics import *
import typer
cli = typer.Typer()


device = torch.device("cuda")

###### GENERAL PARAMETERS ######
max_epochs_num = 200
#mode = "testinst" #"kfold_crossvalidation", "finaltrain", "test", "testinst"
seeds = [ 0, 13, 42, 1997]

datasets_from = []
datasets_next = []

dataloaders_from = []
dataloaders_next = []

connections = []


class CrossValidationSetup:
    def __init__(self, foldidx, seedidx, mode, datapath, iou_th=None):

        self.foldidx  = foldidx
        self.seedidx  = seedidx
        self.mode     = mode
        self.datapath = datapath
        self.iou_th   = iou_th
        self.seed = seeds[self.seedidx]

        self.minx = 27.58
        self.maxx = 29.45

        self.epoch = 0
        self.max_epochs_num = max_epochs_num

        with open(f"{self.datapath}/transformations.yaml") as stream:
            try:
                transformations = yaml.safe_load(stream)["transformations"]
            except yaml.YAMLError as exc:
                print(exc)

        gt_08 = np.asarray(transformations["gt_08"])
        gt_14 = np.asarray(transformations["gt_14"])
        gt_21 = np.asarray(transformations["gt_21"])

        Ts = {"08": gt_08, "14": gt_14, "21": gt_21}

        with open("backbone.yaml") as stream:
            try:
                cfg = edict(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

        self.learning_rate = float(cfg.GENERAL["initial_lr"])

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.mode in ["kfold_crossvalidation", "finaltrain"]:
            from_ = "08"
            next_ = "14"
            foldspath = f"{self.datapath}/{from_}_{next_}/kfoldsplit"
            #datapath = "data/08_14"

            fold_splits = glob.glob(os.path.join(foldspath, "split_*"))
            fold_splits.sort()
            
            for split_idx, foldpath in enumerate(fold_splits):
                data_from = Strawberries(f"{self.datapath}/{from_}_{next_}/strawberries_{from_}", os.path.join(foldpath, "selections_1.json"), Ts[from_], min_x=0, max_x=100)
                data_next = Strawberries(f"{self.datapath}/{from_}_{next_}/strawberries_{next_}", os.path.join(foldpath, "selections_2.json"), Ts[next_], min_x=0, max_x=100)
                datasets_from.append(data_from)
                datasets_next.append(data_next)

                dataloaders_from.append(DataLoader(datasets_from[-1], batch_size=len(datasets_from[-1]), shuffle=True,  collate_fn=datasets_from[-1].custom_collation_fn))
                dataloaders_next.append(DataLoader(datasets_next[-1], batch_size=len(datasets_next[-1]), shuffle=True,  collate_fn=datasets_next[-1].custom_collation_fn))

                connections.append(LossConnection(os.path.join(self.datapath, f"{from_}_{next_}", "connections.json"), datasets_from[-1], datasets_next[-1]))
                connections[-1].printSummary(split_idx)

        if self.mode=="kfold_crossvalidation":
            self.setupKFold()

        elif self.mode == "finaltrain":
            self.setupFinalTrain()

        elif self.mode == "test":
            print("|** instance segmentation test set summary")
            #datapath = "data/testdata_14_21"
            self.test_from  = Strawberries(f"{self.datapath}/14_21/strawberries_14", os.path.join(self.datapath, "selections_1.json"),  Ts["14"], min_x=self.minx, max_x=self.maxx,    training=False)
            self.test_next  = Strawberries(f"{self.datapath}/14_21/strawberries_21", os.path.join(self.datapath, "selections_2.json"),  Ts["21"], min_x=self.minx, max_x=self.maxx,    training=False)
            self.testloader_from = DataLoader(self.test_from, batch_size=len(self.test_from), shuffle=False,  collate_fn=self.test_from.custom_collation_fn)
            self.testloader_next = DataLoader(self.test_next, batch_size=len(self.test_next), shuffle=False,  collate_fn=self.test_next.custom_collation_fn)
            self.test_gt  = LossConnection(os.path.join(self.datapath, "connections.json"), self.test_from, self.test_next)
            self.test_gt.printSummary(0)

        elif self.mode == "testinst":
            print("|** instance segmentation test set summary")
            self.test_from  = Strawberries(f"{self.datapath}/14_21_inst@{iou_th}/strawberries_14", os.path.join(self.datapath, f"14_21_inst@{iou_th}/selections_1.json"),  Ts["14"], min_x=self.minx, max_x=self.maxx,    training=False)
            self.test_next  = Strawberries(f"{self.datapath}/14_21_inst@{iou_th}/strawberries_21", os.path.join(self.datapath, f"14_21_inst@{iou_th}/selections_2.json"),  Ts["21"], min_x=self.minx, max_x=self.maxx,    training=False)
            self.testloader_from = DataLoader(self.test_from, batch_size=len(self.test_from), shuffle=False,  collate_fn=self.test_from.custom_collation_fn)
            self.testloader_next = DataLoader(self.test_next, batch_size=len(self.test_next), shuffle=False,  collate_fn=self.test_next.custom_collation_fn)
            self.test_gt  = LossConnection(os.path.join(self.datapath, f"14_21_inst@{iou_th}", "connections.json"), self.test_from, self.test_next)
            self.test_gt.printSummary(0)

        elif self.mode not in ["test", "testinst"]:
            quit(f"Mode {self.mode} not implemented. Error.")

        self.encoder = Encoder(cfg).to(device)
        self.matcher = Matcher(cfg.MATCHER, self.encoder.descriptor_len).to(device)

        self.optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.matcher.parameters()), lr=self.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)

        self.crossentropy = nn.CrossEntropyLoss()
        self.matchingloss = MatchingLoss()

        self.prev_train_acc,  self.prev_val_acc,  self.prev_test_acc  = 0, 0, 0
        self.prev_train_loss, self.prev_val_loss, self.prev_test_loss = 10000000, 10000000, 10000000

        self.prev_val_f1 = 0


        results_outbasepath = os.path.join(self.datapath, "log", self.encoder.str + "," + self.matcher.str)
        if self.mode == "kfold_crossvalidation":
            self.logpath = f"{results_outbasepath}/fold_{str(self.foldidx).zfill(2)}/seed_{str(self.seed).zfill(5)}"
        else:
            self.logpath = f"{results_outbasepath}/fullset/seed_{str(self.seed).zfill(5)}"
        print("logpath: ", self.logpath)
        os.makedirs(self.logpath, exist_ok=True)

        if self.mode in ["test", "testinst"]:
            checkpoint_path = os.path.join(self.logpath, "best_model_f1.pt")
            if not os.path.isfile(checkpoint_path):
                quit(f"file {checkpoint_path} does not exist! Error.")
            print(f"loading model from {checkpoint_path}", end="... ")

            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.matcher.load_state_dict(checkpoint['matcher_state_dict'])
            self.encoder.eval()
            self.matcher.eval()
            print("done!")

        self.logpath_train = os.path.join(self.logpath, "train")
        self.logpath_valid = os.path.join(self.logpath, "valid")
        self.logpath_test  = os.path.join(self.logpath, "test")
        os.makedirs(self.logpath_train, exist_ok=True)
        os.makedirs(self.logpath_valid, exist_ok=True)
        os.makedirs(self.logpath_test,  exist_ok=True)


        if self.mode in ["kfold_crossvalidation", "finaltrain"]:

            self.logfilepath_train = os.path.join(self.logpath, "log_train.txt")
            self.logfile_train = open(self.logfilepath_train, "w")

            self.logfilepath_valid = os.path.join(self.logpath, "log_valid.txt")
            self.logfile_valid = open(self.logfilepath_valid, "w")
            self.logfilepath_train_best_loss = os.path.join(self.logpath, "log_train_best_loss.txt")
            self.logfilepath_valid_best_loss = os.path.join(self.logpath, "log_valid_best_loss.txt")

        if self.mode ==["test", "testinst"]:
            self.logfilepath_test = os.path.join(self.logpath, f"log_{self.mode}.txt")
            self.logfile_test = open(self.logfilepath_test, "w")

            self.logfilepath_test_best_acc   = os.path.join(self.logpath, "log_test_best_acc.txt")
            self.logfilepath_test_best_loss  = os.path.join(self.logpath, "log_test_best_loss.txt")

        self.logfilepath = os.path.join(self.logpath, "log.txt")
        self.logfile = open(self.logfilepath, "a")
        self.logfile.write(f"foldidx {self.foldidx}\n")
        self.logfile.write(f"seedidx {self.seedidx}\n")
        self.logfile.write(f"seed {seeds[self.seedidx]}\n")
        self.logfile.write(f"initial learning rate {self.learning_rate}\n")
        self.logfile.write(f"optimizer AdamW\n")
        self.logfile.write(f"scheduler StepLR with gamma {0.99}\n")
        self.logfile.write(f"max_epochs_num {self.max_epochs_num}\n")

    def setupKFold(self):
        self.trainsets_idx = np.arange(len(fold_splits), dtype=np.int32)
        self.trainsets_idx = np.delete(self.trainsets_idx, self.foldidx)
        self.validsets_idx = np.array([self.foldidx])
        self.setupTrainValidSets()

    def setupFinalTrain(self):
        self.trainsets_idx = np.arange(len(fold_splits)-1, dtype=np.int32)
        self.validsets_idx = np.array([len(fold_splits)-1])
        self.setupTrainValidSets()

    def setupTrainValidSets(self):
        self.trainsets_from = [datasets_from[i] for i in self.trainsets_idx]
        self.trainsets_next = [datasets_next[i] for i in self.trainsets_idx]
        self.validsets_from = [datasets_from[i] for i in self.validsets_idx]
        self.validsets_next = [datasets_next[i] for i in self.validsets_idx]

        self.trainloaders_from = [dataloaders_from[i] for i in self.trainsets_idx]
        self.trainloaders_next = [dataloaders_next[i] for i in self.trainsets_idx]
        self.train_gts = [connections[i] for i in self.trainsets_idx]
        self.valid_gts = [connections[i] for i in self.validsets_idx]

        print("train on: ", self.trainsets_idx)
        print("valid on: ", self.validsets_idx)
        print()

    def process(self, dlf, tf, dln, tn, conn):
        descriptors_from, from_idxs_b, from_keys_b = self.encoder(dlf, tf)
        descriptors_next, next_idxs_b, next_keys_b = self.encoder(dln, tn)

        ## matching
        next_centered, connmatrix = conn.getdata(from_idxs_b, next_idxs_b)
        
        predicted_matrix_logits = self.matcher(descriptors_from, descriptors_next, next_centered, tf.training)
        
        gt_one_hot = connmatrix.T[1:, :]
        gt_argmax  = gt_one_hot.argmax(dim=1)

        return predicted_matrix_logits, gt_one_hot, gt_argmax, {"from_idxs_b":from_idxs_b, "from_keys_b": from_keys_b, "next_idxs_b":next_idxs_b, "next_keys_b": next_keys_b}


    def do_one_train_epoch(self):

        print(f"epoch {self.epoch} started")# with lr {self.scheduler.get_last_lr()}")

        self.encoder.train()
        self.matcher.train()

        pred_logits = []
        gt_one_hots = []
        gt_argmaxs  = []
        pluss       = []
        loss = 0

        for tlf, tf, tln, tn, gt in zip(self.trainloaders_from, self.trainsets_from, self.trainloaders_next, self.trainsets_next, self.train_gts):
            tf.training=True
            tn.training=True
            predicted_matrix_logits, gt_one_hot, gt_argmax, plus = self.process(tlf, tf, tln, tn, gt)

            weight_zero = 0.08
            weight_other = 0.92/(predicted_matrix_logits.shape[1]-1)
            weights = torch.ones(predicted_matrix_logits.shape[1], device=device)*weight_other
            weights[0]=weight_zero
            celoss = nn.CrossEntropyLoss(weight=weights)

            loss += celoss(predicted_matrix_logits, gt_argmax) + self.matchingloss(predicted_matrix_logits)
            pred_logits.append(predicted_matrix_logits.detach().cpu())
            gt_one_hots.append(gt_one_hot.detach().cpu())
            gt_argmaxs.append(gt_argmax.detach().cpu())
            pluss.append(plus)
            del predicted_matrix_logits, gt_one_hot, gt_argmax, plus
            del tlf, tf, tln, gt
            torch.cuda.empty_cache()

        self.optimizer.zero_grad()
        print("loss", loss.item())
        loss.backward()
        self.optimizer.step()

        with open(os.path.join(self.logpath_train, f"epoch_{str(self.epoch).zfill(3)}.pickle"), 'wb') as handle:
            pickle.dump({
            'predicted_matrix': pred_logits,
            'gt': gt_one_hots,
            }, handle)
        torch.cuda.empty_cache()

        m  = computeMetricsList(pred_logits, gt_one_hots)
        ma = computeMetricsAlessandroList(pred_logits, gt_one_hots)

        if self.prev_train_loss < loss:
            self.prev_train_loss = loss.item()
            
            with open(os.path.join(self.logpath, f"best_train_loss.pickle"), 'wb') as handle:
                pickle.dump({
                'predicted_matrix': pred_logits,
                'gt': gt_one_hots,
                }, handle)
            with open(self.logfilepath_train_best_loss, "w") as f:
                f.write(f"epoch: {self.epoch}\nacc: {m['acc']}\nloss: {loss.item()}\nprecision: {ma['prec']}\nrecall: {ma['rec']}\nf1 {ma['f1']}")
        torch.cuda.empty_cache()
        
        logtxt = f'epoch {self.epoch:3d} - avg loss: {loss: 8.6f} - accuracy {m["acc"]:5.3f} best: {self.prev_train_acc:5.3f} ( {m["cor"]:3d} / {m["tot"]:3d} )  (tn {m["tn"]:3d} / {m["fn"]:3d} | {m["gtz"]:3d})  - precision {ma["prec"]:4.2f}  recall {ma["rec"]:4.2f}  f1 {ma["f1"]:4.2f}'
        print(logtxt)
        self.logfile.write(logtxt + "\n")
        self.logfile_train.write(logtxt + "\n")
        
        return m["acc"]

    def do_one_valid_phase(self):
        self.encoder.eval()
        self.matcher.eval()

        pred_logits = []
        gt_one_hots = []
        gt_argmaxs  = []
        pluss       = []
        loss = 0

        with torch.no_grad():
            for vf, vn, gt in zip(self.validsets_from, self.validsets_next, self.valid_gts):
                vf.training=False
                vn.training=False

                dataloader_from = DataLoader(vf, batch_size=len(vf), shuffle=False,  collate_fn=vf.custom_collation_fn)
                dataloader_next = DataLoader(vn, batch_size=len(vn), shuffle=False,  collate_fn=vn.custom_collation_fn)

                predicted_matrix_logits, gt_one_hot, gt_argmax, plus = self.process(dataloader_from, vf, dataloader_next, vn, gt)
                loss += self.crossentropy(predicted_matrix_logits, gt_argmax) #+ self.matchingloss(predicted_matrix_logits)

                pred_logits.append(predicted_matrix_logits.detach().cpu())
                gt_one_hots.append(gt_one_hot.detach().cpu())
                gt_argmaxs.append(gt_argmax.detach().cpu())
                pluss.append(plus)

        torch.cuda.empty_cache()

        m  = computeMetricsList(pred_logits, gt_one_hots)
        ma = computeMetricsAlessandroList(pred_logits, gt_one_hots)
        
        #self.scheduler.step()

        with open(os.path.join(self.logpath_valid, f"epoch_{str(self.epoch).zfill(3)}.pickle"), 'wb') as handle:
            pickle.dump({
            'predicted_matrix': pred_logits,
            'gt': gt_one_hots,
            }, handle)


        torch.cuda.empty_cache()

        msg = "val"
        if self.prev_val_loss>loss.item():

            logtxt = f"GREAT loss! \033[92m{loss:6.4f}\033[0m > {self.prev_val_loss:6.4f}"
            print(logtxt)
            logtxt = f"GREAT loss! {loss:6.4f} > {self.prev_val_loss:6.4f}"
            self.logfile.write(logtxt + "\n")

            self.prev_val_loss = loss.item()
            
            with open(os.path.join(self.logpath, f"best_valid_loss.pickle"), 'wb') as handle:
                pickle.dump({
                'predicted_matrix': pred_logits,
                'gt': gt_one_hots,
                }, handle)
            with open(self.logfilepath_valid_best_loss, "w") as f:
                f.write(f"epoch: {self.epoch}\nacc: {m['acc']}\nloss: {loss.item()}\nprecision: {ma['prec']}\nrecall: {ma['rec']}\nf1 {ma['f1']}")

            if self.mode == "finaltrain":
                torch.save({
                        'epoch': self.epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'matcher_state_dict': self.matcher.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'f1': ma["f1"],
                        'acc': m["acc"],
                        }, os.path.join(self.logpath, "best_model_loss.pt"))
        

        if self.prev_val_f1<ma["f1"]:
            logtxt = f"GREAT f1! \033[92m{ma['f1']:6.4f}\033[0m > {self.prev_val_f1:6.4f}"
            print(logtxt)
            logtxt = f"GREAT f1! {ma['f1']:6.4f} > {self.prev_val_f1:6.4f}"
            self.logfile.write(logtxt + "\n")
            self.logfile_valid.write(logtxt + "\n")
            self.prev_val_f1 = ma["f1"]

            if self.mode == "finaltrain":
                torch.save({
                        'epoch': self.epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'matcher_state_dict': self.matcher.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'f1': ma["f1"],
                        'acc': m["acc"],
                        }, os.path.join(self.logpath, "best_model_f1.pt"))


        
        logtxt = f'epoch {self.epoch:3d} - {msg} loss: {loss.item(): 8.6f} - accuracy {m["acc"]:5.3f} best: {self.prev_val_acc:5.3f} ( {m["cor"]:3d} / {m["tot"]:3d} )  (tn {m["tn"]:3d} / {m["fn"]:3d} | {m["gtz"]:3d})  - precision {ma["prec"]:4.2f}  recall {ma["rec"]:4.2f}  f1 {ma["f1"]:4.2f}'
        print(logtxt)
        self.logfile.write(logtxt + "\n")
        self.logfile_valid.write(logtxt + "\n")

        return loss, m["cor"], m["tot"]

    def run(self):
        for epoch in range(self.max_epochs_num):
            self.epoch = epoch

            trainacc = self.do_one_train_epoch()
            torch.cuda.empty_cache()

            if True:#epoch>40:
                loss, correct, tot = self.do_one_valid_phase()
                torch.cuda.empty_cache()

            self.logfile.flush()
            self.logfile_train.flush()
            self.logfile_valid.flush()

    def run_test(self):
        self.epoch = -1

        self.encoder.eval()
        self.matcher.eval()

        with torch.no_grad():
            predicted_matrix_logits, gt_one_hot, gt_argmax, plus = self.process(self.testloader_from, self.test_from, self.testloader_next, self.test_next, self.test_gt)
            loss = self.crossentropy(predicted_matrix_logits, gt_argmax) + self.matchingloss(predicted_matrix_logits)

        torch.cuda.empty_cache()
        costmatrix = predicted_matrix_logits.clone()

        cont = 0

        gt_argmax = gt_argmax.detach().cpu().numpy()

        probs = torch.nn.functional.softmax(predicted_matrix_logits, dim=-1).detach().cpu().numpy()
        copynorm = np.copy(probs)

        probs = probs.max(axis=1)

        hungpreds = np.zeros_like(copynorm)
        hungpreds[:, 0] = 1
        

        centers_from = self.test_from.centers[plus["from_idxs_b"]]
        centers_next = self.test_next.centers[plus["next_idxs_b"]]

        dist = np.linalg.norm(centers_next - centers_from[:, None], axis=-1).T

        mask = np.zeros_like(copynorm).astype(float)
        mask[:, 1:] = dist
        copynorm[mask>0.05] = -1*np.inf


        while cont < predicted_matrix_logits.shape[0]:
            if copynorm.argmax() < 0:
                print("breaking with argmax < 0")
                break
            rn, cf = np.unravel_index(copynorm.argmax(), copynorm.shape)
            hungpreds[rn, 0] = 0
            hungpreds[rn, cf] = 1
            copynorm[rn, :] = -1 * np.inf
            if cf>0:
                copynorm[:, cf] = -1 * np.inf
            cont += 1

        hungpreds = hungpreds.argmax(axis=1)

        m = computeMetricsPret2(hungpreds, gt_argmax)
        #print("f1p", m["f1p"], "f1n", m["f1n"], "f1", m["f1"])
        #print(m)

        mask = getMetricMask(hungpreds, gt_argmax)

        import pickle
        with open("instsegm_gcnconv_2_minxmaxx.pickle", "wb") as handle:
            pickle.dump({"gt_keys":gt_argmax, "hungpreds":hungpreds, "mask": mask, "plus":plus}, handle)

        return m["f1p"], m["f1n"], m["f1"]



@cli.command()
def main(
    mode: str = typer.Option(
        ...,
        "--mode",
        help="mode of execution (kfold_crossvalidation, finaltrain, test, testinst)",
    ),
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
    f1ps, f1ns, f1s = [], [], []

    print("mode: ", mode)

    for seedidx, _ in enumerate(seeds):
        split = CrossValidationSetup(-1, seedidx, mode, datapath, iou)
        f1p, f1n, f1 = split.run_test()
        f1ps.append(f1p); f1ns.append(f1n); f1s.append(f1)
        
    f1ps, f1ns, f1s = np.array(f1ps), np.array(f1ns), np.array(f1s)
    print("#####################")
    print("final results")
    print(f"f1p {f1ps.mean()*100:4.1f} +- {f1ps.std()*100:4.1f}")
    print(f"f1n {f1ns.mean()*100:4.1f} +- {f1ns.std()*100:4.1f}")
    print(f"f1  {f1s.mean()*100:4.1f} +- {f1s.std()*100:4.1f}")
    print("#####################")

if __name__ == "__main__":
    cli()

#if mode == "kfold_crossvalidation":
#    for foldidx in range(len(fold_splits)):
#        for seedidx, _ in enumerate(seeds):
#            print(f"| ***** SPLIT {foldidx:2d}  - seedidx {seedidx:6d} *****|")
#            split = CrossValidationSetup(foldidx, seedidx, mode)
#            split.run() 
#
#
#elif mode=="finaltrain":
#    for seedidx, _ in enumerate(seeds):
#        split = CrossValidationSetup(-1, seedidx, mode)
#        split.run()
#
#elif mode in ["test", "testinst"]:
#
#    f1ps, f1ns, f1s = [], [], []
#
#    for seedidx, _ in enumerate(seeds):
#        split = CrossValidationSetup(-1, seedidx, mode)
#        f1p, f1n, f1 = split.run_test()
#        f1ps.append(f1p)
#        f1ns.append(f1n)
#        f1s.append(f1)
#    f1ps = np.array(f1ps)
#    f1ns = np.array(f1ns)
#    f1s  = np.array(f1s)
#    print()
#    print("final results")
#    print("f1p ", f1ps.mean(), "+-", f1ps.std())
#    print("f1n ", f1ns.mean(), "+-", f1ns.std())
#    print("f1  ",  f1s.mean(), "+-",  f1s.std())