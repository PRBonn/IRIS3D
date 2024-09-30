import torch
import numpy as np

def computeMetrics(pred_logits, gt):
    tpm, wpm, tn, fp, fn  = 0, 0, 0, 0, 0  ## true_positive_matching, wrong_positive_matching
    precision, recall, f1 = 0, 0, 0

    pred = torch.nn.functional.softmax(pred_logits, dim=-1)

    for r in range(pred.shape[0]):
        pred_match_idx = pred[r].argmax()
        gt_match_idx   = gt[r].argmax()

        if gt_match_idx==0:
            if pred_match_idx==0:
                tn += 1
            else:
                fp += 1
        else:
            if pred_match_idx==gt_match_idx:
                tpm += 1
            else:
                if pred_match_idx>0:
                    wpm += 1
                else:
                    fn += 1
    
    #tot = pred.shape[0]            ## aka
    tot = tpm + wpm + tn + fp + fn  ## just for fun

    accuracy = (tpm + tn) / tot
    
    if (tpm + fp + wpm) > 0:
        precision = tpm / (tpm + fp + wpm)

    if (tpm + fn) > 0:
        recall = tpm / (tpm + fn)
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"acc":accuracy, "prec":precision,
            "rec":recall, "f1":f1, "tot":tot,
            "tpm":tpm, "wpm":wpm,
            "tn":tn, "fp":fp, "fn":fn,
            "cor":tn+tpm, "z":tn+fn, "gtz":tn+fp}


def computeMetricsAlessandro(pred_logits, gt):
    tp, fp, fn = 0, 0, 0

    pred = torch.nn.functional.softmax(pred, dim=-1)

    for r in range(pred.shape[0]):
        pred_match_idx = pred[r].argmax()
        gt_match_idx   = gt[r].argmax()

        if pred_match_idx>0:
            if gt_match_idx==0:
                fp += 1
                continue
            if not pred_match_idx==gt_match_idx:
                fp += 1
                continue
            tp += 1

    for r in range(pred.shape[0]):
        pred_match_idx = pred[r].argmax()
        gt_match_idx   = gt[r].argmax()

        if gt_match_idx>0:
            if pred_match_idx==0:
                fn += 1
                continue
            if not pred_match_idx==gt_match_idx:
                fn += 1
                continue
 
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall    = 0 if (tp + fn) == 0 else tp / (tp + fn)
    f1        = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

    return {"prec":precision, "rec":recall, "f1":f1, "tp":tp, "fp":fp, "fn":fn}

def computeMetricsList(pred_list:list, gt_list:list) -> dict:
    tpm, wpm, tn, fp, fn  = 0, 0, 0, 0, 0  ## true_positive_matching, wrong_positive_matching
    precision, recall, f1 = 0, 0, 0

    assert len(pred_list) == len(gt_list)

    for pred, gt in zip(pred_list, gt_list):

        pred = torch.nn.functional.softmax(pred, dim=-1)

        for r in range(pred.shape[0]):
            pred_match_idx = pred[r].argmax()
            gt_match_idx   = gt[r].argmax()

            if gt_match_idx==0:
                if pred_match_idx==0:
                    tn += 1
                else:
                    fp += 1
            else:
                if pred_match_idx==gt_match_idx:
                    tpm += 1
                else:
                    if pred_match_idx>0:
                        wpm += 1
                    else:
                        fn += 1
    
    #tot = pred.shape[0]            ## aka
    tot = tpm + wpm + tn + fp + fn  ## just for fun

    accuracy = (tpm + tn) / tot
    
    if (tpm + fp + wpm) > 0:
        precision = tpm / (tpm + fp + wpm)

    if (tpm + fn) > 0:
        recall = tpm / (tpm + fn)
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"acc":accuracy, "prec":precision,
            "rec":recall, "f1":f1, "tot":tot,
            "tpm":tpm, "wpm":wpm,
            "tn":tn, "fp":fp, "fn":fn,
            "cor":tn+tpm, "z":tn+fn, "gtz":tn+fp}


def computeMetricsAlessandroList(pred_list:list, gt_list:list) -> dict:
    tp, fp, fn = 0, 0, 0

    assert len(pred_list) == len(gt_list)

    for pred, gt in zip(pred_list, gt_list):

        pred = torch.nn.functional.softmax(pred, dim=-1)

        for r in range(pred.shape[0]):
            pred_match_idx = pred[r].argmax()
            gt_match_idx   = gt[r].argmax()

            if pred_match_idx>0:
                if gt_match_idx==0:
                    fp += 1
                    continue
                if not pred_match_idx==gt_match_idx:
                    fp += 1
                    continue
                tp += 1

        for r in range(pred.shape[0]):
            pred_match_idx = pred[r].argmax()
            gt_match_idx   = gt[r].argmax()

            if gt_match_idx>0:
                if pred_match_idx==0:
                    fn += 1
                    continue
                if not pred_match_idx==gt_match_idx:
                    fn += 1
                    continue
 
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall    = 0 if (tp + fn) == 0 else tp / (tp + fn)
    f1        = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

    return {"prec":precision, "rec":recall, "f1":f1, "tp":tp, "fp":fp, "fn":fn}

def getMetricMask(predidx, gt):
    tpm, wpm, tn, fp, fn  = 0, 1, 2, 3, 4
    assert predidx.shape == gt.shape

    mask = np.zeros(predidx.shape[0], dtype=np.int32)
    for r in range(predidx.shape[0]):
        pred_match_idx = predidx[r]
        gt_match_idx   = gt[r]
        if gt_match_idx==0:
            if pred_match_idx==0:
                mask[r] = tn
            else:
                mask[r] = fp
        else:
            if pred_match_idx==gt_match_idx:
                mask[r] = tpm
            else:
                if pred_match_idx>0:
                    mask[r] = wpm
                else:
                    mask[r] = fn
    return mask


def computeMetricsPret(predidx, gt):
    tpm, wpm, tn, fp, fn  = 0, 0, 0, 0, 0  ## true_positive_matching, wrong_positive_matching
    precision, recall, f1 = 0, 0, 0

    #predidx = predidx + 1

    print("predidx", predidx.shape[0])
    print("gt     ", gt.shape[0])

    for r in range(predidx.shape[0]):
        pred_match_idx = predidx[r]
        gt_match_idx   = gt[r]


        if gt_match_idx==0:
            if pred_match_idx==0:
                tn += 1
            else:
                fp += 1
        else:
            if pred_match_idx==gt_match_idx:
                tpm += 1
            else:
                if pred_match_idx>0:
                    wpm += 1
                else:
                    fn += 1
    
    #tot = pred.shape[0]            ## aka
    tot = tpm + wpm + tn + fp + fn  ## just for fun

    accuracy = (tpm + tn) / tot
    
    if (tpm + fp + wpm) > 0:
        precision = tpm / (tpm + fp + wpm)

    if (tpm + fn) > 0:
        recall = tpm / (tpm + fn)
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"acc":accuracy, "prec":precision,
            "rec":recall, "f1":f1, "tot":tot,
            "tpm":tpm, "wpm":wpm,
            "tn":tn, "fp":fp, "fn":fn,
            "cor":tn+tpm, "z":tn+fn, "gtz":tn+fp}


def computeMetricsAlessandroPret(predidx, gt):
    tp, fp, fn = 0, 0, 0

    #predidx = predidx + 1

    for r in range(predidx.shape[0]):
        pred_match_idx = predidx[r]
        gt_match_idx   = gt[r]

        if pred_match_idx>0:
            if gt_match_idx==0:
                fp += 1
                continue
            if not pred_match_idx==gt_match_idx:
                fp += 1
                continue
            tp += 1

    for r in range(predidx.shape[0]):
        pred_match_idx = predidx[r].argmax()
        gt_match_idx   = gt[r].argmax()

        if gt_match_idx>0:
            if pred_match_idx==0:
                fn += 1
                continue
            if not pred_match_idx==gt_match_idx:
                fn += 1
                continue
 
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall    = 0 if (tp + fn) == 0 else tp / (tp + fn)
    f1        = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

    return {"prec":precision, "rec":recall, "f1":f1, "tp":tp, "fp":fp, "fn":fn}


def computeMetricsPret2(predidx, gt):
    tpm, wpm, tn, fp, fn  = 0, 0, 0, 0, 0  ## true_positive_matching, wrong_positive_matching
    precision, recall, f1 = 0, 0, 0

    for r in range(predidx.shape[0]):
        pred_match_idx = predidx[r]
        gt_match_idx   = gt[r]

        if gt_match_idx==0:
            if pred_match_idx==0:
                tn += 1
            else:
                fp += 1
        else:
            if pred_match_idx==gt_match_idx:
                tpm += 1
            else:
                if pred_match_idx>0:
                    wpm += 1
                else:
                    fn += 1
    
    tot = tpm + wpm + tn + fp + fn

    f1p = (2*tpm) / (2*tpm + wpm + fn + fp) if tpm + wpm + fn + fp > 0 else 0
    f1n = (2*tn) / (2*tn + fn + fp) if 2*tn + fn + fp > 0 else 0
    f1  = (f1p + f1n) / 2

    return {"f1p":f1p, "f1n":f1n, "f1":f1, "tot":tot,
            "tpm":tpm, "wpm":wpm,
            "tn":tn, "fp":fp, "fn":fn,
            "cor":tn+tpm, "z":tn+fn, "gtz":tn+fp}