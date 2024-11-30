import regex
import time

import pandas as pd
import numpy as np

def eval_pattern(ds, pattern, cfg):
    if not pattern:
        return None
    try:
        regex.compile(pattern)
    except regex.error as e:
        return None
    res = pd.DataFrame.from_records(_eval_pattern_raw(ds, pattern, cfg))
    res = res.sum()
    d = {
        f"{subset}_{k}": v
        for subset in ["match", "char"]
        for k, v in calculate_metrics(res, subset).items()
    } 
    d["mean_f1"] = (d["char_f1"] + d["match_f1"])/2
    return d



def calculate_metrics(res, subset):
    tp = res[f"{subset}_tp"]
    fn = res[f"{subset}_fn"]
    fp = res[f"{subset}_fp"]
    
    return {
        "gt": res[f"{subset}_gt"],
        "pred": res[f"{subset}_pred"],
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "precision": 0 if tp + fp == 0 else tp/(tp+fp),
        "recall": 0 if tp + fn == 0 else tp/(tp+fn),
        "f1": 0 if 2*tp + fp + fn == 0 else (2*tp) / (2*tp + fp + fn),
    }


def _eval_pattern_raw(ds, pattern, cfg):
    res = []
    for ex in ds:
        text = ex["string"]
        gt = ex["match"]
        pred = get_anns(text, pattern, cfg)
        # print(len(pred))
        eres = {
            "match_gt": len(gt),
            "match_pred": len(pred),
            "match_tp": 0,
            "match_fp": 0,
            "match_fn": 0,
            "char_gt": sum(ann["end"] - ann["start"] for ann in gt), 
            "char_pred": sum(ann["end"] - ann["start"] for ann in pred), 
        }
        matches = np.zeros(len(text))
        for ann in gt:
            matches[ann["start"]:ann["end"]] += 1
            
        for pr in pred:
            matches[pr["start"]:pr["end"]] += 2
            if pr in gt:
                eres["match_tp"] += 1
            else:
                eres["match_fp"] += 1
        eres["match_fn"] = eres["match_gt"] - eres["match_tp"]

        eres["char_fn"] = (matches == 1).sum()
        eres["char_fp"] = (matches == 2).sum()
        eres["char_tp"] = (matches == 3).sum()

        res.append(eres)
    return res

def get_eval_examples(ds, pattern, cfg):
    res = []
    for ex in ds:
        text = ex["string"]
        gt = ex["match"]
        preds = get_anns(text, pattern, cfg)

        # types = ["exact", "partial_pred", "partial_gt", "miss_pred", "miss_gt"]
        
        gt_state = [0 for _ in gt]
        for pred in preds:
            pl, pr = pred["start"], pred["end"]
            found = False
            for i, ann in enumerate(gt):
                gl, gr = ann["start"], ann["end"]
                if overlaps(pl, pr, gl, gr):
                    found = True
                    if pl == gl and pr == gr:
                        res.append({"type": "exact", "start": pl, "end": pr, "string": text})
                        gt_state[i] |= 2
                    else:
                        res.append({"type": "partial_pred", "start": pl, "end": pr,"string": text})
                        if not gt_state[i]:
                            res.append({"type": "partial_gt", "start": gl, "end": gr,"string": text})
                            gt_state[i] |= 1
            if not found:
                res.append({"type": "miss_pred", "start": pl, "end": pr,"string": text})
        for ann, state in zip(gt, gt_state):
            if not state:
                gl, gr = ann["start"], ann["end"]
                res.append({"type": "miss_gt", "start": gl, "end": gr,"string": text})
        
    return pd.DataFrame.from_records(res)
           
                
def overlaps(l1, r1, l2, r2): 
    return max(l1, l2) <= min(r1, r2)

def get_anns(text, pattern, cfg, group=0):
    timeout_time = time.time() + cfg.match_timeout
    err = None
    anns = []
    try:
        for m in regex.finditer(pattern, text, timeout=cfg.match_timeout):
            # print(m)
            ann = {}
            ann["start"], ann["end"] = m.span(group)
            anns.append(ann)
            if len(anns) >= cfg.max_doc_anns:
                err = "too many annotations"
                break
            if time.time() >= timeout_time:
                err = "timeout"
                break
    except TimeoutError:
        err = "timeout"
        pass

    return anns