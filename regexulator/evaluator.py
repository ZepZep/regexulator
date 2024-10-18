import re

import pandas as pd
import numpy as np

def eval_pattern(ds, pattern):
    try:
        re.compile(pattern)
    except re.error as e:
        return None
    res = pd.DataFrame.from_records(_eval_pattern_raw(ds, pattern))
    res = res.sum()
    return {
        f"{subset}_{k}": v
        for subset in ["match", "char"]
        for k, v in calculate_metrics(res, subset).items()
    } 



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
        "precision": tp/(tp+fp),
        "recall": tp/(tp+fn),
        "f1": (2*tp) / (2*tp + fp + fn),
    }


def _eval_pattern_raw(ds, pattern):
    res = []
    for ex in ds:
        text = ex["string"]
        gt = ex["match"]
        pred = get_anns(text, pattern)
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

def get_anns(text, pattern, group=0):
    anns = []
    for m in re.finditer(pattern, text):
        # print(m)
        ann = {}
        ann["start"], ann["end"] = m.span(group)
        anns.append(ann)

    return anns