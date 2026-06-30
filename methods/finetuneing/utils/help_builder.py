import torch.nn as nn
import torch


from finetuneing.models.loss import HuberLoss, CELoss, BCELoss, BazLoss, NLLLoss

from finetuneing.utils.pooler import Attentive, AVGPool
import pandas as pd

from finetuneing.datasets.SFTData import get_id_type
import time

def get_loss(downstream_task, det_weight, p_weight, s_weight, distribution_name='GaussianMixture'):
    if downstream_task == "fmp" or downstream_task == "cls":
        return CELoss()
    elif downstream_task == "dpk":
        return BCELoss([[det_weight], [p_weight], [s_weight]])
    elif downstream_task == "baz":
        return BazLoss()
    elif downstream_task == "emg_z" or downstream_task == 'dis_z':
        return NLLLoss(distribution_name)
    else:
        return HuberLoss()
    


def get_labels_tasks(downstream_task):
    """
    downstream task : str
        "dis": seismic epicenter distance regression
        "dpk": seismic detection and picking
        "mag_full": seismic magnitude regression using full waveforms
        "mag_P_only": seismic magnitude regression using P-wave only
        "baz": seismic back-azimuth regression
        "fmp": seismic first motion polarity classification
        "dep": seismic depth regression
        "cls": seismic event classification

    label : list
        List of labels for the downstream task
        "dis": epicenter distance label
        "det": detection label
        "ppk": P-wave picking label
        "spk": S-wave picking label
        "mag_full": seismic magnitude regression using full waveforms label
        "mag_P_only": seismic magnitude regression using P-wave only label
        "baz": back-azimuth label
        "fmp": first motion polarity label
        "dep": depth label
        "cls": event classification label
    
    return : label_list, task_list
    """

    if downstream_task == "dis":
        return ["dis"], ["dis"]
    elif downstream_task == "dis_z":
        return ["dis_z"], ["dis_z"]
    elif downstream_task == "dpk":
        return [["det", "ppk", "spk"]], ["det", "ppk", "spk"]
    elif downstream_task == "emg":
        return ["emg"], ["emg"]
    elif downstream_task == "emg_z":
        return ["emg_z"], ["emg_z"]
    elif downstream_task == "baz":
        return ["baz"], ["baz"]
    elif downstream_task == "fmp":
        return ["fmp"], ["fmp"]
    elif downstream_task == "dep":
        return ["dep"], ["dep"]
    elif downstream_task == "cls":
        return ["cls"], ["cls"]
    else:
        raise NotImplementedError(f"Downstream task {downstream_task} not implemented")