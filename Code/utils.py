import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml




def add_weight_decayAll(model, settings):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name[:3] == "bns":
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': settings["weight_decay"]},
        {'params': no_decay, 'weight_decay': 0.}]

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

        

#Early Topping on both loss and acc
class EarlyStoppingLA():
    def __init__(self, patience, path):
        self.patience = patience
        self.path = path
        self.counter = 0

        self.val_loss_min = np.Inf
        self.val_acc_max = -1.0
            
    def __call__(self, val_loss, val_acc, model):
    #todo loss/acc option
        loss_worse = val_loss >= self.val_loss_min
        acc_worse = val_acc <= self.val_acc_max
        if loss_worse and acc_worse:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            if not loss_worse:
                self.val_loss_min = val_loss
            if not acc_worse:
                self.val_acc_max = val_acc
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self,  model):
        torch.save(model.state_dict(), str(self.path))


def collect_res_dat(p = Path("../results/test/")):
    outer_dfls = []
    for d in tqdm(list(p.glob("setting*"))):
        try:
            settings = yaml.safe_load((d/"settings.yml").open())
        except FileNotFoundError:
            print("FNF")
            continue
        del settings["statrep"]
        del settings["epochs"]
        
        #del settings["early_stopping"]
        dfls = []
        l = list(d.glob("stats_*.pkl"))
        if len(l) == 0:
            continue
        for sp in l:
            df = pd.read_pickle(str(sp))
            df["statrep"] = int(sp.parts[-1][6:-4])
            dfls.append(df)
        df = pd.concat(dfls)
        for k in settings:
            if type(settings[k])==dict or type(settings[k])==list:
                df[k]=str(settings[k])
            else:
                df[k]=settings[k]
                    # try:
            #     df[k]=str(settings[k])
            # except ValueError:
            #     df[k]=str(settings[k])
        outer_dfls.append(df)       
    return pd.concat(outer_dfls, ignore_index=True)









