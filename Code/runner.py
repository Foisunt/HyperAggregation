#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
import torch
from torch import multiprocessing
import sys
import pandas as pd

from data import Dataset
from models import get_model
from train import train_eval_model
from combine_results import combine_results

import os


def main(argv):
    print("argv", len(argv), argv)
    if len(argv) == 2:
        paths2do = find_pars_yaml(argv, 1)
        num_proc = 1
        dev_ls = [0]
    else: 
        paths2do = find_pars_yaml(argv, 3)
        num_proc = int(argv[1])
        dev_ls = [int(x) for x in argv[2].split(".")]
    print("#"*30)
    print("will run "+str(len(paths2do))+" experiments:")
    print(paths2do)
    print("#"*30)
    print("with", num_proc, "process on devices", dev_ls)
    print("#"*30)
    multiprocessing.set_start_method("forkserver")
    with multiprocessing.Pool(num_proc) as p:
        for count, path in enumerate(paths2do):
            print((count+1), "of", len(paths2do), "doing", path)
            yml = yaml.safe_load(path.open())
            print(yml)
            print("#"*30)
            name = path.parts[-1].split(".")[0]
            save_path = Path("../results/"+name+"/")
            run1exp(save_path, yml, p, dev_ls)
            yml["alldone"] = True
            yaml.dump(yml, path.open(mode="w"))
            combine_results(name)


def find_pars_yaml(argv, s):
    paths = []
    ret = []
    for x in range(s, len(argv)):
        paths.extend(Path("../Experiments/").glob(argv[x]))
    for p in paths:
        yml = yaml.safe_load(p.open())
        if yml.get("alldone"):
            continue
        else:
            ret.append(p)
    return ret

# takes one experiments, splits it into the cross product of its settings and starts them
def run1exp(save_path, update_dict, p, dev_ls):
    settings_dict = yaml.safe_load(Path("defaults.yml").open())
    settings_dict.update(update_dict)
    setting_ls = list(settings_dict.items())
    lens = [len(x[1]) for x in setting_ls]
    keys = [x[0] for x in setting_ls]
    n = np.prod(np.array(lens))
    it = list(zip(range(n), [lens]*n, [keys]*n, [settings_dict]*n, [save_path]*n, [dev_ls]*n))
    tmp = list(tqdm(p.imap(f, it), total=len(it)))

        
#glue method to use imap
def f(args):
    i, lens, keys, settings_dict, save_path, dev_ls = args
    conf = num2conf(i, lens)
    current_dict = {keys[i]:settings_dict[keys[i]][conf[i]] for i in range(len(conf))}
    train1setting(current_dict, save_path/("setting_"+str(i)), dev_ls)

def num2conf(num, lens):
    left = num
    res = [0]*len(lens)
    for ix in range(len(lens)-1, -1, -1):
        res[ix] = left % lens[ix]
        left = int(left/lens[ix])
    return res

def train1setting(settings, save_path, dev_ls):
    save_path.mkdir(parents=True, exist_ok=True)
    yaml.dump(settings, (save_path/"settings.yml").open(mode="w"))

    if type(settings["statrep"]) != str:
        reps = range(settings["statrep"])
    else:
        reps = [int(x) for x in settings["statrep"].split("x") if len(x)>0]

    skip = True #get_data does preprocessing, next 8 lines improve restarting speed of canceld/crashed experiments
    for rep in reps:
        p = save_path / ("stats_"+str(rep)+".pkl")
        if not p.exists():
            skip = False
            break
    if skip:
        #print("skip", settings["dataset"])
        return None

    if settings.get("max_split_size_mb", False)!=False:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:"+str(settings["max_split_size_mb"])

    pr_nr = int(multiprocessing.current_process().name.split("-")[1])
    device = torch.device("cuda:"+str(dev_ls[pr_nr%len(dev_ls)]))# if torch.cuda.is_available() else "cpu")
    settings["device"]=device
    ds = Dataset(settings, device)

    for rep in reps:
        p = save_path / ("stats_"+str(rep)+".pkl")
        p_tr = save_path / ("trainhist_"+str(rep)+".ls")
        if p.exists():
            continue
        ds.set_rep(rep, settings["device"])
        torch.manual_seed(rep)
        model = get_model(settings, ds.n_feats, ds.n_cls).to(settings["device"])
        model_path = save_path / ("model_"+str(rep)+".pt")
        train_stats, eval_df = train_eval_model(model, ds, settings, model_path)
        eval_df = pd.DataFrame(data=eval_df, index = [rep])
        
        eval_df.to_pickle(p)
        torch.save(train_stats, p_tr)

if __name__ == "__main__":
    main(sys.argv)


