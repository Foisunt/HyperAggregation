import torch
from torch_geometric.datasets import Planetoid, Amazon, Actor, WikipediaNetwork, HeterophilousGraphDataset,GNNBenchmarkDataset
from Zinc import ZINC # the pyg loader is iffy, you need to manually download it and modify the official Zinc loader accordingly, see https://github.com/pyg-team/pytorch_geometric/issues/1602

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from pathlib import Path
import torch_geometric.transforms as T
from torch_geometric.utils import contains_self_loops



# mask is boolean mask
# idx is a long tensor of idxs
class Dataset:
    def __init__(self, settings, dev):
        transform_ls = []
        if settings["ds_undir"]:
            transform_ls.append(T.ToUndirected())
        if settings["ds_self"]:
            transform_ls.append(T.AddSelfLoops())
        self.ds, self.splits, name, n_f, n_cls = self.load_ds(settings["dataset"], settings["setup"], transform_ls, dev)
        self.setup = settings["setup"] #trans/ind-real/ind-82
        assert self.setup in ["transd", "ind-real", "ind-82", "ind-gra"]
        self.n_feats = n_f
        self.n_cls = n_cls
        self.n2v = False
        if settings.get("cat_n2v", False):
            self.n2v = torch.load(Path("dataset") / name / "n2v_trans.pt")


    #loads the dataset and split
    #82 splits are loaded, need to be prcomputed once for consistency
    def load_ds(self, name, setup, transform_ls, dev):
        splits = None
        #print("t", transform)
        if setup != "ind-gra":
            transform = None
            if transform_ls != []:
                transform = T.Compose(transform_ls)

            if name in ["Cora", "CiteSeer", "PubMed"]:
                ds = Planetoid(root='dataset/'+name+"/", name=name, transform=transform)[0]
                splits = torch.load("dataset/"+name+"/own_23_splits.pt")
            elif name in ["Roman-empire", "Minesweeper"]:
                ds = HeterophilousGraphDataset(root='dataset/'+name+"/", name=name, transform=transform)[0]
                splits = {"train":ds.train_mask, "valid":ds.val_mask, "test":ds.test_mask}
            elif name in ["Photo", "Computers"]:
                ds = Amazon(root='dataset/'+name+"/", name=name, transform=transform)[0]
                splits = torch.load("dataset/"+name+"/own_23_splits.pt")
            elif name in ["Chameleon", "Squirrel"]:
                ds = WikipediaNetwork(root="dataset/"+name+"/", name = name, transform=transform)[0]
                splits = {"train":ds.train_mask, "valid":ds.val_mask, "test":ds.test_mask}
            elif name == "Actor":
                ds = Actor(root="dataset/Actor/", transform=transform)[0]
                splits = {"train":ds.train_mask, "valid":ds.val_mask, "test":ds.test_mask}
            elif name in ["Wisconsin", "Cornell", "Texas"]:
                ds = WebKB(root="dataset/"+name+"/", name = name, transform=transform)[0]
                splits = {"train":ds.train_mask, "valid":ds.val_mask, "test":ds.test_mask}
            elif name in ["Arxiv", "Products"]:
                name = "ogbn-"+name.lower()
                ds = PygNodePropPredDataset(name = name, transform=transform)
                sp = ds.get_idx_split()
                ds = ds[0]
                name = name.replace("-", "_")
                tr = torch.zeros(len(ds.x),1, dtype=torch.bool)
                tr[sp["train"]] = 1
                va = torch.zeros(len(ds.x),1, dtype=torch.bool)
                va[sp["valid"]] = 1
                te = torch.zeros(len(ds.x),1, dtype=torch.bool)
                te[sp["test"]] = 1
                
                splits = {"train":tr.expand(-1,3), "valid":va.expand(-1,3), "test":te.expand(-1,3)}
            else:
                raise NotImplementedError("Dataset '" + name+ "' is unkown")
            if setup == "ind-82":
                splits = torch.load("dataset/"+name+"/own_82_splits.pt")
            n_f, n_cls = ds.x.shape[1], int(ds.y.max())+1
        else:
            transform_ls.append(T.ToDevice(dev))
            transform = T.Compose(transform_ls)
            
            if name == "ZINC":
                train = ZINC("dataset/ZINC/", subset=True, split='train', transform=transform)
                val = ZINC("dataset/ZINC/", subset=True, split='val', transform=transform)
                test = ZINC("dataset/ZINC/", subset=True, split='test', transform=transform)
                ds = {"train":train, "val": val, "test":test}
                n_cls = 1 #1 task regression
                n_f = 21 # 1, but this gets one hot encoded by the model

            elif name in ["CIFAR10", "MNIST"]:
                train = GNNBenchmarkDataset("dataset/"+name+"/", name = name, split='train', transform=transform)
                val = GNNBenchmarkDataset("dataset/"+name+"/", name = name, split='val', transform=transform)
                test = GNNBenchmarkDataset("dataset/"+name+"/", name = name, split='test', transform=transform)
                ds = {"train":train, "val": val, "test":test}
                n_cls = 10
                n_f = 5 if name == "CIFAR10" else 3
            else:
                raise NotImplementedError("Dataset '" + name+ "' is unkown")
                
        return ds, splits, name, n_f, n_cls
        
    def set_rep(self, rep, dev):
        if self.setup != "ind-gra":
            catx = torch.zeros(self.ds.x.shape[0],0)
            if self.n2v != False:
                catx = torch.cat((catx, self.n2v[rep]), 1)
            
            x = torch.cat((self.ds.x, catx), 1)#.to(dev)
            self.n_feats = x.shape[1]
        if self.setup == "transd":
            d = Data(x=x, y=self.ds.y.squeeze(), edge_index=self.ds.edge_index, train_mask=self.splits["train"][:,rep], val_mask=self.splits["valid"][:,rep], test_mask=self.splits["test"][:,rep], dist=0).to(dev)
            self.ds_train = d
            self.ds_val = d
            self.ds_test = d
        elif self.setup == "ind-real":
            ixs = self.splits["train"][:,rep].argwhere()
            self.ds_train = Data(x=x, y=self.ds.y.squeeze(), edge_index=self.ds.edge_index, train_mask=self.splits["train"][:,rep], dist=0).subgraph(ixs).to(dev)
            ixs = (self.splits["train"][:,rep]|self.splits["valid"][:,rep]).argwhere()
            self.ds_val = Data(x=x, y=self.ds.y.squeeze(), edge_index=self.ds.edge_index, val_mask=self.splits["valid"][:,rep], dist=0).subgraph(ixs).to(dev)
            self.ds_test = Data(x=x, y=self.ds.y.squeeze(), edge_index=self.ds.edge_index, test_mask=self.splits["test"][:,rep], dist=0).to(dev)
        elif self.setup == "ind-82":
            ixs = self.splits["structure"][:,rep]|self.splits["train"][:,rep].argwhere()
            self.ds_train = Data(x=x, y=self.ds.y.squeeze(), edge_index=self.ds.edge_index, train_mask=self.splits["train"][:,rep], dist=0).subgraph(ixs).to(dev)
            ixs = (self.splits["structure"][:,rep]|self.splits["train"][:,rep]|self.splits["valid"][:,rep]).argwhere()
            self.ds_val = Data(x=x, y=self.ds.y.squeeze(), edge_index=self.ds.edge_index, val_mask=self.splits["valid"][:,rep], dist=0).subgraph(ixs).to(dev)
            self.ds_test = Data(x=x, y=self.ds.y.squeeze(), edge_index=self.ds.edge_index, test_mask=self.splits["test"][:,rep], dist=0).to(dev)
        elif self.setup == "ind-gra":
            self.ds_train = self.ds["train"]
            self.ds_val = self.ds["val"]
            self.ds_test = self.ds["test"]
            
        else:
            raise NotImplementedError("setup "+self.setup+" not supported")
        #prepare train, val, test data and masks


    def get_train(self):
        return self.ds_train#, self.ds_train.train_mask
    def get_val(self):
        return self.ds_val#, self.ds_val.val_mask
    def get_test(self):
        return self.ds_test#, self.ds_test.test_mask

