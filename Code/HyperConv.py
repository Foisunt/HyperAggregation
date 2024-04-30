import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing

from typing import Optional
from torch import Tensor

from torch_geometric.utils import dropout_edge
from torch_geometric.utils import sort_edge_index

from utils import add_weight_decayAll

import warnings


act_map = {"gelu":nn.GELU()}

#apply layernorm, activation, dropout
class Trans(nn.Module):
    def __init__(self, drop, norm_shape, act, aff):
        super().__init__()
        self.d = Dropout(drop)
        self.a = act_map[act]
        self.n = LayerNorm(norm_shape, elementwise_affine=aff)
    def forward(self, x):
        return self.d(self.a(self.n(x)))

#mlp base
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, drop, act, aff, hhid = None):
        super().__init__()
        if hhid == None:
            hhid = in_channels 
        self.l1 = nn.Linear(in_channels, hhid)
        self.l2 = nn.Linear(hhid, out_channels)
        self.tr = Trans(drop, hhid, act, aff)
    def forward(self, x):
        return self.l2(self.tr(self.l1(x)))

#aggregation using hypernetwork
class HyperAggregation(Aggregation):
    def __init__(self, hid_channels:int, mix_channels: int, dropout: float, dropout_mix: float, aggr_mean: bool, ln_aff:bool, act:str, tr_in: bool, **kwargs):
        #print("hyper init", hid_channels, mix_channels)
        super().__init__()
        self.hid_channels = hid_channels
        self.mix_channels = mix_channels
        self.mlp = MLP(hid_channels, mix_channels, dropout, act, ln_aff)
        self.drop_mix = nn.Dropout(dropout_mix)
        self.aggr_mean = aggr_mean
        if tr_in:
            self.tr_in_mlp = Trans(dropout, hid_channels, act, ln_aff)
            self.tr_in_mix = Trans(dropout_mix, hid_channels, act, ln_aff)
        else:
            self.tr_in_mlp = nn.Identity()
            self.tr_in_mix = nn.Identity()
        # if tr_mix:
        #     self.tr_mix = Trans(dropout_mix, mix_channels, act, ln_aff)
        # else:
        self.act_mix = act_map[act]

    def forward(self, x: Tensor, index: Optional[Tensor] = None, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None, dim: int = -2, max_num_elements: Optional[int] = None,) -> Tensor:
        
        if len(index) == 0:
            warnings.warn("empty batch in hyper aggregation, skipped aggregating")
            return x

        
        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim, max_num_elements=max_num_elements)
        W1 = self.mlp(self.tr_in_mlp(x))
        
        act = torch.matmul(W1.transpose(-2,-1), self.tr_in_mix(x))
        tmp = torch.matmul(W1, self.act_mix(act))
        
        if self.aggr_mean:
            return tmp.mean(1)
       
        return tmp[:,0,:]


class HyperConv(MessagePassing):

#    def __init__(self, in_c: int, hid_c: int, out_c: int, mix_c: int, dropout: float = 0.0, dropout_mix: float = 0.0, aggr_mean: bool = True, ln_aff:bool = False, tr_mid:bool = False, **kwargs):
    def __init__(self, in_channels: int, hid_c: int, out_channels: int, mix_c: int, dropout: float, drop_mix: float, aggr_mean: bool, ln_aff:bool, act:str, tr_mid:bool, tr_agg_in:bool, skip_mix:bool, **kwargs):
        kwargs.setdefault('aggr', HyperAggregation(hid_c, mix_c, dropout, drop_mix, aggr_mean, ln_aff, act, tr_agg_in))
        super().__init__(node_dim=0, **kwargs)

        self.skip_mix = skip_mix
        self.l1 = Linear(in_channels, hid_c)
        if skip_mix:
            self.l2 = Linear(2*hid_c, out_channels)
        else:
            self.l2 = Linear(hid_c, out_channels)

        self.tr_mid = nn.Identity()
        if tr_mid:
            if skip_mix:
                self.tr_mid = Trans(dropout, 2*hid_c, act, ln_aff)
            else:
                self.tr_mid = Trans(dropout, hid_c, act, ln_aff)


    #forwardsplit
    def forward_n(self, x: Tensor, edge_index: Tensor):
        x = self.l1(x)
        x = self.propagate(edge_index, x=x)
        x = self.l2(self.tr_mid(x))
        return x
    def forward_s(self, x: Tensor, edge_index: Tensor):
        x = self.l1(x)
        y = self.propagate(edge_index, x=x)
        x = self.l2(self.tr_mid(torch.cat((x,y),-1)))
        return x

    def forward(self, x: Tensor, edge_index: Tensor):
        if self.skip_mix:
            return self.forward_s(x, edge_index)
        else:
            return self.forward_n(x, edge_index)

    def act(self, x, edge_index):
        x = self.l1(x)
        y, w, a = self.propagate(edge_index, x=x, ret_act = True)
        x = self.l2(self.tr_mid(torch.cat((x,y),-1)))
        return x, w, a


    def __repr__(self) -> str:
        #return (f'{self.__class__.__name__}({self.in_channels},{self.out_channels})')
        return (f'{self.__class__.__name__}')



class HyperConvMixer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, settings):
        super().__init__()
        tmp = [int(x) for x in settings["model"].split("_")[1:]] #HCM_nl_hid_mix
        self.num_hc = tmp[0]
        assert self.num_hc >= 1
        self.h_in = in_channels
        self.h_out = out_channels
        self.h_hid = tmp[1]
        self.h_mix = tmp[2]

        self.drop_edge_p = settings["drop_adj"]
        self.drop_model_p = settings["drop_model"]
        self.drop_mix_p = settings["drop_mix"]
        self.drop_in = Dropout(settings["drop_input"])
        self.aggr_mean = settings["aggr_mean"]
        self.tr_agg_in = settings["tr_agg_in"]
        #self.tr_agg_mix = settings["tr_agg_mix"] #unused -> del
        self.tr_mid = settings["tr_mid"]
        self.type = settings["model_type"]
        self.ln_aff = settings["ln_aff"]
        self.act = settings["act"]
        self.skip_mix = settings["skip_mix"]

        self.hc_args = {"dropout":self.drop_model_p, "drop_mix":self.drop_mix_p, "aggr_mean":self.aggr_mean, "ln_aff":self.ln_aff, "act":self.act, "tr_mid":self.tr_mid, 
                        "tr_agg_in":self.tr_agg_in, "skip_mix":self.skip_mix}

        #norm and dropout x
        if settings["norm_in"]:
            self.in_ln = LayerNorm(in_channels, elementwise_affine=self.ln_aff)
            self.in_x = lambda x: self.drop_in(self.in_ln(x))
        else:
            self.in_x = self.drop_in
        #edge drop and sort
        self.in_e = lambda e: sort_edge_index(dropout_edge(e, p = self.drop_edge_p, force_undirected=True)[0], sort_by_row=False)

        
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()


        self.loss_fn = F.nll_loss
        self.score_fn = lambda preds,lbls:(preds.argmax(1) == lbls).sum()/len(lbls)
        if settings["dataset"] == "ZINC":
            self.loss_fn = F.l1_loss
            self.score_fn = lambda x,y:0
            self.h_out = self.h_hid
            self.lin_init()
            self.h_out = out_channels
            self.fwd = self.fwd_zinc
            self.last = Linear(self.h_hid, self.h_out)
        elif settings["dataset"] in ["CIFAR10", "MNIST"]:
            self.h_out = self.h_hid
            self.lin_init()
            self.h_out = out_channels
            self.fwd = self.fwd_img
            self.last = Linear(self.h_hid, self.h_out)
        elif self.type=="res":
            self.res_init()
            self.fwd = self.fwd_res
        elif self.type=="lin":
            self.lin_init()
            self.fwd = self.fwd_lin
        else:
            raise NotImplementedError("type " + self.type)
 


    def res_init(self):
        self.layers.append(Linear(self.h_in, self.h_hid))
        for _ in range(self.num_hc):
            self.norms.append(Trans(self.drop_model_p, self.h_hid, self.act, self.ln_aff))
            self.layers.append(HyperConv(self.h_hid, self.h_hid, self.h_hid, self.h_mix, **self.hc_args))
        self.layers.append(Linear(self.h_hid, self.h_out))

    def lin_init(self):
        if self.num_hc == 1:
            self.layers.append(HyperConv(self.h_in, self.h_hid, self.h_out, self.h_mix, **self.hc_args))
        else:
            self.layers.append(HyperConv(self.h_in, self.h_hid, self.h_hid, self.h_mix, **self.hc_args))
            for _ in range(self.num_hc-2):
                self.norms.append(Trans(self.drop_model_p, self.h_hid, self.act, self.ln_aff))
                self.layers.append(HyperConv(self.h_hid, self.h_hid, self.h_hid, self.h_mix, **self.hc_args))
            self.norms.append(Trans(self.drop_model_p, self.h_hid, self.act, self.ln_aff))
            self.layers.append(HyperConv(self.h_hid, self.h_hid, self.h_out, self.h_mix, **self.hc_args))
    
    def fwd_res(self, data):
        x, e = self.in_x(data.x), self.in_e(data.edge_index)
        x = self.layers[0](x)
        for i in range(self.num_hc):
            x = x + self.layers[i+1](self.norms[i](x), e)
        return F.log_softmax(self.layers[-1](x), dim=1)
       
    def fwd_lin(self, data):
        x, e = self.in_x(data.x), self.in_e(data.edge_index)
        x = self.layers[0](x, e)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x), e)
        return F.log_softmax(x, dim=1)

#fwd method for img graph cls tasks (graphs with x and pos features)
    def fwd_img(self, data):
        x, e = self.in_x(torch.cat([data.x, data.pos], 1)), self.in_e(data.edge_index)
        x = self.layers[0](x, e)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x), e)
        x = x.mean(0) #mean pooling
        return F.log_softmax(self.last(x), dim=0)

    #fwd methods for ZINC (graph regression)
    def fwd_zinc(self, data):
        x, e = self.in_x(F.one_hot(data.x.squeeze(), num_classes=21).float()), self.in_e(data.edge_index)
        x = self.layers[0](x, e)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x), e)
        x = x.mean(0) #mean pooling
        return self.last(x).squeeze()


    def embed_res(self, data):
        x, e = self.in_x(data.x), self.in_e(data.edge_index)
        x = self.layers[0](x)
        for i in range(self.num_hc):
            x = x + self.layers[i+1](self.norms[i](x), e)
        return x
    def embed_lin(self, data):
        x, e = self.in_x(data.x), self.in_e(data.edge_index)
        x = self.layers[0](x, e)
        for i in range(self.num_hc-2):
            x = self.layers[i+1](self.norms[i](x), e)
        return x
    def embed_img(self, data):
        x, e = self.in_x(torch.cat([data.x, data.pos], 1)), self.in_e(data.edge_index)
        x = self.layers[0](x, e)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x), e)
        x = x.mean(0) #mean pooling
        return x
    def embed_zinc(self, data):
        x, e = self.in_x(F.one_hot(data.x.squeeze(), num_classes=21).float()), self.in_e(data.edge_index)
        x = self.layers[0](x, e)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x), e)
        x = x.mean(0) #mean pooling
        return x

    #get hypernetwork activations for vis
    def act_lin(self, data):
        ws = []
        acts = []
        x, e = self.in_x(data.x), self.in_e(data.edge_index)
        x, w, a = self.layers[0].act(x, e)
        ws.append(w)
        acts.append(a)
        for i in range(self.num_hc-2):
            x, w, a = self.layers[i+1].act(self.norms[i](x), e)
            ws.append(w)
            acts.append(a)
        return x, acts, ws

    
    @property
    def device(self):
        return next(self.parameters()).device
   
    def forward(self, data):
        return self.fwd(data)

    def create_wd_groups(self, settings):
        return add_weight_decayAll(self, settings)

    def loss(self, p, l):
        return self.loss_fn(p,l)
    def score(self, p, l):
        return self.score_fn(p,l)




