import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from utils import add_weight_decayAll


from typing import Optional
from torch import Tensor



act_map = {"gelu":nn.GELU()}

#apply layernorm, activation, dropout
class Trans(nn.Module):
    def __init__(self, drop, norm_shape, act,ln_aff):
        super().__init__()
        self.d = Dropout(drop)
        self.a = act_map[act]
        self.n = LayerNorm(norm_shape, elementwise_affine = ln_aff)
    def forward(self, x):
        return self.d(self.a(self.n(x)))


#mlp base
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, drop, act,ln_aff, hhid = None):
        super().__init__()
        if hhid == None:
            hhid = in_channels 
        self.l1 = nn.Linear(in_channels, hhid)
        self.l2 = nn.Linear(hhid, out_channels)
        self.tr = Trans(drop, hhid, act,ln_aff)
    def forward(self, x):
        return self.l2(self.tr(self.l1(x)))




class MIX(nn.Module):
    def __init__(self,hdim, mixdim, drop_model, drop_mix, cat_c, act, ln_aff, tr_in):
        super().__init__()
        hdim2=hdim
        if cat_c:
            hdim2 = 2*hdim
        self.mlp1 = MLP(hdim2, mixdim, drop_model, act,ln_aff, hdim)
        self.forward = self.forward_c if cat_c else self.forward_

        if tr_in:
            self.tr_in_mlp = Trans(drop_model, hdim2, act,ln_aff)
            self.tr_in_mix = Trans(drop_mix, hdim, act,ln_aff)
        else:
            self.tr_in_mlp = nn.Identity()
            self.tr_in_mix = nn.Identity()
        # if tr_mix:
        #     self.tr_mix = Trans(drop_mix, mixdim, act, ln_aff)
        # else:
        self.act_mix = act_map[act]

    
    def forward_c(self, x):
        xc = torch.cat((x, x[0:1,:].repeat((len(x),1))), dim=1)
        W1 = self.mlp1(self.tr_in_mlp(xc))
        return torch.matmul(W1, self.act_mix(torch.matmul(W1.transpose(-2,-1), self.tr_in_mix(x))))

    def forward_(self, x):
        W1 = self.mlp1(self.tr_in_mlp(x))
        return torch.matmul(W1, self.act_mix(torch.matmul(W1.transpose(-2,-1), self.tr_in_mix(x))))

    
    
class MIX_L(nn.Module):
    def __init__(self,in_channels, out_channels, hhid, hmix, drop_model, drop_mix, cat_c, act, ln_aff, tr_mid, tr_agg_in, skip_mix):
        super().__init__()
        #self.skip_mix = skip_mix
        self.mix = MIX(hhid, hmix, drop_model, drop_mix, cat_c, act, ln_aff, tr_agg_in)
        self.l1 = nn.Linear(in_channels, hhid)
        if skip_mix:
            self.l2 = nn.Linear(2*hhid, out_channels)
            self.fwd = self.forward_s
        else: 
            self.l2 = nn.Linear(hhid, out_channels)
            self.fwd = self.forward_s
        self.tr_mid = nn.Identity()
        if tr_mid:
            if skip_mix:
                self.tr_mid = Trans(drop_model, 2*hhid, act, ln_aff)
            else:
                self.tr_mid = Trans(drop_model, hhid, act, ln_aff)
                
    def forward_n(self, x):
        return self.l2(self.tr_mid(self.mix(self.l1(x))))

    def forward_s(self, x):
        tmp = self.l1(x)
        return self.l2(self.tr_mid(torch.cat((tmp, self.mix(tmp)), -1)))

    def forward(self, x: Tensor):
        return self.fwd(x)
        

#MIX_L_hdim_mlpdim_mixmlpdim_mixdim
class MIXER(nn.Module):
    def __init__(self, in_channels, out_channels, settings):
        super().__init__()

        tmp = [int(x) for x in settings["model"].split("_")[1:]] #HCM_nl_hid_mix
        
        self.num_hc = tmp[0]
        assert self.num_hc >= 1
        self.h_in = in_channels
        self.h_out = out_channels
        self.h_hid = tmp[1]
        self.h_mix = tmp[2]
        self.drop_model_p = float(settings["drop_model"])
        self.drop_mix_p = float(settings["drop_mix"])
        self.drop_in = Dropout(settings["drop_input"])
        self.tr_agg_in = settings["tr_agg_in"]
        #self.tr_agg_mix = settings["tr_agg_mix"] #unused -> del
        self.tr_mid = settings["tr_mid"]
        self.type = settings["model_type"]
        self.ln_aff = settings["ln_aff"]
        self.act = settings["act"]
        self.skip_mix = settings["skip_mix"]

        self.cat_c = settings["cat_c"] #whether to use the center node for aggr pred as well
        

        #norm and dropout x
        if settings["norm_in"]:
            self.in_ln = LayerNorm(self.h_in, elementwise_affine=self.ln_aff)
            self.in_x = lambda x: self.drop_in(self.in_ln(x))
        else:
            self.in_x = self.drop_in
        
        self.type = settings["model_type"]
        self.norms = nn.ModuleList()
        self.layers = nn.ModuleList()

        self.hc_args = {"hhid":self.h_hid, "hmix":self.h_mix, "drop_model":self.drop_model_p, "drop_mix":self.drop_mix_p, "cat_c":self.cat_c, "act":self.act, "ln_aff":self.ln_aff,"tr_mid":self.tr_mid, "tr_agg_in":self.tr_agg_in, "skip_mix":self.skip_mix}

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
        self.layers.append(Linear(self.h_in, self.h_hid)) #m
        for _ in range(self.num_hc):
            self.norms.append(Trans(self.drop_model_p, self.h_hid, self.act, self.ln_aff))
            self.layers.append(MIX_L(self.h_hid, self.h_hid, **self.hc_args))
        self.layers.append(Linear(self.h_hid, self.h_out))

    def lin_init(self):
        if self.num_hc == 1:
            self.layers.append(MIX_L(self.h_in, self.h_out, **self.hc_args)) #m
        else:
            self.layers.append(MIX_L(self.h_in, self.h_hid, **self.hc_args)) #m
            for _ in range(self.num_hc-2):
                self.norms.append(Trans(self.drop_model_p, self.h_hid, self.act, self.ln_aff))
                self.layers.append(MIX_L(self.h_hid, self.h_hid, **self.hc_args))
            self.norms.append(Trans(self.drop_model_p, self.h_hid, self.act, self.ln_aff))
            self.layers.append(MIX_L(self.h_hid, self.h_out, **self.hc_args))
    
    def create_wd_groups(self, settings):
        return add_weight_decayAll(self, settings)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, data):
        return self.fwd(data)
    
    def fwd_res(self, data):
        x = self.in_x(data.x)
        x = self.layers[0](x)
        for i in range(self.num_hc):
            x = x + self.layers[i+1](self.norms[i](x))
        return F.log_softmax(self.layers[-1](x), dim=1)

       
    def fwd_lin(self, data):
        x = self.in_x(data.x)
        x = self.layers[0](x)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x))
        return F.log_softmax(x, dim=1)

#fwd method for img graph cls tasks (graphs with x and pos features)
    def fwd_img(self, data):
        x = self.in_x(torch.cat([data.x, data.pos], 1))
        x = self.layers[0](x)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x))
        x = x.mean(0) #mean pooling
        return F.log_softmax(self.last(x), dim=0)

    #fwd methods for ZINC (graph regression)
    def fwd_zinc(self, data):
        x = self.in_x(F.one_hot(data.x.squeeze(), num_classes=21).float())
        x = self.layers[0](x)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x))
        x = x.mean(0) #mean pooling
        return self.last(x).squeeze()
    
    def loss(self, p, l):
        return self.loss_fn(p,l)
    def score(self, p, l):
        return self.score_fn(p,l)






