import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch.nn import Dropout, Linear, LayerNorm, BatchNorm1d, Identity
from torch_geometric.utils import dropout_edge
from utils import add_weight_decayAll, get_feature_dis
from hyper_mixer import MIXER
from HyperConv import HyperConvMixer
from HyperConv import HyperAggregation

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



class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, settings, Layer, Layer_p):
        super().__init__()
        tmp = [int(x) for x in settings["model"].split("_")[1:]] #XX_nl_hid
        self.num_hc = tmp[0]
        assert self.num_hc >= 1
        self.h_in = in_channels
        self.h_out = out_channels
        self.h_hid = tmp[1]
        
        self.drop_model_p = settings["drop_model"]
        self.drop_edge_p = settings["drop_adj"]
        self.drop_in = Dropout(settings["drop_input"])
        self.type = settings["model_type"]
        self.ln_aff = settings["ln_aff"]
        self.act = settings["act"]

        self.Layer = Layer
        self.Layer_p = Layer_p
        
        #norm and dropout x
        if settings["norm_in"]:
            self.in_ln = LayerNorm(in_channels, elementwise_affine=self.ln_aff)
            self.in_x = lambda x: self.drop_in(self.in_ln(x))
        else:
            self.in_x = self.drop_in
        
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.loss_fn = F.nll_loss
        self.score_fn = lambda preds,lbls:(preds.argmax(1) == lbls).sum()/len(lbls)
        if settings["dataset"] == "ZINC":
            self.loss_fn = F.l1_loss #= MAE
            self.score_fn = lambda x,y:0
            self.h_out = self.h_hid
            self.lin_init()
            self.h_out = out_channels
            self.fwd = self.fwd_zinc
            self.last = Linear(self.h_hid, self.h_out)
        elif settings["dataset"] in ["CIFAR10", "MNIST"]:
            #self.loss_fn = F.nll_loss
            #self.score_fn = lambda preds,lbls:(preds.argmax(1) == lbls).sum()/len(lbls)
            self.h_out = self.h_hid
            self.lin_init()
            self.h_out = out_channels
            self.fwd = self.fwd_img
            self.last = Linear(self.h_hid, self.h_out)
        elif self.type=="res":
            self.res_init()
            self.fwd = self.fwd_res
        else:
            self.lin_init()
            self.fwd = self.fwd_lin
            

    def loss(self, p, l):
        return self.loss_fn(p,l)
    def score(self, p, l):
        return self.score_fn(p,l)

    def res_init(self):
        self.layers.append(Linear(self.h_in, self.h_hid))
        for _ in range(self.num_hc):
            self.norms.append(Trans(self.drop_model_p, self.h_hid, self.act, self.ln_aff))
            self.layers.append(self.Layer(self.h_hid, self.h_hid, **self.Layer_p))
        self.layers.append(Linear(self.h_hid, self.h_out))

    def lin_init(self):
        if self.num_hc == 1:
            self.layers.append(self.Layer(self.h_in, self.h_out, **self.Layer_p))
        else:
            self.layers.append(self.Layer(self.h_in, self.h_hid, **self.Layer_p))
            for _ in range(self.num_hc-2):
                self.norms.append(Trans(self.drop_model_p, self.h_hid, self.act, self.ln_aff))
                self.layers.append(self.Layer(self.h_hid, self.h_hid, **self.Layer_p))
            self.norms.append(Trans(self.drop_model_p, self.h_hid, self.act, self.ln_aff))
            self.layers.append(self.Layer(self.h_hid, self.h_out, **self.Layer_p))

    #fwd method with residual connections for vertex classification
    def fwd_res(self, data):
        x, e = self.in_x(data.x), dropout_edge(data.edge_index, p = self.drop_edge_p, force_undirected=True)[0]
        x = self.layers[0](x)
        for i in range(self.num_hc):
            x = x + self.layers[i+1](self.norms[i](x), e)
        return F.log_softmax(self.layers[-1](x), dim=1)

    #fwd method without residual connection for vertex classification
    def fwd_lin(self, data):
        x, e = self.in_x(data.x), dropout_edge(data.edge_index, p = self.drop_edge_p, force_undirected=True)[0]
        x = self.layers[0](x, e)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x), e)
        return F.log_softmax(x, dim=1)
    
    #fwd method for img graph cls tasks (graphs with x and pos features)
    def fwd_img(self, data):
        x, e = self.in_x(torch.cat([data.x, data.pos], 1)), data.edge_index
        x = self.layers[0](x, e)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x), e)
        x = x.mean(0) #mean pooling
        return F.log_softmax(self.last(x), dim=0)

    #fwd methods for ZINC (graph regression)
    def fwd_zinc(self, data):
        x, e = self.in_x(F.one_hot(data.x.squeeze(), num_classes=21).float()), data.edge_index
        x = self.layers[0](x, e)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x), e)
        x = x.mean(0) #mean pooling
        return self.last(x).squeeze()

    def embed_res(self, data):
        x, e = self.in_x(data.x), dropout_edge(data.edge_index, p = self.drop_edge_p, force_undirected=True)[0]
        x = self.layers[0](x)
        for i in range(self.num_hc):
            x = x + self.layers[i+1](self.norms[i](x), e)
        return x
        
    def embed_lin(self, data):
        x, e = self.in_x(data.x), dropout_edge(data.edge_index, p = self.drop_edge_p, force_undirected=True)[0]
        x = self.layers[0](x, e)
        for i in range(self.num_hc-2):
            x = self.layers[i+1](self.norms[i](x), e)
        return x
    
    def embed_img(self, data):
        x, e = self.in_x(torch.cat([data.x, data.pos], 1)), data.edge_index
        x = self.layers[0](x, e)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x), e)
        x = x.mean(0) #mean pooling
        return x

    def embed_zinc(self, data):
        x, e = self.in_x(F.one_hot(data.x.squeeze(), num_classes=21).float()), data.edge_index
        x = self.layers[0](x, e)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x), e)
        x = x.mean(0) #mean pooling
        return x
        
    
    def forward(self, data):
        return self.fwd(data)
        
    @property
    def device(self):
        return next(self.parameters()).device
        
    def create_wd_groups(self, settings):
        return add_weight_decayAll(self, settings)



class MLP(Model):
    def __init__(self, in_channels, out_channels, settings):
        ls = [int(x) for x in settings["model"].split("_")[1:]]
        super().__init__(in_channels, out_channels, settings, Layer = Linear, Layer_p={})

        if settings["setup"] != "ind-gra":
            if self.type=="res":
                self.fwd = self.fwd_res_MLP
            else:
                self.fwd = self.fwd_lin_MLP
        elif settings["dataset"] == "ZINC":
            self.fwd = self.fwd_zinc_MLP
            
        elif settings["dataset"] in ["CIFAR10", "MNIST"]:
            self.fwd = self.fwd_img_MLP
        else:
            raise NotImplementedError("Setup / Dataset '" + settings["setup"] +" / "+ settings["dataset"]+ "' is unkown")
            
    def fwd_res_MLP(self, data):
        x = self.in_x(data.x)
        x = self.layers[0](x)
        for i in range(self.num_hc):
            x = x + self.layers[i+1](self.norms[i](x))
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)
        
    def fwd_lin_MLP(self, data):
        x = self.in_x(data.x)
        x = self.layers[0](x)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x))
        return F.log_softmax(x, dim=1)

    def fwd_img_MLP(self, data):
        x = self.in_x(torch.cat([data.x, data.pos], 1))
        x = self.layers[0](x)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x))
        x = x.mean(0) #mean pooling
        return F.log_softmax(self.last(x), dim=0)

    def fwd_zinc_MLP(self, data):
        x = self.in_x(F.one_hot(data.x.squeeze(), num_classes=21).float())
        x = self.layers[0](x)
        for i in range(self.num_hc-1):
            x = self.layers[i+1](self.norms[i](x))
        x = x.mean(0) #mean pooling
        return self.last(x).squeeze()

    def forward(self, data):
        return self.fwd(data)

            
class GCN(Model):
    def __init__(self, in_channels, out_channels, settings):
        cach = ((settings["setup"]=="transd") & (settings["batch_size"]=="full")) #cach if the graph does not change (ppi is multigraph, inductive and sampling change graph by def)
        super().__init__(in_channels, out_channels, settings, Layer = GCNConv, Layer_p={"cached":cach, "add_self_loops":False}) #self loops are added (if set) during loading the dataset


     
       
#returns an initialized model
def get_model(settings, num_features, num_classes):
    name_d = {
        "MLP_" :MLP,
        "GCN_":GCN,
        "MIX_":MIXER,
        "MIXB":MIXER,
        "HCM_":HyperConvMixer,
    }
    m = name_d[settings["model"][:4]]
    
    return m(num_features, num_classes, settings)
    