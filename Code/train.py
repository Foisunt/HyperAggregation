import torch
import torch.nn.functional as F
from torch import optim
import torch_geometric as pyg
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import time

from utils import EarlyStoppingLA
from torch_geometric.utils import contains_self_loops



def get_optim(model, settings):
    p = model.create_wd_groups(settings)
    tmp = settings["base_optim"].split("_")
    op, lr = tmp[0], float(tmp[1])
    if op == "Adam":
        return optim.Adam(p, lr=lr)
    elif op == "RAdam":
        return optim.RAdam(p, lr=lr)
    elif op == "NAdam":
        return optim.NAdam(p, lr=lr)
    elif op == "AdamW":
        return optim.AdamW(p, lr=lr)
    elif op == "RMSprop":
        return optim.RMSprop(p, lr=lr, momentum=0.9)
    elif op == "SGD":
        return optim.SGD(p, lr=lr, momentum=0.9)
    elif op == "NSGD":
        return optim.SGD(p, lr=ör, momentum=0.9, nesterov=True) #nan
    else:
        print(op, "unknown")
    return o

def val_vertex(data, mask_name, model, settings, bs):
    with torch.no_grad():
        model.eval()
        if bs == "full":
            preds = model(data)[data[mask_name]]
            lbls = data.y[data[mask_name]]#.to(torch.float32)
        else:
            
            per_b = bs 
            if settings["model"][:3]=="MIX":
                list_of_batches = enhanced_loader(data, data[mask_name], n_neighbours = settings["num_N"], drop_edge_rate = settings["drop_adj"], batch_size=bs)
            else:
                list_of_batches = NeighborLoader(data, num_neighbors=settings["num_N"], batch_size=bs,  input_nodes=data[mask_name].to("cpu"))#, directed=False)#subgraph_type="induced")
            preds = []
            lbls = []
            for i, batch in enumerate(list_of_batches):
                preds.append(model(batch).to("cpu")[:batch.batch_size])
                lbls.append(batch.y[:batch.batch_size].to("cpu"))#.to(torch.float32))
            preds = torch.cat(preds)
            lbls = torch.cat(lbls)
        acc = model.score(preds, lbls)
        loss = model.loss(preds, lbls)
    return float(loss), float(acc)

def nhop(data):
    bix = data.num_sampled_nodes
    data.dist = torch.cat([torch.full((bix[i],), i) for i in range(len(bix))])+1 #self = 1, N1 = 2, ... Nk = k+1
    return data

#fixed mini batch size of 1
def enhanced_loader(graph, mask, n_neighbours, drop_edge_rate, batch_size=1):
    trans = nhop#[hop0, hop1, hop2][len(n_neighbours)]
    tmpD = pyg.data.Data(x=graph.x, edge_index = pyg.utils.dropout_edge(graph.edge_index, p=drop_edge_rate)[0], y=graph.y)
    return list(NeighborLoader(tmpD, num_neighbors=n_neighbours, batch_size=batch_size, input_nodes=mask.to("cpu"), transform=trans, shuffle=True))


def train_epoch_vertex(model, data, optimizer, settings):
    model.train()
    bs = settings["batch_size"]
    if bs == "full":
        out = model(data.to(settings["device"]))
        loss = model.loss(out[data.train_mask], data.y[data.train_mask])#.to(torch.float32))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return float(loss)
    else:
        tmp = bs.split("x")
        n_mini = int(tmp[0])
        per_mini = int(tmp[1])
        if settings["model"][:3]=="MIX":
            list_of_batches = enhanced_loader(data, data.train_mask, n_neighbours = settings["num_N"], drop_edge_rate = settings["drop_adj"])
        else:
            list_of_batches = NeighborLoader(data, num_neighbors=settings["num_N"], batch_size=per_mini,  input_nodes=data.train_mask.to("cpu"), shuffle = True)
        
        lo = 0
        for i , batch in enumerate(list_of_batches):
            torch.cuda.empty_cache()
            out = model(batch)
            loss = model.loss(out[:batch.batch_size], batch.y[:batch.batch_size])/n_mini #the first batch size data are the "real" training data with the correct neighborhood
            loss.backward()
            lo += float(loss)
            if (i+1) % n_mini ==0:
                optimizer.step()
                optimizer.zero_grad()
        return lo/len(list_of_batches)


def train_epoch_graph(model, data, optimizer, settings):
    model.train()
    sh = torch.randperm(len(data))
    bs = settings["batch_size"]
    lo = 0
    for i, d in enumerate(data[sh]):
        out = model(d)
        loss = model.loss(out, d.y.squeeze(-1)) # get loss from model
        loss.backward()
        lo += float(loss)
        if (i+1) % bs == 0:
            optimizer.step()
            optimizer.zero_grad()
    return lo/len(data)

def val_graph(data, mask, model, settings, bs):
    with torch.no_grad():
        model.eval()
        preds = []
        lbls = []
        for i, d in enumerate(data):
            preds.append(model(d).to("cpu"))
            lbls.append(d.y.to("cpu"))
        preds = torch.stack(preds).squeeze()
        lbls = torch.stack(lbls).squeeze()
        score = model.score(preds, lbls)
        loss = model.loss(preds, lbls)
    return float(loss), float(score)
    



#stats, model = train(model, ds, settings, model_path)
def train_eval_model(model, ds, settings, model_path):
    optimizer = get_optim(model, settings)
    ES = EarlyStoppingLA(settings["early_stopping_patience"], model_path)
    if settings["setup"] == "ind-gra":
        train_epoch = train_epoch_graph
        val = val_graph
    else:
        train_epoch = train_epoch_vertex
        val = val_vertex
        
    train_stats = []

    t0 = time.time()

    end = -1
    for e in range(settings["epochs"]):
        train_loss = train_epoch(model, ds.get_train(), optimizer,settings)
        val_loss, val_acc = val(ds.get_val(), "val_mask", model, settings, bs = settings["batch_size_val"])
        train_stats.append((train_loss, val_loss, val_acc))
        if ES(val_loss, val_acc, model):
            model.load_state_dict(torch.load(model_path))
            end = e - settings["early_stopping_patience"]
            break
        if val_loss == float("NaN"):
            print("val loss NaN in epoch", e)
            break
    t1 = time.time()
    torch.cuda.empty_cache()
    val_stats = val(ds.get_val(), "val_mask", model, settings, bs = settings.get("batch_size_val", None))
    test_stats = val(ds.get_test(), "test_mask", model, settings, bs = settings.get("batch_size_test", None))
    #print(test_stats)
    if settings["save_models"] == False:
        model_path.unlink(missing_ok=True)
    t2 = time.time()
    eval_df = {"trained_epochs":end, "val_loss": val_stats[0], "val_acc":val_stats[1], "test_loss":test_stats[0], "test_acc":test_stats[1], "train_time":t1-t0, "valtest_time":t2-t1} #"train_time":t_train, "inf_time":t_inf,
    
    return train_stats, eval_df

