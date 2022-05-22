import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# ----------------------------------------------------
from utils import *

## load in data
df_train = pd.read_csv("train.csv")

df_new = df_train.groupby("time")
embds = []; X = None
for t, group in df_new:
    G = make_graph_(group)
    G_RW = get_embd_RW(G) # graph after random walk
    spec_G = torch.tensor(spec(G_RW)) # spectral embeddings
    for x, y, d in zip(group['x'], group['y'], group['direction']):
        src = coor2idx(x+1, y+1)
        _x, _y = move([x+1, y+1], d[0])
        if d[1] != 'B':
            _x,_y = move([_x, _y], d[1])
        dst = coor2idx(_x, _y)
        # embd = spec_G[src] + spec_G[dst] # addition to form the edges
        embd = torch.concat((spec_G[src], spec_G[dst]), dim=0) # concat to form the edges
        embds.append(embd.view(1, -1))
    break
X = torch.concat(embds, dim=0)
torch.save(X, "./spectral_embds.pt")
