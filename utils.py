import torch
import torch.nn as nn
import numpy as np

def coor2idx(x, y, xlim=5, ylim=6):
    '''
    input:
        x, y: coordinates 
        N: total number of nodes in the graph
    output:
        idx: index of the given coordinates in the graph (starting from 0)
    '''
    return xlim*y + x


def move(coor, direction):
    if direction == 'E':
        coor[0] += 1
    elif direction == 'W':
        coor[0] -= 1
    elif direction == 'S':
        coor[1] += 1
    elif direction == 'N':
        coor[1] -= 1
    else:
        raise Exception("Invalid Direction!")
    return coor


def make_graph_(group):
    '''
    input: 
        group: dataframe
    output:
        adj_m: adjacent matrix, NxN (with padding)
    '''
    ## load in values
    x = group['x'].tolist(); y = group['y'].tolist()
    direction = group['direction'].tolist()
    # congestion = group['congestion'].tolist()

    ## initialization
    n_nodes = (max(x)+3)*(max(y)+3)
    # print(n_nodes)
    adj_m = torch.zeros((n_nodes, n_nodes)) # padding with zero

    ## generate nodes
    for i, (cx, cy) in enumerate(list(zip(group['x'], group['y']))):
        cx += 1; cy += 1
        src = coor2idx(cx, cy)
        # print(cx, cy, src)
        _direction = direction[i]
        # print(_direction)
        cx, cy = move([cx, cy], _direction[0])
        if _direction[1] != 'B':
            cx, cy = move([cx, cy], _direction[1])
        dst = coor2idx(cx, cy)
        # print(cx, cy, dst)
        adj_m[src, dst] = 1; adj_m[dst, src] = 1
        # break
    
    return adj_m


def get_embd_RW(G, steps=8, gamma=0.3):
    '''
    input:
        G: adjacent matrix
    output:
        P: the probabilistic matrix after multiple steps
    '''
    # print(G.shape)
    w, h = G.shape[0], G.shape[1]
    P = torch.zeros(G.shape)
    D = torch.zeros(G.shape)
    _s = torch.sum(G, dim=1)
    # print(_s)
    for i in range(w):
        D[i][i] = 1/_s[i] if _s[i] != 0 else 0
    
    ## random walk
    _P = torch.eye(w)
    for _ in range(steps):
        P += _P; D = D@G
        _P = gamma*(D) + (1-gamma)*_P
        
    # print(P)
    return P/steps


def spec(A, n=3):
    D = np.diag(A.sum(axis=1))
    L = D-A.numpy()
    vals, vecs = np.linalg.eig(L)
    vals = vals[np.argsort(vals)]
    vecs = vecs[:,np.argsort(vals)]
    return vecs[:,1:n+1]

