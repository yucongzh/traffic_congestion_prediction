import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# ----------------------------------------------------
from utils import *
from dataset import my_dataset
from trainer import LSTMTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## load csv files
df_train = pd.read_csv("./train.csv")

## get spectral embeddings of edges
X = torch.load("./spectral_embds.pt")
X = X.to(device)

## get congestion labels
y = torch.tensor(df_train['congestion'].tolist())
y = torch.reshape(y,((int)(y.shape[0]/65),65))

## make datasets, 2: validation, 8: training
bsz = 64
y_train, y_test = train_test_split(y, test_size=0.2, shuffle = False)
train_dataset = my_dataset(y_train, bsz, 100)
test_dataset = my_dataset(y_test, bsz, 100)

trainer = LSTMTrainer(in_dim     = 65*6, 
                      out_dim    = 65, 
                      latent_dim = 128,
                      n_layer    = 1, 
                      n_epoch    = 100, 
                      bsz        = bsz,
                      device     = device)
trainer.train(train_dataset, test_dataset, X)
