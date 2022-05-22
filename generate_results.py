import torch
import torch.nn as nn
import pandas as pd
# ----------------------------------------------------
from trainer import LSTMTrainer
from models.lstm import LSTMNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMNetwork(in_dim     = 65*6, 
                    out_dim    = 65, 
                    latent_dim = 128, 
                    n_layer    = 1,
                    bsz        = 64)
model.load_state_dict(torch.load("./saved_models/best_lstm.pt"))
model.eval()

trainer = LSTMTrainer(in_dim     = 65*6, 
                      out_dim    = 65, 
                      latent_dim = 128,
                      n_layer    = 1, 
                      n_epoch    = 100, 
                      bsz        = 64,
                      device     = device,
                      network    = model.to(device))

## get spectral embeddings of edges
X = torch.load("./spectral_embds.pt")
X = X.to(device)

## get the last congestion situation
df_train = pd.read_csv("./train.csv")
Y = torch.tensor(df_train['congestion'].tolist())
Y = torch.reshape(Y,((int)(Y.shape[0]/65),65))
Y = Y[-200:,:]

size = 2340 # there are 2340 test samples
Y = torch.reshape(Y,(-1,65,1)).cuda()
results = trainer.predict(Y, size, X)

f = open("results.txt", "w")
f.write("row_id,congestion\n")
id = 848835
for i in range(int(size/65)):
	for j in range(65):
		f.write(str(id) + "," + str(results[i][j]) + "\n")
		id += 1
f.close()
