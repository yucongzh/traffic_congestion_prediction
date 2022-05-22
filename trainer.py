import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
# ----------------------------------------------------
from models.lstm import LSTMNetwork

class LSTMTrainer:

    def __init__(self, in_dim, out_dim, latent_dim, n_layer, bsz, n_epoch, device, outdir="./saved_models/best_lstm.pt", network=None):
        if network is None:
            self.network    = LSTMNetwork(in_dim, out_dim, latent_dim, n_layer, bsz).to(device)
        else:
            self.network = network
        self.best_model = None
        self.opt_loss   = float('inf')
        self.n_epoch    = n_epoch
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.device     = device
        self.outdir     = outdir
        ## lstm parameters
        self.latent_dim = latent_dim
        self.n_layer    = n_layer
        self.bsz        = bsz

    def train(self, train_dataset, test_dataset, X):
        criterion = nn.MSELoss(reduction = 'mean')
        optimizer = Adam(self.network.parameters(), lr=1e-3)
        for epoch in range(self.n_epoch):
            self.network.train()
            # print('Epoch', epoch)
            state1 = tuple([torch.zeros([self.n_layer, self.bsz, self.latent_dim]).to(self.device),\
                            torch.zeros([self.n_layer, self.bsz, self.latent_dim]).to(self.device)])
            iterator = iter(train_dataset)
            training_loss = [];
            for x, y in iterator:
                optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                output, state1, extra = self.network(x, X, state1, self.bsz)
                output = torch.flatten(output, start_dim=0, end_dim=1)
                y = torch.flatten(y,0,1)
                y = y.float()
                loss = criterion(output, y) + torch.norm(extra)*1e-2
                # loss = criterion(output, y)
                training_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']], 5)
                optimizer.step()
            curr_train_loss = np.mean(training_loss)

            self.network.eval()
            with torch.no_grad():
                testing_loss = []
                iterator = iter(test_dataset)
                for x, y in iterator:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output, state1, extra = self.network(x, X, state1, self.bsz)
                    output = torch.flatten(output, start_dim=0, end_dim=1)
                    y = torch.flatten(y,0,1)
                    loss = criterion(output, y)
                    testing_loss.append(loss.item())
                curr_test_loss = np.mean(testing_loss)
                if self.opt_loss > curr_test_loss:
                    self.opt_loss = curr_test_loss
                    torch.save(self.network.state_dict(), self.outdir)

            fmt = "EPOCH {}/{}\t\t TrainingLoss: {:.2f}\t\t TestingLoss: {:.2f}"
            print(fmt.format(epoch+1, self.n_epoch, curr_train_loss, curr_test_loss))
    
    def predict(self, past, size, X):
        self.network.eval()
        x = past
        result = []
        state = tuple([torch.zeros([1, 1, self.latent_dim]).to(self.device),torch.zeros([1, 1, self.latent_dim]).to(self.device)])
        for i in range(int(size/self.out_dim)):
            x, state, extra = self.network(x, X, state, 1)
            result.append(x.reshape(-1, self.out_dim)[-1,:].cpu().detach().tolist())
        return result
    