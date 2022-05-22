import math
from torch.utils.data import Dataset

class my_dataset(Dataset):

    def __init__(self, x_data, bsz, bptt_len=32):
        self.bsz = bsz
        self.bptt_len = bptt_len
        X = x_data
        ncontig = X.size(0) // bsz
        X = X[:ncontig*bsz, :].view(bsz, -1, 65) # batch_size x ncontig * 65
        self.X = X.transpose(0,1).transpose(1,2).contiguous() # ncontig x 65 * batch_size
    def __len__(self):
        return int(math.ceil(self.X.size(0) / self.bptt_len))
          
    def __iter__(self):
        for i in range(0, self.X.size(0)-1):
            seqlen = min(self.bptt_len, self.X.size(0) - i - 1)
            x = self.X[i:i+seqlen,:,:] # seqlen x batch_size
            y = self.X[i+1:i+seqlen+1,:,:] # seqlen x batch_size
            yield x, y
