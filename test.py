import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Encoder, Encoder_LSTM, Decoder
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
#plt.ion()

### Hyperparameters ###
seq_len = 5
heatup_seq_len = 5
batch_size = 1
num_workers = 1
lr = 1e-5 #learning rate
epoch = 10
displaying = False

### DATALOADER ###
ds = PoolDataset(seq_len=seq_len, heatup_seq_len=heatup_seq_len)
dataloader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    #collate_fn=collate_fn,
    drop_last = True
    )

### MODELS ###
m = torch.load("weights.chkpt")
encoder = Encoder()
encoder.load_state_dict(m['encoder'])
encoder.cuda()
decoder = Decoder()
decoder.load_state_dict(m['decoder'])
decoder = decoder.cuda()
lstm = Encoder_LSTM()
lstm.load_state_dict(m['lstm'])
lstm = lstm.cuda()
del m

### TRAINING LOOP ###
losses = []
with torch.no_grad():
    for epochnum in range(epoch):
        for sequence in tqdm(dataloader):
            sequence = sequence.cuda()
            starter, sequence = sequence[:,:heatup_seq_len], sequence[:,heatup_seq_len:]

            #jump start the lstm            
            starter = starter.reshape(-1, 16, 256, 256)
            starter = encoder(starter)
            starter = starter.reshape(batch_size, -1, 256)    
            _, (h,c) = lstm(starter)

            for i in range(sequence.shape[1]-1):
                seq = sequence[:, i]
                seq = encoder(seq)
                seq = seq.unsqueeze(1)
                seq, (h,c) = lstm(seq, h, c)
                seq = seq.reshape(-1, 256)
                seq = decoder(seq)
                seq = seq.sum(dim=1, keepdim=True).squeeze()
                seq = seq.cpu()
                seq = seq.numpy()

                plt.figure(1)
                plt.clf()

                plt.subplot(211)
                plt.imshow(seq)

                plt.subplot(212)
                plt.imshow(sequence[0,i+1].sum(dim=0).cpu().numpy())

                plt.show()