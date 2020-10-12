import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Encoder, Encoder_LSTM, Decoder
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.ion()

### Hyperparameters ###
seq_len = 2
heatup_seq_len = 9
batch_size = 2
num_workers = 4*4
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
encoder = Encoder().cuda()
decoder = Decoder().cuda()
lstm = Encoder_LSTM().cuda()

### OPTIM ###
bceloss = nn.BCELoss()
optimizer = optim.Adam(
    list(encoder.parameters())+list(decoder.parameters())+list(lstm.parameters()),
    lr=lr,
    weight_decay=1e-5
    )

### TRAINING LOOP ###
losses = []
with torch.autograd.enable_grad():
    for epochnum in range(epoch):
        for sequence in tqdm(dataloader):
            sequence = sequence.cuda()
            #jump start the lstm
            with torch.no_grad(): 
                starter, sequence = sequence[:,:heatup_seq_len], sequence[:,heatup_seq_len:]
                
                starter = starter.reshape(-1, 16, 256, 256)
                starter = encoder(starter)
                starter = starter.reshape(batch_size, -1, 256)    
                _, (h,c) = lstm(starter)

            seq = sequence.reshape(-1, 16, 256, 256)
            seq = encoder(seq)
            seq = seq.reshape(batch_size, -1, 256)
            seq, _ = lstm(seq, h, c)
            seq = seq.reshape(-1, 256)
            seq = decoder(seq)
            seq = seq.reshape(batch_size, -1, 16, 256, 256)

            loss = 0
            for i in range(seq.shape[1]-1):
                loss += bceloss(seq[:, i], sequence[:, i+1])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            if displaying:
                plt.clf()
                plt.plot(losses)
                plt.pause(0.001)
                plt.show()

        torch.save(
            {
                'encoder':encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'lstm': lstm.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses
            },
            'weights.chkpt'
        )
