import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from e3d_lstm import E3DLSTM
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
plt.ion()

### Hyperparameters ###
seq_len = 4
heatup_seq_len = 0
batch_size = 1
num_workers = 8
lr = 1e-3 #learning rate
epoch = 10
displaying = False
weight_path = "weights_e3d_lstm.chkpt"

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
e3d_lstm = E3DLSTM((1,seq_len,256,256), 16, 1, (1,3,3), 2)
if os.path.isfile(weight_path):
    w = torch.load(weight_path)
    e3d_lstm.load_state_dict(w['e3d_lstm'])
    del w
e3d_lstm = e3d_lstm

### OPTIM ###
loss_func = nn.MSELoss()
optimizer = optim.Adam(
    e3d_lstm.parameters(),
    lr=lr,
    weight_decay=1e-5
    )
if os.path.isfile(weight_path):
    w = torch.load(weight_path)
    optimizer.load_state_dict(w['optimizer'])
    del w

### TRAINING LOOP ###
losses = []
with torch.autograd.enable_grad():
    for epochnum in range(epoch):
        for seq_num, sequence in enumerate(tqdm(dataloader)):
            sequence = sequence.transpose(0,1).unsqueeze(0)

            out = e3d_lstm(sequence)

            loss = 0
            for i in range(seq.shape[1]-1):
                loss += loss_func(seq[:, i], sequence[:, i+1])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            if displaying and not seq_num%5:
                plt.clf()
                plt.plot(losses)
                plt.pause(0.001)
                plt.show()

        print("saving...")
        torch.save(
            {
                'e3d_lstm':e3d_lstm.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses
            },
            weight_path
        )
        print("saved")
