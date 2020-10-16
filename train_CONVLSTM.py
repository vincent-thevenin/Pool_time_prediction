import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from model import ConvLSTM
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
plt.ion()

### Hyperparameters ###
seq_len = 5
heatup_seq_len = 5
batch_size = 4
num_workers = 8*2
lr = 1e-5 #learning rate
epoch = 10
displaying = True
weight_path_lstm = "weights_convlstm.chkpt"


### DATALOADER ###
ds = PoolDataset(seq_len=seq_len, heatup_seq_len=heatup_seq_len, sum_channels=True)
dataloader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    #collate_fn=collate_fn,
    drop_last = True,
    pin_memory=True
)

### MODELS ###
lstm1 = ConvLSTM(1,32,3)
lstm2 = ConvLSTM(32,1,1)

if os.path.isfile(weight_path_lstm):
    w = torch.load(weight_path_lstm)
    lstm1.load_state_dict(w['lstm1'])
    lstm2.load_state_dict(w['lstm2'])
    del w
lstm1 = lstm1.cuda()
lstm2 = lstm2.cuda()

### OPTIM ###
loss_func = nn.MSELoss()
optimizer = optim.Adam(
    list(lstm1.parameters())+list(lstm2.parameters()),
    lr=lr,
    weight_decay=1e-5
    )
if os.path.isfile(weight_path_lstm):
    w = torch.load(weight_path_lstm)
    optimizer.load_state_dict(w['optimizer'])
    del w

### TRAINING LOOP ###
losses = []
last_save_time = time.time()
with torch.autograd.enable_grad():
    for epochnum in range(epoch):
        bar = tqdm(dataloader)
        for seq_num, sequence in enumerate(bar):
            sequence = sequence.cuda()
            #jump start the lstm
            with torch.no_grad(): 
                starter, sequence = sequence[:,:heatup_seq_len], sequence[:,heatup_seq_len:]

                starter, (h1,c1) = lstm1(starter)
                _, (h2, c2) = lstm2(starter)

            inference = []
            loss = 0
            seq = sequence[:,:1]
            for i in range(seq_len-1):
                seq, (h1,c1) = lstm1(seq, (h1,c1))
                seq, (h2,c2) = lstm2(seq, (h2,c2))

                loss += loss_func(seq, sequence[:, i+1:i+2])

                if i>seq_len-5:
                    inference.append(seq.detach())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            bar.set_postfix(
                {
                    "loss": loss.item()
                }
            )
            with torch.no_grad():
                if displaying and not seq_num%20:
                    """plt.clf()
                    plt.plot(losses)
                    plt.pause(0.001)
                    plt.show()"""
                    plt.figure(1)
                    plt.clf()

                    plt.subplot(231)
                    plt.imshow(inference[0][0,0].sum(dim=0).cpu().numpy())

                    plt.subplot(232)
                    plt.imshow(inference[1][0,0].sum(dim=0).cpu().numpy())

                    plt.subplot(233)
                    plt.imshow(inference[2][0,0].sum(dim=0).cpu().numpy())


                    plt.subplot(234)
                    plt.imshow(sequence[0,-3].sum(dim=0).cpu().numpy())

                    plt.subplot(235)
                    plt.imshow(sequence[0,-2].sum(dim=0).cpu().numpy())

                    plt.subplot(236)
                    plt.imshow(sequence[0,-1].sum(dim=0).cpu().numpy())

                    plt.pause(1e-3)
                    plt.show()

            if time.time() - last_save_time > 60*5: #save every 5 minutes
                last_save_time = time.time()
                print("saving...")
                torch.save(
                    {
                        'lstm1': lstm1.state_dict(),
                        'lstm2': lstm2.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'losses': losses
                    },
                    weight_path_lstm
                )
                print("saved")

