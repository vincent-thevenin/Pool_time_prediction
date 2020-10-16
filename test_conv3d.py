import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Conv3D
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
plt.ion()

### Hyperparameters ###
seq_len = 100
heatup_seq_len = 0
batch_size = 1
num_workers = 8*2
lr = 1e-2 #learning rate
epoch = 1
displaying = True
weight_path_conv3d = "weights_conv3d.chkpt"

### DATALOADER ###
ds = PoolDataset(seq_len=seq_len, heatup_seq_len=heatup_seq_len, sum_channels=False)
dataloader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    #collate_fn=collate_fn,
    drop_last = True
    )

### MODELS ###
model = Conv3D()

if os.path.isfile(weight_path_conv3d):
    w = torch.load(weight_path_conv3d)
    model.load_state_dict(w['model'])
    del w
model = model.cuda()


### TRAINING LOOP ###
losses = []
last_save_time = time.time()
with torch.no_grad():
    for epochnum in range(epoch):
        bar = tqdm(dataloader)
        for seq_num, sequence in enumerate(bar):
            sequence = sequence.cuda() #B, T, C, H, W
            sequence = sequence.transpose(1,2) #B, C, T, H, W

            seq = sequence[:,:,:3]

            if displaying:
                for i in range(seq_len-3):
                    plt.figure(1)
                    plt.clf()

                    plt.subplot(211)
                    seqo = model(seq)
                    plt.imshow(seqo[0, :, -1].sum(dim=0).cpu().numpy())
                    seq = torch.cat((seq[:, :, 1:], seqo), dim=2)
                
                    plt.subplot(212)
                    plt.imshow(sequence[0, :, i+3].sum(dim=0).cpu().numpy())

                    plt.pause(1e-3)
                    plt.show()