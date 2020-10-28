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
seq_len = 5
heatup_seq_len = 0
batch_size = 1
num_workers = 8
lr = 1e-5 #learning rate
epoch = 10
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
"""nn.Sequential(
    nn.Conv3d(
        in_channels=16,
        out_channels=16,
        kernel_size=(3, 3, 3), #D, H, W (kernel)
        padding=(0,1,1),
        #padding_mode='reflect'
    ),
    nn.LeakyReLU(),
    nn.Conv3d(
        in_channels=16,
        out_channels=16,
        kernel_size=(3, 3, 3), #D, H, W (kernel)
        padding=(0,1,1),
        #padding_mode='reflect'
    ),
    nn.Sigmoid()
)"""

if os.path.isfile(weight_path_conv3d):
    w = torch.load(weight_path_conv3d)
    model.load_state_dict(w['model'])
    del w
model = model.cuda()

### OPTIM ###
loss_func = nn.MSELoss()
optimizer = optim.Adam(
    list(model.parameters()),
    lr=lr,
    weight_decay=1e-5
    )
if os.path.isfile(weight_path_conv3d):
    w = torch.load(weight_path_conv3d)
    optimizer.load_state_dict(w['optimizer'])
    del w

### TRAINING LOOP ###
losses = []
last_save_time = time.time()
with torch.autograd.enable_grad():
    for epochnum in range(epoch):
        bar = tqdm(dataloader)
        for seq_num, sequence in enumerate(bar):
            sequence = sequence.cuda() #B, T, C, H, W
            sequence = sequence.transpose(1,2) #B, C, T, H, W
            
            seq = model(sequence)

            loss = 0
            #for i in range(seq.shape[1]-1):
            loss = loss_func(seq[:, :, :-1], sequence[:, :, 3:])

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

                    plt.subplot(211)
                    plt.imshow(seq[0, :, 0].sum(dim=0).cpu().numpy())

                    plt.subplot(212)
                    plt.imshow(sequence[0, :, 3].sum(dim=0).cpu().numpy())

                    plt.pause(1e-3)
                    plt.show()

            if time.time() - last_save_time > 60*5: #save every 5 minutes
                last_save_time = time.time()
                print("saving...")
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'losses': losses
                    },
                    weight_path_conv3d
                )
                print("saved")