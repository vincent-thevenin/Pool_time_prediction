import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Encoder, Decoder
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
plt.ion()

### Hyperparameters ###
seq_len = 1
heatup_seq_len = 0
batch_size = 8*2
num_workers = batch_size
lr = 1e-3 #learning rate
epoch = 50
displaying = True


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
encoder = Encoder()
decoder = Decoder()
if os.path.isfile("weights_ae.chkpt"):
    w = torch.load("weights_ae.chkpt")
    encoder.load_state_dict(w['encoder'])
    decoder.load_state_dict(w['decoder'])
    del w
encoder = encoder.cuda()
decoder = decoder.cuda()

### OPTIM ###
loss_func = nn.MSELoss()
optimizer = optim.Adam(
    list(encoder.parameters())+list(decoder.parameters()),
    lr=lr,
    #weight_decay=0
    )
if os.path.isfile("weights_ae.chkpt"):
    w = torch.load("weights_ae.chkpt")
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
            seq = sequence.squeeze(1).requires_grad_()

            seq_o = encoder(seq)
            seq_2 = decoder(seq_o)

            loss = loss_func(seq_2, seq)

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
                if displaying and not seq_num%10:
                    """plt.clf()
                    plt.plot(losses)
                    plt.pause(0.001)
                    plt.show()"""
                    plt.figure(1)
                    plt.clf()

                    plt.subplot(211)
                    plt.imshow(seq_2[0].sum(dim=0).cpu().numpy())

                    plt.subplot(212)
                    plt.imshow(seq[0].sum(dim=0).cpu().numpy())

                    plt.pause(1e-5)
                    plt.show()

            if time.time() - last_save_time > 60*5: #save every 5 minutes
                last_save_time = time.time()
                print("saving...")
                torch.save(
                    {
                        'encoder':encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'losses': losses
                    },
                    'weights_ae.chkpt'
                )
                print("saved")
