import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Encoder, Encoder_LSTM, Decoder
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
plt.ion()

### Hyperparameters ###
seq_len = 10
heatup_seq_len = 9
batch_size = 2
num_workers = 8*3
lr = 1e-5 #learning rate
epoch = 10
displaying = False
weight_path_lstm = "weights_lstm.chkpt"
weight_path_ae = "weights_ae.chkpt"


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
lstm = Encoder_LSTM()
if os.path.isfile(weight_path_lstm):
    w = torch.load(weight_path_lstm)
    lstm.load_state_dict(w['lstm'])
    del w
if os.path.isfile(weight_path_ae):
    w = torch.load(weight_path_ae)
    encoder.load_state_dict(w['encoder'])
    decoder.load_state_dict(w['decoder'])
    del w
encoder = encoder.cuda()
decoder = decoder.cuda()
lstm = lstm.cuda()

### OPTIM ###
loss_func = nn.L1Loss()
optimizer = optim.Adam(
    list(lstm.parameters()),
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
                
                starter = starter.reshape(-1, 1, 256, 256)
                starter = encoder(starter)
                starter = starter.reshape(batch_size, -1, 2048)    
                _, (h,c) = lstm(starter)

                sequence = sequence.reshape(-1, 1, 256, 256).requires_grad_()
                sequence = encoder(sequence)
                sequence = sequence.reshape(batch_size, -1, 2048)

            seq, _ = lstm(sequence, h, c)

            loss = 0
            #for i in range(seq.shape[1]-1):
            loss = loss_func(seq[:, :-1], sequence[:, 1:])

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
                    plt.imshow(decoder(seq[:,-2])[0].sum(dim=0).cpu().numpy())

                    plt.subplot(212)
                    plt.imshow(decoder(sequence[:,-1])[0].sum(dim=0).cpu().numpy())

                    plt.pause(1e-5)
                    plt.show()

            if time.time() - last_save_time > 60*5: #save every 5 minutes
                last_save_time = time.time()
                print("saving...")
                torch.save(
                    {
                        'lstm': lstm.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'losses': losses
                    },
                    weight_path_lstm
                )
                print("saved")
