import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from model import PredRNN, AdaptiveWingLoss, SelfAttention, Discriminator
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

### Hyperparameters ###
seq_len = 10
heatup_seq_len = 10*2
batch_size = 4
num_workers = 0
lr = 1e-3 #learning rate
epoch = 10000
displaying = True
weight_path_lstm = "weights_predrnn.chkpt"


### DATALOADER ###
ds = PoolDataset(
    seq_len=seq_len,
    heatup_seq_len=heatup_seq_len,
    sum_channels=True,
    transform=lambda frames: F.avg_pool2d((frames/frames.max()-0.5)*2, 4),
    variance=10
)
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
lstm = nn.Sequential(
    PredRNN(1, 16, 4,4, num_layers=1),
    PredRNN(16,32, 4,4, num_layers=1),
    #PredRNN(32,64, 4,4, num_layers=3),
    #PredRNN(64,32, 4,4, num_layers=1),
    PredRNN(32,16, 4,4, num_layers=1),
    PredRNN(16, 1, 4,4, num_layers=1),
)
att = SelfAttention(4)
disc = Discriminator()

if os.path.isfile(weight_path_lstm):
    w = torch.load(weight_path_lstm)
    lstm.load_state_dict(w['lstm'])
    att.load_state_dict(w['att'])
    disc.load_state_dict(w['disc'])
    del w
lstm = lstm.cuda()
att = att.cuda()
disc = disc.cuda()


### OPTIM ###
loss_func = AdaptiveWingLoss()
optimizer = optim.Adam(
    list(lstm.parameters())+list(att.parameters()),
    lr=lr,
    #weight_decay=1e-5
)
optimizerDisc = optim.Adam(
    list(disc.parameters()),
    lr=lr,
    #weight_decay=1e-5
)
if os.path.isfile(weight_path_lstm):
    w = torch.load(weight_path_lstm)
    optimizer.load_state_dict(w['optimizer'])
    optimizerDisc.load_state_dict(w['optimizerDisc'])
    del w

### TRAINING LOOP ###
losses = []
last_save_time = time.time()
with torch.autograd.enable_grad():
    for epochnum in range(epoch):
        bar = tqdm(dataloader)
        for seq_num, sequence in enumerate(bar):
            #jump start the lstm
            with torch.no_grad(): 
                starter, sequence = sequence[:,:heatup_seq_len//2], sequence[:,heatup_seq_len//2:]

                state = []

                starter = list(torch.split(starter, 1, dim=1))

                s = starter[0].cuda()
                s, (h,c,m) = lstm[0](s)
                state.append([h,c])
                for layer in lstm[1:]:
                    s, (h,c,m) = layer(s, (None, None, m))
                    state.append([h,c])

                for t in range(1, len(starter)):
                    #m = torch.tanh(att(m))
                    s = starter[t].cuda()
                    for i,layer in enumerate(lstm):
                        s, (h,c,m) = layer(s, (state[i][0], state[i][1], m))
                        state[i] = [h,c]


                inference = []

            loss = 0
            sequence = sequence.cuda()
            s = sequence[:,:1]
            ss = []
            for t in range(sequence.shape[1]-2):
                s_prev = s.detach()
                #s = sequence[:,:t+1]
                # m = torch.tanh(att(m))
                for i,layer in enumerate(lstm):
                    s, (h,c,m) = layer(s, (state[i][0], state[i][1], m))
                    state[i] = [h,c]

                if t==0:
                    inference.append(s[0,0].detach().cpu())
                if t==(sequence.shape[1]-1)//2:
                    inference.append(s[0,0].detach().cpu())
                if t==(sequence.shape[1]-1-2):
                    inference.append(s[0,0].detach().cpu())

                #loss += loss_func((s+1)/2, (1+sequence[:, t+1:t+2])/2)
                
                """loss -= disc(torch.cat((sequence[:,t], s[:,0], sequence[:,t+2]), dim=1)).mean()*10
                ss.append(s.detach())"""
                d1 = (s-s_prev)**2
                if d1.max():
                    d1 = d1 / d1.max()
                d2 = (sequence[:, t+1:t+2] - sequence[:, t:t+1])**2
                if d2.max():
                    d2 = d2 / d2.max()
                loss_cont = loss_func(d1, d2) + torch.nn.functional.l1_loss(s, sequence[:, t+1:t+2])
                loss += loss_cont * (1+t*0.1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            optimizerDisc.zero_grad()

            loss_d = 0
            """for t,s in enumerate(ss):
                loss_d = loss_d + disc(torch.cat((sequence[:,t], s[:,0], sequence[:,t+2]), dim=1)).mean() - disc(sequence[:,t:t+3].squeeze(dim=2)).mean()
            loss_d.backward()
            optimizerDisc.step()
            optimizerDisc.zero_grad()"""

            bar.set_postfix(
                {
                    "loss": loss.item(),
                    #"lossD": loss_d.item()
                }
            )
            with torch.no_grad():
                if displaying and not seq_num%20:
                    """plt.clf()
                    plt.plot(losses)
                    plt.pause(0.001)
                    plt.show()"""
                    plt.subplot(231)
                    plt.imshow(inference[0].sum(dim=0).numpy(), cmap='gray')

                    plt.subplot(232)
                    plt.imshow(inference[1].sum(dim=0).numpy(), cmap='gray')

                    plt.subplot(233)
                    plt.imshow(inference[2].sum(dim=0).numpy(), cmap='gray')


                    plt.subplot(234)
                    plt.imshow(sequence[0, 1].sum(dim=0).cpu().numpy(), cmap='gray')

                    plt.subplot(235)
                    plt.imshow(sequence[0, (sequence.shape[1]-1)//2+1].sum(dim=0).cpu().numpy(), cmap='gray')

                    plt.subplot(236)
                    plt.imshow(sequence[0, -1].sum(dim=0).cpu().numpy(), cmap='gray')

                    plt.savefig("tmp.png")

            if time.time() - last_save_time > 60*5: #save every 5 minutes
                last_save_time = time.time()
                print("saving...")
                torch.save(
                    {
                        'att': att.state_dict(),
                        'disc': disc.state_dict(),
                        'lstm': lstm.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'optimizerDisc': optimizerDisc.state_dict(),
                        'losses': losses
                    },
                    weight_path_lstm
                )
                print("saved")