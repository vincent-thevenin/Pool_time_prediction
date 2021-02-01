import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from model import ResOpFlow, AdaptiveWingLoss, MBDisc
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from random import random
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import torchvision

### VIZ ###
writer = SummaryWriter("runs/resopflow_att_disc_add_longer")
ppepoch = 60 #points per epoch

### Hyperparameters ###
seq_len = 20
heatup_seq_len = 10
batch_size = 6
num_workers = 8
lr = 1e-3 #learning rate
lr_d = 1e-3
epoch = 1000
displaying = True
weight_path = "weights_resopflow.chkpt"
epsilon = 0.01 #1 --> take only real
stop_idx_dist = torch.distributions.categorical.Categorical(
    torch.nn.functional.softmax(
        epsilon*torch.arange(seq_len-1, 0, -1.0)
    )
)

### DATALOADER ###
ds = PoolDataset(
    seq_len=seq_len,
    heatup_seq_len=heatup_seq_len,
    sum_channels=True,
    transform=lambda frames: F.avg_pool2d((frames/frames.max()-0.5)*2, 4),
    variance=10
)
ds_val = PoolDataset(
    seq_len=heatup_seq_len+1,
    heatup_seq_len=0,
    sum_channels=True,
    transform=lambda frames: F.avg_pool2d((frames/frames.max()-0.5)*2, 4),
    variance=10,
    train=False
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
dataloader_val = DataLoader(
    ds_val,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    #collate_fn=collate_fn,
    drop_last = True,
    pin_memory=True
)
len_dataloader = len(dataloader)
len_dataloader_val = len(dataloader_val)
### MODELS ###
resopflow = ResOpFlow(seq_len=heatup_seq_len)
disc = MBDisc()

load_fail = False
if os.path.isfile(weight_path):
    w = torch.load(weight_path)
    try:
        resopflow.load_state_dict(w['resopflow'])
        disc.load_state_dict(w["disc"])
    except:
        print("Failed to load")
        load_fail = True
        resopflow = ResOpFlow(seq_len=heatup_seq_len)
        disc = MBDisc()
    del w
resopflow = resopflow.cuda()
disc = disc.cuda()

### OPTIM ###
loss_func = AdaptiveWingLoss()
optimizer = optim.Adam(
    list(resopflow.parameters()),
    lr=lr,
    #weight_decay=1e-5
)
optimizerD = optim.Adam(
    disc.parameters(),
    lr = lr_d,
)

if os.path.isfile(weight_path) and not load_fail:
    w = torch.load(weight_path)
    optimizer.load_state_dict(w['optimizer'])
    optimizerD.load_state_dict(w["optimizerD"])
    del w

### TRAINING LOOP ###
losses = []
last_save_time = time.time()
with torch.autograd.enable_grad():
    for epochnum in range(26,epoch):
        with tqdm(dataloader) as bar:
            for seq_num, sequence in enumerate(bar):
                sequence = sequence.cuda()

                with torch.no_grad():
                    if epsilon >= 0.01:
                        epsilon = max(0.01, epsilon-(10/(len_dataloader)))
                        stop_idx_dist = torch.distributions.categorical.Categorical(
                            torch.nn.functional.softmax(
                                epsilon*torch.arange(seq_len-1, 0, -1.0)
                            )
                        )
                    fakes = sequence[:, :heatup_seq_len, 0]
                    stop_idx = heatup_seq_len
                    stop_idx = int(stop_idx_dist.sample().item())+heatup_seq_len
                    if stop_idx > heatup_seq_len:
                        for _ in range(heatup_seq_len, stop_idx):
                            fakes = torch.cat((fakes[:, 1:], resopflow(fakes)[0]), dim=1)


                # if random() < epsilon:
                #     sequence_now = sequence[:, seq_len]
                # else:
                #     sequence_now = fakes[:, :1]
                # for t in range(seq_len+1, seq_len+heatup_seq_len-1):
                #     if random() < epsilon:
                #         sequence_now = torch.cat((sequence_now, sequence[:, t]), dim=1)
                #     else:
                #         sequence_now = torch.cat((sequence_now, fakes[:, t-seq_len].unsqueeze(1)), dim=1)

                pred, flow = resopflow(fakes)

                loss_c = loss_func(pred, sequence[:, stop_idx]) *100
                lossd = torch.nn.functional.relu(1-disc(pred)).mean() *5e-2
                loss_e = torch.nn.functional.mse_loss(
                    (pred+1).mean(dim=-1).mean(dim=-1),
                    (sequence[:, stop_idx]+1).mean(dim=-1).mean(dim=-1)
                ) * 10
                #loss_s = loss_func(pred - sequence[:, stop_idx-1], sequence[:, stop_idx] - sequence[:, stop_idx-1])

                (loss_c+loss_e+lossd).backward()
                optimizer.step()
                optimizer.zero_grad()
                optimizerD.zero_grad()


                #train disc
                pred = pred.detach()
                loss_real = torch.nn.functional.relu(1-disc(sequence[:, stop_idx])).mean()
                loss_fake = torch.nn.functional.relu(1+disc(pred)).mean()

                
                (loss_fake + loss_real).backward()

                optimizerD.step()
                optimizerD.zero_grad()


                if not seq_num % (len_dataloader//ppepoch):
                    writer.add_scalar('Loss/c', loss_c.item(), epochnum*len_dataloader + seq_num)
                    writer.add_scalar('Loss/e', loss_e.item(), epochnum*len_dataloader + seq_num)
                    writer.add_scalar('Loss/d', lossd.item(), epochnum*len_dataloader + seq_num)
                    writer.add_scalar('Loss/disc', loss_real.item()+loss_fake.item(), epochnum*len_dataloader + seq_num)

                bar.set_postfix(
                    lossC= loss_c.item(),
                    lossE = loss_e.item(),
                    #lossS = loss_s.item(),
                    real= loss_real.item(),
                    fake= loss_fake.item(),
                    eps=epsilon,
                    idx=stop_idx
                )
                with torch.no_grad():
                    if displaying and not seq_num%10:
                        plt.subplot(231)
                        plt.imshow(pred[0, 0].cpu().numpy(), cmap='gray')

                        plt.subplot(232)
                        plt.imshow(flow[0, 0].cpu().numpy(), cmap='gray')

                        plt.subplot(233)
                        plt.imshow(flow[0, -1].cpu().numpy(), cmap='gray')
                        
                        plt.subplot(234)
                        plt.imshow(sequence[0, stop_idx, 0].cpu().numpy(), cmap='gray')

                        plt.savefig("tmp.png")

                if time.time() - last_save_time > 60*5: #save every 5 minutes
                    last_save_time = time.time()
                    print("saving...")
                    torch.save(
                        {
                            'resopflow': resopflow.state_dict(),
                            'disc': disc.state_dict(),
                            'optimizerD': optimizerD.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        },
                        weight_path
                    )
                    print("saved")
            
        with torch.no_grad():
            with tqdm(dataloader_val) as bar:
                loss = 0
                for seq_num, sequence in enumerate(bar):
                    sequence = sequence.cuda()
                    pred, _ = resopflow(sequence[:, :heatup_seq_len].squeeze(2))
                    loss += loss_func(pred, sequence[:, -1])
                writer.add_scalar('Loss/val', loss.item()/len_dataloader_val, epochnum*len_dataloader_val)
        
        print("saving...")
        torch.save(
            {
                'resopflow': resopflow.state_dict(),
                'disc': disc.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            weight_path+'_'+str(datetime.now())
        )
        print("saved")
