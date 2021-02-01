import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from model import ResOpFlow
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys
from time import time


### Hyperparameters ###
seq_len = 48
heatup_seq_len = 0
batch_size = 1
num_workers = 8*2
lr = 1e-4 #learning rate
epoch = 10
displaying = True
weight_path = "weights_resopflow.chkpt"
save_path = "test_result"
if not save_path in os.listdir():
    os.mkdir(save_path)


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
resopflow = ResOpFlow(seq_len=10)

if os.path.isfile(weight_path):
    w = torch.load(weight_path)
    resopflow.load_state_dict(w['resopflow'])
    del w
resopflow = resopflow.cuda()


### TRAINING LOOP ###
with torch.no_grad():
    for epochnum in range(epoch):
        bar = tqdm(dataloader)
        for seq_num, sequence in enumerate(bar):
            
            sequence_before = sequence[:, :10].reshape(batch_size, 10, sequence.shape[-2], sequence.shape[-1]).cuda()

            fps = 0
            for t in tqdm(range(9, sequence.shape[1])):

                start = time()
                s, _ = resopflow(sequence_before)
                fps = 1/(time() - start)*0.8 + fps*0.2
            
                plt.figure(1)

                plt.subplot(211)
                plt.imshow(s[0].sum(dim=0).cpu().numpy())

                plt.subplot(212)
                plt.imshow(sequence[0,t].sum(dim=0).cpu().numpy())

                plt.savefig(save_path+"/"+str(t)+".png")

                sequence_before = torch.cat((sequence_before[:, 1:], s), dim=1)

                bar.set_postfix(fps=fps)

            sys.exit()
