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

### Hyperparameters ###
seq_len = 100
heatup_seq_len = 10
batch_size = 1
num_workers = 8*2
lr = 1e-3 #learning rate
epoch = 10
displaying = True
weight_path_lstm = "weights_convlstm.chkpt"


### DATALOADER ###
ds = PoolDataset(seq_len=seq_len, heatup_seq_len=heatup_seq_len, sum_channels=True, transform=lambda frames: (frames/frames.max()-0.5)*2)
dataloader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    #collate_fn=collate_fn,
    drop_last = True,
    pin_memory=True
)

### MODELS ###
lstm = nn.Sequential(
    ConvLSTM(1,32,10),
    ConvLSTM(32,1,1)
)

if os.path.isfile(weight_path_lstm):
    w = torch.load(weight_path_lstm)
    lstm.load_state_dict(w['lstm'])
    del w
lstm = lstm.cuda()


### TRAINING LOOP ###
losses = []
last_save_time = time.time()
with torch.no_grad():
    for epochnum in range(epoch):
        bar = tqdm(dataloader)
        for seq_num, sequence in enumerate(bar):
            sequence = sequence.cuda()
            #jump start the lstm
            starter, sequence = sequence[:,:heatup_seq_len], sequence[:,heatup_seq_len:]

            state = []
            for layer in lstm:
                starter, (h,c) = layer(starter)
                state.append([h,c])

            seq = sequence[:,:1]
            

            for i in tqdm(range(seq_len-1)):
                for j,layer in enumerate(lstm):
                    seq, (h,c) = layer(seq, state[j])
                    state[j] = [h,c]

                plt.figure(1)

                plt.subplot(211)
                plt.imshow(seq[0,0].sum(dim=0).cpu().numpy())

                plt.subplot(212)
                plt.imshow(sequence[0,i+1].sum(dim=0).cpu().numpy())

                plt.savefig("test_results_convlstm/"+str(i)+".png")

            sys.exit()