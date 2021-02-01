import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from model import PredRNN, SelfAttention
from build_dataset import PoolDataset, collate_fn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys
import time

### Hyperparameters ###
seq_len = 48
heatup_seq_len = 10
batch_size = 1
num_workers = 8*2
lr = 1e-4 #learning rate
epoch = 10
displaying = True
weight_path_lstm = "weights_predrnn.chkpt"
save_path = "test_results_predrnn_10frames_delta_wing_disc_AllNorm2"
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
    shuffle=False,
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

if os.path.isfile(weight_path_lstm):
    w = torch.load(weight_path_lstm)
    lstm.load_state_dict(w['lstm'])
    att.load_state_dict(w['att'])
    del w
lstm = lstm.cuda()
att = att.cuda()


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
            seq = list(torch.split(sequence, 1, dim=1))


            for t in tqdm(range(sequence.shape[1])):
                #m = torch.tanh(att(m))
                for i,layer in enumerate(lstm):
                    s, (h,c,m) = layer(s, (state[i][0], state[i][1], m))
                    state[i] = [h,c]
            
                plt.figure(1)

                plt.subplot(211)
                plt.imshow(s[0,0].sum(dim=0).cpu().numpy())

                plt.subplot(212)
                plt.imshow(sequence[0,t].sum(dim=0).cpu().numpy())

                plt.savefig(save_path+"/"+str(t)+".png")

            sys.exit()
