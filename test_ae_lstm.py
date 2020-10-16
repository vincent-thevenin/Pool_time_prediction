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
import sys

#plt.ion()

### Hyperparameters ###
seq_len = 100
heatup_seq_len = 9
batch_size = 1
num_workers = 8*2
lr = 1e-5 #learning rate
epoch = 10
displaying = True
weight_path_lstm = "weights_lstm.chkpt"
weight_path_ae = "weights_ae.chkpt"


### DATALOADER ###
ds = PoolDataset(seq_len=seq_len, heatup_seq_len=heatup_seq_len)
dataloader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
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
            
            starter = starter.reshape(-1, 1, 256, 256)
            starter = encoder(starter)
            starter = starter.reshape(batch_size, -1, 2048)    
            _, (h,c) = lstm(starter)

            seq = sequence.reshape(-1, 1, 256, 256).requires_grad_()
            seq = encoder(seq)
            seq = seq.reshape(batch_size, -1, 2048)

            for i in range(seq_len):
                seq[:, i:i+1], (h,c) = lstm(seq[:, i:i+1], h, c)

            with torch.no_grad():
                if displaying:
                    if not os.path.isdir("test_results_ae_lstm"):
                        os.mkdir('test_results_ae_lstm')
                    for i in tqdm(range(seq_len-1)):
                        """plt.clf()
                        plt.plot(losses)
                        plt.pause(0.001)
                        plt.show()"""
                        plt.figure(1)

                        plt.subplot(211)
                        plt.imshow(decoder(seq[:,i])[0].sum(dim=0).cpu().numpy())

                        plt.subplot(212)
                        plt.imshow(sequence[0,i+1].sum(dim=0).cpu().numpy())

                        plt.savefig("test_results_ae_lstm/"+str(i)+".png")

                    sys.exit()