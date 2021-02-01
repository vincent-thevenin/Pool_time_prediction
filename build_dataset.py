from matplotlib import pyplot as pyplot
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
import random
import threading

from pool.pool.config import resolution

class PoolDataset(Dataset):
    def __init__(self, seq_len, heatup_seq_len=10, num_thread=None, sum_channels=True, transform=None, variance=1000, size=256, train=True, split=0.9):
        self.turns = []
        self.seq_len = seq_len
        self.heatup_seq_len = heatup_seq_len
        self.max_seq_len = 0
        self.file_idx = []
        self.sum_channels = sum_channels
        self.transform = transform
        self.variance = variance
        self.size=size
        
        games = os.listdir('dataset')
        games.sort()
        if train:
            games = games[:int(len(games)*split)]
        else:
            games = games[int(len(games)*split):]
            
        for g in games:
            g_turns = os.listdir('dataset/'+str(g))
            g_turns.sort() #DEBUG take last moves
            g_turns = g_turns[-100:] #DEBUG take last moves
            for t in g_turns:
                turn_seq_len = int(re.search("_[0-9]+", t)[0][1:])
                if turn_seq_len > heatup_seq_len:
                    self.turns.append('dataset/'+str(g)+'/'+str(t))
                    self.max_seq_len = max(self.max_seq_len, turn_seq_len)
        
        for t in self.turns: #create list of (file_name, idx_inside_of_file)
            turn_seq_len = int(re.search("_[0-9]+", t)[0][1:])
            sub = turn_seq_len-self.heatup_seq_len
            
            """for i in range(sub//self.seq_len): #add first idxes
                self.file_idx.append((t, i*self.seq_len))
            if sub%self.seq_len: #add remaining frame, collate_fn should duplicate missing ones
                self.file_idx.append((t, turn_seq_len-self.heatup_seq_len-sub%self.seq_len))"""
            self.file_idx.append((t, 0)) #DEBUG


    def create_frames(self, turn_file, idx):
        with open(turn_file, "rb") as f:
            turn_list = pickle.load(f)

        turn_list = turn_list[idx:idx+self.seq_len+self.heatup_seq_len]

        frames = torch.zeros(((self.seq_len+self.heatup_seq_len), 16, self.size, self.size)) #b, seq_len, channels, 256, 256
        
        for i,turn in enumerate(turn_list):
            for j,position in enumerate(turn):
                x,y = np.mgrid[0:self.size, 0:self.size]
                pos = np.dstack((x, y))
                rv = multivariate_normal([position[0]/(resolution[0]-1)*(self.size-1), position[1]/(resolution[1]-1)*(self.size-1)], [[(self.variance/(resolution[0]-1)*200)**2, 0.0], [0.0, (self.variance/(resolution[1]-1)*200)**2]])
                frame = rv.pdf(pos)
                if self.variance == 10:
                    frame = (frame > 5e-3).astype(np.float32)
                frames[i, j, :, :] = torch.from_numpy(frame)
        for i2 in range(i+1, (self.seq_len+self.heatup_seq_len)):
            frames[i2] = frames[i]

        if self.sum_channels:
            frames = frames.sum(dim=1, keepdim=True)

        if self.transform is not None:
            return self.transform(frames)
        else:
            return (frames/frames.max()-0.5)*2
    
    def __len__(self):
        length = 0
        for t in self.turns:
            sub = int(re.search("_[0-9]+", t)[0][1:]) - self.heatup_seq_len
            length += sub//self.seq_len + int(sub%self.seq_len != 0)

        length = len(self.file_idx) #DEBUG
        return length

    def __getitem__(self, idx):
        #DEBUG
        self.variance *= 1 - 1e-3
        self.variance = max(10, self.variance)

        return self.create_frames(self.file_idx[idx][0], self.file_idx[idx][1])


def collate_fn(batch_list):
    """max_len = 0
    for data in batch_list:
        max_len = max(data.shape[1], max_len) #TODO Check if it is one for seq length

    for i,data in enumerate(batch_list):
        if data.shape[1] != max_len:
            last_frame = data[:,-1:] #TODO Check if it is one for seq length
            last_frame = last_frame.expand((1, max_len-data.shape[1], 16, 256, 256)) #TODO Check if it is one for seq length
            batch_list[i] = torch.cat((data, last_frame), dim=1)"""
    
    return torch.cat(batch_list, dim=0)
