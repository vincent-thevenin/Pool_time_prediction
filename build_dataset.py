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
    def __init__(self, seq_len, heatup_seq_len=10, num_thread=None, sum_channels=True):
        self.turns = []
        self.seq_len = seq_len
        self.heatup_seq_len = heatup_seq_len
        self.max_seq_len = 0
        self.file_idx = []
        self.sum_channels = sum_channels
        
        games = os.listdir('dataset')
        for g in games:
            g_turns = os.listdir('dataset/'+str(g))
            for t in g_turns:
                turn_seq_len = int(re.search("_[0-9]+", t)[0][1:])
                if turn_seq_len > heatup_seq_len:
                    self.turns.append('dataset/'+str(g)+'/'+str(t))
                    self.max_seq_len = max(self.max_seq_len, turn_seq_len)
        
        for t in self.turns: #create list of (file_name, idx_inside_of_file)
            turn_seq_len = int(re.search("_[0-9]+", t)[0][1:])
            sub = turn_seq_len-self.heatup_seq_len
            for i in range(sub//self.seq_len): #add first idxes
                self.file_idx.append((t, i*self.seq_len))
            if sub%self.seq_len: #add remaining frame, collate_fn should duplicate missing ones
                self.file_idx.append((t, turn_seq_len-self.heatup_seq_len-sub%self.seq_len))


    def create_frames(self, turn_file, idx):
        with open(turn_file, "rb") as f:
                turn_list = pickle.load(f)

        turn_list = turn_list[idx:idx+self.seq_len+self.heatup_seq_len]

        frames = torch.zeros((self.seq_len+self.heatup_seq_len, 16, 256,256)) #b, seq_len, channels, 256, 256
        
        for i,turn in enumerate(turn_list):
            for j,position in enumerate(turn):
                x,y = np.mgrid[0:256, 0:256]
                pos = np.dstack((x, y))
                rv = multivariate_normal([position[0]/(resolution[0]-1)*255, position[1]/(resolution[1]-1)*255], [[10.0, 0.0], [0.0, 10.0]]) #TODO Adapt variance
                frame = rv.pdf(pos)
                frames[i, j, :, :] = torch.from_numpy(frame)
        for i2 in range(i+1, self.seq_len+self.heatup_seq_len):
            frames[i2] = frames[i]
        
        if self.sum_channels:
            frames = frames.sum(dim=1, keepdim=True)

        return (frames/frames.max() - 0.5)*2
    
    def __len__(self):
        length = 0
        for t in self.turns:
            sub = int(re.search("_[0-9]+", t)[0][1:]) - self.heatup_seq_len
            length += sub//self.seq_len + int(sub%self.seq_len != 0)
        return length

    def __getitem__(self, idx):

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