import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(16,32, (3,3), padding=1)
        self.conv2 = nn.Conv2d(32,64, (3,3), padding=1)
        self.conv3 = nn.Conv2d(64,128, (3,3), padding=1)
        self.conv4 = nn.Conv2d(128,128, (3,3), padding=1)

        self.lin1 = nn.Linear(128*16*16, 256)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.max_pool(out) #32, 128, 128

        out = self.conv2(out)
        out = self.relu(out)
        out = self.max_pool(out) #64, 64, 64

        out = self.conv3(out)
        out = self.relu(out)
        out = self.max_pool(out) #128, 32, 32

        out = self.conv4(out)
        out = self.relu(out)
        out = self.max_pool(out) #128, 16, 16

        out = out.view(-1, 128*16*16)
        out = self.lin1(out)
        out = self.relu(out) #256

        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.lin1 = nn.Linear(256, 16*16*128)

        self.conv1 = nn.Conv2d(128,128, (3,3), padding=1)
        self.conv2 = nn.Conv2d(128,64, (3,3), padding=1)
        self.conv3 = nn.Conv2d(64,32, (3,3), padding=1)
        self.conv4 = nn.Conv2d(32,16, (3,3), padding=1)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = out.view(-1, 128, 16, 16)

        out = self.upsample(out)
        out = self.conv1(out)
        out = self.relu(out)

        out = self.upsample(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = self.upsample(out)
        out = self.conv3(out)
        out = self.relu(out)

        out = self.upsample(out)
        out = self.conv4(out)
        out = self.sigmoid(out)

        return out

class Encoder_LSTM(nn.Module):
    def __init__(self):
        super(Encoder_LSTM, self).__init__()

        self.lstm = nn.LSTM(256, hidden_size=256, batch_first=True)
    
    def forward(self, x, hidden=None, cell=None):

        if hidden is not None:
            out, (h,c) = self.lstm(x, (hidden, cell))
        else:
            out, (h,c) = self.lstm(x)

        return out, (h,c)