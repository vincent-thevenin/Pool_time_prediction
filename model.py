import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.relu = nn.LeakyReLU()
        self.max_pool = nn.AvgPool2d(2)

        self.conv1 = nn.Conv2d(1,32, (3,3), padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(32,64, (3,3), padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(64,128, (3,3), padding=1, padding_mode='zeros')
        self.conv4 = nn.Conv2d(128,128, (3,3), padding=1, padding_mode='zeros')
        self.conv5 = nn.Conv2d(128,128, (3,3), padding=1, padding_mode='zeros')

        self.lin1 = nn.Linear(128*8*8, 2048)

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

        out = self.conv5(out)
        out = self.relu(out)
        out = self.max_pool(out) #128, 8, 8

        out = out.reshape(-1, 128*8*8)
        out = self.lin1(out)
        out = self.relu(out) #256

        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.lin1 = nn.Linear(2048, 16*16*128)

        self.conv1 = nn.Conv2d(128,128, (3,3), padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(128,64, (3,3), padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(64,32, (3,3), padding=1, padding_mode='zeros')
        self.conv4 = nn.Conv2d(32,1, (3,3), padding=1, padding_mode='zeros')
        self.conv5 = nn.Conv2d(1,1, (3,3), padding=1, padding_mode='zeros')

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = out.reshape(-1, 128, 16, 16)

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
        out = self.relu(out)

        out = self.conv5(out)
        out = self.sigmoid(out)

        return out

class Encoder_LSTM(nn.Module):
    def __init__(self):
        super(Encoder_LSTM, self).__init__()

        self.lstm = nn.LSTM(2048, hidden_size=2048, batch_first=True)
    
    def forward(self, x, hidden=None, cell=None):

        if hidden is not None:
            out, (h,c) = self.lstm(x, (hidden, cell))
        else:
            out, (h,c) = self.lstm(x)

        return out, (h,c)

class Conv3D(nn.Module):
    def __init__(self):
        super(Conv3D, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv3d(
            in_channels=16,
            out_channels=16,
            kernel_size=(3, 3, 3), #D, H, W (kernel)
            padding=(0,1,1),
            #padding_mode='zeros'
        )

    def forward(self, seq):
        return self.sigmoid(self.conv1(seq))
    

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLSTMCell, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        """self.w_xi = nn.Conv2d(in_channels, out_channels, (3,3), padding=1, padding_mode='zeros')
        self.w_hi = 
        self.w_ci = 
        self.b_i  =

        self.w_xf =
        self.w_hf =
        self.w_cf =
        self.b_f  =

        self.w_xc =
        self.w_hc =
        self.b_c =

        self.w_xo =
        self.w_ho =
        self.w_co =
        self.b_f  ="""
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels=in_channels+out_channels,
            out_channels=4*out_channels,
            kernel_size=(3,3),
            padding=1
        )

    def forward(self, x, state):
        """
        x: Tensor(B, Cin, D, W)
        state: tuple(Tensor(B, Cout, D, W), Tensor(B, Cout, D, W)) or None
        """
        h, c = state

        bottom = torch.cat((x, h), dim=1) #cat channel-wise

        i, f, g, o = torch.split(self.conv(bottom), self.out_channels, dim=1)

        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = self.tanh(g)
        o = self.sigmoid(o)

        c = c*f + i*g
        h = o*self.tanh(c)

        return h,c

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1):
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.cells = [
            ConvLSTMCell(in_channels, out_channels)
        ]

        for _ in range(1, num_layers):
            self.cells.append(ConvLSTMCell(out_channels, out_channels))

        self.cells = nn.Sequential(
            *self.cells
        )

    def forward(self, x, state=None):
        if state is None:
            state = (
                torch.zeros(
                    [x.shape[0], self.out_channels*self.num_layers, x.shape[-2], x.shape[-1]],
                    requires_grad=True
                ).to(x.device),
                torch.zeros(
                    [x.shape[0], self.out_channels*self.num_layers, x.shape[-2], x.shape[-1]],
                    requires_grad=True
                ).to(x.device)
            )

        h = list(torch.split(state[0], self.out_channels, dim=1))
        c = list(torch.split(state[1], self.out_channels, dim=1))


        h[0],c[0] = self.cells[0](x[:, 0], (h[0], c[0]))
        for layer in range(1, self.num_layers):
            h[layer],c[layer] = self.cells[layer](h[layer-1], (h[layer], c[layer]))
        out = h[-1].unsqueeze(1)


        for time in range(1,x.shape[1]):
            h[0],c[0] = self.cells[0](x[:, time], (h[0], c[0]))
            for layer in range(1, self.num_layers):
                h[layer],c[layer] = self.cells[layer](h[layer-1], (h[layer], c[layer]))
            out = torch.cat((out, h[-1].unsqueeze(1)), dim=1)

        h = torch.cat(h, dim=1)
        c = torch.cat(c, dim=1)

        return out, (h,c)