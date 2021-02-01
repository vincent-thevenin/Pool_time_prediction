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
        #self.norm = nn.InstanceNorm2d(out_channels*4, affine=True)

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

        """h = list(torch.split(state[0], self.out_channels, dim=1))
        c = list(torch.split(state[1], self.out_channels, dim=1))"""

        h = [h_ for h_ in torch.split(state[0], self.out_channels, dim=1)]
        c = [c_ for c_ in torch.split(state[1], self.out_channels, dim=1)]

        h[0],c[0] = self.cells[0](x[:, 0], (h[0], c[0]))
        for layer in range(1, self.num_layers):
            h[layer],c[layer] = self.cells[layer](torch.clone(h[layer-1]), (h[layer], c[layer]))
        out = torch.clone(h[-1]).unsqueeze(1)


        for time in range(1,x.shape[1]):
            h[0],c[0] = self.cells[0](x[:, time], (h[0], c[0]))
            for layer in range(1, self.num_layers):
                h[layer],c[layer] = self.cells[layer](torch.clone(h[layer-1]), (h[layer], c[layer]))
            out = torch.cat((out, torch.clone(h[-1]).unsqueeze(1)), dim=1)

        h = torch.cat(h, dim=1)
        c = torch.cat(c, dim=1)

        return out, (h,c)

class PredRNNCell(nn.Module):
    def __init__(self, in_channels, out_channels, m_in_channels=None, m_out_channels=None):
        super(PredRNNCell, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.out_channels = out_channels
        if m_in_channels is None:
            m_in_channels = out_channels
        self.m_in_channels = m_in_channels
        if m_out_channels is None:
            m_out_channels = out_channels
        self.m_out_channels = m_out_channels

        #self.selfatt = SelfAttention(m_in_channels)
        self.normx = nn.BatchNorm2d(in_channels, affine=True)
        self.normm = nn.BatchNorm2d(m_in_channels, affine=True)
        self.normh = nn.BatchNorm2d(out_channels, affine=True)
        self.normc = nn.BatchNorm2d(out_channels, affine=True)

        self.conv_m_resize = nn.Conv2d(
            in_channels=m_in_channels,
            out_channels=m_out_channels,
            kernel_size=(1,1),
            padding=0
        )

        self.conv_h = nn.Conv2d(
            in_channels=in_channels+out_channels,
            out_channels=4*out_channels,
            kernel_size=(3,3),
            padding=1
        )

        self.conv_m = nn.Conv2d(
            in_channels=in_channels+m_out_channels,
            out_channels=3*m_out_channels,
            kernel_size=(3,3),
            padding=1
        )

        self.conv_out = nn.Conv2d(
            in_channels=out_channels+m_out_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
            padding=1
        )

        self.conv_11 = nn.Conv2d(
            in_channels=out_channels+m_out_channels,
            out_channels=out_channels,
            kernel_size=(1,1),
            padding=0
        )
    
    def forward(self, x, state):
        """
        x: Tensor(B, Cin, D, W)
        state: tuple(Tensor(B, Cout, D, W), Tensor(B, Cout, D, W)) or None
        H,C,M
        """
        h, c, m = state

        x = self.normx(x)
        m = self.normm(m)
        h = self.normh(h)
        c = self.normc(c)

        #m = self.selfatt(m)

        if self.m_in_channels != self.m_out_channels:
            m = self.conv_m_resize(m)
            m = self.tanh(m)

        bottom_h = torch.cat((x, h), dim=1) #cat channel-wise
        bottom_m = torch.cat((x, m), dim=1) #cat channel-wise

        i_h, f_h, g_h, o = torch.split(self.conv_h(bottom_h), self.out_channels, dim=1)

        i_m, f_m, g_m = torch.split(self.conv_m(bottom_m), self.m_out_channels, dim=1)

        i_h = self.sigmoid(i_h)
        f_h = self.sigmoid(f_h)
        g_h = self.tanh(g_h)
        c = c*f_h + i_h*g_h

        g_m = self.tanh(g_m)
        i_m = self.sigmoid(i_m)
        f_m = self.sigmoid(f_m)

        m = f_m*m + i_m*g_m
        o = self.sigmoid(o + self.conv_out(torch.cat((c, m), dim=1)))
        h = o*self.tanh(
            self.conv_11(
                torch.cat((c, m),dim=1)
            )
        )

        return h,c,m

class PredRNN(nn.Module):
    def __init__(self, in_channels, out_channels, m_in_channels=None, m_out_channels=None, num_layers=1):
        super(PredRNN, self).__init__()

        self.out_channels = out_channels
        if m_in_channels is None:
            m_in_channels = out_channels
        self.m_in_channels = m_in_channels
        if m_out_channels is None:
            m_out_channels = out_channels
        self.m_out_channels = m_out_channels

        self.num_layers = num_layers

        self.cells = [
            PredRNNCell(in_channels, out_channels, m_in_channels, m_out_channels)
        ]

        for _ in range(1, num_layers):
            self.cells.append(PredRNNCell(out_channels, out_channels, m_out_channels, m_out_channels))

        self.cells = nn.Sequential(
            *self.cells
        )

    def forward(self, x, state=None):
        """
        state: H,C,M
        """
        if state is None:
            state = (
                torch.zeros(
                    [x.shape[0], self.out_channels*self.num_layers, x.shape[-2], x.shape[-1]],
                    requires_grad=True
                ).to(x.device),
                torch.zeros(
                    [x.shape[0], self.out_channels*self.num_layers, x.shape[-2], x.shape[-1]],
                    requires_grad=True
                ).to(x.device),
                #x[:,0].expand(x.shape[0], self.m_in_channels, x.shape[-2], x.shape[-1])
                torch.zeros(
                    [x.shape[0], self.m_in_channels, x.shape[-2], x.shape[-1]],
                    requires_grad=True
                ).to(x.device)
            )
        if state[0] is None: #Only m is initialized
            state = (
                torch.zeros(
                    [x.shape[0], self.out_channels*self.num_layers, x.shape[-2], x.shape[-1]],
                    requires_grad=True
                ).to(x.device),
                torch.zeros(
                    [x.shape[0], self.out_channels*self.num_layers, x.shape[-2], x.shape[-1]],
                    requires_grad=True
                ).to(x.device),
                state[2]
            )

        """h = list(torch.split(state[0], self.out_channels, dim=1))
        c = list(torch.split(state[1], self.out_channels, dim=1))"""

        h = [h_ for h_ in torch.split(state[0], self.out_channels, dim=1)]
        c = [c_ for c_ in torch.split(state[1], self.out_channels, dim=1)]
        m = state[2]

        h[0],c[0],m = self.cells[0](x[:, 0], (h[0], c[0], m))
        for layer in range(1, self.num_layers-1):
            h[layer], c[layer], m = self.cells[layer](h[layer-1], (h[layer], c[layer], m))
        """if self.num_layers != 1:
            h[-1], c[-1], m = self.cells[-1](h[-1-1], (h[-1], c[-1], m))"""
        out = torch.clone(h[-1]).unsqueeze(1)


        for time in range(1,x.shape[1]):
            h[0],c[0],m = self.cells[0](x[:, time], (h[0], c[0],m))
            for layer in range(1, self.num_layers-1):
                h[layer],c[layer],m = self.cells[layer](h[layer-1], (h[layer], c[layer], m))
            """if self.num_layer != 1:
                h[-1],c[-1],m = self.cells[-1](h[-1-1], (h[-1], c[-1], m))"""
            out = torch.cat((out, torch.clone(h[-1]).unsqueeze(1)), dim=1)

        h = torch.cat(h, dim=1)
        c = torch.cat(c, dim=1)

        return out, (h,c,m)

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega: float = 14, theta: float = 0.5, eps: float = 1, alpha: float = 2.1) -> None:
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.eps = eps
        self.alpha = alpha

    def forward(self, _y: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        delta_y = torch.abs(_y - y) if mask is None else mask * torch.abs(_y - y)
        
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        
        loss1 = self.omega * torch.log(1 + (delta_y1 / self.omega) ** (self.alpha - y1))

        A = self.omega * (1 / (1 + (self.theta / self.eps) ** (self.alpha - y2))) * (self.alpha - y2) * ((self.theta / self.eps) ** (self.alpha - y2 - 1)) * (1 / self.eps)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.eps, self.alpha - y2))
        loss2 = A * delta_y2 - C

        return (torch.sum(loss1) + torch.sum(loss2)) / (len(loss1) + len(loss2))

class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()
        
        #conv f
        self.conv_f = nn.Conv2d(in_channel, max(1,in_channel//8), 1)
        #conv_g
        self.conv_g = nn.Conv2d(in_channel, max(1,in_channel//8), 1)
        #conv_h
        self.conv_h = nn.Conv2d(in_channel, in_channel, 1)
        
        self.softmax = nn.Softmax(-2) #sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x) #BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x) #BxC'xHxW
        h_projection = self.conv_h(x) #BxCxHxW
        
        f_projection = torch.transpose(f_projection.view(B,-1,H*W), 1, 2) #BxNxC', N=H*W
        g_projection = g_projection.view(B,-1,H*W) #BxC'xN
        h_projection = h_projection.view(B,-1,H*W) #BxCxN
        
        attention_map = torch.bmm(f_projection, g_projection) #BxNxN
        attention_map = self.softmax(attention_map) #sum_i_N (A i,j) = 1
        
        #sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map) #BxCxN
        out = out.view(B,C,H,W)
        
        out = self.gamma*out + x
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(2)
        self.tanh = nn.Tanh()
        self.norm = nn.InstanceNorm2d(8, affine=True)
        
        self.conv1 = nn.Conv2d(3,8,3,padding=1)
        self.conv2 = nn.Conv2d(8,16,3,padding=1)
        self.conv3 = nn.Conv2d(16,8,3,padding=1)
        self.conv4 = nn.Conv2d(8,1,3,padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.norm(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = out.mean(dim=-1).mean(dim=-1)
        out = self.tanh(out)

        return out

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride:int = 1, depthwise = True):
        super(MBConv, self).__init__()
        self.stride = stride
        if stride==1:
            self.relu6 = nn.ReLU6(inplace=False)
        else:
            self.relu6 = nn.ReLU6(inplace=True)

        if in_channels != out_channels:
            self.convr = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.convr = lambda x: x

        self.conv1 = nn.Conv2d(in_channels, in_channels*6, 1)
        if depthwise:
            self.conv2 = nn.Conv2d(in_channels*6, in_channels*6, kernel_size, padding=(kernel_size-1)//2, groups=in_channels*6, stride=stride)
        else:
            self.conv2 = nn.Conv2d(in_channels*6, in_channels*6, kernel_size, padding=(kernel_size-1)//2, groups=1, stride=stride)
        self.conv3 = nn.Conv2d(in_channels*6, out_channels, 1)


    def forward(self, x):
        res = self.relu6(self.conv1(x))
        res = self.relu6(self.conv2(res))
        res = self.conv3(res)

        x = self.convr(x)

        return res + x if self.stride == 1 else res

def adaIN(feature, mean_style, std_style, eps = 1e-5):
    B,C,H,W = feature.shape
    
    feature = feature.view(B,C,-1)
            
    std_feat = (torch.std(feature, dim = 2) + eps).view(B,C,1)
    mean_feat = torch.mean(feature, dim = 2).view(B,C,1)
    
    mean_style = mean_style.view(B, -1, 1)
    std_style = std_style.view(B, -1, 1)

    adain = std_style * (feature - mean_feat)/std_feat + mean_style
    
    adain = adain.view(B,C,H,W)
    return adain

class ResOpFlow(nn.Module):
    def __init__(self, seq_len=10):
        super(ResOpFlow, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        #opflow
        self.c1_1 = nn.Conv2d(seq_len-1, 32, 3, padding=1) #64
        self.c1_2 = MBConv(32, 64, 3) #64
        self.norm1_2 = nn.BatchNorm2d(64)
        self.c1_3 = MBConv(64, 128, 3)
        self.norm1_3 = nn.BatchNorm2d(128)
        self.att_1 = SelfAttention(128)
        
        # final
        self.c2_1 = nn.Conv2d(1, 32, 3, padding=1) #64
        self.c2_2 = MBConv(32, 64, 3, stride=1)
        self.norm2_2 = nn.BatchNorm2d(64)
        self.c2_3 = MBConv(64, 128, 3, stride=1)
        self.norm2_3 = nn.BatchNorm2d(128)

        # common
        self.c3_1 = MBConv(128, 128, 3, depthwise=False)
        self.norm3_1 = nn.BatchNorm2d(128)
        self.c3_2 = MBConv(128, 64, 3, depthwise=False)
        self.norm3_2 = nn.BatchNorm2d(64)
        self.c3_3 = nn.Conv2d(64, 32, 3, padding=1)
        self.c3_4 = nn.Conv2d(32, 1, 3, padding=1)
        

    def forward(self, seq):
        seq, final = seq[:, :-1], seq[:, -1:]

        seq = self.relu(self.c1_1(seq))
        seq = self.c1_2(seq)
        seq = self.norm1_2(seq)
        seq = self.c1_3(seq)
        seq = self.norm1_3(seq)
        seq = self.att_1(seq)

        final = self.relu(self.c2_1(final))
        final = self.c2_2(final)
        final = self.norm2_2(final)
        final = self.c2_3(final)
        final = self.norm2_3(final)

        #final = torch.cat((seq, final), dim=1)
        final = seq + final

        final = self.relu(self.c3_1(final))
        final = self.norm3_1(final)
        final = self.relu(self.c3_2(final))
        final = self.norm3_2(final)
        final = self.relu(self.c3_3(final))
        final = self.tanh(self.c3_4(final))

        return final, seq.detach()

class MBDisc(nn.Module):
    def __init__(self):
        super(MBDisc, self).__init__()
        self.relu = nn.LeakyReLU()

        #opflow
        self.c1 = nn.Conv2d(1, 16, 3, padding=1) #64
        self.att = SelfAttention(16)
        self.c2 = MBConv(16, 32, 3, stride=2) #32
        self.norm1 = nn.BatchNorm2d(32)
        self.c3 = MBConv(32, 64, 3, stride=2) #16
        self.norm2 = nn.BatchNorm2d(64)
        self.c4 = MBConv(64, 128, 3, stride=2) #8
        self.c5 = MBConv(128, 256, 3, stride=2) #4
        self.c6 = nn.Conv2d(256, 512, 4)

        #fc
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
    def forward(self, x):
        
        x = self.relu(self.c1(x))
        x = self.att(x)
        x = self.c2(x)
        x = self.norm1(x)
        x = self.c3(x)
        x = self.norm2(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.relu(self.c6(x))

        x = x.mean(dim=-1).mean(dim=-1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x