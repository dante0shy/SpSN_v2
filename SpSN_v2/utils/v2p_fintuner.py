from torch import nn
import torch


class V2PFinetuner(nn.Module):
    def __init__(self, out_channels):
        nn.Module.__init__(self)
        self.out_finetune_mlp = nn.Sequential(
            torch.nn.Linear(3 + out_channels, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 64),
            nn.ReLU(),
            torch.nn.Linear(64, out_channels),
        )
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def forward(self, x):
        return self.out_finetune_mlp(x)


class V2PFinetuner_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)
        self.out_finetune_mlp = nn.Sequential(
            torch.nn.Linear(in_channels, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 64),
            nn.ReLU(),
            torch.nn.Linear(64, out_channels),
        )
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def forward(self, x):
        return self.out_finetune_mlp(x)


class V2PFinetuner_v2(nn.Module):
    def __init__(self, in_channels, out_channels, k=4):
        nn.Module.__init__(self)
        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(in_channels, 64), nn.ReLU(),
        )
        self.out_finetune_mlp_conv = nn.Sequential(
            torch.nn.Conv1d(64, 64, k), nn.ReLU(),
        )
        self.out_finetune_mlp_tune = torch.nn.Linear(64, 64)

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(64, 64), nn.ReLU(), torch.nn.Linear(64, out_channels)
        )
        self.k = k
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def forward(self, x):
        tmp_in = self.out_finetune_mlp_in(x)
        tmp = self.out_finetune_mlp_tune(
            self.out_finetune_mlp_conv(tmp_in.view(-1, 64, self.k)).view(-1, 64)
        ) + tmp_in[:, :1, :].view(-1, 64)
        return self.out_finetune_mlp_out(tmp)


class V2PFinetuner_v3(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=4):
        nn.Module.__init__(self)
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.out_finetune_mlp_offset = nn.Sequential(torch.nn.Linear(3, 64), nn.Tanh(),)

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(f_channels, 64), nn.ReLU(),
        )
        self.out_finetune_mlp_conv = nn.Sequential(
            torch.nn.Conv1d(64, 64, k), nn.ReLU(),
        )
        self.out_finetune_mlp_tune = torch.nn.Linear(64, 64)

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(64, 64), nn.ReLU(), torch.nn.Linear(64, out_channels)
        )
        self.k = k
        self.dist_range = 5
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def forward(self, x, d):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        pre = x[:, :, : self.pre_channels]
        feature = x[:, :, self.pre_channels : self.f_channels + self.pre_channels]
        offset = x[:, :, self.f_channels + self.pre_channels :]
        offset_w = self.d_weight(offset)
        tmp_in = self.out_finetune_mlp_in(feature)
        offset_w = self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
            self.out_finetune_mlp_conv((tmp_in * offset_w).view(-1, 64, self.k)).view(
                -1, 64
            )
        ) + tmp_in[:, :1, :].view(-1, 64)

        return self.out_finetune_mlp_out(tmp) + pre[:, :1, :].view(
            -1, self.pre_channels
        )


class V2PFinetuner_v4(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=4, main_channel=96):
        nn.Module.__init__(self)
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(f_channels, self.main_channel), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_conv = nn.Sequential(
            torch.nn.Conv1d(self.main_channel, self.main_channel, k), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_tune = torch.nn.Linear(
            self.main_channel, self.main_channel, bias=False
        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def forward(self, x, d):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        pre = x[:, :, : self.pre_channels]
        feature = x[:, :, self.pre_channels : self.f_channels + self.pre_channels]
        offset = x[:, :, self.f_channels + self.pre_channels :]
        offset_w = self.d_weight(offset)
        tmp_in = self.out_finetune_mlp_in(feature)
        offset_w = self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
            self.out_finetune_mlp_conv(
                (tmp_in * offset_w).view(-1, self.main_channel, self.k)
            ).view(-1, self.main_channel)
        ) + tmp_in[:, :1, :].view(-1, self.main_channel)

        return self.out_finetune_mlp_out(tmp) + pre[:, :1, :].view(
            -1, self.pre_channels
        )


class V2PFinetuner_v5(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6, main_channel=64):
        nn.Module.__init__(self)
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(f_channels, self.main_channel), nn.LeakyReLU(),
        )
        # self.out_finetune_mlp_conv = nn.Sequential(
        #     torch.nn.Conv1d(self.main_channel, self.main_channel, k), nn.LeakyReLU(),
        # )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

            # torch.nn.Linear(
            # self.main_channel, self.main_channel#, bias=False
        # )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def forward(self, x, d):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        pre = x[:, :, : self.pre_channels]
        feature = x[:, :, self.pre_channels : self.f_channels + self.pre_channels]
        offset = x[:, :, self.f_channels + self.pre_channels :]
        offset_w = self.d_weight(offset)
        tmp_in = self.out_finetune_mlp_in(feature)
        offset_w =  self.out_finetune_mlp_offset(offset_w)
        # tmp = self.out_finetune_mlp_tune(
        #     self.out_finetune_mlp_conv(
        #         (tmp_in * offset_w).view(-1, self.main_channel, self.k)
        #     ).view(-1, self.main_channel)
        # ) + tmp_in[:, :1, :].view(-1, self.main_channel)
        tmp = self.out_finetune_mlp_tune(
                (tmp_in * offset_w).max(dim= 1)[0]
        ) + tmp_in[:,0,:]#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) #+ pre[:, :1, :].view( -1, self.pre_channels)

class V2PFinetuner_v5_1(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6):
        nn.Module.__init__(self)
        main_channel=f_channels
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(f_channels, self.main_channel), nn.LeakyReLU(),
        )
        # self.out_finetune_mlp_conv = nn.Sequential(
        #     torch.nn.Conv1d(self.main_channel, self.main_channel, k), nn.LeakyReLU(),
        # )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

            # torch.nn.Linear(
            # self.main_channel, self.main_channel#, bias=False
        # )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def forward(self, x, d):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        pre = x[:, :, : self.pre_channels]
        feature = x[:, :, self.pre_channels : self.f_channels + self.pre_channels]
        offset = x[:, :, self.f_channels + self.pre_channels :]
        offset_w = self.d_weight(offset)
        tmp_in = self.out_finetune_mlp_in(feature)
        offset_w =  self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
                (tmp_in * offset_w)
        ).max(dim= 1)[0] + feature[:,0,:]#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) #+ pre[:, :1, :].view( -1, self.pre_channels)

class V2PFinetuner_v6(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6, main_channel=64):
        nn.Module.__init__(self)
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel*2), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(f_channels, self.main_channel), nn.LeakyReLU(),
        )
        # self.out_finetune_mlp_conv = nn.Sequential(
        #     torch.nn.Conv1d(self.main_channel, self.main_channel, k), nn.LeakyReLU(),
        # )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel *2, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def offset_trans(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def forward(self, x, d):
        pre = x[:, :, : self.pre_channels]
        feature = x[:, :, self.pre_channels : self.f_channels + self.pre_channels]
        offset = x[:, :, self.f_channels + self.pre_channels :]
        offset_w = offset#self.d_weight(offset)
        tmp_in = self.out_finetune_mlp_in(feature)
        offset_w =  self.out_finetune_mlp_offset(offset_w)

        tmp = self.out_finetune_mlp_tune(
                ( offset_w* torch.cat((tmp_in,tmp_in-tmp_in[:,:1,:]),dim=2))
        ).max(dim= 1)[0] + tmp_in[:,0,:]#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) #+ pre[:, :1, :].view( -1, self.pre_channels)


class V2PFinetuner_v7(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6, main_channel=64):
        nn.Module.__init__(self)
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.out_finetune_mlp_offset = nn.Sequential(
            # nn.Tanh(),
            torch.nn.Linear(3, self.main_channel),
            nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(f_channels, self.main_channel), nn.LeakyReLU(),
        )
        # self.out_finetune_mlp_conv = nn.Sequential(
        #     torch.nn.Conv1d(self.main_channel, self.main_channel, k), nn.LeakyReLU(),
        # )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel , self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

        self.out_finetune_mlp_diff_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel , self.main_channel),
            nn.LeakyReLU(),
        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),

        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def offset_trans(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def forward(self, x, d):
        pre = x[:, :, : self.pre_channels]
        feature = x[:, :, self.pre_channels : self.f_channels + self.pre_channels]
        offset = x[:, :, self.f_channels + self.pre_channels :]

        offset_w = torch.tanh(offset)
        offset_v = 1- offset_w[:,:,:2]
        offset_z = offset_w[:,:,2:]#self.d_weight(offset)
        tmp_in = self.out_finetune_mlp_in(feature)
        offset_w =  self.out_finetune_mlp_offset(torch.cat([offset_v,offset_z],dim=-1))

        tmp = self.out_finetune_mlp_tune(
                ( offset_w* (tmp_in+self.out_finetune_mlp_diff_tune(tmp_in-tmp_in[:,:1,:])))
        ).max(dim= 1)[0] + tmp_in[:,0,:]#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) + pre[:, :1, :].view( -1, self.pre_channels)

class V2PFinetuner_v7_1(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6, main_channel=64):
        nn.Module.__init__(self)
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.out_finetune_mlp_offset = nn.Sequential(
            # nn.Tanh(),
            torch.nn.Linear(3, self.main_channel),
            nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(f_channels, self.main_channel), nn.LeakyReLU(),
        )
        # self.out_finetune_mlp_conv = nn.Sequential(
        #     torch.nn.Conv1d(self.main_channel, self.main_channel, k), nn.LeakyReLU(),
        # )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel , self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

        self.out_finetune_mlp_diff_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel , self.main_channel),
            nn.LeakyReLU(),
        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),

        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def offset_trans(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def forward(self, x, d):
        pre = x[:, :, : self.pre_channels]
        feature = x[:, :, self.pre_channels : self.f_channels + self.pre_channels]
        offset = x[:, :, self.f_channels + self.pre_channels :]

        offset_w = torch.tanh(offset)
        offset_v = offset_w[:,:,:2]#1-
        offset_z = offset_w[:,:,2:]#self.d_weight(offset)
        tmp_in = self.out_finetune_mlp_in(feature)
        offset_w =  self.out_finetune_mlp_offset(torch.cat([offset_v,offset_z],dim=-1))

        tmp = self.out_finetune_mlp_tune(
                ( offset_w* (tmp_in+self.out_finetune_mlp_diff_tune(tmp_in-tmp_in[:,:1,:])))
        ).max(dim= 1)[0] + tmp_in[:,0,:]#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) + pre[:, :1, :].view( -1, self.pre_channels)

class V2PFinetuner_v4_01(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=4, main_channel=96):
        nn.Module.__init__(self)
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(f_channels, self.main_channel), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_conv = nn.Sequential(
            torch.nn.Conv1d(self.main_channel, self.main_channel, k), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_tune = torch.nn.Linear(
            self.main_channel, self.main_channel, bias=False
        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def forward(self, x, d):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        pre = x[:, :, : self.pre_channels]
        feature = x[:, :, self.pre_channels : self.f_channels + self.pre_channels]
        offset = x[:, :, self.f_channels + self.pre_channels :]
        offset_w = self.d_weight(offset)
        tmp_in = self.out_finetune_mlp_in(feature)
        offset_w = self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
            self.out_finetune_mlp_conv(
                (tmp_in * offset_w).view(-1, self.main_channel, self.k)
            ).view(-1, self.main_channel)
        ) + tmp_in[:, :1, :].view(-1, self.main_channel)

        return self.out_finetune_mlp_out(tmp) + pre[:, :1, :].view(
            -1, self.pre_channels
        )


class V2PFinetuner_v5_2(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6,n_class = 26):
        nn.Module.__init__(self)
        main_channel=f_channels
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.n_class = n_class
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(n_class, self.main_channel), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def forward(self, x,x_sur, d):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        # pre = x[:, : -self.n_class]
        # pre_pre = x[:, -self.n_class:]
        pre = x[:, self.n_class: ]
        pre_pre = x[:,  :self.n_class]
        feature_sur = x_sur[:, :,  : self.n_class]
        offset_sur = x_sur[:, :, self.n_class:]
        offset_w = torch.tanh(offset_sur)
        offset_v = offset_w[:, :, :2]  # 1-
        offset_z = offset_w[:, :, 2:]  #
        offset_w = torch.cat([offset_v, offset_z], dim=-1)
        # offset_w = self.d_weight(offset_w)
        tmp_in = self.out_finetune_mlp_in(feature_sur)
        offset_w =  self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
                (tmp_in * offset_w)
        ).max(dim= 1)[0] + pre#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) +pre_pre#+ pre[:, :1, :].view( -1, self.pre_channels)


class V2PFinetuner_v5_3(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6,n_class = 26):
        nn.Module.__init__(self)
        main_channel=f_channels
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.n_class = n_class
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(n_class, self.main_channel), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)
    def rloc_trans(self,pos):
        offset_w = torch.tanh(pos)
        offset_v = (1 - torch.relu(offset_w[:, :, :2]))+  (-1 + torch.relu(-offset_w[:, :, :2])) # 1-
        offset_z = offset_w[:, :, 2:]  #
        offset_w = torch.cat([offset_v, offset_z], dim=-1)
        return offset_w


    def forward(self, x,x_sur, d):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        pre = x[:, : ,:self.n_class]
        pre_pre = x[:, :, self.n_class:-3]
        pre_loc = x[:, :, -3:]
        feature_sur = x_sur[:, :,  : self.n_class]
        offset_sur = x_sur[:, :, self.n_class:]
        offset_w = torch.tanh(offset_sur)
        offset_v = offset_w[:, :, :2]  # 1-
        offset_z = offset_w[:, :, 2:]  #
        offset_w = torch.cat([offset_v, offset_z], dim=-1)
        # offset_w = self.d_weight(offset_w)
        tmp_in = self.out_finetune_mlp_in(feature_sur)
        offset_w =  self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
                (tmp_in * offset_w)
        ).max(dim= 1)[0] + pre#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) +pre_pre#+ pre[:, :1, :].view( -1, self.pre_channels)

class V2PFinetuner_v5_4(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6,n_class = 26):
        nn.Module.__init__(self)
        main_channel=f_channels
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.n_class = n_class
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(n_class, self.main_channel), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def rloc_trans(self,pos):
        offset_w = torch.tanh(pos)
        offset_w = (1 - torch.relu(offset_w))+  (-1 + torch.relu(-offset_w)) # 1-
        # offset_z = offset_w[:, :, 2:]  #
        # offset_w = torch.cat([offset_v, offset_z], dim=-1)
        return offset_w

    def forward(self, x,x_sur, d=None):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        # pre = x[:, : -self.n_class]
        # pre_pre = x[:, -self.n_class:]
        pre = x[:, self.n_class: ]
        pre_pre = x[:,  :self.n_class]
        feature_sur = x_sur[:, :,  : self.n_class]
        offset_sur = x_sur[:, :, self.n_class:]
        offset_w = self.rloc_trans(offset_sur)
        # offset_v = offset_w[:, :, :2]  # 1-
        # offset_z = offset_w[:, :, 2:]  #
        # offset_w = torch.cat([offset_v, offset_z], dim=-1)
        # offset_w = self.d_weight(offset_w)
        tmp_in = self.out_finetune_mlp_in(feature_sur)
        offset_w =  self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
                (tmp_in * offset_w)
        ).max(dim= 1)[0] + pre#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) +pre_pre


class V2PFinetuner_v5_4_cp(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6,n_class = 26):
        nn.Module.__init__(self)
        main_channel=f_channels
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.n_class = n_class
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(n_class, self.main_channel), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def rloc_trans(self,pos):
        offset_w = torch.tanh(pos)
        offset_w = (1 - torch.relu(offset_w))+  (-1 + torch.relu(-offset_w)) # 1-
        # offset_z = offset_w[:, :, 2:]  #
        # offset_w = torch.cat([offset_v, offset_z], dim=-1)
        return offset_w

    def forward(self, x,x_sur, d=None):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        # pre = x[:, : -self.n_class]
        # pre_pre = x[:, -self.n_class:]
        pre = x[:, self.n_class: ]
        pre_pre = x[:,  :self.n_class]
        feature_sur = x_sur[:, :,  : self.n_class]
        # offset_sur = x_sur[:, :, self.n_class:]
        # offset_w = self.rloc_trans(offset_sur)
        # offset_v = offset_w[:, :, :2]  # 1-
        # offset_z = offset_w[:, :, 2:]  #
        # offset_w = torch.cat([offset_v, offset_z], dim=-1)
        # offset_w = self.d_weight(offset_w)
        tmp_in = self.out_finetune_mlp_in(feature_sur)
        # offset_w =  self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
                tmp_in
        ).mean(dim= 1) + pre#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp)


class V2PFinetuner_v5_4_cp_2(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6,n_class = 26):
        nn.Module.__init__(self)
        main_channel=f_channels
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.n_class = n_class
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(n_class, self.main_channel), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def rloc_trans(self,pos):
        offset_w = torch.tanh(pos)
        offset_w = (1 - torch.relu(offset_w))+  (-1 + torch.relu(-offset_w)) # 1-
        # offset_z = offset_w[:, :, 2:]  #
        # offset_w = torch.cat([offset_v, offset_z], dim=-1)
        return offset_w

    def forward(self, x,x_sur, d=None):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        # pre = x[:, : -self.n_class]
        # pre_pre = x[:, -self.n_class:]
        pre = x[:, self.n_class: ]
        pre_pre = x[:,  :self.n_class]
        feature_sur = x_sur[:, :,  : self.n_class]
        # offset_sur = x_sur[:, :, self.n_class:]
        # offset_w = self.rloc_trans(offset_sur)
        # offset_v = offset_w[:, :, :2]  # 1-
        # offset_z = offset_w[:, :, 2:]  #
        # offset_w = torch.cat([offset_v, offset_z], dim=-1)
        # offset_w = self.d_weight(offset_w)
        tmp_in = self.out_finetune_mlp_in(feature_sur)
        # offset_w =  self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
            tmp_in
        ).max(dim= 1)[0] + pre#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) +pre_pre

class V2PFinetuner_v5_4_cp_3(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6,n_class = 26):
        nn.Module.__init__(self)
        main_channel=f_channels
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.n_class = n_class
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(n_class, self.main_channel), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def rloc_trans(self,pos):
        offset_w = torch.tanh(pos)
        offset_w = (1 - torch.relu(offset_w))+  (-1 + torch.relu(-offset_w)) # 1-
        # offset_z = offset_w[:, :, 2:]  #
        # offset_w = torch.cat([offset_v, offset_z], dim=-1)
        return offset_w

    def forward(self, x,x_sur, d=None):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        # pre = x[:, : -self.n_class]
        # pre_pre = x[:, -self.n_class:]
        pre = x[:, self.n_class: ]
        pre_pre = x[:,  :self.n_class]
        feature_sur = x_sur[:, :,  : self.n_class]
        # offset_sur = x_sur[:, :, self.n_class:]
        # offset_w = self.rloc_trans(offset_sur)
        # offset_v = offset_w[:, :, :2]  # 1-
        # offset_z = offset_w[:, :, 2:]  #
        # offset_w = torch.cat([offset_v, offset_z], dim=-1)
        # offset_w = self.d_weight(offset_w)
        tmp_in = self.out_finetune_mlp_in(feature_sur)
        # offset_w =  self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
            tmp_in
        ).max(dim= 1)[0] #tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) +pre_pre

class V2PFinetuner_v5_5(nn.Module):
    def __init__(self, pre_channels, f_channels, out_channels, k=6,n_class = 26):
        nn.Module.__init__(self)
        main_channel=f_channels
        self.pre_channels = pre_channels
        self.f_channels = f_channels
        self.out_channels = out_channels
        self.main_channel = main_channel
        self.n_class = n_class
        self.out_finetune_mlp_offset = nn.Sequential(
            torch.nn.Linear(3, self.main_channel, bias=False), nn.Tanh(),
        )

        self.out_finetune_mlp_in = nn.Sequential(
            torch.nn.Linear(n_class * 2, self.main_channel), nn.LeakyReLU(),

            torch.nn.Linear(self.main_channel, self.main_channel), nn.LeakyReLU(),
        )
        self.out_finetune_mlp_tune = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),

        )

        self.out_finetune_mlp_out = nn.Sequential(
            torch.nn.Linear(self.main_channel, self.main_channel),
            nn.LeakyReLU(),
            torch.nn.Linear(self.main_channel, out_channels, bias=False),
        )
        self.k = k
        self.dist_range = 5
        # self.init()
        # self.out_finetune_mlp = torch.nn.Linear(3 + out_channels, out_channels)

    def init(self):
        for p in self.parameters():
            print(p)
            torch.nn.init.kaiming_uniform_(p)#, gain=1.0

    def d_weight(self, d):
        return torch.clip(1 - d / self.dist_range, min=0)

    def offset_weight(self, d):

        pn = ((d >= 0) - 0.5) * 2
        return pn * torch.clip(1 - torch.abs(d) / self.dist_range, min=0)

    def rloc_trans(self,pos):
        offset_w = torch.tanh(pos)
        offset_w = (1 - torch.relu(offset_w))+  (-1 + torch.relu(-offset_w)) # 1-
        # offset_z = offset_w[:, :, 2:]  #
        # offset_w = torch.cat([offset_v, offset_z], dim=-1)
        return offset_w

    def forward(self, x,x_sur, d):
        # d_w = self.d_weight(torch.Tensor(d).cuda())
        # pre = x[:, : -self.n_class]
        # pre_pre = x[:, -self.n_class:]
        pre = x[:, self.n_class: ]
        pre_pre = x[:,  :self.n_class]
        feature_sur = x_sur[:, :,  : self.n_class]
        offset_sur = x_sur[:, :, self.n_class:]
        offset_w = self.rloc_trans(offset_sur)
        # offset_v = offset_w[:, :, :2]  # 1-
        # offset_z = offset_w[:, :, 2:]  #
        # offset_w = torch.cat([offset_v, offset_z], dim=-1)
        # offset_w = self.d_weight(offset_w)
        tmp_in = self.out_finetune_mlp_in(
                torch.cat(
                    [
                        feature_sur,
                        feature_sur - torch.softmax(pre_pre,dim=1).view(-1,1,self.n_class)
                    ],
                    dim=2))
        offset_w = self.out_finetune_mlp_offset(offset_w)
        tmp = self.out_finetune_mlp_tune(
                (tmp_in * offset_w)
        ).max(dim= 1)[0] + pre#tmp_in[:, :1, :].view(-1, self.main_channel) +

        return self.out_finetune_mlp_out(tmp) +pre_pre