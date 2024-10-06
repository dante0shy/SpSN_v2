from SpSN_v2.core.SpSq_unet_squantial import (
    MinkUNetDecoder,
    MinkUNetEncoder_v2,
    MinkUNetEncoder_spsqNet,
)
from torch import nn
import torch
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
import MinkowskiEngine as ME
from SpSN_v2.utils.spsqModule import GCA as GCA
from SpSN_v2.utils.spsqModule import CLI_v3 as CLI
from SpSN_v2.core.resnet import ResNetBase


class Model(nn.Module):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    # PLANES = (16, 32, 64, 128, 64, 64, 48, 48)
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

    def __init__(self, in_channels, out_channels, D, frames=1):
        nn.Module.__init__(self)
        self.surpport_encoder = MinkUNetEncoder_v2(
            in_channels=in_channels,
            out_channels=out_channels,
            D=3,
            BLOCK=self.BLOCK,
            PLANES=self.PLANES,
            LAYERS=self.LAYERS,

        )
        self.main_encoder = MinkUNetEncoder_spsqNet_v2(
            in_channels=in_channels,
            out_channels=out_channels,
            D=3,
            BLOCK=self.BLOCK,
            PLANES=self.PLANES,
            LAYERS=self.LAYERS,

        )
        # self.surpport_encoder.freez_param()
        self.surppt_tune = ME.MinkowskiConvolution(
            self.PLANES[3], self.PLANES[3], kernel_size=1, dimension=D
        )
        self.surppt_tune_agg = ME.MinkowskiConvolution(
            self.PLANES[3], self.PLANES[3], kernel_size=3, dimension=D
        )
        self.s_attrntion_tune = []
        # self.s_attrntion_tune_1 = ME.MinkowskiConvolution(
        #     self.PLANES[0], self.PLANES[0], kernel_size=1, stride=1, dimension=D)
        self.s_attrntion_tune_2 = ME.MinkowskiConvolution(
            self.PLANES[0], self.PLANES[0], kernel_size=1, stride=1, dimension=D
        )
        self.s_attrntion_tune_3 = ME.MinkowskiConvolution(
            self.PLANES[1], self.PLANES[1], kernel_size=1, stride=1, dimension=D
        )
        self.s_attrntion_tune_4 = ME.MinkowskiConvolution(
            self.PLANES[2], self.PLANES[2], kernel_size=1, stride=1, dimension=D
        )

        self.mask = ME.MinkowskiSigmoid()
        self.main_tune = ME.MinkowskiConvolution(
            self.PLANES[3] * 2, self.PLANES[3], kernel_size=1, stride=1, dimension=D
        )
        self.decoder = MinkUNetDecoder_add(
            in_channels=in_channels,
            out_channels=out_channels,
            D=3,
            BLOCK=self.BLOCK,
            PLANES=self.PLANES,
            LAYERS=self.LAYERS,

        )
        self.mask_enhance = torch.IntTensor([2]).cuda()
        # self.mut = ME.MinkowskiBroadcastMultiplication(D)
        self.GCA = GCA()
        self.CLI = CLI()

    def surrppot_mask(self, x, coords=None):
        f = self.surppt_tune(x)
        f = self.surppt_tune_agg(f, coords=coords)
        # f = self.mask(f)
        # f.F = f.F* self.mask_enhance
        return self.mask(f) * self.mask_enhance  # f#

    def forward(self, x):
        support_frames = [x[i] for i in range(1, len(x))]
        pre_frame = x[0]
        res_s = [self.surpport_encoder(t) for t in support_frames]
        # res_s_mask = [ self.s_attrntion_tune[idx](r) if idx <4 else r for idx,r in enumerate(res_s[-1])]
        res_s_mask = []
        # res_s_mask.append(self.s_attrntion_tune_1(res_s[-1][0]))
        res_s_mask.append(self.s_attrntion_tune_2(res_s[-1][1]))
        res_s_mask.append(self.s_attrntion_tune_3(res_s[-1][2]))
        res_s_mask.append(self.s_attrntion_tune_4(res_s[-1][3]))
        res_s_mask.append(res_s[-1][4])
        res_m = list(self.main_encoder(pre_frame, res_s_mask))
        # masks = [self.surrppot_mask(m[-1], res_m[-1].coords) for m in res_s]
        # for m in masks:
        #     tmp = ME.SparseTensor(m.F,coords_manager = res_m[-1].coords_man,coords_key=res_m[-1].coords_key)
        #     # res_m[-1] =  self.mut(res_m[-1] , m)
        res_m[-1] = self.main_tune(res_m[-1])
        res_f, fea_o = self.decoder(res_m)

        return res_f,fea_o

class MinkUNetEncoder_spsqNet_v2(ResNetBase):
    # BLOCK = None
    # PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, BLOCK=None, PLANES=None, LAYERS=None, D=3):
        assert BLOCK
        assert PLANES
        self.BLOCK = BLOCK
        self.PLANES = PLANES
        if LAYERS:
            self.LAYERS = LAYERS
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def freez_param(self,):
        for param in self.parameters():
            param.requires_grad = False

    def refresh_p(self, net):
        self.load_state_dict(net.state_dict())

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D
        )

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.GCA = GCA()
        self.CLI = CLI()

    def forward(self, x, surppot):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        # out_p1 = self.GCA(out_p1,surppot[0])

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        out_b1p2 = self.GCA(out_b1p2, surppot[0])

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        out_b2p4 = self.GCA(out_b2p4, surppot[1])

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        out_b3p8 = self.GCA(out_b3p8, surppot[2])

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)
        out = self.CLI(out, surppot[3])
        # out = self.out_tune(out)
        return out_p1, out_b1p2, out_b2p4, out_b3p8, out

class MinkUNetDecoder_add(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, BLOCK=None, PLANES=None,LAYERS=None, D=3):
        assert BLOCK
        assert PLANES
        self.BLOCK = BLOCK
        self.PLANES = PLANES
        if LAYERS:
            self.LAYERS = LAYERS
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.PLANES[3]

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7], out_channels, kernel_size=1, bias=True, dimension=D
        )
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # tensor_stride=8
        out = self.convtr4p16s2(x[4])
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, x[3])
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, x[2])
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, x[1])
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, x[0])
        out = self.block8(out)

        return self.final(out),out

