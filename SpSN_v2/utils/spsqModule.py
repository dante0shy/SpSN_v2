import torch.nn as nn
import torch
import MinkowskiEngine as ME
import numpy as np


class GCA(nn.Module):
    def __init__(self, dimension=3):
        super(GCA, self).__init__()
        # self.dimension = dimension
        self.gloable_pooling = ME.MinkowskiGlobalAvgPooling()
        self.sigmoid = ME.MinkowskiSigmoid()

    def forward(self, input, vecter):
        attention = self.sigmoid(self.gloable_pooling(input))
        point_idx = input.C[:, 0]
        output_feature = input.F * attention.F[point_idx.long()]
        output = ME.SparseTensor(
            output_feature,
            coordinate_manager=input.coordinate_manager,
            coordinate_map_key=input.coordinate_map_key,
        )
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s


class GCA_V1(nn.Module):
    def __init__(self, dimension=3):
        super(GCA_V1, self).__init__()
        # self.dimension = dimension
        self.gloable_pooling = ME.MinkowskiGlobalMaxPooling()
        self.sigmoid = ME.MinkowskiSigmoid()

    def forward(self, input, vecter):
        attention = self.sigmoid(self.gloable_pooling(input))
        point_idx = input.C[:, 0]
        output_feature = input.F * attention.F[point_idx.long()]
        output = ME.SparseTensor(
            output_feature,
            coordinate_manager=input.coordinate_manager,
            coordinate_map_key=input.coordinate_map_key,
        )
        return output

    def __repr__(self):
        s = "GlobalAddLayer"
        return s


class CLI(nn.Module):
    def __init__(self, full_scale=128, topk=3, r=0.5):
        super(CLI, self).__init__()
        # self.dimension = dimension
        # self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r

        # self.ad = nn.Linear(20, 30)

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale
    def pairwise_distances_l2_cli(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)/ (self.full_scale**2)

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.C[:, 0]
        point_idx_b = b.C[:, 0]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.C[:, 1:]//16
        coordinate_b = b.C[:, 1:]//16
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            # dist_c = self.pairwise_distances_l2_cli(
            #     coordinate_a[point_idx_a == i].float(),
            #     coordinate_b[point_idx_b == i].float(),
            # )
            #
            # dist_c = torch.cdist(
            #         coordinate_a[point_idx_a == i].float(),
            #         coordinate_b[point_idx_b == i].float(), ).pow(2) / (self.full_scale ** 2)

            dist_c = torch.cdist(
                    coordinate_a[point_idx_a == i].float(),
                    coordinate_b[point_idx_b == i].float(), ) / (self.full_scale )

            tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            idx_pick = torch.argsort(dist_c, 1)[:, :topk]
            dist_w = torch.gather(dist_c, 1, idx_pick).cuda()
            dist_w = self.r - torch.clamp(dist_w, 0.0, self.r)

            tmp_b = b.F[point_idx_b == i][idx_pick.reshape(-1)].reshape(
                -1, topk, b.F.size()[-1]
            ) * dist_w.view(-1, topk, 1)
            tmp_b = torch.sum(tmp_b, 1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        tmp = ME.SparseTensor(
            tmp,
            coordinate_manager=a.coordinate_manager,
            coordinate_map_key=a.coordinate_map_key,
        )
        output = ME.cat(a, tmp)
        return output

    def __repr__(self):
        s = "CLILayer"
        return s

class CLI_mr_o_v2(nn.Module):
    def __init__(self, full_scale=128, topk=3, r=0.5):
        super(CLI_mr_o_v2, self).__init__()
        # self.dimension = dimension
        # self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r

        # self.ad = nn.Linear(20, 30)

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale
    def pairwise_distances_l2_cli(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)/ (self.full_scale**2)

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.C[:, 0]
        point_idx_b = b.C[:, 0]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.C[:, 1:]//16
        coordinate_b = b.C[:, 1:]//16
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            # dist_c = self.pairwise_distances_l2_cli(
            #     coordinate_a[point_idx_a == i].float(),
            #     coordinate_b[point_idx_b == i].float(),
            # )
            #
            # dist_c = torch.cdist(
            #         coordinate_a[point_idx_a == i].float(),
            #         coordinate_b[point_idx_b == i].float(), ).pow(2) / (self.full_scale ** 2)

            dist_c = torch.cdist(
                    coordinate_a[point_idx_a == i].float(),
                    coordinate_b[point_idx_b == i].float(), ) .pow(2) / (self.full_scale ** 2)

            tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            idx_pick = torch.argsort(dist_c, 1)[:, :topk]
            dist_w = torch.gather(dist_c, 1, idx_pick).cuda()
            dist_w = self.r - torch.clamp(dist_w, 0.0, self.r)

            tmp_b = b.F[point_idx_b == i][idx_pick.reshape(-1)].reshape(
                -1, topk, b.F.size()[-1]
            ) * dist_w.view(-1, topk, 1)
            tmp_b = torch.sum(tmp_b, 1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        tmp = ME.SparseTensor(
            tmp,
            coordinate_manager=a.coordinate_manager,
            coordinate_map_key=a.coordinate_map_key,
        )
        output = ME.cat(a, tmp)
        return output

    def __repr__(self):
        s = "CLILayer"
        return s

class CLI_m_v1(nn.Module):
    def __init__(self, full_scale=128, topk=3, r=0.5):
        super(CLI_m_v1, self).__init__()
        # self.dimension = dimension
        # self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r

        self.ad = nn.Linear(20, 30)

        self.mlp_f_fuse_1 = torch.nn.Linear(256 , 256)
        self.mlp_f_fuse = torch.nn.Linear(256, 256)


    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale

    def pairwise_distances_l2_cli(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)/ (self.full_scale**2)

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.C[:, 0]
        point_idx_b = b.C[:, 0]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.C[:, 1:]//16
        coordinate_b = b.C[:, 1:]//16
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            dist_c = self.pairwise_distances_l2_cli(
                coordinate_a[point_idx_a == i].float(),
                coordinate_b[point_idx_b == i].float(),
            )
            dist = dist_c
            tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            idx_pick = torch.argsort(dist, 1)[:, :topk]
            dist_w = torch.gather(dist_c, 1, idx_pick).cuda()
            dist_w = self.r - torch.clamp(dist_w, 0.0, self.r)

            b_f = b.F[point_idx_b == i][idx_pick.reshape(-1)].reshape(
                -1, topk, b.F.size()[-1]
            ) * dist_w.view(-1, topk, 1)
            tmp_b = self.mlp_f_fuse(
                torch.relu(self.mlp_f_fuse_1(b_f))
                * dist_w.view(-1, topk, 1)
            )
            tmp_b = torch.sum(tmp_b, 1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        tmp = ME.SparseTensor(
            tmp,
            coordinate_manager=a.coordinate_manager,
            coordinate_map_key=a.coordinate_map_key,
        )
        output = ME.cat(a, tmp)
        return output

    def __repr__(self):
        s = "CLILayer"
        return s


class CLI_v1(nn.Module):
    def __init__(self, full_scale=128, topk=3, r=0.5, in_dim=256):
        super(CLI_v1, self).__init__()
        # self.dimension = dimension
        # self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.C[:, 0]
        point_idx_b = b.C[:, 0]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.C[:, 1:]
        coordinate_b = b.C[:, 1:]
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            a_f = a.F[point_idx_a == i].repeat(1, topk).view(-1, topk, a.F.size()[-1])
            dist_c = self.pairwise_distances_l2(
                coordinate_a[point_idx_a == i].float(),
                coordinate_b[point_idx_b == i].float(),
            )
            dist = dist_c
            tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            idx_pick = torch.argsort(dist, 1)[:, :topk]
            dist_w = torch.gather(dist_c, 1, idx_pick).cuda()
            dist_w = self.r - torch.clamp(dist_w, 0.0, self.r)
            b_f = b.F[point_idx_b == i][idx_pick.reshape(-1)].view(
                -1, topk, b.F.size()[-1]
            )
            tmp_b = torch.relu(
                self.mlp_f_fuse(torch.cat([b_f, a_f - b_f], dim=2))
            ) * dist_w.view(-1, topk, 1)
            tmp_b = torch.sum(tmp_b, 1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        tmp = ME.SparseTensor(
            tmp,
            coordinate_manager=a.coordinate_manager,
            coordinate_map_key=a.coordinate_map_key,
        )
        output = ME.cat(a, tmp)
        return output

    def __repr__(self):
        s = "CLILayer"
        return s

class CLI_cp_v1(nn.Module):
    def __init__(self, full_scale=128, topk=3, r=0.5, in_dim=256):
        super(CLI_cp_v1, self).__init__()
        # self.dimension = dimension
        # self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r
        self.mlp_f_fuse_1 = torch.nn.Linear(in_dim, in_dim)
        self.mlp_f_fuse = torch.nn.Linear(in_dim, in_dim)

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.C[:, 0]
        point_idx_b = b.C[:, 0]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.C[:, 1:]
        coordinate_b = b.C[:, 1:]
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            a_f = a.F[point_idx_a == i].repeat(1, topk).view(-1, topk, a.F.size()[-1])
            dist_c = self.pairwise_distances_l2(
                coordinate_a[point_idx_a == i].float(),
                coordinate_b[point_idx_b == i].float(),
            )
            dist = dist_c
            tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            idx_pick = torch.argsort(dist, 1)[:, :topk]
            dist_w = torch.gather(dist_c, 1, idx_pick).cuda()
            dist_w = self.r - torch.clamp(dist_w, 0.0, self.r)
            b_f = b.F[point_idx_b == i][idx_pick.reshape(-1)].view(
                -1, topk, b.F.size()[-1]
            )
            tmp_b = self.mlp_f_fuse(
                torch.relu(self.mlp_f_fuse_1(b_f))
                * dist_w.view(-1, topk, 1)
            )
            tmp_b = torch.sum(tmp_b, 1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        tmp = ME.SparseTensor(
            tmp,
            coordinate_manager=a.coordinate_manager,
            coordinate_map_key=a.coordinate_map_key,
        )
        output = ME.cat(a, tmp)
        return output

    def __repr__(self):
        s = "CLILayer"
        return s


class CLI_v2(nn.Module):
    def __init__(self, full_scale=128, topk=3, r=1.0, in_dim=256):
        super(CLI_v2, self).__init__()
        # self.dimension = dimension
        # self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r
        # self.mlp_f_fuse = torch.nn.Linear(in_dim * 2, in_dim)
        self.mlp_f_fuse_1 = torch.nn.Linear(in_dim * 2, in_dim)
        self.mlp_f_fuse = torch.nn.Linear(in_dim, in_dim)

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.C[:, 0]
        point_idx_b = b.C[:, 0]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.C[:, 1:] // 16
        coordinate_b = b.C[:, 1:] // 16
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            a_f = a.F[point_idx_a == i].repeat(1, topk).view(-1, topk, a.F.size()[-1])
            dist_c = self.pairwise_distances_l2(
                coordinate_a[point_idx_a == i].float(),
                coordinate_b[point_idx_b == i].float(),
            )
            dist = dist_c
            tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            idx_pick = torch.argsort(dist, 1)[:, :topk]
            dist_w = torch.gather(dist_c, 1, idx_pick).cuda()
            dist_w = self.r - torch.clamp(dist_w, 0.0, self.r)
            b_f = b.F[point_idx_b == i][idx_pick.reshape(-1)].view(
                -1, topk, b.F.size()[-1]
            )
            tmp_b = self.mlp_f_fuse(
                torch.relu(self.mlp_f_fuse_1(torch.cat([b_f, a_f - b_f], dim=2)))
                * dist_w.view(-1, topk, 1)
            ) * dist_w.view(-1, topk, 1)
            tmp_b = torch.max(tmp_b, 1)[0]
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        tmp = ME.SparseTensor(
            tmp,
            coordinate_manager=a.coordinate_manager,
            coordinate_map_key=a.coordinate_map_key,
        )
        output = ME.cat(a, tmp)
        return output

    def __repr__(self):
        s = "CLILayer"
        return s


class CLI_v3(nn.Module):
    def __init__(self, full_scale=128, topk=5, r=0.5, in_dim=256):
        super(CLI_v3, self).__init__()
        # self.dimension = dimension
        # self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r
        self.mlp_f_fuse_1 = torch.nn.Linear(in_dim * 2, in_dim)
        self.mlp_f_fuse = torch.nn.Linear(in_dim, in_dim)

    # def pairwise_distances_l2(self, x, y):
    #     x_norm = (x ** 2).sum(1).view(-1, 1)
    #     y_t = torch.transpose(y, 0, 1)
    #     y_norm = (y ** 2).sum(1).view(1, -1)
    #     dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    #     return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf) / (self.full_scale**2)

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.C[:, 0]
        point_idx_b = b.C[:, 0]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.C[:, 1:] // 16
        coordinate_b = b.C[:, 1:] // 16
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            a_f = a.F[point_idx_a == i].repeat(1, topk).view(-1, topk, a.F.size()[-1])

                # dist_c = self.pairwise_distances_l2(
                #     coordinate_a[point_idx_a == i].float(),
                #     coordinate_b[point_idx_b == i].float(),
                # )
            # dist_c = torch.cdist(
            #         coordinate_a[point_idx_a == i].float(),
            #         coordinate_b[point_idx_b == i].float(),).pow(2)/(self.full_scale**2)
            dist_c = torch.cdist(
                    coordinate_a[point_idx_a == i].float(),
                    coordinate_b[point_idx_b == i].float(),)/ self.full_scale

            idx_pick = torch.argsort(dist_c, 1)[:, :topk]

            # dist_w, idx_pick = torch.topk(dist, topk)#

            dist_w = torch.gather(dist_c, 1, idx_pick).cuda()
            dist_w = self.r - torch.clamp(dist_w, 0.0, self.r)
            b_f = b.F[point_idx_b == i][idx_pick.reshape(-1)].view(
                -1, topk, b.F.size()[-1]
            )
            # tmp_f = []

            tmp_b = self.mlp_f_fuse(
                torch.relu(self.mlp_f_fuse_1(torch.cat([b_f, a_f - b_f], dim=2)))
                * dist_w.view(-1, topk, 1)
            )
            tmp_b = torch.sum(tmp_b, 1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        tmp = ME.SparseTensor(
            tmp,
            coordinate_manager=a.coordinate_manager,
            coordinate_map_key=a.coordinate_map_key,
        )
        output = ME.cat(a, tmp)
        return output

    def __repr__(self):
        s = "CLILayer"
        return s

class CLI_v3_mr_2(nn.Module):
    def __init__(self, full_scale=128, topk=5, r=0.5, in_dim=256):
        super(CLI_v3_mr_2, self).__init__()
        # self.dimension = dimension
        # self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r
        self.mlp_f_fuse_1 = torch.nn.Linear(in_dim * 2, in_dim)
        self.mlp_f_fuse = torch.nn.Linear(in_dim, in_dim)

    # def pairwise_distances_l2(self, x, y):
    #     x_norm = (x ** 2).sum(1).view(-1, 1)
    #     y_t = torch.transpose(y, 0, 1)
    #     y_norm = (y ** 2).sum(1).view(1, -1)
    #     dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    #     return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf) / (self.full_scale**2)

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.C[:, 0]
        point_idx_b = b.C[:, 0]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.C[:, 1:] // 16
        coordinate_b = b.C[:, 1:] // 16
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            a_f = a.F[point_idx_a == i].repeat(1, topk).view(-1, topk, a.F.size()[-1])

                # dist_c = self.pairwise_distances_l2(
                #     coordinate_a[point_idx_a == i].float(),
                #     coordinate_b[point_idx_b == i].float(),
                # )
            # dist_c = torch.cdist(
            #         coordinate_a[point_idx_a == i].float(),
            #         coordinate_b[point_idx_b == i].float(),).pow(2)/(self.full_scale**2)
            dist_c = torch.cdist(
                    coordinate_a[point_idx_a == i].float(),
                    coordinate_b[point_idx_b == i].float(),).pow(2)/(self.full_scale**2)

            idx_pick = torch.argsort(dist_c, 1)[:, :topk]

            # dist_w, idx_pick = torch.topk(dist, topk)#

            dist_w = torch.gather(dist_c, 1, idx_pick).cuda()
            dist_w = self.r - torch.clamp(dist_w, 0.0, self.r)
            b_f = b.F[point_idx_b == i][idx_pick.reshape(-1)].view(
                -1, topk, b.F.size()[-1]
            )
            # tmp_f = []

            tmp_b = self.mlp_f_fuse(
                torch.relu(self.mlp_f_fuse_1(torch.cat([b_f, a_f - b_f], dim=2)))
                * dist_w.view(-1, topk, 1)
            )
            tmp_b = torch.sum(tmp_b, 1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        tmp = ME.SparseTensor(
            tmp,
            coordinate_manager=a.coordinate_manager,
            coordinate_map_key=a.coordinate_map_key,
        )
        output = ME.cat(a, tmp)
        return output

    def __repr__(self):
        s = "CLILayer"
        return s

class CLI_v4(nn.Module):
    def __init__(self, full_scale=128, topk=5, r=0.5, in_dim=256):
        super(CLI_v4, self).__init__()
        # self.dimension = dimension
        # self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r
        self.mlp_f_fuse_1 = torch.nn.Linear(in_dim * 2, in_dim)
        self.mlp_f_fuse = torch.nn.Linear(in_dim, in_dim)

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.C[:, 0]
        point_idx_b = b.C[:, 0]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.C[:, 1:] // 16
        coordinate_b = b.C[:, 1:] // 16
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            a_f = a.F[point_idx_a == i].repeat(1, topk).view(-1, topk, a.F.size()[-1])
            dist_c = self.pairwise_distances_l2(
                coordinate_a[point_idx_a == i].float(),
                coordinate_b[point_idx_b == i].float(),
            )
            dist = dist_c
            tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            idx_pick = torch.argsort(dist, 1)[:, :topk]
            dist_w = torch.gather(dist_c, 1, idx_pick).cuda()
            dist_w = self.r - torch.clamp(dist_w, 0.0, self.r)
            b_f = b.F[point_idx_b == i][idx_pick.reshape(-1)].view(
                -1, topk, b.F.size()[-1]
            )
            tmp_b = self.mlp_f_fuse(
                torch.relu(self.mlp_f_fuse_1(torch.cat([b_f, a_f - b_f], dim=2)))
                * dist_w.view(-1, topk, 1)
            )
            tmp_b = torch.sum(tmp_b, 1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        tmp = ME.SparseTensor(
            tmp,
            coordinate_manager=a.coordinate_manager,
            coordinate_map_key=a.coordinate_map_key,
        )
        output = ME.cat(a, tmp)
        return output

    def __repr__(self):
        s = "CLILayer"
        return s


class CLI_mr_v1(nn.Module):
    def __init__(self, full_scale=128, topk=5, r=0.5, in_dim=256):
        super(CLI_v3, self).__init__()
        # self.dimension = dimension
        # self.s2d = scn.SparseToDense(dimension, nPlanes)
        self.full_scale = full_scale
        self.topk = topk
        self.r = r
        self.mlp_f_fuse_1 = torch.nn.Linear(in_dim * 2, in_dim)
        self.mlp_f_fuse = torch.nn.Linear(in_dim, in_dim)

    # def pairwise_distances_l2(self, x, y):
    #     x_norm = (x ** 2).sum(1).view(-1, 1)
    #     y_t = torch.transpose(y, 0, 1)
    #     y_norm = (y ** 2).sum(1).view(1, -1)
    #     dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    #     return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / self.full_scale

    def pairwise_distances_l2(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.sqrt(torch.clamp(dist, 0.0, np.inf)) / (self.full_scale**2)

    def forward(self, a, b):
        # assert input.features.shape[-1] == vecter.shape[-1]
        point_idx_a = a.C[:, 0]
        point_idx_b = b.C[:, 0]
        max_idx = torch.max(point_idx_a).max().item()
        coordinate_a = a.C[:, 1:] // 16
        coordinate_b = b.C[:, 1:] // 16
        tmp = []
        for i in range(max_idx + 1):
            if self.topk > coordinate_b[point_idx_b == i].size()[0]:
                topk = coordinate_b[point_idx_b == i].size()[0]
            else:
                topk = self.topk
            a_f = a.F[point_idx_a == i].repeat(1, topk).view(-1, topk, a.F.size()[-1])
            dist_c = self.pairwise_distances_l2(
                coordinate_a[point_idx_a == i].float(),
                coordinate_b[point_idx_b == i].float(),
            )
            dist = dist_c
            tmp_f = []
            # dist_w, idx_pick = torch.topk(dist, topk)#

            idx_pick = torch.argsort(dist, 1)[:, :topk]
            dist_w = torch.gather(dist_c, 1, idx_pick).cuda()
            dist_w = self.r - torch.clamp(dist_w, 0.0, self.r)
            b_f = b.F[point_idx_b == i][idx_pick.reshape(-1)].view(
                -1, topk, b.F.size()[-1]
            )
            tmp_b = self.mlp_f_fuse(
                torch.relu(self.mlp_f_fuse_1(torch.cat([b_f, a_f - b_f], dim=2)))
                * dist_w.view(-1, topk, 1)
            )
            tmp_b = torch.sum(tmp_b, 1)
            tmp.append(tmp_b)
        tmp = torch.cat(tmp, 0)
        tmp = ME.SparseTensor(
            tmp,
            coordinate_manager=a.coordinate_manager,
            coordinate_map_key=a.coordinate_map_key,
        )
        output = ME.cat(a, tmp)
        return output

    def __repr__(self):
        s = "CLILayer"
        return s