# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
m = 16  # 16 or 32
residual_blocks = False  # True or False
block_reps = 1  # Conv block repetition factor: 1 or 2
import os, sys, glob, tqdm
import numpy as np
import random, time

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ""
    )
)
import torch
import json,shutil
import MinkowskiEngine as ME
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from SpSN_v2.core import sq_model_mr_v1_add_34c as mdoel
# from SqSN_v2.utils.v2p_fintuner import V2PFinetuner_v5_4 as V2PFinetuner
from SpSN_v2.utils.focal_loss import FocalLoss
# import torch_cluster
# from pytor import FocalLossV3
# from focal.focal_loss import FocalLoss
# from ME_Squeeze.model import unet
# from ME_Squeeze.dataset import data_squeeze_paper as data
from SpSN_v2.dataset import data_mutli_frame_v1_34 as data
from SpSN_v2.utils import np_ioueval

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
evealer = np_ioueval.evaler
device = torch.device("cuda" if use_cuda else "cpu")

log_pos = "/home/dante0shy/dataset/shy/SpSN_v2"
log_pos = os.path.join(
    log_pos, os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
)
log_path = os.path.join(log_pos, "snap")
if not os.path.exists(log_path):
    os.makedirs(log_path,exist_ok=True)
exp_name = "unet_scale{}_m{}_rep{}_mr_v1_03_t_34c".format(data.scale[0], m, block_reps)
log_dir = os.path.join(log_path, exp_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
logs_dir = os.path.join(log_dir, "log.json")
if os.path.exists(logs_dir):
    logs = json.load(open(logs_dir, "r"))
else:
    logs = {}


unet = mdoel.Model(in_channels=1, out_channels=np_ioueval.N_CLASSES, D=3)

# print(unet)

# if use_cuda:
# unet=unet.cuda()
unet = unet.to(device)
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(
    alpha=0.25, gamma=2.0, apply_nonlin=lambda x: torch.softmax(x, dim=1)
)

unet.surpport_encoder.freez_param()

# optimizer = optim.Adam(unet.parameters())
optimizer = optim.Adam(filter(lambda p: p.requires_grad, unet.parameters()))


print(
    "#classifer parameters",
    sum([x.nelement() for x in unet.parameters() if x.requires_grad]),
)

training_epochs = 50
epoch_s = 0
snap = glob.glob(os.path.join(log_dir, "net*.pth"))
snap = list(sorted(snap, key=lambda x: int(x.split("-")[-1].split(".")[0])))
train_first = True
pertrain = True
pertrain_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pretrained",
    "net-000000038.pth",
)
if pertrain and not snap:
    print("Pertrained from {}".format(pertrain_dir))
    model_dict = unet.state_dict()
    pretrained_dict = {
        k: v for k, v in torch.load(pertrain_dir).items() if k in model_dict
    }
    model_dict.update(pretrained_dict)
    unet.load_state_dict(model_dict)
    pretrained_dict_s = {
        k: v
        for k, v in unet.main_encoder.state_dict().items()
        if k in unet.surpport_encoder.state_dict()
    }
    unet.surpport_encoder.load_state_dict(pretrained_dict_s)
elif snap:
    print("Restore from " + snap[-1])
    unet.load_state_dict(torch.load(snap[-1]))
    epoch_s = int(snap[-1].split("/")[-1].split(".")[0].split("-")[-1])
    optimizer.load_state_dict(torch.load(snap[-1].replace("net-", "optim-")))
    train_first = False

for epoch in range(epoch_s, training_epochs):

    unet.train()
    stats = {}
    start = time.time()
    train_loss = 0
    mid = time.time()
    # for i,batch in enumerate(data.train_data_loader):
    if epoch not in logs.keys():
        logs[epoch] = {"epoch": epoch, "TrainLoss": 0, "mIoU": 0, "ValData": ""}
    if train_first:
        with tqdm.tqdm(total=len(data.train) // data.batch_size) as pbar:
            for i, batch in enumerate(data.get_train_data_loader()):
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                unet.surpport_encoder.refresh_p(unet.main_encoder)

                batch["y"] = batch["y"].to(device)
                batch["x"] = [
                    ME.SparseTensor(x[1].float(), coordinates=x[0], device=device)
                    for x in zip(*batch["x"])
                ]
                predictions,_ = unet(batch["x"])
                loss = criterion(predictions.F, batch["y"])
                train_loss += loss.item()  #
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"T": "{0:1.5f}".format(train_loss / (i + 1))})
                pbar.update(1)
                del batch
        print(
            epoch, "Train loss", train_loss / (i + 1), "time=", time.time() - start, "s"
        )
        torch.save(
            unet.state_dict(), os.path.join(log_dir, "net-%09d" % epoch + ".pth")
        )
        torch.save(
            optimizer.state_dict(), os.path.join(log_dir, "optim-%09d" % epoch + ".pth")
        )
        logs[epoch]["TrainLoss"] = train_loss / (i + 1)
        json.dump(logs, open(logs_dir, "w"))
    else:
        print("test first")
        train_first = True

    if True:
        # if scn.is_power2(epoch):
        with torch.no_grad():
            unet.eval()
            torch.cuda.empty_cache()
            unet.surpport_encoder.refresh_p(unet.main_encoder)

            start = time.time()
            evealer.reset()
            # unet.refresh_p()
            # for rep in range(1, 1 + data.val_reps):
            #     for i, batch in tqdm.tqdm(enumerate(data.get_val_data_loader()))://data.batch_size
            with tqdm.tqdm(total=len(data.val)) as pbar:
                for i, batch in enumerate(data.get_val_data_loader()):
                    batch["x"] = [
                        ME.SparseTensor(x[1].float(), coordinates=x[0], device=device)
                        for x in zip(*batch["x"])
                    ]
                    # input = ME.SparseTensor(feats, coords=locs).to(device)  # .cuda()#
                    predictions,_ = unet(batch["x"])
                    batch_ind = predictions.C[:, 0]
                    predictions = predictions.F.max(1)[1].cpu().numpy()
                    for b in torch.unique(batch_ind):
                        y = batch["o_labels"][b]
                        # pre = predictions[batch_ind == b]
                        # idx_la = batch["trees"][b].query(
                        #     batch["o_locs"][b], 1, return_distance=False
                        # )
                        pre = predictions[(batch_ind == b).cpu()][batch["rev"][b]]
                        evealer.addBatch(pre, y)
                    # pass
                        pbar.set_postfix({"T": "{0:1.5f}".format(evealer.getIoU()[0])})
                        pbar.update(1)
                    # break
                    # store.index_add_(0,batch['point_ids'],predictions.cpu())
                print("0", "Val MegaMulAdd=", "time=", time.time() - start, "s")
                m_iou, iou = evealer.getIoU()
                print("mean IOU", m_iou)
                logs[epoch]["mIoU"] = m_iou

                tp, fp, fn = evealer.getStats()
                total = tp + fp + fn
                print("classes          IoU")
                print("----------------------------")
                for i in range(np_ioueval.N_CLASSES):
                    label_name = np_ioueval.CLASS_LABELS[i]
                    logs[epoch][
                        "ValData"
                    ] += "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})\n".format(
                        label_name, iou[i], tp[i], total[i]
                    )
                    print(
                        "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
                            label_name, iou[i], tp[i], total[i]
                        )
                    )
    json.dump(logs, open(logs_dir, "w"))
