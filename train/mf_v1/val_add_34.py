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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
from SpSN_v2.utils.v2p_fintuner import V2PFinetuner_v5_4 as V2PFinetuner
from SpSN_v2.dataset import data_mutli_frame_v1_val as data
from SpSN_v2.utils import np_ioueval
import torch_cluster



use_cuda = torch.cuda.is_available()
evealer = np_ioueval.evaler
evaler_main = np_ioueval.iouEval(np_ioueval.N_CLASSES, np_ioueval.UNKNOWN_ID)

device = torch.device("cuda" if use_cuda else "cpu")

# log_pos = '/home/dante0shy/dataset/shy/SpSN_v2/mf_mr'
# log_path = os.path.join(log_pos, "snap")
# if not os.path.exists(log_path):
#     os.mkdir(log_path)
# # exp_name_main_model = "unet_scale{}_m{}_rep{}_3".format(data.scale[0], m, block_reps)
# # main_model_path = os.path.join(log_path, exp_name_main_model)
# # assert os.path.exists(main_model_path)
#
# exp_name = "finetuner_scale{}_m{}_rep{}_mr_v1_cp".format(data.scale[0], m, block_reps)
# log_dir = os.path.join(log_path, exp_name)
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)
# logs_dir = os.path.join(log_dir, "log_val.json")
# if os.path.exists(logs_dir):
#     logs = json.load(open(logs_dir, "r"))
# else:
#     logs = {}


out_name = "mid_val_mf_v1_34_cp"
out_dit = os.path.join(data.data_base,out_name)
data.set_out_name(out_name)
if not os.path.exists(out_dit):
    os.mkdir(out_dit)
out_dit_pre = os.path.join(out_dit, "sequences")
out_dit_prob = os.path.join(out_dit, "sequences_prob")
if not os.path.exists(out_dit_pre):
    os.mkdir(out_dit_pre)
if not os.path.exists(out_dit_prob):
    os.mkdir(out_dit_prob)
sq_pre_format = os.path.join(out_dit_pre, "{:02d}")
sq_prob_format = os.path.join(out_dit_prob, "{:02d}")
log_file_format = "predictions/{}.label"
log_prob_file_format = "predictions_prob/{}.npy"
sqs = []
files = []
for sq, x, y, _ in data.train:
    files.append((sq, x[-1].split("/")[-1].split(".")[0]))
    sqs.append(sq)
for sq, x, y, _ in data.val:
    files.append((sq, x[-1].split("/")[-1].split(".")[0]))
    sqs.append(sq)
sqs = set(sqs)
update = False
def set_dir(update_s):
    for x in sqs:

        p = sq_pre_format.format(x)
        pp = sq_prob_format.format(x)
        if not update_s:
            if os.path.exists(p):
                shutil.rmtree(p)
            if os.path.exists(pp):
                shutil.rmtree(pp)
            if os.path.exists(os.path.join(p, "predictions")):
                shutil.rmtree(os.path.join(p, "predictions"))
            if os.path.exists(os.path.join(pp, "predictions_prob")):
                shutil.rmtree(os.path.join(pp, "predictions_prob"))
        if not os.path.exists(p):
            os.mkdir(p)
        if not os.path.exists(pp):
            os.mkdir(pp)
        if not os.path.exists(os.path.join(p, "predictions")):
            os.mkdir(os.path.join(p, "predictions"))
        if not os.path.exists(os.path.join(pp, "predictions_prob")):
            os.mkdir(os.path.join(pp, "predictions_prob"))


set_dir(update)



v2p_k =4

unet = mdoel.Model(in_channels=1, out_channels=np_ioueval.N_CLASSES, D=3)
v2p_fintuner = V2PFinetuner(pre_channels=np_ioueval.N_CLASSES, f_channels=unet.PLANES[-1], out_channels=np_ioueval.N_CLASSES,k=v2p_k)

# print(unet)

# if use_cuda:
# unet=unet.cuda()
unet = unet.to(device)
v2p_fintuner = v2p_fintuner.to(device)
# criterion = nn.CrossEntropyLoss()

print(
    "#classifer parameters",
    sum([x.nelement() for x in unet.parameters() if x.requires_grad]),
)
print(
    "#classifer parameters",
    sum([x.nelement() for x in v2p_fintuner.parameters() if x.requires_grad]),
)


training_epochs = 50
epoch_s = 0
# snap = glob.glob(os.path.join(main_model_path, "net*.pth"))/home/dante0shy/remote_workplace/SpSN_v2/pretrained
# snap = list(sorted(snap, key=lambda x: int(x.split("-")[-1].split(".")[0])))
main_snap = [os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'pretrained',
    'v9-34c-000000028.pth')]
print("Restore from " + main_snap[-1])
unet.load_state_dict(torch.load(main_snap[-1]))
# for p in unet.parameters(): p.requires_grad = False

train_first = True
pertrain = False

# snap = glob.glob(os.path.join(log_dir, "v2p-fintuner*.pth"))
# snap = list(sorted(snap, key=lambda x: int(x.split("-")[-1].split(".")[0])))
add_snap= [os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'pretrained',
    'v2p-fintuner-01-34-000000013.pth')]
for s in reversed(add_snap):
    print("Restore from " + s)
    v2p_fintuner.load_state_dict(torch.load(s))
    epoch= int(s.split("/")[-1].split(".")[0].split("-")[-1])
    # logs[epoch] = {}
    unet.eval()
    if True:
        # if scn.is_power2(epoch):
        with torch.no_grad():
            unet.eval()
            v2p_fintuner.eval()
            torch.cuda.empty_cache()
            start = time.time()
            evealer.reset()
            pre_prediction_next  = None
            or_surpport  = None
            with tqdm.tqdm(total=len(data.val)) as pbar:#//data.batch_size+1
                time_list = []

                for i, batch in enumerate(data.get_val_data_loader()):
                    tt= [0,0,0]
                    batch["x"] = [
                        ME.SparseTensor(x[1].float(), coordinates=x[0], device=device)
                        for x in zip(*batch["x"])
                    ]
                    # input = ME.SparseTensor(feats, coords=locs).to(device)  # .cuda()#
                    stime = time.time()
                    predictions,feature_o = unet(batch["x"])
                    feature_o = ME.cat(predictions, feature_o)
                    batch_ind = feature_o.C[:, 0]
                    # print("a {}".format(time.time()-stime))
                    tt[0]=time.time()-stime
                    pre_prediction = None
                    for b in torch.unique(batch_ind):
                        f  = data.val[batch['id'][b]][1][-1]
                        sq_f = f.split('/')[-1].split('.')[0]
                        sq_p = int(f.split('/')[-3])
                        log_file = os.path.join(sq_pre_format.format(sq_p), log_file_format.format(sq_f))
                        y = batch["o_labels"][b]
                        # pre = predictions[batch_ind == b]
                        #dist, idx_la = batch["trees"][b].query(batch["o_locs"][b], 1)

                        #mid_feature = feature_o.F[(batch_ind == b).cpu()][
                        #    idx_la.flatten()
                        #].view(-1,feature_o.shape[-1])
                        mid_feature = feature_o.F[(batch_ind == b).cpu()][batch["rev"][b]]
                        pre = predictions.F[(batch_ind == b).cpu()][batch["rev"][b]]

                        orilos = torch.from_numpy(batch["ori_locs"][b]).cuda().float()
                        orilos_sur = torch.from_numpy(batch["ori_locs_sur"][b]).cuda().float()
                        col, row, = torch_cluster.knn(
                            orilos_sur.cpu(),
                            orilos.cpu(),
                            v2p_k,
                            torch.zeros(batch["ori_locs_sur"][b].shape[0], dtype=torch.int64),
                            torch.zeros(batch["o_locs"][b].shape[0], dtype=torch.int64),

                        )
                        # print("b {}".format(time.time() - stime))
                        # tt.append(time.time() - stime- tt[-1])
                        tt[1] = time.time() - stime
                        idx_la_sur = row[col.argsort()]
                        re_idx = col[col.argsort()]
                        if not isinstance(pre_prediction_next,torch.Tensor):
                            mid_feature_Sur = torch.Tensor(np.eye(data.N_CLASSES)[batch['labels_sur'][b].astype(int)]).cuda()[
                                idx_la_sur.flatten()
                            ].view(-1, v2p_k, data.N_CLASSES)
                        else:
                            mid_feature_Sur = torch.Tensor(torch.eye(data.N_CLASSES)[pre_prediction_next]).cuda()[
                                idx_la_sur.flatten()
                            ].view(-1, v2p_k, data.N_CLASSES)
                        # or_surpport = batch["ori_locs"][b]

                        orilos = torch.FloatTensor(batch["ori_locs"][b]).cuda()
                        orilos_sur = torch.FloatTensor(batch["ori_locs_sur"][b]).cuda()
                        relative_pos = (
                            orilos[re_idx].view(-1,v2p_k,3)
                            - orilos_sur[idx_la_sur.flatten()].view(-1,v2p_k,3)
                        )


                        mid_feature_sur = torch.cat([mid_feature_Sur,relative_pos,], dim=2)
                        n_prediction = v2p_fintuner(mid_feature,mid_feature_sur)
                        # print("c {}".format(time.time() - stime))
                        # tt.append(time.time() - stime- sum(tt))
                        tt[2] = time.time() - stime

                        pre_prediction = n_prediction.argmax(1).cpu()
                        evealer.addBatch(pre_prediction, y)

                        # tmp = np.zeros_like(pre_prediction)
                        # for k, v in data.config['learning_map_inv'].items():
                        #     tmp[pre_prediction == k] = v
                        # pre_prediction = tmp
                        # pre_prediction.astype(np.uint32).tofile(log_file)

                        pre = predictions.F[(batch_ind == b).cpu()][batch["rev"][b]]
                        pre = pre.max(1)[1].cpu().numpy()
                        evaler_main.addBatch(pre, y)
                    pass

                    pre_prediction_next = torch.clone(pre_prediction).detach()
                    # or_surpport = or_surpport.detach()
                    time_list.append(tt)

                    pbar.set_postfix({
                        "T": "{0:1.5f}".format(evealer.getIoU()[0]),
                        'm': "{0:1.3f}".format(sum([ tt[0] for x in time_list[10:]])/max(len(time_list)-10,1)),
                        't': "{0:1.3f}".format(sum([ tt[1] for x in time_list[10:]])/max(len(time_list)-10,1)),
                        'y': "{0:1.3f}".format(sum([ tt[2] for x in time_list[10:]])/max(len(time_list)-10,1))
                    })

                    pbar.update(1)
                    # break
                    # store.index_add_(0,batch['point_ids'],predictions.cpu())
                print("0", "Val MegaMulAdd=", "time=", time.time() - start, "s")
                print('latency {0:1.3f} {0:1.3f} {0:1.3f}'.format(
                    sum([x[0] for x in time_list[10:]]) / (len(time_list)-10),
                    sum([x[1] for x in time_list[10:]]) / (len(time_list)-10),
                    sum([x[2] for x in time_list[10:]]) / (len(time_list)-10),)
                )
                m_iou, iou = evealer.getIoU()
                print("mean IOU", m_iou)
                m_iou_main, iou_main = evaler_main.getIoU()
                print("mean main IOU", m_iou_main)
                # logs[epoch]["mIoU"] = m_iou

                tp, fp, fn = evealer.getStats()
                total = tp + fp + fn
                print("classes          IoU")
                print("----------------------------")
                # logs[epoch][
                #     "ValData"
                # ] = ''
                for i in range(np_ioueval.N_CLASSES):
                    label_name = np_ioueval.CLASS_LABELS[i]
                    # logs[epoch][
                    #     "ValData"
                    # ] += "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})\n".format(
                    #     label_name, iou[i], tp[i], total[i]
                    # )
                    print(
                        "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
                            label_name, iou[i], tp[i], total[i]
                        )
                    )
    # json.dump(logs, open(logs_dir, "w"))
    # 211 - (189 - 72)
# bin_file_path = lidar_sample_data_token + '_lidarseg.bin"
# np.array(predicted_labels).astype(np.uint8).tofile(bin_file_path)