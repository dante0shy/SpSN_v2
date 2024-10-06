# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options

elastic_deformation = False
import MinkowskiEngine as ME

# print(ME.__file__)
import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage
import os, time, yaml, tqdm, random
from concurrent.futures import ThreadPoolExecutor
from SqSN_v2.utils.generate_sequential import *
from SqSN_v2.config import *
from MinkowskiEngine.utils import sparse_quantize, sparse_collate
from sklearn.neighbors import KDTree

val_base = "/home/dante0shy/dataset/KITTI"
data_base = "/home/dante0shy/dataset/KITTI"
input_base = os.path.join(data_base, "sequences")
config = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), "config/semantic-kitti-all.yaml"))
)
dirs = glob.glob(os.path.join(input_base, "*"))
downsample_mode = False
downsample = 0.8
batch_size_t = 4
shift_scale = [20, 20, 20]
scale = [1 / 20, 1 / 20, 1 / 20]
frames = 2
batch_size = 4
files = {"train": [], "valid": [], "test": []}
for dir_ in dirs:
    datas = glob.glob(os.path.join(dir_, "velodyne", "*"))
    labels = glob.glob(os.path.join(dir_, "labels", "*"))
    times = os.path.join(dir_, "times.txt")
    clib = parse_calibration(os.path.join(dir_, "calib.txt"))
    poses = parse_poses(os.path.join(dir_, "poses.txt"), clib)

    # labels_dict = {x[-12:-6]: x for x in labels}
    sq = int(dir_[-2:])
    split = [k for k, v in config["split"].items() if sq in v][0]
    if split not in files.keys():
        continue
    # f = [(sq, data, labels_dict[data[-10:-4]] if len(labels_dict) else '') for data in datas]
    datas = list(sorted(datas, key=lambda x: int(x[-10:-4])))
    labels = list(sorted(labels, key=lambda x: int(x[-12:-6])))
    f = []
    datas = [datas[0]] + datas
    poses = [poses[0]] + poses
    if split != "test":
        labels = [labels[0].replace('sequences','mid_val_1_frame/sequences').replace('labels','predictions')] + labels


    for i, data in enumerate(datas):
        if  i == len(datas)-1:
            continue
        pre = i - frames + 1 if i - frames + 1 >= 0 else 0
        data_frames = datas[i : i + 2]
        pose_frames = poses[i : i + 2]
        if split == "test":
            label_frames = [""] * 2
        else:
            label_frames = labels[i : i + 2]

        f.append((sq, data_frames, label_frames, pose_frames))
    files[split].extend(f)

train = files["train"]  # glob.glob(os.path.join(input_base,'train/*.pth'))

print("Training examples:", len(train))

val = files["valid"]  # glob.glob(os.path.join(input_base,'valid/*.pth'))
print("Validation examples:", len(val))

out_name = None
def set_out_name(n):
    global out_name
    out_name = n

def aug_rotate(a, seed):

    a = np.matmul(a, seed)  # +full_scale/2#+np.random.uniform(-2,2,3)
    return a[:, :3]  # .mean(0)


def trainMerge(tbl):
    locs = []
    feats = []
    labels_pre = []
    labels_sur = []
    inds = []
    o_locs = []
    o_labels = []
    trees = []
    revs = []
    revs_sur = []
    trees_sur  = []
    ori_coords = []
    ori_coords_sur = []
    # idx=[]
    for idx, i in enumerate(tbl):
        # torch.load(train[i])
        locs_pre = []
        feats_pre = []
        # labels_pre = []

        scans_b = train[i][1]
        labels_b = train[i][2]
        poss_b = train[i][3]
        pre_frame = len(scans_b) - 1

        seed = np.eye(3) + np.random.randn(3, 3) * 0.01
        seed[0][0] *= np.random.randint(0, 2) * 2 - 1
        # seed *= shift_scale #scale
        theta = np.random.rand() * 2 * math.pi
        seed = np.matmul(
            seed,
            [
                [math.cos(theta), math.sin(theta), 0],
                [-math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ],
        )
        min_c = None

        def get_data(frame_idx, mean=None):
            scan = np.fromfile(scans_b[frame_idx], dtype=np.float32)
            scan = scan.reshape((-1, 4))
            r = scan[:, 3] - np.min(scan[:, 3])  # np.ones_like(scan[:, 3] )#

            if frame_idx != pre_frame:
                scan[:, 3] = 1
                for f_pos in range(frame_idx + 1, len(scans_b)):
                    diff = np.matmul(inv(poss_b[f_pos]), poss_b[f_pos - 1])
                    scan = np.matmul(diff, scan.T).T
                # diff = np.matmul(inv(poss_b[-1]), poss_b[frame_idx])
                # scan = np.matmul(diff, scan.T).T
            coords = scan[:, :3]
            coords = aug_rotate(coords, seed)
            global min_c
            if frame_idx == pre_frame:
                min_c = coords.min(0)
            coords = coords - min_c
            # offset = np.random.rand(3) * 0.05
            # coords += offset
            # idxs = r ==1
            # idxs= np.ones_like(scan[:, 3] )#
            if downsample_mode:
                idxs = np.random.uniform(0, 1, coords.shape[0]) <= downsample
                coords = coords[idxs]
                r = r[idxs]
            # elif coords.shape[0]>120000:
            #     idxs = np.random.uniform(0, 1, coords.shape[0]) <= downsample
            #     coords = coords[idxs]
            #     r = r[idxs]
            # coords = torch.from_numpy(coords).long()

            if True:#frame_idx == pre_frame:/home/dante0shy/dataset/KITTI/mid_val_n_train_mf_v10/sequences
                tmp = labels_b[frame_idx] \
                    .replace('sequences', '{}/{}'.format(out_name, 'sequences')) \
                    .replace('labels', 'predictions')
                label = np.fromfile(
                    labels_b[frame_idx]
                    if (frame_idx == pre_frame)
                       or not os.path.exists(tmp)
                    else tmp,
                    dtype=np.uint32)
                sem_label = label & 0xFFFF
                # for k, v in config["learning_map"].items():
                #     sem_label[sem_label == k] = v
                sem_label = np.vectorize(config["learning_map"].__getitem__)(sem_label)

                if downsample_mode:
                    sem_label = sem_label[idxs]

                ori_coord = np.copy(coords)
                coords /= scale
                coords = coords.astype(np.int32)
                # coords, r, sem_label = sparse_quantize(coords, feats=r.reshape(-1,1), labels=sem_label.astype(np.int32), ignore_label=0, quantization_size=scale)#ME.utils.
                ind = sparse_quantize(
                    coords,
                    features=r.reshape(-1, 1),
                    labels=sem_label.astype(np.int32),
                    ignore_label=0,
                    return_index=True,
                    return_inverse=True
                )  # ME.utils. quantization_size=scale,
                return (
                    ind[0],
                    ind[1],
                    ind[2],
                    ind[3],
                    coords,
                    sem_label,
                    ind[4],
                    ori_coord,
                )

        tmp_pre = get_data(pre_frame)
        locs_pre.append(tmp_pre[0])
        feats_pre.append(tmp_pre[1])
        labels_pre.append(tmp_pre[2])
        inds.append(tmp_pre[3])
        o_locs.append(tmp_pre[4])
        o_labels.append(tmp_pre[5])
        revs.append(tmp_pre[6])
        ori_coords.append(tmp_pre[7])
        if pre_frame:
            tmp = list(map(lambda x: get_data(x, None), range(pre_frame)))
        else:
            tmp = []

        if tmp:
            # a,b,_,_=list(zip(*tmp))
            for d in tmp:
                locs_pre.append(d[0])
                feats_pre.append(d[1].astype(float))
                labels_sur.append(d[5].astype(float))
                revs_sur.append(d[6])
                ori_coords_sur.append(d[7].astype(float))
        for i in range(frames - 1):
            pos = pre_frame - i - 1
            if pos < 0:
                coords, r = sparse_quantize(
                    np.array([[0, 0, 0]]),
                    features=np.array([[0,]]),
                    ignore_label=0,
                    quantization_size=scale,
                )
                locs_pre.append(coords)
                feats_pre.append(r.astype(float))
                labels_sur.append(np.copy(r.astype(float)))
                revs_sur.append(np.copy(r.astype(float)))
                ori_coords_sur.append(np.copy(r.astype(float)))
        locs.append(locs_pre)
        feats.append(feats_pre)
    feats = [[x[0].reshape(-1, 1), x[1].reshape(-1, 1)] for x in feats]
    feats = list(zip(*feats))
    locs = list(zip(*locs))

    a, b, c = sparse_collate(locs[0], feats[0], labels=labels_pre)
    ap = []
    bp = []
    if len(locs) > 1:
        ap, bp = list(
            zip(*[sparse_collate(locs[x], feats[x]) for x in range(1, frames)])
        )

    locs = [a]
    locs.extend(ap)
    feats = [b]
    feats.extend(bp)
    labels = c
    return {
        "x": [locs, feats],
        "y": labels.long(),
        "id": tbl,
        "ind": inds,
        "o_locs": o_locs,
        "o_labels": o_labels,
        "rev": revs,
        "rev_sur": revs_sur,
        "ori_locs": ori_coords,
        "ori_locs_sur": ori_coords_sur,
        "labels_sur": labels_sur,
    }


def valMerge(tbl):
    locs = []
    feats = []
    labels_pre = []
    labels_sur = []
    inds = []
    o_locs = []
    o_labels = []
    trees = []
    trees_sur = []
    revs = []
    revs_sur = []
    ori_coords = []
    ori_coords_sur = []
    # idx=[]
    for idx, i in enumerate(tbl):
        # torch.load(train[i])
        locs_pre = []
        feats_pre = []
        # labels_pre = []

        scans_b = val[i][1]
        labels_b = val[i][2]
        poss_b = val[i][3]
        pre_frame = len(scans_b) - 1
        min_c = None

        def get_data(frame_idx, mean=None):
            scan = np.fromfile(scans_b[frame_idx], dtype=np.float32)
            scan = scan.reshape((-1, 4))
            r = scan[:, 3] - np.min(scan[:, 3])  # np.ones_like(scan[:, 3] )#

            if frame_idx != pre_frame:
                scan[:, 3] = 1
                for f_pos in range(frame_idx + 1, len(scans_b)):
                    diff = np.matmul(inv(poss_b[f_pos]), poss_b[f_pos - 1])
                    scan = np.matmul(diff, scan.T).T
            coords = scan[:, :3]
            # coords = coords - coords.min(0)
            ori_coord = coords

            global min_c
            if frame_idx == pre_frame:
                min_c = coords.min(0)
            coords = coords - min_c

            r = r
            # coords = torch.from_numpy(coords).long()

            if True:#frame_idx == pre_frame:/home/dante0shy/dataset/KITTI/sequences
                tmp = labels_b[frame_idx]\
                    .replace('sequences', '{}/{}'.format(out_name, 'sequences'))\
                    .replace('labels','predictions')
                label = np.fromfile(
                    labels_b[frame_idx]
                    if (frame_idx == pre_frame)
                       or not  os.path.exists(tmp)
                    else tmp,
                    dtype=np.uint32)
                sem_label = label & 0xFFFF
                sem_label = np.vectorize(config["learning_map"].__getitem__)(sem_label)

                coords /= scale
                coords = coords.astype(np.int32)
                ind = sparse_quantize(
                    np.ascontiguousarray(coords),
                    features=np.ascontiguousarray(r.reshape(-1, 1)),
                    labels=np.ascontiguousarray(sem_label.astype(np.int32)),
                    ignore_label=0,
                    return_index=True,
                    return_inverse= True
                )  # ME.utils.                    quantization_size=scale,

                return (
                    ind[0],
                    ind[1],
                    ind[2],
                    ind[3],
                    coords,
                    sem_label,
                    ind[4],
                    ori_coord,
                )
            else:
                if coords.shape[0]:

                    coords, r = sparse_quantize(
                        np.ascontiguousarray(coords),
                        features=r.reshape(-1, 1),
                        ignore_label=0,
                        quantization_size=scale,
                    )
                    return coords, r.astype(np.float), [0], None, None, None, None
                else:
                    coords, r = sparse_quantize(
                        np.array([0, 0, 0]),
                        features=np.array([[0,]]),
                        ignore_label=0,
                        quantization_size=scale,
                    )
                    return coords, r, [0], None, None, None, None

        tmp_pre = get_data(pre_frame)
        locs_pre.append(tmp_pre[0])
        feats_pre.append(tmp_pre[1])
        labels_pre.append(tmp_pre[2])
        inds.append(tmp_pre[3])
        o_locs.append(tmp_pre[4])
        o_labels.append(tmp_pre[5])
        # trees.append(tmp_pre[6])
        revs.append(tmp_pre[6])
        ori_coords.append(tmp_pre[7])
        if pre_frame:
            tmp = list(map(lambda x: get_data(x, None), range(pre_frame)))
        else:
            tmp = []

        if tmp:
            # a,b,_,_=list(zip(*tmp))
            for d in tmp:
                locs_pre.append(d[0])
                feats_pre.append(d[1].astype(float))
                labels_sur.append(d[5].astype(float))
                revs_sur.append(d[6])
                ori_coords_sur.append(d[7].astype(float))
        for i in range(frames - 1):
            pos = pre_frame - i - 1
            if pos < 0:
                coords, r = sparse_quantize(
                    np.array([[0, 0, 0]]),
                    features=np.array([[0,]]),
                    ignore_label=0,
                    quantization_size=scale,
                )
                locs_pre.append(coords)
                feats_pre.append(r.astype(float))
                labels_sur.append(None)
                revs_sur.append(None)
                ori_coords_sur.append(None)
        locs.append(locs_pre)
        feats.append(feats_pre)
    feats = [[x[0].reshape(-1, 1), x[1].reshape(-1, 1)] for x in feats]
    feats = list(zip(*feats))
    locs = list(zip(*locs))

    a, b, c = sparse_collate(locs[0], feats[0], labels=labels_pre)
    ap = []
    bp = []
    if len(locs) > 1:
        ap, bp = list(
            zip(*[sparse_collate(locs[x], feats[x]) for x in range(1, frames)])
        )

    locs = [a]
    locs.extend(ap)
    feats = [b]
    feats.extend(bp)
    labels = c

    return {
        "x": [locs, feats],
        "y": labels.long(),
        "id": tbl,
        "ind": inds,
        "o_locs": o_locs,
        "o_labels": o_labels,
        "rev": revs,
        "rev_sur": revs_sur,
        "ori_locs": ori_coords,
        "ori_locs_sur": ori_coords_sur,
        "labels_sur": labels_sur,
    }


def get_val_data_loader():
    return torch.utils.data.DataLoader(
        list(range(len(val))),
        batch_size= 1,#batch_size//2 if batch_size//2 >0 else
        collate_fn=valMerge,
        num_workers=0,
        shuffle=False,
    )


# val_data_loader = torch.utils.data.DataLoader(
#     list(range(len(val))),batch_size=batch_size, collate_fn=valMerge, num_workers=0,shuffle=False)#batch_size


def get_train_data_loader():
    return torch.utils.data.DataLoader(
        list(range(len(train))),
        batch_size=batch_size,
        collate_fn=trainMerge,
        num_workers=5,
        shuffle=True,
    )
