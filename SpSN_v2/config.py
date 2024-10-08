# model
block_reps = 1
residual_blocks = True
m = 16  # 16 or 32

# data
downsample = 0.6
dimension = 3
full_scale = 2048  # Input field size
frames = 2
scale = [1 / 20, 1 / 20, 1 / 20]  # Voxel size = 1/scale
shift_scale = [1 / 20, 1 / 20, 1 / 20]
batch_size = 16

val_reps = 1  # Number of test views, 1 or more
CLASS_LABELS_DICT = {
    "single": [
        "unlabeled",
        "car",
        "bicycle",
        "motorcycle",
        "truck",
        "other-vehicle",
        "person",
        "bicyclist",
        "motorcyclist",
        "road",
        "parking",
        "sidewalk",
        "other-ground",
        "building",
        "fence",
        "vegetation",
        "trunk",
        "terrain",
        "pole",
        "traffic-sign",
    ],
    "muti": [
        "unlabeled",
        "car",
        "bicycle",
        "motorcycle",
        "truck",
        "other-vehicle",
        "person",
        "bicyclist",
        "motorcyclist",
        "road",
        "parking",
        "sidewalk",
        "other-ground",
        "building",
        "fence",
        "vegetation",
        "trunk",
        "terrain",
        "pole",
        "traffic-sign",
        "moving-car",
        "moving-bicyclist",
        "moving-person",
        "moving-motorcyclist",
        "moving-other-vehicle",
        "moving-truck",
    ],
}
CLASS_LABELS = CLASS_LABELS_DICT["muti"]
# CLASS_LABELS = CLASS_LABELS_DICT['single']
# CLASS_LABELS = ["unlabeled", "car","bicycle","motorcycle","truck","other-vehicle","person","bicyclist","motorcyclist","road","parking","sidewalk","other-ground","building","fence",
#                 "vegetation","trunk","terrain","pole","traffic-sign","moving-car","moving-bicyclist","moving-person","moving-motorcyclist","moving-other-vehicle","moving-truck"]

UNKNOWN_ID = 0
N_CLASSES = len(CLASS_LABELS)
