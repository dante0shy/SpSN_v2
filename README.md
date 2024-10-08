# Learning Temporal Variations for 4D Point Cloud Segmentation

PyTorch implementation of the paper [Learning Temporal Variations for 4D Point Cloud Segmentation](https://link.springer.com/article/10.1007/s11263-024-02149-w), which is the extension of our previous paper 'SpSequenceNet: Semantic Segmentation Network on 4D Point Clouds'.

## Dependencies

You can follow the Installation of [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)

## Scripts

Training:

Firstly, the dataset setting is in the ```val_base``` and ```data_base``` in the dataset code, and ```log_pos``` in the train or val code.
Modify it to the direction of your own dataset.
Secondly, run as following:

```
# for main network training
cd train/mf_v1
python unet_v1_34.py

# for TVPR training
cd train/mf_v1
python unet_add_v1_34.py
```

Evaluation:

If you are validating your own trainined model, change ```main_snap```and ```add_snap``` for the path of your ckpt,and run as following:

```
cd train/mf_v1
python val_add_34.py
```


Our trained model is in [here](https://drive.google.com/drive/folders/1GBA4gCi9_dylUzQNoS8XTveqf_uww2im?usp=sharing)


