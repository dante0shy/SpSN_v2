# Learning Temporal Variations for 4D Point Cloud Segmentation

PyTorch implementation of the paper [Learning Temporal Variations for 4D Point Cloud Segmentation](https://link.springer.com/article/10.1007/s11263-024-02149-w), which is the extension of our previous paper 'SpSequenceNet: Semantic Segmentation Network on 4D Point Clouds'.

## Dependencies

You can follow the Installation of [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)

## Scripts

Training:

Firstly, the dataset setting is in the data_base and val_base of config.yaml.
Modify it to the direction of your own dataset.
Secondly, run as following:

```
cd train/mf_v1
python unet_add_v1_34.py
```

Evaluation:

If you are validating your own trainined model, change ```main_snap```and ```add_snap``` for the path of your ckpt,and run as following:

```
cd train/mf_v1
python val_add_34.py
```

If you want to use our trained model, add 'val_model_dir' under 'model' in the config.yaml.
The val_model_dir is the directory of your model.

Our trained model is in [here](https://drive.google.com/file/d/1mgOg9bsozfiXxc5EhtpVAbIcg1BXAbDu/view?usp=sharing)


