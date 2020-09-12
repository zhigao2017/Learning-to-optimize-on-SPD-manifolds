# Learning-to-optimize-on-SPD-manifolds

This repository is the implementation of CVPR 2020 paper: "Learning to Optimize on SPD Manifolds".

We provide the code about the clustering task on the Kylberg texture dataset.

Prerequisites
-------
Our code requires PyTorch v1.0 and Python 3.

Training our model.
-------

We train the optimizer by
```
python train.py
```

You can modify ```config.py``` to set more detailed hyper-parameters.

The trained optimizer is saved in the ```snapshot``` folder.


Evaluate our model.
-------

We evaluate the optimizer by
```
python evaluation.py
```

You can modify ```config_evaluation.py``` to set more detailed hyper-parameters.


Reference.
-------

If this code is helpful, we'd appreciate it if you could cite our paper

```
@InProceedings{Gao_2020_CVPR,
author = {Gao, Zhi and Wu, Yuwei and Jia, Yunde and Harandi, Mehrtash},
title = {Learning to Optimize on SPD Manifolds},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

```
