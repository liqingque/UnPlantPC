# UnPlantPC: Unsupervised Plant Point Cloud Completion Based on KeyPoint Sampling and Region-Aware Contrastive Chamfer Distance Loss
The official implementation of the paper：
 UnPlantPC: Unsupervised Plant Point Cloud Completion Based on KeyPoint Sampling and Region-Aware Contrastive Chamfer Distance Loss

Contact: xiaomengli@cau.edu.cn Any questions or discussion are welcome!

-----
+ [2025.02.14] We have uploaded the dataset, which can be downloaded from the following link: https://drive.google.com/file/d/1aJSs3iaMcyP2C7Hkzo98SDxwBytxeigB/view?usp=sharing.

+ [2025.02.24] We have initialized the repo. The related resources will be released after the manuscript is accepted.


<img src="PlantPCom.png" alt="Dataset" width="800" height="600">




## Abstract
In smart agriculture, the precise acquisition of complete 3D plant phenotypic data is critical for applications such as intelligent breeding and growth monitoring. However, due to equipment constraints, environmental noise, and self-occlusion, the collected 3D point cloud data of plants is often incomplete. This incompleteness significantly hinders key tasks in plant phenotypic analysis, including organ segmentation and surface reconstruction, necessitating effective data completion methods.
Supervised point cloud completion methods face challenges due to the inherent incompleteness of collected data and the need for extensive labeled datasets. To address these issues, we propose UnPlantPC, an unsupervised plant point cloud completion model built on a self-supervised encoder-decoder paradigm. To effectively capture regions with complex geometric structures in plant point clouds, the model employs a keypoint down-sampling strategy that integrates Euclidean and cosine distances, ensuring the extracted key points are both representative and directionally informative. Additionally, a geometric-aware attention module enhances feature extraction in these regions, further improving the model’s ability to capture intricate geometric details. To align plant point cloud distributions under self-supervised learning, we introduce a novel Region-Aware Contrastive Distance (RCCD), which provides accurate supervisory information. This innovation enables the model to deliver more precise completion results.
UnPlantPC demonstrates state-of-the-art performance across several metrics on the PlantPCom dataset, achieving a notable 20.67\% improvement in CDL$_2$ compared to existing models.

## Contributions
1. We propose a high-quality plant 3D point cloud completion dataset, PlantPCom, addressing the scarcity of suitable datasets for this domain.
   
2. We present UnPlantPC, the first self-supervised framework capable of learning to complete point clouds using only partial plant point clouds, without requiring full annotations.

3. To tackle the complex edge shapes characteristic of plant point clouds, we design a feature extraction network based on Keypoint Sampling and Geometric-Aware Attention (KSGA). This module focuses on sampling and extracting features from key points.
4. Inspired by contrastive learning, we introduce a novel Regional Contrastive Chamfer Distance loss (RCCD) to align complex plant point cloud distributions effectively, enabling robust supervision.
5. Our model achieves state-of-the-art (SOTA) performance across three metrics on the PlantPCom dataset. Notably, it improves the critical $CDL_2$ metric by 20.67\% compared to the previous best model.

`

## Usage

### Requirements

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
```
The solution for a common bug in chamfer distance installation can be found in Issue [#6](https://github.com/yuxumin/PoinTr/issues/6)
```
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Note: If you still get `ModuleNotFoundError: No module named 'gridding'` or something similar then run these steps

```
    1. cd into extensions/Module (eg extensions/gridding)
    2. run `python setup.py install`
```

That will fix the `ModuleNotFoundError`.




### Inference

To inference sample(s) with pretrained model

```
python tools/inference.py \
${POINTR_CONFIG_FILE} ${POINTR_CHECKPOINT_FILE} \
[--pc_root <path> or --pc <file>] \
[--save_vis_img] \
[--out_pc_root <dir>] \
```


### Evaluation

To evaluate a pre-trained PoinTr model on the Three Dataset with single GPU, run:

```
bash ./scripts/test.sh <GPU_IDS>  \
    --ckpts <path> \
    --config <config> \
    --exp_name <name> \
    [--mode <easy/median/hard>]
```


### Training

To train a point cloud completion model from scratch, run:

```
# Use DistributedDataParallel (DDP)
bash ./scripts/dist_train.sh <NUM_GPU> <port> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
# or just use DataParallel (DP)
bash ./scripts/train.sh <GPUIDS> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
```
## Acknowledgement
A large part of the code is borrowed from [Anchorformer](https://github.com/chenzhik/AnchorFormer), [PoinTr](https://github.com/ifzhang/ByteTrack), [P2C] Thanks for their wonderful works!

## Citation
The related resources will be released after the manuscript is accepted. 
