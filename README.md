# XNAS

**XNAS** is an effective, modular and flexible neural architecture search (NAS) codebase, which aims to provide a common framework and baselines for the NAS community. 
<!-- It is written in [PyTorch](https://pytorch.org/) and . We hope **XNAS** become a zoo for NAS  -->
This project is now supported by PengCheng Lab

## Supported Algorithms

### Stable

### Beta

- DARTS
  - `python search/DARTS.py --cfg configs/search/DARTS.yaml`
- PCDARTS
  - `python search/PDARTS.py --cfg configs/search/PDARTS.yaml`
- PDARTS
  - `python search/PCDARTS.py --cfg configs/search/PCDARTS.yaml`
- SNG
- ASNG
- MDENAS
- DDPNAS
- MIGONAS
- GridSearch
- DrNAS
  - `python search/DrNAS/nb201space.py --cfg configs/search/DrNAS/nb201_cifar10_Dirichlet.yaml`
  - `python search/DrNAS/nb201space.py --cfg configs/search/DrNAS/nb201_cifar100_Dirichlet.yaml`
  - `python search/DrNAS/DARTSspace.py --cfg configs/search/DrNAS/DARTS_cifar10.yaml`
- TENAS
  - `python search/TENAS.py --cfg configs/search/TENAS/nb201_cifar10.yaml`
- RMINAS
  - `./search/RMINAS/download_weight.sh # prepare weights of teacher models`
  - `./python search/RMINAS/RMINAS_nb201.py --cfg configs/search/RMINAS/nb201_cifar10.yaml`
  - `./python search/RMINAS/RMINAS_darts.py --cfg configs/search/RMINAS/darts_cifar10.yaml`
- DROPNAS
  - `./python search/DropNAS.py --cfg configs/search/DROPNAS.yaml`

## Supported Search Spaces

### Stable

### Beta

#### Cell-based Search Space

- NAS-Bench-1Shot1
- DARTS
- NAS-Bench-201

#### Chain-structured Search Space

- MobileNetV3
- OFA
- ProxylessNAS

--- 

## Installation

```bash
git clone https://git.openi.org.cn/PCL_AutoML/XNAS
cd XNAS
# set root path
export PYTHONPATH=$PYTHONPATH:/Path/to/XNAS
```

---

## Usage Examples

```bash
# set gpu devices. Multiple GPUs are under test and may cause errors now.
export CUDA_VISIBLE_DEVICES=0
# unit test example
python tools/test_func/sng_function_optimization.py
# train example
python train/DARTS_train.py --cfg configs/search/darts.yaml
# replace config example
python train/DARTS_train.py --cfg configs/search/darts.yaml OUT_DIR /username/project/XNAS/experiment/darts/test1
```

---

## Experiment Results

### DARTS Search Space 
On cifar10, the network is trained by using the default training set of [pt.darts](https://github.com/zhengxiawu/pytorch_cls/tree/master/pytorch-cifar-v2).

We reimplement several widely used NAS methods including:

* Darts, [paper](https://arxiv.org/abs/1806.09055), [official_code](https://github.com/quark0/darts)

### Results on CIFAR10

|Method |Trial|params(M)|search(hrs)|train(hrs)|Top1 |Flops(M)|download|Search Top1 |Search Space|
| ------|:---:|:-------:|:--------: |:-------: |:---:|:---:   |:---:   |:---:       |:---:       |
| darts |1    |4.39     |21         |39        |96.97|689.335 |-       |90.32       |cell-based  |
| darts |2    |4.25     |26.36      |39        |97.31|680.073 |-       |90.47       |cell-based  |
| darts |3    |4.450    |27.63      |39.7      |97.32|708.468 |-       |90.09       |cell-based  |
| darts |4    |4.467    |21.48      |48        |97.39|717.454 |-       |90.52       |cell-based  |
|paper  |-    |3.3      |96         |-         |97.24|-       |-       |-           |cell-based  |
| sng   |1    |3.042    |2.5        |33.45     |96.87|506.002 |-       |87.52       |cell-based  |
| sng   |2    |2.477    |3.0        |26.62     |96.73|397.068 |-       |87.81       |cell-based  |
| sng   |3    |2.087    |3.0        |21.75     |96.56|339.201 |-       |87.00       |cell-based  |
| sng   |4    |3.230    |2.5        |27.47     |97.30|509.071 |-       |88.51       |cell-based  |
| asng  |1    |2.001    |2.5        |18.98     |96.61|330.575 |-       |85.78       |cell-based  |
| asng  |2    |2.749    |2.5        |25.66     |96.48|450.153 |-       |87.47       |cell-based  |
| asng  |3    |2.991    |2.5        |27.88     |97.31|476.695 |-       |85.52       |cell-based  |
| asng  |4    |2.189    |2.5        |23.88     |96.55|350.647 |-       |86.41       |cell-based  |
| MIGO  |1    |3.266    |1.5        |28.20     |97.35|531.217 |-       |84.75       |cell-based  |
| MIGO  |2    |3.274    |1.5        |26.33     |97.41|523.973 |-       |84.61       |cell-based  |
| MIGO  |3    |2.848    |1.5        |25.91     |97.36|451.480 |-       |84.89       |cell-based  |
| MIGO  |4    |2.749    |1.5        |30.19     |97.28|439.619 |-       |84.44       |cell-based  |
| pcdarts(official)|1   |4.052    |3.61       |41.28     |97.20|638.823 |        |85.296      |cell-based  |
| pcdarts(official)|2   |3.247    |3.6        |27.96     |97.23|512.444 |        |84.552      |cell-based  |
| pcdarts(official)|3   |4.368    |3.63       |38.68     |97.25|688.561 |        |84.792      |cell-based  |
| pcdarts(official)|4   |4.148    |3.16       |34.58     |97.49|649.108 |        |85.280      |cell-based  |
| xnas-pcdarts     |  1 | 3.779   |    3.46   |    31.9  |97.55|595.498 |        | 85.192     | cell-based |
| xnas-pcdarts     |  2 | 3.641   |    3.03   |   33.08  |96.92|573.933 |        | 85.036     | cell-based |
| xnas-pcdarts     |  3 | 4.536   |    3.02   |   41.18  |97.37|722.790 |        | 84.592     | cell-based |
| xnas-pcdarts     |  4 | 3.143   |    3.48   |   32.26  |97.25|505.227 |        | 85.088     | cell-based |
| pdarts(official) |1   |4.052    |3.50       |-         |97.41|555.270 |        |-           |cell-based  |
| pdarts(official) |2   |3.247    |3.31       |-         |97.25|529.419 |        |-           |cell-based  |
| pdarts(official) |3   |4.368    |3.39       |-         |97.25|545.732 |        |-           |cell-based  |
| pdarts(official) |4   |4.148    |4.08       |-         |97.29|642.555 |        |-           |cell-based  |
| dynamic_ASNG     |1   |2.901    |0.0        |31.77     |96.86|465.193 |-       |78.65       |cell-based  |
| dynamic_ASNG     |2   |2.208    |0.0        |18.00     |96.78|351.145 |-       |79.2        |cell-based  |
| dynamic_ASNG     |3   |2.365    |0.0        |19.93     |96.20|387.364 |-       |79.87       |cell-based  |
| dynamic_ASNG     |4   |3.466    |0.0        |31.35     |97.11|565.058 |-       |79.87       |cell-based  |
| dynamic_SNG      |1   |2.245    |0.0        |23.98     |96.28|352.693 |-       |78.95       |cell-based  |
| dynamic_SNG      |2   |2.927    |0.0        |24.13     |96.87|473.156 |-       |78.07       |cell-based  |
| dynamic_SNG      |3   |2.724    |0.0        |28.07     |97.45|442.826 |-       |77.68       |cell-based  |
| dynamic_SNG      |4   |3.323    |0.0        |31.85     |96.65|528.784 |-       |79.78       |cell-based  |
| RMINAS           |-   |-        |1.92       |31.9      |97.36|-       |-       |-           |cell-based  |

### TODO

1. data parallel support
2. fix Nvidia DALI backend support
3. add **kwargs for space_builder
4. test code for imagenet

## BibTex
```
@inproceedings{zheng2022rminas,
  title={Neural Architecture Search with Representation Mutual Information},
  author={Xiawu Zheng, Xiang Fei, Lei Zhang, Chenglin Wu, Fei Chao, Jianzhuang Liu, Wei Zeng, Yonghong Tian, Rongrong Ji},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
@article{zheng2021migo,
  title={MIGO-NAS: Towards fast and generalizable neural architecture search},
  author={Zheng, Xiawu and Ji, Rongrong and Chen, Yuhang and Wang, Qiang and Zhang, Baochang and Chen, Jie and Ye, Qixiang and Huang, Feiyue and Tian, Yonghong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
@inproceedings{zheng2020rethinking,
  title={Rethinking performance estimation in neural architecture search},
  author={Zheng, Xiawu and Ji, Rongrong and Wang, Qiang and Ye, Qixiang and Li, Zhenguo and Tian, Yonghong and Tian, Qi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11356--11365},
  year={2020}
}
```