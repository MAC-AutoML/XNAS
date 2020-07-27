# Benchmark

## Introduction

This file documents a collection of baselines searched neural architectures on different search spaces and datasets. On cifar10, the network is trained by using the default training set of [pt.darts](https://github.com/zhengxiawu/pytorch_cls/tree/master/pytorch-cifar-v2).

We reimplement several widely used NAS methods including:

* Darts, [paper](https://arxiv.org/abs/1806.09055), [official_code](https://github.com/quark0/darts)

### Results on CIFAR10

|Method |Seed |params(M)|search(hrs)|train(hrs)|Top1 |Flops(M)|download|Search Top1 |Search Space|
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
| dynamic_ASNG   |1    |2.901    |0.0        |31.77     |96.86|465.193 |-       |78.65       |cell-based  |
| dynamic_ASNG   |2    |2.208    |0.0        |18.00     |96.78|351.145 |-       |79.2        |cell-based  |
| dynamic_ASNG   |3    |2.365    |0.0        |19.93     |96.20|387.364 |-       |79.87       |cell-based  |
| dynamic_ASNG   |4    |3.466    |0.0        |31.35     |97.11|565.058 |-       |79.87       |cell-based  |
| dynamic_SNG   |1    |2.245    |0.0        |23.98     |96.28|352.693 |-       |78.95       |cell-based  |
| dynamic_SNG   |2    |2.927    |0.0        |24.13     |96.87|473.156 |-       |78.07       |cell-based  |
| dynamic_SNG   |3    |2.724    |0.0        |28.07     |97.45|442.826 |-       |77.68       |cell-based  |
| dynamic_SNG   |4    |3.323    |0.0        |31.85     |96.65|528.784 |-       |79.78       |cell-based  |
