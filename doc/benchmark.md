# Benchmark

## Introduction

This file documents a collection of baselines searched neural architectures on different search spaces and datasets. On cifar10, the network is trained by using the default training set of [pt.darts](https://github.com/zhengxiawu/pytorch_cls/tree/master/pytorch-cifar-v2).

We reimplement several widely used NAS methods including:

* Darts, [paper](https://arxiv.org/abs/1806.09055), [official_code](https://github.com/quark0/darts)

### Results on CIFAR10

|Method |Seed |params(M)|search(hrs)|train(hrs)|Top1 |Flops(M)|download|
| ------|:---:|:-------:|:--------: |:-------: |:---:|:---:   |:---:   |
| darts |1    |4.39     |21         |39        |96.97|689.335 |-       |
| darts |2    |4.25     |26.36      |39        |97.31|680.073 |-       |
| darts |3    |4.450    |27.63      |39.7      |97.32|708.468 |-       |
| darts |4    |4.467    |21.48      |48        |97.39|717.454 |-       |
|paper  |-    |3.3      |96         |-         |97.24|-       |-       |
